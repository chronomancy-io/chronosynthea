//! End-to-end streaming bench: generate patients in chunks via
//! `BatchGenerator::generate_full_chunked`, emit Parquet on the
//! way, never hold more than `chunk_size` patients in memory.
//!
//! Memory peak ≈ chunk_size × ~24 KB/patient. At chunk_size = 10_000,
//! peaks at ~240 MB regardless of total patient count, so a 10M-patient
//! run fits in 240 MB instead of the ~240 GB the in-memory path would
//! need. This is the bench that proves Parquet at population scale.
//!
//! Build/run:
//!     cargo run --release --features parquet --bin parquet_stream_bench -- 1000000

#[cfg(not(feature = "parquet"))]
fn main() {
    eprintln!("parquet_stream_bench requires --features parquet");
    std::process::exit(2);
}

#[cfg(feature = "parquet")]
fn main() {
    use chronosynthea_mss::parquet_writer::{
        SyntheaParquetFullWriter, SyntheaParquetWriter,
        SyntheaStatsParquetWriter,
    };
    use chronosynthea_mss::reproducibility::CohortManifest;
    use chronosynthea_mss::{BatchConfig, BatchGenerator, CalibratedRegistry};
    use std::path::PathBuf;
    use std::time::Instant;

    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let chunk_size: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.push("data");
    p.push("prevalence");
    p.push("calibrated_registry.json");
    if !p.exists() {
        eprintln!("registry not found at {}", p.display());
        std::process::exit(1);
    }
    let registry = CalibratedRegistry::load(&p).unwrap();
    let fingerprint = registry.to_fingerprint();
    let seed: u64 = 42;
    let config = BatchConfig {
        seed,
        ..Default::default()
    };
    let generator = BatchGenerator::new(fingerprint, config);

    let bench_root: PathBuf = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache/chronosynthea-bench");

    eprintln!(
        "[stream] n={}  chunk_size={}  output dir={}  (real NVMe)",
        n,
        chunk_size,
        bench_root.display()
    );
    eprintln!(
        "[stream] peak RAM ≈ {:.0} MB ({} patients × ~24 KB)",
        chunk_size as f64 * 24.0 / 1000.0,
        chunk_size
    );

    println!();
    println!(
        "{:>16}  {:>10}  {:>11}  {:>10}  {:>10}",
        "variant", "wall (s)", "pat/sec", "size (MB)", "bytes/pat"
    );
    println!("{}", "-".repeat(67));

    let fp_hash = *generator.fingerprint_hash();

    // ---- Parquet full (patients.parquet) ----
    {
        let dir = bench_root.join("stream-parquet-full");
        let _ = std::fs::remove_dir_all(&dir);
        let mut w = SyntheaParquetWriter::create(&dir).unwrap();
        let t = Instant::now();
        generator.generate_full_chunked(n, chunk_size, |chunk| {
            for pat in &chunk {
                w.write_patient(pat).unwrap();
            }
        });
        w.finish().unwrap();
        let dt = t.elapsed();
        let path = dir.join("parquet/patients.parquet");
        let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let mut manifest =
            CohortManifest::new(&fp_hash, seed, n, "parquet-full");
        manifest.output_bytes = bytes;
        manifest.output_sha256 = Some(sha256_of_file(&path));
        manifest
            .write_json(dir.join("parquet/manifest.json"))
            .unwrap();
        println!(
            "{:>16}  {:>10.3}  {:>11.0}  {:>10.2}  {:>10.0}",
            "parquet full",
            dt.as_secs_f64(),
            n as f64 / dt.as_secs_f64(),
            bytes as f64 / 1e6,
            bytes as f64 / n as f64,
        );
    }

    // ---- Parquet slim ----
    {
        let dir = bench_root.join("stream-parquet-slim");
        let _ = std::fs::remove_dir_all(&dir);
        let mut w = SyntheaParquetWriter::create_slim(&dir).unwrap();
        let t = Instant::now();
        generator.generate_full_chunked(n, chunk_size, |chunk| {
            for pat in &chunk {
                w.write_patient(pat).unwrap();
            }
        });
        w.finish().unwrap();
        let dt = t.elapsed();
        let bytes = std::fs::metadata(dir.join("parquet/patients.parquet"))
            .map(|m| m.len())
            .unwrap_or(0);
        println!(
            "{:>16}  {:>10.3}  {:>11.0}  {:>10.2}  {:>10.0}",
            "parquet slim",
            dt.as_secs_f64(),
            n as f64 / dt.as_secs_f64(),
            bytes as f64 / 1e6,
            bytes as f64 / n as f64,
        );
    }

    // ---- Parquet 6-file (patients + encounters + conditions + obs + meds + procs) ----
    {
        let dir = bench_root.join("stream-parquet-6file");
        let _ = std::fs::remove_dir_all(&dir);
        let mut w = SyntheaParquetFullWriter::create(&dir).unwrap();
        let archetypes = generator.archetypes();
        let code_table = generator.code_table();
        let t = Instant::now();
        generator.generate_full_chunked(n, chunk_size, |chunk| {
            for pat in &chunk {
                w.write_patient(pat, archetypes, code_table).unwrap();
            }
        });
        w.finish().unwrap();
        let dt = t.elapsed();
        let bytes = dir_size_recursive(&dir.join("parquet"));
        println!(
            "{:>16}  {:>10.3}  {:>11.0}  {:>10.2}  {:>10.0}",
            "parquet 6-file",
            dt.as_secs_f64(),
            n as f64 / dt.as_secs_f64(),
            bytes as f64 / 1e6,
            bytes as f64 / n as f64,
        );
    }

    // ---- Parquet stats (summary.parquet) ----
    {
        let dir = bench_root.join("stream-parquet-stats");
        let _ = std::fs::remove_dir_all(&dir);
        let mut w = SyntheaStatsParquetWriter::create(&dir).unwrap();
        let t = Instant::now();
        generator.generate_full_chunked(n, chunk_size, |chunk| {
            for pat in &chunk {
                w.write_patient(pat).unwrap();
            }
        });
        w.finish().unwrap();
        let dt = t.elapsed();
        let bytes = std::fs::metadata(dir.join("parquet/summary.parquet"))
            .map(|m| m.len())
            .unwrap_or(0);
        println!(
            "{:>16}  {:>10.3}  {:>11.0}  {:>10.2}  {:>10.0}",
            "parquet stats",
            dt.as_secs_f64(),
            n as f64 / dt.as_secs_f64(),
            bytes as f64 / 1e6,
            bytes as f64 / n as f64,
        );
    }

    println!();
    println!("(end-to-end wall time: generate + write streamed in chunks.)");
}

#[cfg(feature = "parquet")]
fn sha256_of_file(p: &std::path::Path) -> String {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut hasher = Sha256::new();
    if let Ok(mut f) = std::fs::File::open(p) {
        let mut buf = [0u8; 1 << 16];
        loop {
            match f.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => hasher.update(&buf[..n]),
            }
        }
    }
    let h: [u8; 32] = hasher.finalize().into();
    let mut s = String::with_capacity(64);
    for b in &h {
        use std::fmt::Write;
        write!(&mut s, "{:02x}", b).unwrap();
    }
    s
}

#[cfg(feature = "parquet")]
fn dir_size_recursive(p: &std::path::Path) -> u64 {
    use std::fs;
    let mut total = 0u64;
    let Ok(entries) = fs::read_dir(p) else { return 0 };
    for e in entries.flatten() {
        if let Ok(meta) = e.metadata() {
            total += if meta.is_file() {
                meta.len()
            } else if meta.is_dir() {
                dir_size_recursive(&e.path())
            } else {
                0
            };
        }
    }
    total
}
