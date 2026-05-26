//! CSV vs Parquet (full / slim / stats-only) throughput + size bench.
//!
//! Writes everything under `~/.cache/chronosynthea-bench/` on real
//! NVMe (never `/tmp` — that's tmpfs and fills fast).
//!
//! Build/run:
//!     cargo run --release --features parquet --bin parquet_bench -- 10000

#[cfg(not(feature = "parquet"))]
fn main() {
    eprintln!("parquet_bench requires --features parquet");
    std::process::exit(2);
}

#[cfg(feature = "parquet")]
fn main() {
    use chronosynthea_mss::parquet_writer::{
        SyntheaParquetWriter, SyntheaStatsParquetWriter,
    };
    use chronosynthea_mss::{
        patient_uuid, BatchConfig, BatchGenerator, CalibratedRegistry, SyntheaCsvWriter,
    };
    use std::path::PathBuf;
    use std::time::Instant;

    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    let skip_csv =
        std::env::args().any(|a| a == "--parquet-only") || n > 50_000;

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
    let config = BatchConfig {
        seed: 42,
        ..Default::default()
    };
    let generator = BatchGenerator::new(fingerprint, config);

    let t_gen = Instant::now();
    let patients = generator.generate_full(n);
    eprintln!(
        "[gen] {} patients in {:.3}ms",
        n,
        t_gen.elapsed().as_secs_f64() * 1000.0
    );

    // Bench output root: ~/.cache/chronosynthea-bench/ on real NVMe.
    let bench_root: PathBuf = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache/chronosynthea-bench");

    let archetypes = generator.archetypes();
    let code_table = generator.code_table();

    println!();
    println!(
        "{:>16}  {:>10}  {:>10}  {:>10}  {:>10}",
        "variant", "wall (s)", "pat/sec", "size (MB)", "bytes/pat"
    );
    println!("{}", "-".repeat(64));

    // CSV — full 15-file write via the existing writer.
    if !skip_csv {
        let dir = bench_root.join("csv-full");
        let _ = std::fs::remove_dir_all(&dir);
        let mut w = SyntheaCsvWriter::create(&dir).unwrap();
        let t = Instant::now();
        w.write_patients_parallel(&patients, archetypes, code_table)
            .unwrap();
        w.flush().unwrap();
        let dt = t.elapsed();
        let bytes = dir_size(&dir.join("csv"));
        println!(
            "{:>16}  {:>10.3}  {:>10.0}  {:>10.2}  {:>10.0}",
            "csv (15 files)",
            dt.as_secs_f64(),
            n as f64 / dt.as_secs_f64(),
            bytes as f64 / 1e6,
            bytes as f64 / n as f64,
        );
    } else {
        println!(
            "{:>16}  {:>10}  {:>10}  {:>10}  {:>10}",
            "csv (15 files)", "SKIP", "SKIP", "SKIP", "SKIP"
        );
    }

    // Parquet full — patients.parquet only.
    let dir = bench_root.join("parquet-full");
    let _ = std::fs::remove_dir_all(&dir);
    let mut w = SyntheaParquetWriter::create(&dir).unwrap();
    let t = Instant::now();
    for pat in &patients {
        w.write_patient(pat).unwrap();
    }
    w.finish().unwrap();
    let dt = t.elapsed();
    let bytes = std::fs::metadata(dir.join("parquet/patients.parquet"))
        .map(|m| m.len())
        .unwrap_or(0);
    println!(
        "{:>16}  {:>10.3}  {:>10.0}  {:>10.2}  {:>10.0}",
        "parquet full",
        dt.as_secs_f64(),
        n as f64 / dt.as_secs_f64(),
        bytes as f64 / 1e6,
        bytes as f64 / n as f64,
    );

    // Parquet slim — patients.parquet with PII columns dropped.
    let dir = bench_root.join("parquet-slim");
    let _ = std::fs::remove_dir_all(&dir);
    let mut w = SyntheaParquetWriter::create_slim(&dir).unwrap();
    let t = Instant::now();
    for pat in &patients {
        w.write_patient(pat).unwrap();
    }
    w.finish().unwrap();
    let dt = t.elapsed();
    let bytes = std::fs::metadata(dir.join("parquet/patients.parquet"))
        .map(|m| m.len())
        .unwrap_or(0);
    println!(
        "{:>16}  {:>10.3}  {:>10.0}  {:>10.2}  {:>10.0}",
        "parquet slim",
        dt.as_secs_f64(),
        n as f64 / dt.as_secs_f64(),
        bytes as f64 / 1e6,
        bytes as f64 / n as f64,
    );

    // Parquet stats — one summary row per patient, no event-level data.
    let dir = bench_root.join("parquet-stats");
    let _ = std::fs::remove_dir_all(&dir);
    let mut w = SyntheaStatsParquetWriter::create(&dir).unwrap();
    let t = Instant::now();
    for pat in &patients {
        w.write_patient(pat).unwrap();
    }
    w.finish().unwrap();
    let dt = t.elapsed();
    let bytes = std::fs::metadata(dir.join("parquet/summary.parquet"))
        .map(|m| m.len())
        .unwrap_or(0);
    println!(
        "{:>16}  {:>10.3}  {:>10.0}  {:>10.2}  {:>10.0}",
        "parquet stats",
        dt.as_secs_f64(),
        n as f64 / dt.as_secs_f64(),
        bytes as f64 / 1e6,
        bytes as f64 / n as f64,
    );

    println!();
    println!(
        "(disk: ~/.cache/chronosynthea-bench/ on real NVMe — \
         /dev/nvme0n1p2, not tmpfs.)"
    );
}

#[cfg(feature = "parquet")]
fn dir_size(p: &std::path::Path) -> u64 {
    use std::fs;
    fs::read_dir(p)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}
