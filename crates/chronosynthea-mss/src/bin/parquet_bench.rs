//! CSV vs Parquet patient-output proof-of-concept.
//!
//! Generates N patients, writes patients.csv (existing path) AND
//! patients.parquet (new path), reports throughput + on-disk size.
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
    use chronosynthea_mss::parquet_writer::SyntheaParquetWriter;
    use chronosynthea_mss::{
        patient_uuid, BatchConfig, BatchGenerator, CalibratedRegistry, SyntheaCsvWriter,
    };
    use std::path::PathBuf;
    use std::time::Instant;

    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    // For 100K+ patients the CSV path's 58+ GB of output won't fit in
    // tmpfs. Skip it; just measure Parquet at scale.
    let skip_csv = std::env::args().any(|a| a == "--parquet-only") || n > 50_000;

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

    let archetypes = generator.archetypes();
    let code_table = generator.code_table();

    let mut csv_patients_size = 0u64;
    // Default bench output root: ~/.cache/chronosynthea-bench/ (real
    // disk). NOT /tmp — that's a tmpfs that fills fast on multi-GB
    // CSV runs and bricks the rest of the machine.
    let bench_root: PathBuf = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache/chronosynthea-bench");

    if !skip_csv {
        // CSV path — patient.csv via the existing 15-file writer.
        let csv_dir = bench_root.join("csv");
        let _ = std::fs::remove_dir_all(&csv_dir);
        let mut csv_writer = SyntheaCsvWriter::create(&csv_dir).unwrap();
        let t_csv = Instant::now();
        for p in &patients {
            let uuid = patient_uuid(p.id);
            csv_writer
                .write_patient(p, &uuid, archetypes, code_table)
                .unwrap();
        }
        csv_writer.flush().unwrap();
        let dt_csv = t_csv.elapsed();
        csv_patients_size = std::fs::metadata(csv_dir.join("csv/patients.csv"))
            .map(|m| m.len())
            .unwrap_or(0);
        eprintln!(
            "[csv]     {} patients in {:.3}s = {:.0}/s (FULL 15-file write)",
            n,
            dt_csv.as_secs_f64(),
            n as f64 / dt_csv.as_secs_f64()
        );
        eprintln!(
            "[csv]     patients.csv on disk: {:.2} MB ({:.2} KB/patient)",
            csv_patients_size as f64 / 1e6,
            csv_patients_size as f64 / 1000.0 / n as f64
        );
    } else {
        eprintln!("[csv]     SKIPPED (n={} would overflow tmpfs)", n);
    }
    let _ = archetypes;
    let _ = code_table;

    // Parquet path — patients.parquet only.
    let parquet_dir = bench_root.join("parquet");
    let _ = std::fs::remove_dir_all(&parquet_dir);
    let mut pw = SyntheaParquetWriter::create(&parquet_dir).unwrap();
    let t_parquet = Instant::now();
    for p in &patients {
        pw.write_patient(p).unwrap();
    }
    pw.finish().unwrap();
    let dt_parquet = t_parquet.elapsed();
    let parquet_patients_size =
        std::fs::metadata(parquet_dir.join("parquet/patients.parquet"))
            .map(|m| m.len())
            .unwrap_or(0);
    eprintln!(
        "[parquet] {} patients in {:.3}s = {:.0}/s (patients.parquet only)",
        n,
        dt_parquet.as_secs_f64(),
        n as f64 / dt_parquet.as_secs_f64()
    );
    eprintln!(
        "[parquet] patients.parquet on disk: {:.2} MB ({:.2} KB/patient)",
        parquet_patients_size as f64 / 1e6,
        parquet_patients_size as f64 / 1000.0 / n as f64
    );

    if parquet_patients_size > 0 && csv_patients_size > 0 {
        let ratio = csv_patients_size as f64 / parquet_patients_size as f64;
        eprintln!(
            "[ratio]   parquet is {:.2}× smaller than patients.csv \
             (would be larger across all 15 files since claims_transactions, \
             observations, etc dictionary-encode even better)",
            ratio
        );
    }
}
