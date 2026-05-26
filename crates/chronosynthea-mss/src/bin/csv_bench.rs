//! Standalone CSV-write throughput benchmark.
//!
//! Mirrors `tests/csv_throughput.rs` but as a real binary so profilers
//! see only the workload — no `libtest::PrettyFormatter` overhead in the
//! samples, no `--nocapture` indirection. Run via:
//!
//!     cargo run --profile profiling --bin csv_bench
//!
//! or under samply:
//!
//!     samply record target/profiling/csv_bench

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use chronosynthea_mss::{
    patient_uuid, BatchConfig, BatchGenerator, CalibratedRegistry, SyntheaCsvWriter,
};
use std::path::PathBuf;
use std::time::Instant;

fn registry_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.push("data");
    p.push("prevalence");
    p.push("calibrated_registry.json");
    p
}

fn main() {
    let path = registry_path();
    if !path.exists() {
        eprintln!("registry not found at {}", path.display());
        std::process::exit(1);
    }
    let registry = CalibratedRegistry::load(&path).unwrap();
    let fingerprint = registry.to_fingerprint();
    let config = BatchConfig {
        seed: 42,
        ..Default::default()
    };
    let generator = BatchGenerator::new(fingerprint, config);

    let n = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10_000);

    let t0 = Instant::now();
    let patients = generator.generate_full(n);
    let gen_dt = t0.elapsed();
    let total_events: usize = patients.iter().map(|p| p.total_events()).sum();
    eprintln!(
        "[gen] {} patients in {:.3}s = {:.0}/sec; {:.1}M events",
        n,
        gen_dt.as_secs_f64(),
        n as f64 / gen_dt.as_secs_f64(),
        total_events as f64 / 1e6
    );

    let archetypes = generator.archetypes();
    let code_table = generator.code_table();

    // Serial baseline.
    let out_dir = std::env::temp_dir().join("chronosynthea-csv-bench");
    let _ = std::fs::remove_dir_all(&out_dir);
    let mut writer = SyntheaCsvWriter::create(&out_dir).unwrap();
    let t1 = Instant::now();
    for p in &patients {
        let uuid = patient_uuid(p.id);
        writer
            .write_patient(p, &uuid, archetypes, code_table)
            .unwrap();
    }
    writer.flush().unwrap();
    let dt_serial = t1.elapsed();
    eprintln!(
        "[serial]   {} patients in {:.3}s = {:.0}/sec; {:.2} Mevents/sec",
        n,
        dt_serial.as_secs_f64(),
        n as f64 / dt_serial.as_secs_f64(),
        total_events as f64 / 1e6 / dt_serial.as_secs_f64()
    );

    // Parallel path (only for n large enough that workers amortize).
    let out_dir_p = std::env::temp_dir().join("chronosynthea-csv-bench-par");
    let _ = std::fs::remove_dir_all(&out_dir_p);
    let mut writer_p = SyntheaCsvWriter::create(&out_dir_p).unwrap();
    let t2 = Instant::now();
    writer_p
        .write_patients_parallel(&patients, archetypes, code_table)
        .unwrap();
    writer_p.flush().unwrap();
    let dt_parallel = t2.elapsed();
    eprintln!(
        "[parallel] {} patients in {:.3}s = {:.0}/sec; {:.2} Mevents/sec  ({:.2}× vs serial)",
        n,
        dt_parallel.as_secs_f64(),
        n as f64 / dt_parallel.as_secs_f64(),
        total_events as f64 / 1e6 / dt_parallel.as_secs_f64(),
        dt_serial.as_secs_f64() / dt_parallel.as_secs_f64()
    );
}
