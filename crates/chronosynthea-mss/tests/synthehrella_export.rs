//! Tool runner: generates N patients and writes the SynthEHRella
//! binary matrix + temporal records to the directory specified by the
//! `CHRONOSYNTHEA_SYNTHEHRELLA_OUT` environment variable.
//!
//! Used as the export step by `scripts/synthehrella_evaluate.py`.
//! Run via `cargo test --test synthehrella_export -- --ignored`.
//!
//! Env vars:
//!   * `CHRONOSYNTHEA_SYNTHEHRELLA_OUT` — output directory (default
//!     `workspace/synthehrella-eval/`)
//!   * `CHRONOSYNTHEA_SYNTHEHRELLA_N`  — patient count (default 10_000)
//!   * `CHRONOSYNTHEA_JOINT_MODE`      — `marginal-only`,
//!     `pairwise-empirical`, or `causal-dag` (default `marginal-only`)
//!   * `CHRONOSYNTHEA_SEED`             — base seed (default 42)

use chronosynthea_mss::synthehrella::{
    write_binary_matrix, write_temporal_records, MatrixOptions,
};
use chronosynthea_mss::{BatchConfig, BatchGenerator, CalibratedRegistry};
use std::path::PathBuf;

fn registry_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.push("data");
    p.push("prevalence");
    p.push("calibrated_registry.json");
    p
}

#[test]
#[ignore]
fn synthehrella_export() {
    let path = registry_path();
    if !path.exists() {
        eprintln!("Skipping: calibrated_registry.json not found");
        return;
    }
    let out_dir: PathBuf = std::env::var_os("CHRONOSYNTHEA_SYNTHEHRELLA_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("workspace/synthehrella-eval"));
    let n: usize = std::env::var("CHRONOSYNTHEA_SYNTHEHRELLA_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    let seed: u64 = std::env::var("CHRONOSYNTHEA_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    let registry = CalibratedRegistry::load(&path).unwrap();
    let fingerprint = registry.to_fingerprint();
    let config = BatchConfig {
        seed,
        ..Default::default()
    };
    let generator = BatchGenerator::new(fingerprint, config);

    let t0 = std::time::Instant::now();
    let patients = generator.generate_full(n);
    let gen_dt = t0.elapsed();
    eprintln!(
        "generated {} patients in {:.2}s = {:.0}/sec",
        patients.len(),
        gen_dt.as_secs_f64(),
        n as f64 / gen_dt.as_secs_f64()
    );

    let archetypes = generator.archetypes();
    let code_table = generator.code_table();

    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let bm_path = out_dir.join("binary_matrix.csv");
    let tr_path = out_dir.join("temporal_records.csv");

    let t1 = std::time::Instant::now();
    let bm_rows = write_binary_matrix(
        &patients,
        archetypes,
        code_table,
        &bm_path,
        MatrixOptions::default(),
    )
    .expect("write binary matrix");
    eprintln!(
        "wrote {} binary-matrix rows to {:?} in {:.2}s",
        bm_rows,
        bm_path,
        t1.elapsed().as_secs_f64()
    );

    let t2 = std::time::Instant::now();
    let tr_rows = write_temporal_records(&patients, archetypes, code_table, &tr_path)
        .expect("write temporal records");
    eprintln!(
        "wrote {} temporal-record rows to {:?} in {:.2}s",
        tr_rows,
        tr_path,
        t2.elapsed().as_secs_f64()
    );
}
