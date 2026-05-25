//! Throughput smoke test for the full FullPatient pipeline (encounters,
//! events, REASONCODE) after the encounter-level event sequencing work.
//! Compare against the historical 11.4M/sec full-pipeline figure quoted
//! in MANIFESTO.md. Ignored by default; run via `--ignored`.

use chronosynthea_mss::{BatchConfig, BatchGenerator, CalibratedRegistry};
use std::path::PathBuf;
use std::time::Instant;

#[test]
#[ignore]
fn full_pipeline_throughput() {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.push("data");
    p.push("prevalence");
    p.push("calibrated_registry.json");
    if !p.exists() {
        eprintln!("Skipping: calibrated_registry.json not found");
        return;
    }
    let registry = CalibratedRegistry::load(&p).unwrap();
    let fingerprint = registry.to_fingerprint();
    let config = BatchConfig {
        seed: 42,
        ..Default::default()
    };
    let gen = BatchGenerator::new(fingerprint, config);

    // Warmup
    let _ = gen.generate_full(1_000);

    let n: usize = 100_000;
    let t0 = Instant::now();
    let patients = gen.generate_full(n);
    let dt = t0.elapsed();
    assert_eq!(patients.len(), n);
    let rate = n as f64 / dt.as_secs_f64();
    let total_events: usize = patients
        .iter()
        .map(|p| p.encounters.iter().map(|e| e.events.len()).sum::<usize>())
        .sum();
    let mean_events_per_patient = total_events as f64 / n as f64;
    println!(
        "generated {} full patients in {:.2}s = {:.0}/sec (mean {:.1} events/patient)",
        patients.len(),
        dt.as_secs_f64(),
        rate,
        mean_events_per_patient
    );
}
