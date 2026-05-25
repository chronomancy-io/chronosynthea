use chronosynthea_mss::{BatchConfig, BatchGenerator, CalibratedRegistry};
use std::path::PathBuf;
use std::time::Instant;

fn calibrated_registry_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.push("data");
    p.push("prevalence");
    p.push("calibrated_registry.json");
    p
}

/// Micro-bench: time generate_stats_only (conditions + atomic counters only)
/// for 1M patients, averaged over 5 runs after a 100K-patient warmup.
#[test]
#[ignore]
fn micro_stats_only_1m() {
    let path = calibrated_registry_path();
    if !path.exists() {
        eprintln!("Skipping: calibrated_registry.json not found at {:?}", path);
        return;
    }

    let registry = CalibratedRegistry::load(&path).unwrap();
    let fingerprint = registry.to_fingerprint();
    let config = BatchConfig::default();
    let generator = BatchGenerator::new(fingerprint, config);

    // Warmup
    let _ = generator.generate_stats_only(100_000);

    let iters = 5;
    let n: usize = 1_000_000;
    let mut total_ns: u128 = 0;
    for _ in 0..iters {
        let t0 = Instant::now();
        let stats = generator.generate_stats_only(n);
        total_ns += t0.elapsed().as_nanos();
        assert_eq!(stats.total_patients as usize, n);
    }
    let avg_ns = total_ns / iters as u128;
    let pps = (n as f64) / (avg_ns as f64 / 1e9);
    println!(
        "generate_stats_only ({} patients, avg of {} runs): {:.2} ms ({:.2}M patients/sec)",
        n,
        iters,
        avg_ns as f64 / 1e6,
        pps / 1e6
    );
}
