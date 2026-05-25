use chronosynthea_mss::{BatchConfig, BatchGenerator, CalibratedRegistry, JavaValidation};
use std::path::PathBuf;

fn calibrated_registry_path() -> PathBuf {
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
fn seed_sweep_50() {
    let path = calibrated_registry_path();
    if !path.exists() {
        eprintln!("Skipping: calibrated_registry.json not found at {:?}", path);
        return;
    }

    let registry = CalibratedRegistry::load(&path).unwrap();
    let fingerprint = registry.to_fingerprint();
    let n_seeds: u64 = 50;
    let n_patients: usize = 100_000;

    let mut max_devs: Vec<f64> = Vec::with_capacity(n_seeds as usize);
    let mut kls: Vec<f64> = Vec::with_capacity(n_seeds as usize);
    let mut chis: Vec<f64> = Vec::with_capacity(n_seeds as usize);

    for seed in 0..n_seeds {
        let config = BatchConfig {
            seed,
            ..Default::default()
        };
        let generator = BatchGenerator::new(fingerprint.clone(), config);
        let stats = generator.generate_stats_only(n_patients);
        let validator =
            JavaValidation::from_fingerprint(fingerprint.clone()).with_tolerance(0.10);
        let result = validator.validate(&stats);
        max_devs.push(result.max_deviation);
        kls.push(result.kl_divergence);
        chis.push(result.chi_squared);
    }

    let summary = |xs: &mut Vec<f64>| -> (f64, f64, f64, f64, f64) {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = xs[0];
        let max = *xs.last().unwrap();
        let median = xs[xs.len() / 2];
        let p95 = xs[(xs.len() as f64 * 0.95) as usize];
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        (min, median, mean, p95, max)
    };

    let (md_min, md_med, md_mean, md_p95, md_max) = summary(&mut max_devs);
    let (kl_min, kl_med, kl_mean, kl_p95, kl_max) = summary(&mut kls);
    let (ch_min, ch_med, ch_mean, ch_p95, ch_max) = summary(&mut chis);

    println!();
    println!("=== SEED SWEEP (n_seeds={n_seeds}, n_patients={n_patients}, tolerance=10%) ===");
    println!();
    println!("Max deviation (per-condition, max over 214 conditions):");
    println!("  min={md_min:.4}%  median={md_med:.4}%  mean={md_mean:.4}%  p95={md_p95:.4}%  max={md_max:.4}%");
    println!("KL divergence:");
    println!("  min={kl_min:.6}  median={kl_med:.6}  mean={kl_mean:.6}  p95={kl_p95:.6}  max={kl_max:.6}");
    println!("Chi-squared:");
    println!("  min={ch_min:.2}  median={ch_med:.2}  mean={ch_mean:.2}  p95={ch_p95:.2}  max={ch_max:.2}");
    println!();
}
