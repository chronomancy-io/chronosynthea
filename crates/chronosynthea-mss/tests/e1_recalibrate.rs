//! E1 — iterative marginal recalibration for d5 `pairwise-empirical` mode.
//!
//! When the joint sampler is active, dependents get added on top of the
//! independent draws, inflating their marginals. This test runs a fixed-point
//! loop:
//!
//!   1. generate N patients in joint mode under current per-archetype
//!      thresholds,
//!   2. measure observed marginals,
//!   3. compute per-condition correction factor = target_marginal / observed,
//!   4. apply the correction to every archetype's prevalence for that
//!      condition,
//!   5. iterate until max |target − observed| < `TOL`.
//!
//! Writes the converged per-condition multiplier vector to
//! `/tmp/chronosynthea-recalibration.json` so downstream consumers can
//! reproduce the same calibrated state offline.

use chronosynthea_mss::{BatchConfig, BatchGenerator, CalibratedRegistry};
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

const N_CALIBRATION: usize = 50_000;
const MAX_ITERS: usize = 30;
const TOL: f64 = 0.005; // 0.5 percentage-point per-condition tolerance

#[test]
#[ignore]
fn e1_recalibrate_marginals() {
    let path = calibrated_registry_path();
    if !path.exists() {
        eprintln!("Skipping: calibrated_registry.json not found");
        return;
    }

    let coocc_env = std::env::var("CHRONOSYNTHEA_COOCCURRENCE_PATH").ok();
    println!(
        "CHRONOSYNTHEA_COOCCURRENCE_PATH = {}",
        coocc_env.as_deref().unwrap_or("<unset>")
    );

    let registry = CalibratedRegistry::load(&path).unwrap();
    let fingerprint = registry.to_fingerprint();
    let mode = if fingerprint.cooccurrence.is_empty() {
        "marginal-only"
    } else {
        "pairwise-empirical"
    };
    println!("d5 mode: {mode}");

    // Capture target marginals from the fingerprint BEFORE recalibration.
    // These are the per-condition prevalences the framework should preserve.
    let target_marginals: Vec<f64> = fingerprint
        .conditions
        .iter()
        .map(|c| c.prevalence)
        .collect();
    let num_conditions = target_marginals.len();
    let condition_codes: Vec<String> =
        fingerprint.conditions.iter().map(|c| c.code.clone()).collect();
    let fingerprint_displays: Vec<String> =
        fingerprint.conditions.iter().map(|c| c.display.clone()).collect();

    let config = BatchConfig {
        seed: 42,
        ..Default::default()
    };
    let mut generator = BatchGenerator::new(fingerprint, config);

    // Accumulated multipliers (one per condition), starting at identity.
    let mut multipliers: Vec<f32> = vec![1.0; num_conditions];

    let mut coocc_multipliers: Vec<f32> = vec![1.0; num_conditions];

    for iter in 0..MAX_ITERS {
        // Generate a calibration batch and observe per-condition marginals.
        let stats = generator.generate_stats_only(N_CALIBRATION);
        let observed: Vec<f64> = (0..num_conditions)
            .map(|i| {
                if i < stats.condition_counts.len() {
                    stats.condition_counts[i] as f64 / N_CALIBRATION as f64
                } else {
                    0.0
                }
            })
            .collect();

        // Two-knob fit: per-condition base prevalence scale + per-dependent
        // cooccurrence boost scale. Both are updated each iteration so the
        // additive boost has somewhere to land when base goes to 0.
        let mut max_abs_err = 0.0f64;
        let mut prevalence_adj: Vec<f32> = vec![1.0; num_conditions];
        let mut boost_adj: Vec<f32> = vec![1.0; num_conditions];
        for i in 0..num_conditions {
            let t = target_marginals[i];
            let o = observed[i];
            let abs_err = (t - o).abs();
            if abs_err > max_abs_err {
                max_abs_err = abs_err;
            }
            let raw = if o > 1e-6 { t / o } else { 1.0 };
            // Damped multiplicative step. Both knobs move in the same direction
            // (need to lower o → lower both base and boost), but at half-rate
            // so they share the correction rather than fighting each other.
            let damped = 1.0 + 0.35 * (raw - 1.0);
            let clamped = damped.clamp(0.5, 2.0) as f32;
            prevalence_adj[i] = clamped;
            boost_adj[i] = clamped;
        }

        // Apply to both knobs.
        {
            let arch_arc = generator.archetypes_arc_mut();
            std::sync::Arc::get_mut(arch_arc)
                .expect("registry has no other readers during calibration")
                .scale_per_condition_prevalence(&prevalence_adj);
        }
        {
            let coocc_arc = generator.cooccurrence_arc_mut();
            std::sync::Arc::get_mut(coocc_arc)
                .expect("cooccurrence has no other readers during calibration")
                .scale_dependent_boosts(&boost_adj);
        }

        for i in 0..num_conditions {
            multipliers[i] *= prevalence_adj[i];
            coocc_multipliers[i] *= boost_adj[i];
        }

        let n_outside_tol = observed
            .iter()
            .zip(target_marginals.iter())
            .filter(|(o, t)| (*t - *o).abs() > TOL)
            .count();

        println!(
            "iter {iter:>2}: max_abs_err = {:.4}%, {} conditions outside ±{:.1}% tolerance",
            max_abs_err * 100.0,
            n_outside_tol,
            TOL * 100.0
        );

        if max_abs_err < TOL {
            println!(
                "converged after {} iterations; max_abs_err {:.4}% < {:.1}%",
                iter + 1,
                max_abs_err * 100.0,
                TOL * 100.0
            );
            break;
        }
    }

    // Final measurement at higher n to characterise the converged state.
    let stats = generator.generate_stats_only(100_000);
    let final_observed: Vec<f64> = (0..num_conditions)
        .map(|i| stats.condition_counts[i] as f64 / 100_000.0)
        .collect();
    let mut max_dev = 0.0f64;
    let mut worst_idx = 0usize;
    for i in 0..num_conditions {
        let dev = (target_marginals[i] - final_observed[i]).abs();
        if dev > max_dev {
            max_dev = dev;
            worst_idx = i;
        }
    }
    println!();
    println!("=== Final converged state (n=100k) ===");
    println!(
        "max_deviation = {:.4}% (condition: {} target {:.4} observed {:.4})",
        max_dev * 100.0,
        condition_codes[worst_idx],
        target_marginals[worst_idx],
        final_observed[worst_idx]
    );

    // Persist the multipliers.
    use std::fs::File;
    use std::io::Write;
    let mut f = File::create("/tmp/chronosynthea-recalibration.json").unwrap();
    #[derive(serde::Serialize)]
    struct RecalibrationOut {
        prevalence_multipliers: Vec<(String, f32)>,
        boost_multipliers: Vec<(String, f32)>,
    }
    let out = RecalibrationOut {
        prevalence_multipliers: condition_codes
            .iter()
            .zip(multipliers.iter().copied())
            .map(|(c, m)| (c.clone(), m))
            .collect(),
        boost_multipliers: condition_codes
            .iter()
            .zip(coocc_multipliers.iter().copied())
            .map(|(c, m)| (c.clone(), m))
            .collect(),
    };
    serde_json::to_writer(&mut f, &out).unwrap();
    let _ = writeln!(f);
    println!("wrote recalibration to /tmp/chronosynthea-recalibration.json");

    // Now emit pairwise CSV at the calibrated state so we can compare joint
    // correlation against Java Synthea (E1 dual-mode compare).
    let n_pairwise = 10_000;
    let patients = generator.generate_compact(n_pairwise);
    let mut marginal = vec![0u64; num_conditions];
    let mut pairwise: std::collections::HashMap<(u16, u16), u64> =
        std::collections::HashMap::new();
    for p in &patients {
        let conds = p.conditions.as_slice();
        for &c in conds {
            marginal[c as usize] += 1;
        }
        for i in 0..conds.len() {
            for j in (i + 1)..conds.len() {
                let a = conds[i].min(conds[j]);
                let b = conds[i].max(conds[j]);
                if a == b {
                    continue;
                }
                *pairwise.entry((a, b)).or_insert(0) += 1;
            }
        }
    }
    let n_f = n_pairwise as f64;
    let mut out_csv = File::create(
        "/tmp/e1-chronosynthea-pairwise-pairwise-empirical-calibrated.csv",
    )
    .unwrap();
    writeln!(
        out_csv,
        "cond_a_code,cond_a_display,cond_b_code,cond_b_display,joint_count,joint_prev,marginal_a,marginal_b,expected_under_indep,lift"
    )
    .unwrap();
    let mut pairs_vec: Vec<((u16, u16), u64)> = pairwise.into_iter().collect();
    pairs_vec.sort_by(|x, y| y.1.cmp(&x.1));
    for ((a, b), count) in &pairs_vec {
        let pa = marginal[*a as usize] as f64 / n_f;
        let pb = marginal[*b as usize] as f64 / n_f;
        let joint = *count as f64 / n_f;
        let expected = pa * pb;
        let lift = if expected > 0.0 { joint / expected } else { 0.0 };
        let (code_a, disp_a) = (
            &condition_codes[*a as usize],
            &fingerprint_displays[*a as usize],
        );
        let (code_b, disp_b) = (
            &condition_codes[*b as usize],
            &fingerprint_displays[*b as usize],
        );
        writeln!(
            out_csv,
            "{},\"{}\",{},\"{}\",{},{:.6},{:.6},{:.6},{:.6},{:.4}",
            code_a, disp_a, code_b, disp_b, count, joint, pa, pb, expected, lift
        )
        .unwrap();
    }
    println!(
        "wrote calibrated pairwise CSV to /tmp/e1-chronosynthea-pairwise-pairwise-empirical-calibrated.csv ({} pairs)",
        pairs_vec.len()
    );
}
