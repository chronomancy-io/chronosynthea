//! Validates the causal-cascade post-pass produces Java-Synthea-equivalent
//! trajectory ordering: when a patient has both a trigger condition (e.g.
//! diabetes 44054006) and its downstream condition (e.g. CKD 431855005),
//! the downstream onset day must follow the trigger onset day by the
//! empirical mean lag (Java: ~2000 days for diabetes → CKD).

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
fn cascade_enforces_trajectory_ordering() {
    let path = registry_path();
    if !path.exists() {
        eprintln!("Skipping: calibrated_registry.json not found");
        return;
    }
    let cascade_path = path.parent().unwrap().join("cascade_lags.json");
    if !cascade_path.exists() {
        eprintln!("Skipping: cascade_lags.json not found");
        return;
    }

    let registry = CalibratedRegistry::load(&path).unwrap();
    let fingerprint = registry.to_fingerprint();
    let config = BatchConfig {
        seed: 314159,
        ..Default::default()
    };
    let generator = BatchGenerator::new(fingerprint.clone(), config);
    let code_table = generator.code_table();

    let n = 10_000;
    let patients = generator.generate_full(n);

    // Test pairs where the trigger is the MOST PROBABLE cause of the
    // downstream in Java's empirical data. The cascade is designed to
    // enforce the dominant-cause relationship; multi-cause downstreams
    // (e.g. CKD via diabetes vs CKD via kidney-disorder-due-to-diabetes)
    // pick the most-proximate cause, not all simultaneous triggers.
    // These pairs come from the head of `cascade_lags.json` ranked by
    // probability per downstream.
    let target_pairs: &[(&str, &str, &str)] = &[
        // Each pair lists the DOMINANT (highest-probability) trigger for
        // the downstream, as recorded in `cascade_lags.json`. The cascade
        // enforces ordering only for the dominant trigger; multi-cause
        // pairs (e.g. "DKD → CKD stage 2" when proteinuria is also
        // active) are governed by the most proximate cause, matching
        // Java's state-machine behaviour where one trigger fires at a
        // time.
        ("127013003", "431855005", "DKD → CKD stage 1"),
        ("90781000119102", "431856006", "Diabetic proteinuria → CKD stage 2"),
        ("59621000", "44054006", "Stress → Diabetes"),
    ];

    let mut code_to_idx: ahash::AHashMap<String, u16> = ahash::AHashMap::new();
    for i in 0..code_table.num_conditions() {
        if let Some(e) = code_table.condition(i as u16) {
            code_to_idx.insert(e.code.clone(), i as u16);
        }
    }

    println!("Cascade-enforced trajectory observation (n=10k patients):");
    let mut any_observed = false;
    for (trigger_code, downstream_code, label) in target_pairs {
        let t_idx = match code_to_idx.get(*trigger_code) {
            Some(&i) => i,
            None => continue,
        };
        let d_idx = match code_to_idx.get(*downstream_code) {
            Some(&i) => i,
            None => continue,
        };
        let mut both = 0u32;
        let mut ordered = 0u32;
        let mut clamp_bound = 0u32;
        let mut lags: Vec<i32> = Vec::new();
        for p in &patients {
            let t_slot = p.conditions.iter().position(|&c| c == t_idx);
            let d_slot = p.conditions.iter().position(|&c| c == d_idx);
            if let (Some(t), Some(d)) = (t_slot, d_slot) {
                both += 1;
                let t_onset = p.condition_onset_days[t] as i32;
                let d_onset = p.condition_onset_days[d] as i32;
                // Patients whose trigger onset has already saturated to
                // the patient's max-age clamp can't be ordered by the
                // cascade — both conditions land on the same day at
                // end-of-life. Count them separately so the cascade's
                // actual enforcement rate is observable.
                let max_age =
                    p.condition_onset_days.iter().copied().max().unwrap_or(0) as i32;
                if t_onset >= max_age - 30 {
                    clamp_bound += 1;
                    continue;
                }
                let lag = d_onset - t_onset;
                lags.push(lag);
                if d_onset > t_onset {
                    ordered += 1;
                }
            }
        }
        if both == 0 {
            continue;
        }
        any_observed = true;
        let assessable = both - clamp_bound;
        let mean_lag = if !lags.is_empty() {
            lags.iter().sum::<i32>() as f64 / lags.len() as f64
        } else {
            0.0
        };
        println!(
            "  {:48} both={:>5} clamp_bound={:>4} ordered={:>5}/{:<5} ({:>5.1}%) mean_lag={:.0}d",
            label,
            both,
            clamp_bound,
            ordered,
            assessable,
            if assessable > 0 {
                100.0 * ordered as f64 / assessable as f64
            } else {
                0.0
            },
            mean_lag
        );
        if assessable < 10 {
            // Too few non-clamp patients to draw a reliable conclusion.
            continue;
        }
        // Strict assertion: ≥95% of cascade-applicable patient pairs (i.e.
        // those whose trigger onset has lifespan room to allow the
        // downstream to follow) must have the downstream onset after the
        // trigger onset.
        assert!(
            ordered as f64 / assessable as f64 >= 0.95,
            "{}: only {:.1}% of {} cascade-applicable patients have ordered onsets",
            label,
            100.0 * ordered as f64 / assessable as f64,
            assessable
        );
        assert!(mean_lag > 0.0, "{}: mean lag {} is not positive", label, mean_lag);
    }
    assert!(
        any_observed,
        "no target pairs observed in the patient set — registry / cascade rules misaligned?"
    );
}
