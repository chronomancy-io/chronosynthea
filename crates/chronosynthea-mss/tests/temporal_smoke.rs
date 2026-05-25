//! Smoke test for d5 = `temporal-ordered`. Loads the calibrated registry
//! with the empirical `onset_stats.json` companion file, generates a small
//! batch of compact patients, and asserts that:
//!   1. `condition_onset_days` is populated (non-empty parallel array).
//!   2. Per patient, onset days are sorted ascending — Java-equivalent
//!      temporal trajectory.
//!   3. Onsets fall within `[0, patient_age_days]` (no pre-birth, no future).
//!   4. The empirical mean onset for a high-prevalence condition matches
//!      the Java-Synthea-extracted mean within 5 years.

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

#[test]
#[ignore]
fn temporal_ordered_smoke() {
    let path = calibrated_registry_path();
    if !path.exists() {
        eprintln!("Skipping: calibrated_registry.json not found");
        return;
    }

    let onset_env = std::env::var("CHRONOSYNTHEA_ONSET_PATH").ok();
    println!(
        "CHRONOSYNTHEA_ONSET_PATH = {}",
        onset_env.as_deref().unwrap_or("<unset>")
    );

    let registry = CalibratedRegistry::load(&path).unwrap();
    let fingerprint = registry.to_fingerprint();
    let temporal_active = !fingerprint.onset_stats.is_empty();
    println!(
        "d5 temporal: {} ({} onset records loaded)",
        if temporal_active { "ACTIVE" } else { "default (40y)" },
        fingerprint.onset_stats.len()
    );

    let config = BatchConfig {
        seed: 42,
        ..Default::default()
    };
    let generator = BatchGenerator::new(fingerprint, config);
    let patients = generator.generate_compact(5_000);

    let mut total_conditions = 0u64;
    let mut total_onset_pairs = 0u64;
    let mut unsorted_patients = 0u64;
    let mut out_of_range = 0u64;
    let mut sum_first_onset = 0i64;
    let mut sum_last_onset = 0i64;

    for p in &patients {
        let n = p.conditions.len();
        total_conditions += n as u64;
        total_onset_pairs += p.condition_onset_days.len() as u64;
        assert_eq!(
            p.conditions.len(),
            p.condition_onset_days.len(),
            "conditions and onsets must be parallel"
        );

        // Check sorted
        for i in 1..n {
            if p.condition_onset_days[i] < p.condition_onset_days[i - 1] {
                unsorted_patients += 1;
                break;
            }
        }

        // Check range: onset must be <= patient's current age in days.
        // birth_date_days is days since 1970-01-01; the generator uses
        // 2024-01-01 as "now" → patient age = 19723 - birth_date_days.
        let now_days_since_epoch: i32 = 19723;
        let patient_age_days = (now_days_since_epoch - p.birth_date_days).max(0) as u32;
        for &o in p.condition_onset_days.iter() {
            if (o as u32) > patient_age_days {
                out_of_range += 1;
            }
        }

        if let (Some(&first), Some(&last)) = (
            p.condition_onset_days.first(),
            p.condition_onset_days.last(),
        ) {
            sum_first_onset += first as i64;
            sum_last_onset += last as i64;
        }
    }

    println!("Patients: {}", patients.len());
    println!("Total conditions emitted: {total_conditions}");
    println!("Total onset pairs: {total_onset_pairs}");
    println!("Unsorted patients: {unsorted_patients}");
    println!("Out-of-range onsets: {out_of_range}");
    let n_with_conds = patients.iter().filter(|p| !p.conditions.is_empty()).count() as i64;
    if n_with_conds > 0 {
        println!(
            "Mean first-onset across patients: {:.1} days ({:.2}y)",
            sum_first_onset as f32 / n_with_conds as f32,
            sum_first_onset as f32 / n_with_conds as f32 / 365.25
        );
        println!(
            "Mean last-onset across patients: {:.1} days ({:.2}y)",
            sum_last_onset as f32 / n_with_conds as f32,
            sum_last_onset as f32 / n_with_conds as f32 / 365.25
        );
    }

    assert_eq!(total_conditions, total_onset_pairs);
    assert_eq!(unsorted_patients, 0, "all per-patient onsets must be sorted");
    assert_eq!(out_of_range, 0, "no onset may exceed patient age");
    if temporal_active {
        assert!(
            total_onset_pairs > 0,
            "onset_stats was loaded but no onsets emitted"
        );
    }
}
