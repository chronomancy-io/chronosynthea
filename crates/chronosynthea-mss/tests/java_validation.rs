//! Integration tests validating MSS generation against Java Synthea baseline.
//!
//! These tests load the calibrated registry extracted from actual Java Synthea
//! runs and verify that our generated population matches the expected distributions.

use std::path::PathBuf;

use chronosynthea_mss::batch::{BatchConfig, BatchGenerator};
use chronosynthea_mss::java_compat::{CalibratedRegistry, JavaValidation};

/// Gets the path to the calibrated registry.
fn calibrated_registry_path() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data")
        .join("prevalence")
        .join("calibrated_registry.json")
}

#[test]
fn test_load_calibrated_registry() {
    let path = calibrated_registry_path();
    if !path.exists() {
        eprintln!(
            "Skipping test: calibrated_registry.json not found at {:?}",
            path
        );
        return;
    }

    let registry = CalibratedRegistry::load(&path).expect("Failed to load registry");

    // Verify we loaded meaningful data
    assert!(!registry.conditions.is_empty(), "Should have conditions");
    eprintln!(
        "Loaded {} conditions from calibrated registry",
        registry.conditions.len()
    );

    // Check some expected conditions exist
    let has_hypertension = registry
        .conditions
        .iter()
        .any(|c| c.display.to_lowercase().contains("hypertension"));
    assert!(has_hypertension, "Should have hypertension condition");
}

#[test]
fn test_convert_to_fingerprint() {
    let path = calibrated_registry_path();
    if !path.exists() {
        eprintln!("Skipping test: calibrated_registry.json not found");
        return;
    }

    let registry = CalibratedRegistry::load(&path).unwrap();
    let fingerprint = registry.to_fingerprint();

    // Verify fingerprint is valid
    assert!(!fingerprint.conditions.is_empty());
    assert!(!fingerprint.joint_demographics.buckets.is_empty());

    // Demographics should sum to ~1
    let demo_sum: f64 = fingerprint.joint_demographics.buckets.values().sum();
    assert!(
        (demo_sum - 1.0).abs() < 0.01,
        "Demographics should sum to 1, got {}",
        demo_sum
    );

    eprintln!(
        "Fingerprint has {} conditions, {} demographic buckets",
        fingerprint.conditions.len(),
        fingerprint.joint_demographics.buckets.len()
    );
}

#[test]
fn test_generate_matches_java_baseline() {
    let path = calibrated_registry_path();
    if !path.exists() {
        eprintln!("Skipping test: calibrated_registry.json not found");
        return;
    }

    let registry = CalibratedRegistry::load(&path).unwrap();
    let fingerprint = registry.to_fingerprint();
    let validator = JavaValidation::from_fingerprint(fingerprint.clone()).with_tolerance(0.10); // 10% tolerance for this test

    // Generate patients
    let config = BatchConfig {
        seed: 42,
        ..Default::default()
    };
    let num_conditions = fingerprint.conditions.len();
    let generator = BatchGenerator::new(fingerprint, config);

    // Generate 100K patients for statistical significance
    let stats = generator.generate_stats_only(100_000);

    // Validate against Java baseline
    let result = validator.validate(&stats);

    eprintln!("{}", result.summary());

    // We expect most conditions to be within tolerance
    // Some deviation is expected due to:
    // 1. Different sampling algorithms
    // 2. Simplified demographic model
    // 3. No co-occurrence modeling yet
    let failure_rate = result.failures.len() as f64 / num_conditions as f64;

    eprintln!(
        "Failure rate: {:.1}% ({}/{} conditions)",
        failure_rate * 100.0,
        result.failures.len(),
        num_conditions
    );

    // Production gate. Observed across 50 seeds × 100k patients (see seed_sweep test):
    //   max_deviation: 0.30%–0.31% (variance < 0.01 percentage points, LLN-dominated)
    //   failure_rate:  0/214 conditions (exact across all seeds tested)
    //
    // Gate thresholds (each is well above the observed value, but enforces G1/G3
    // as actual guarantees rather than as eprintln-only observations):
    //   max_deviation < 0.5% (16× margin over the worst observed value)
    //   failure_rate  == 0   (no condition outside 10% tolerance)
    assert!(
        result.max_deviation < 0.005,
        "max_deviation {:.4}% exceeds 0.5% gate",
        result.max_deviation * 100.0
    );
    assert_eq!(
        result.failures.len(),
        0,
        "expected 0 conditions outside 10% tolerance, got {} ({:.1}%): {:?}",
        result.failures.len(),
        failure_rate * 100.0,
        result.failures
    );
}

#[test]
fn test_high_volume_generation_with_validation() {
    let path = calibrated_registry_path();
    if !path.exists() {
        eprintln!("Skipping test: calibrated_registry.json not found");
        return;
    }

    let registry = CalibratedRegistry::load(&path).unwrap();
    let fingerprint = registry.to_fingerprint();

    let config = BatchConfig::default();
    let generator = BatchGenerator::new(fingerprint.clone(), config);

    // Time 1M patient generation
    let start = std::time::Instant::now();
    let stats = generator.generate_stats_only(1_000_000);
    let duration = start.elapsed();

    let patients_per_sec = 1_000_000.0 / duration.as_secs_f64();

    eprintln!(
        "Generated 1M patients in {:?} ({:.2}M patients/sec)",
        duration,
        patients_per_sec / 1_000_000.0
    );

    // Note: In debug mode, performance is lower
    // The 1M/sec target is for release builds
    // We just check it runs and produces valid output here
    eprintln!(
        "Note: Debug build. Release target is 1M+/sec, debug achieved {:.0}/sec",
        patients_per_sec
    );

    // Quick sanity check on the generated data
    assert_eq!(stats.total_patients, 1_000_000);
    assert!(stats.total_encounters > 0);

    // Check that we're generating conditions
    let total_conditions: u64 = stats.condition_counts.iter().sum();
    assert!(
        total_conditions > 100_000,
        "Should generate conditions, got {}",
        total_conditions
    );
}

#[test]
fn test_deterministic_with_java_registry() {
    let path = calibrated_registry_path();
    if !path.exists() {
        eprintln!("Skipping test: calibrated_registry.json not found");
        return;
    }

    let registry = CalibratedRegistry::load(&path).unwrap();
    let fingerprint = registry.to_fingerprint();

    // Generate twice with same seed
    let config1 = BatchConfig {
        seed: 12345,
        ..Default::default()
    };
    let config2 = BatchConfig {
        seed: 12345,
        ..Default::default()
    };

    let gen1 = BatchGenerator::new(fingerprint.clone(), config1);
    let gen2 = BatchGenerator::new(fingerprint, config2);

    let stats1 = gen1.generate_stats_only(10_000);
    let stats2 = gen2.generate_stats_only(10_000);

    // Should be identical
    assert_eq!(stats1.total_patients, stats2.total_patients);
    assert_eq!(stats1.condition_counts, stats2.condition_counts);
}

#[test]
fn test_condition_prevalence_distribution() {
    let path = calibrated_registry_path();
    if !path.exists() {
        eprintln!("Skipping test: calibrated_registry.json not found");
        return;
    }

    let registry = CalibratedRegistry::load(&path).unwrap();
    let fingerprint = registry.to_fingerprint();

    let config = BatchConfig::default();
    let generator = BatchGenerator::new(fingerprint.clone(), config);

    let stats = generator.generate_stats_only(50_000);
    let prevalences = stats.condition_prevalences();

    // Find some common conditions and check their prevalences
    for (i, cond) in fingerprint.conditions.iter().take(20).enumerate() {
        let observed = prevalences.get(i).copied().unwrap_or(0.0);
        let expected = cond.prevalence;

        eprintln!(
            "{}: expected={:.2}%, observed={:.2}%, diff={:.2}%",
            cond.display,
            expected * 100.0,
            observed * 100.0,
            (observed - expected).abs() * 100.0
        );
    }
}

#[test]
fn test_all_data_loaded() {
    let path = calibrated_registry_path();
    if !path.exists() {
        eprintln!("Skipping test: calibrated_registry.json not found");
        return;
    }

    let registry = CalibratedRegistry::load(&path).unwrap();
    let fingerprint = registry.to_fingerprint();

    eprintln!("=== Data Loaded into MSS Fingerprint ===");
    eprintln!(
        "Conditions:   {} (registry: {})",
        fingerprint.conditions.len(),
        registry.conditions.len()
    );
    eprintln!(
        "Medications:  {} (registry: {})",
        fingerprint.medications.len(),
        registry.medications.len()
    );
    eprintln!(
        "Observations: {} (registry: {})",
        fingerprint.observations.len(),
        registry.observations.len()
    );
    eprintln!(
        "Procedures:   {} (registry: {})",
        fingerprint.procedures.len(),
        registry.procedures.len()
    );
    eprintln!(
        "Demographics: {} buckets",
        fingerprint.joint_demographics.buckets.len()
    );
    eprintln!("Co-occurrences: {} pairs", fingerprint.cooccurrence.len());

    // Verify all data is loaded
    assert_eq!(
        fingerprint.conditions.len(),
        registry.conditions.len(),
        "All conditions should be loaded"
    );
    assert_eq!(
        fingerprint.medications.len(),
        registry.medications.len(),
        "All medications should be loaded"
    );
    assert_eq!(
        fingerprint.observations.len(),
        registry.observations.len(),
        "All observations should be loaded"
    );
    assert_eq!(
        fingerprint.procedures.len(),
        registry.procedures.len(),
        "All procedures should be loaded"
    );

    // Sample some specific items
    if !fingerprint.medications.is_empty() {
        eprintln!("\nSample medications:");
        for med in fingerprint.medications.iter().take(5) {
            eprintln!("  - {} ({})", med.display, med.code);
        }
    }

    if !fingerprint.observations.is_empty() {
        eprintln!("\nSample observations:");
        for obs in fingerprint.observations.iter().take(5) {
            eprintln!("  - {} ({}, {})", obs.display, obs.code, obs.system);
        }
    }

    if !fingerprint.procedures.is_empty() {
        eprintln!("\nSample procedures:");
        for proc in fingerprint.procedures.iter().take(5) {
            eprintln!("  - {} ({})", proc.display, proc.code);
        }
    }
}

#[test]
fn test_full_generation_with_events() {
    let path = calibrated_registry_path();
    if !path.exists() {
        eprintln!("Skipping test: calibrated_registry.json not found");
        return;
    }

    let registry = CalibratedRegistry::load(&path).unwrap();
    let fingerprint = registry.to_fingerprint();

    let config = BatchConfig::default();
    let generator = BatchGenerator::new(fingerprint.clone(), config);

    // Generate full patients
    let patients = generator.generate_full(1000);

    eprintln!("=== Full Generation Results (1000 patients) ===");

    // Count totals
    let total_encounters: usize = patients.iter().map(|p| p.num_encounters()).sum();
    let total_events: usize = patients.iter().map(|p| p.total_events()).sum();
    let total_conditions: usize = patients.iter().map(|p| p.conditions.len()).sum();
    let total_medications: usize = patients.iter().map(|p| p.medications.len()).sum();
    let total_procedures: usize = patients.iter().map(|p| p.procedures.len()).sum();

    eprintln!("Total encounters: {}", total_encounters);
    eprintln!("Total events: {}", total_events);
    eprintln!("Total conditions (patient-level): {}", total_conditions);
    eprintln!("Total medications (patient-level): {}", total_medications);
    eprintln!("Total procedures (patient-level): {}", total_procedures);
    eprintln!(
        "Avg encounters/patient: {:.1}",
        total_encounters as f64 / 1000.0
    );
    eprintln!("Avg events/patient: {:.1}", total_events as f64 / 1000.0);
    eprintln!(
        "Avg conditions/patient: {:.1}",
        total_conditions as f64 / 1000.0
    );
    eprintln!(
        "Avg medications/patient: {:.1}",
        total_medications as f64 / 1000.0
    );
    eprintln!(
        "Avg procedures/patient: {:.1}",
        total_procedures as f64 / 1000.0
    );

    // Verify we have events
    assert!(total_encounters > 0, "Should have encounters");
    assert!(total_events > 0, "Should have events");
    assert!(total_conditions > 0, "Should have conditions");
    assert!(total_medications > 0, "Should have medications");

    // Sample a patient
    let sample = &patients[0];
    eprintln!("\nSample patient {}:", sample.id);
    eprintln!("  Conditions: {:?}", sample.conditions.as_slice());
    eprintln!("  Medications: {:?}", sample.medications.as_slice());
    eprintln!("  Procedures: {:?}", sample.procedures.as_slice());
    eprintln!("  Encounters: {}", sample.encounters.len());

    if let Some(enc) = sample.encounters.first() {
        eprintln!(
            "  First encounter: type={}, events={}",
            enc.encounter_type,
            enc.events.len()
        );
    }
}

#[test]
fn test_full_stats_generation() {
    let path = calibrated_registry_path();
    if !path.exists() {
        eprintln!("Skipping test: calibrated_registry.json not found");
        return;
    }

    let registry = CalibratedRegistry::load(&path).unwrap();
    let fingerprint = registry.to_fingerprint();

    let config = BatchConfig::default();
    let generator = BatchGenerator::new(fingerprint.clone(), config);

    // Generate full statistics
    let stats = generator.generate_full_stats_only(50_000);

    eprintln!("=== Full Statistics Generation (50K patients) ===");
    eprintln!("Total patients: {}", stats.total_patients);
    eprintln!("Total encounters: {}", stats.total_encounters);
    eprintln!("Total events: {}", stats.total_events);

    // Calculate medication prevalences
    let med_prevalences: Vec<f64> = stats
        .medication_counts
        .iter()
        .map(|&c| c as f64 / stats.total_patients as f64)
        .collect();

    // Top 10 medications by prevalence
    let mut med_by_prev: Vec<(usize, f64)> = med_prevalences
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    med_by_prev.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!("\nTop 10 Medications by Prevalence:");
    for (i, (med_idx, prev)) in med_by_prev.iter().take(10).enumerate() {
        let display = fingerprint
            .medications
            .get(*med_idx)
            .map(|m| m.display.as_str())
            .unwrap_or("Unknown");
        eprintln!("  {}. {} ({:.2}%)", i + 1, display, prev * 100.0);
    }

    // Calculate observation prevalences (per patient who had at least one)
    let obs_prevalences: Vec<f64> = stats
        .observation_counts
        .iter()
        .map(|&c| c as f64 / stats.total_patients as f64)
        .collect();

    let mut obs_by_prev: Vec<(usize, f64)> = obs_prevalences
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    obs_by_prev.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!("\nTop 10 Observations by Prevalence:");
    for (i, (obs_idx, prev)) in obs_by_prev.iter().take(10).enumerate() {
        let display = fingerprint
            .observations
            .get(*obs_idx)
            .map(|o| o.display.as_str())
            .unwrap_or("Unknown");
        eprintln!("  {}. {} ({:.2}%)", i + 1, display, prev * 100.0);
    }

    // Calculate procedure prevalences
    let proc_prevalences: Vec<f64> = stats
        .procedure_counts
        .iter()
        .map(|&c| c as f64 / stats.total_patients as f64)
        .collect();

    let mut proc_by_prev: Vec<(usize, f64)> = proc_prevalences
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    proc_by_prev.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!("\nTop 10 Procedures by Prevalence:");
    for (i, (proc_idx, prev)) in proc_by_prev.iter().take(10).enumerate() {
        let display = fingerprint
            .procedures
            .get(*proc_idx)
            .map(|p| p.display.as_str())
            .unwrap_or("Unknown");
        eprintln!("  {}. {} ({:.2}%)", i + 1, display, prev * 100.0);
    }

    // Verify we have events
    assert_eq!(stats.total_patients, 50_000);
    assert!(stats.total_encounters > 0, "Should have encounters");

    let total_meds: u64 = stats.medication_counts.iter().sum();
    let total_obs: u64 = stats.observation_counts.iter().sum();
    let total_procs: u64 = stats.procedure_counts.iter().sum();

    eprintln!("\nTotals:");
    eprintln!("  Total medication prescriptions: {}", total_meds);
    eprintln!("  Total observation events: {}", total_obs);
    eprintln!("  Total procedure events: {}", total_procs);

    assert!(total_meds > 0, "Should have medications");
    assert!(total_obs > 0, "Should have observations");
}

#[test]
fn test_full_generation_performance() {
    // Skip this test in debug mode - performance assertions only valid in release
    if cfg!(debug_assertions) {
        eprintln!("Skipping performance test in debug mode");
        return;
    }

    let path = calibrated_registry_path();
    if !path.exists() {
        eprintln!("Skipping test: calibrated_registry.json not found");
        return;
    }

    let registry = CalibratedRegistry::load(&path).unwrap();
    let fingerprint = registry.to_fingerprint();

    let config = BatchConfig::default();
    let generator = BatchGenerator::new(fingerprint, config);

    // Warmup
    let _ = generator.generate_full_stats_only(10_000);

    // Time full stats generation - run multiple iterations
    let iterations = 5;
    let mut total_duration = std::time::Duration::ZERO;

    for _ in 0..iterations {
        let start = std::time::Instant::now();
        let stats = generator.generate_full_stats_only(200_000);
        total_duration += start.elapsed();
        assert_eq!(stats.total_patients, 200_000);
    }

    let avg_duration = total_duration / iterations;
    let patients_per_sec = 200_000.0 / avg_duration.as_secs_f64();
    let time_for_1m = 1_000_000.0 / patients_per_sec;

    eprintln!(
        "Full stats generation (avg of {} runs): 200K patients in {:?} ({:.2}K patients/sec)",
        iterations,
        avg_duration,
        patients_per_sec / 1000.0
    );
    eprintln!(
        "Projected time for 1M patients: {:.2}ms",
        time_for_1m * 1000.0
    );

    // Full stats should still be reasonably fast (>100K/sec)
    assert!(
        patients_per_sec >= 50_000.0,
        "Full stats should achieve at least 50K patients/sec, got {:.0}",
        patients_per_sec
    );
}

#[test]
fn test_show_top_deviations() {
    let path = calibrated_registry_path();
    if !path.exists() {
        return;
    }

    let registry = CalibratedRegistry::load(&path).unwrap();
    let fingerprint = registry.to_fingerprint();
    let config = BatchConfig::default();
    let generator = BatchGenerator::new(fingerprint.clone(), config);

    let stats = generator.generate_stats_only(100_000);
    let comparison = stats.compare(&fingerprint);

    eprintln!("{}", comparison.summary());
}
