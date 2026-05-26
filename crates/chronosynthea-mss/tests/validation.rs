//! Statistical validation tests for MSS-based generation.
//!
//! These tests verify that generated patients match the expected
//! statistical distributions from the MSS fingerprint.

use std::collections::BTreeMap;
use chronosynthea_mss::batch::{BatchConfig, BatchGenerator};
use chronosynthea_mss::fingerprint::{
    ConditionStats, DemographicBucket, EncounterStats, JointDemographics, MssFingerprint,
};
use chronosynthea_mss::stats::StreamingStatistics;

/// Creates a test fingerprint with known distributions.
fn create_test_fingerprint() -> MssFingerprint {
    let mut buckets: BTreeMap<DemographicBucket, f64> = BTreeMap::new();
    buckets.insert(
        DemographicBucket::new("18-44", "male", "white", "nonhispanic"),
        0.25,
    );
    buckets.insert(
        DemographicBucket::new("18-44", "female", "white", "nonhispanic"),
        0.25,
    );
    buckets.insert(
        DemographicBucket::new("45-64", "male", "white", "nonhispanic"),
        0.25,
    );
    buckets.insert(
        DemographicBucket::new("65+", "female", "white", "nonhispanic"),
        0.25,
    );

    MssFingerprint {
        version: "1.0".to_string(),
        source: "test".to_string(),
        total_patients: 10000,
        total_encounters: 100000,
        joint_demographics: JointDemographics {
            buckets,
            total_patients: 10000,
        },
        conditions: vec![
            ConditionStats {
                code: "COND001".to_string(),
                display: "Test Condition 1".to_string(),
                prevalence: 0.30,
                by_age_bucket: BTreeMap::new(),
                by_gender: BTreeMap::new(),
                by_race: BTreeMap::new(),
                chronic: true,
                mean_onset_age: 50.0,
            },
            ConditionStats {
                code: "COND002".to_string(),
                display: "Test Condition 2".to_string(),
                prevalence: 0.10,
                by_age_bucket: BTreeMap::new(),
                by_gender: BTreeMap::new(),
                by_race: BTreeMap::new(),
                chronic: false,
                mean_onset_age: 40.0,
            },
            ConditionStats {
                code: "COND003".to_string(),
                display: "Test Condition 3".to_string(),
                prevalence: 0.05,
                by_age_bucket: BTreeMap::new(),
                by_gender: BTreeMap::new(),
                by_race: BTreeMap::new(),
                chronic: true,
                mean_onset_age: 60.0,
            },
        ],
        medications: vec![],
        observations: vec![],
        procedures: vec![],
        cooccurrence: BTreeMap::new(),
        cooccurrence_dependent_scale: BTreeMap::new(),
        onset_stats: Vec::new(),
        encounter_stats: EncounterStats {
            mean_by_age: BTreeMap::new(),
            type_distribution: BTreeMap::new(),
            mean_events_per_encounter: 5.0,
        },
    }
}

#[test]
fn test_condition_prevalence_matches_fingerprint() {
    let fp = create_test_fingerprint();
    let config = BatchConfig::default();
    let generator = BatchGenerator::new(fp.clone(), config);

    // Generate a large sample for statistical validity
    let stats = generator.generate_stats_only(100_000);

    // Check that observed prevalences are within tolerance of expected
    let prevalences = stats.condition_prevalences();

    for (i, cond) in fp.conditions.iter().enumerate() {
        let observed = prevalences[i];
        let expected = cond.prevalence;
        let tolerance = 0.02; // 2% tolerance

        assert!(
            (observed - expected).abs() < tolerance,
            "Condition {} prevalence mismatch: observed={:.4}, expected={:.4}, diff={:.4}",
            cond.code,
            observed,
            expected,
            (observed - expected).abs()
        );
    }
}

#[test]
fn test_deterministic_generation() {
    let fp = create_test_fingerprint();

    // Generate with same seed twice
    let config1 = BatchConfig {
        seed: 12345,
        ..Default::default()
    };
    let config2 = BatchConfig {
        seed: 12345,
        ..Default::default()
    };

    let generator1 = BatchGenerator::new(fp.clone(), config1);
    let generator2 = BatchGenerator::new(fp, config2);

    let patients1 = generator1.generate_compact(100);
    let patients2 = generator2.generate_compact(100);

    // Verify same patients are generated
    for (p1, p2) in patients1.iter().zip(patients2.iter()) {
        assert_eq!(p1.id, p2.id, "Patient IDs should match");
        assert_eq!(p1.sex, p2.sex, "Patient sex should match");
        assert_eq!(p1.race, p2.race, "Patient race should match");
        assert_eq!(
            p1.conditions.len(),
            p2.conditions.len(),
            "Patient condition count should match"
        );
    }
}

#[test]
fn test_different_seeds_produce_different_patients() {
    let fp = create_test_fingerprint();

    let config1 = BatchConfig {
        seed: 11111,
        ..Default::default()
    };
    let config2 = BatchConfig {
        seed: 22222,
        ..Default::default()
    };

    let generator1 = BatchGenerator::new(fp.clone(), config1);
    let generator2 = BatchGenerator::new(fp, config2);

    let stats1 = generator1.generate_stats_only(10_000);
    let stats2 = generator2.generate_stats_only(10_000);

    // Should have same total but different individual condition counts
    assert_eq!(stats1.total_patients, stats2.total_patients);

    // At least some conditions should have different counts
    let different_counts = stats1
        .condition_counts
        .iter()
        .zip(stats2.condition_counts.iter())
        .filter(|(&a, &b)| a != b)
        .count();

    assert!(
        different_counts > 0,
        "Different seeds should produce different condition distributions"
    );
}

#[test]
fn test_statistical_comparison() {
    let fp = create_test_fingerprint();
    let config = BatchConfig::default();
    let generator = BatchGenerator::new(fp.clone(), config);

    let stats = generator.generate_stats_only(50_000);
    let comparison = stats.compare(&fp);

    // Should pass validation with large enough sample
    assert!(
        comparison.max_deviation < 0.05,
        "Max deviation should be < 5%: {}",
        comparison.max_deviation
    );

    // KL divergence should be low
    assert!(
        comparison.kl_divergence < 0.1,
        "KL divergence should be low: {}",
        comparison.kl_divergence
    );
}

#[test]
fn test_patient_counts_accumulate() {
    let fp = create_test_fingerprint();
    let config = BatchConfig::default();
    let generator = BatchGenerator::new(fp, config);

    let stats = generator.generate_stats_only(1_000);

    assert_eq!(stats.total_patients, 1_000);
    assert!(
        stats.total_encounters > 0,
        "Should have generated encounters"
    );
    assert!(stats.total_events > 0, "Should have generated events");
}

#[test]
fn test_compact_patient_structure() {
    let fp = create_test_fingerprint();
    let config = BatchConfig::default();
    let generator = BatchGenerator::new(fp, config);

    let patients = generator.generate_compact(10);

    assert_eq!(patients.len(), 10);

    for patient in &patients {
        // Check basic structure
        assert!(patient.sex <= 1, "Sex should be 0 or 1");
        assert!(patient.race <= 5, "Race should be valid index");
        assert!(patient.ethnicity <= 2, "Ethnicity should be valid index");

        // Conditions should be valid indices
        for &cond_idx in &patient.conditions {
            assert!(cond_idx < 100, "Condition index should be valid");
        }
    }
}

#[test]
fn test_archetype_coverage() {
    use chronosynthea_mss::archetype::ArchetypeRegistry;

    let fp = create_test_fingerprint();
    let registry = ArchetypeRegistry::from_fingerprint(&fp);

    // Should have archetypes for each demographic bucket
    assert!(
        registry.len() >= 4,
        "Should have at least 4 archetypes (one per demo bucket)"
    );

    // Each archetype should have conditions
    for archetype in registry.archetypes() {
        assert!(
            archetype.active_conditions() > 0,
            "Archetype should have at least one condition"
        );
    }
}

#[test]
fn test_large_scale_generation() {
    let fp = create_test_fingerprint();
    let config = BatchConfig::default();
    let generator = BatchGenerator::new(fp, config);

    // Generate 100K patients (should complete quickly)
    let start = std::time::Instant::now();
    let stats = generator.generate_stats_only(100_000);
    let duration = start.elapsed();

    assert_eq!(stats.total_patients, 100_000);

    // Should complete in reasonable time (< 1 second for 100K)
    assert!(
        duration.as_secs_f64() < 1.0,
        "100K patients should generate in < 1 second, took {:?}",
        duration
    );

    let patients_per_sec = 100_000.0 / duration.as_secs_f64();
    eprintln!(
        "Generated 100K patients in {:?} ({:.0} patients/sec)",
        duration, patients_per_sec
    );
}

#[test]
fn test_streaming_statistics_merge() {
    use chronosynthea_mss::stats::StreamingStatistics;

    let mut stats1 = StreamingStatistics::new(10);
    stats1.total_patients = 500;
    stats1.condition_counts[0] = 150;
    stats1.condition_counts[1] = 50;

    let mut stats2 = StreamingStatistics::new(10);
    stats2.total_patients = 500;
    stats2.condition_counts[0] = 145;
    stats2.condition_counts[2] = 25;

    stats1.merge(&stats2);

    assert_eq!(stats1.total_patients, 1000);
    assert_eq!(stats1.condition_counts[0], 295);
    assert_eq!(stats1.condition_counts[1], 50);
    assert_eq!(stats1.condition_counts[2], 25);
}
