//! Benchmarks for MSS-based patient generation.
//!
//! Target: 1M patients in < 1 second (1M+ patients/sec)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use ahash::AHashMap;
use chronosynthea_mss::batch::{BatchConfig, BatchGenerator};
use chronosynthea_mss::fingerprint::{
    ConditionStats, DemographicBucket, EncounterStats, JointDemographics, MssFingerprint,
};

/// Creates a realistic test fingerprint with typical condition distributions.
fn create_realistic_fingerprint() -> MssFingerprint {
    let mut buckets = AHashMap::new();

    // Realistic demographic distribution
    for age in &["0-17", "18-44", "45-64", "65+"] {
        for gender in &["male", "female"] {
            for race in &["white", "black", "asian", "hispanic", "other"] {
                for ethnicity in &["nonhispanic", "hispanic"] {
                    let weight = match (*age, *gender) {
                        ("18-44", _) => 0.35,
                        ("45-64", _) => 0.25,
                        ("65+", _) => 0.15,
                        _ => 0.25,
                    } / 10.0; // Distribute among race/ethnicity combinations

                    buckets.insert(DemographicBucket::new(age, gender, race, ethnicity), weight);
                }
            }
        }
    }

    // Common conditions with realistic prevalences
    let conditions = vec![
        ConditionStats {
            code: "38341003".to_string(),
            display: "Hypertension".to_string(),
            prevalence: 0.30,
            by_age_bucket: AHashMap::new(),
            by_gender: AHashMap::new(),
            by_race: AHashMap::new(),
            chronic: true,
            mean_onset_age: 50.0,
        },
        ConditionStats {
            code: "44054006".to_string(),
            display: "Type 2 Diabetes".to_string(),
            prevalence: 0.10,
            by_age_bucket: AHashMap::new(),
            by_gender: AHashMap::new(),
            by_race: AHashMap::new(),
            chronic: true,
            mean_onset_age: 55.0,
        },
        ConditionStats {
            code: "195967001".to_string(),
            display: "Asthma".to_string(),
            prevalence: 0.08,
            by_age_bucket: AHashMap::new(),
            by_gender: AHashMap::new(),
            by_race: AHashMap::new(),
            chronic: true,
            mean_onset_age: 15.0,
        },
        ConditionStats {
            code: "13644009".to_string(),
            display: "Hypercholesterolemia".to_string(),
            prevalence: 0.25,
            by_age_bucket: AHashMap::new(),
            by_gender: AHashMap::new(),
            by_race: AHashMap::new(),
            chronic: true,
            mean_onset_age: 45.0,
        },
        ConditionStats {
            code: "73211009".to_string(),
            display: "Type 1 Diabetes".to_string(),
            prevalence: 0.01,
            by_age_bucket: AHashMap::new(),
            by_gender: AHashMap::new(),
            by_race: AHashMap::new(),
            chronic: true,
            mean_onset_age: 12.0,
        },
        ConditionStats {
            code: "40930008".to_string(),
            display: "COPD".to_string(),
            prevalence: 0.06,
            by_age_bucket: AHashMap::new(),
            by_gender: AHashMap::new(),
            by_race: AHashMap::new(),
            chronic: true,
            mean_onset_age: 60.0,
        },
        ConditionStats {
            code: "53741008".to_string(),
            display: "Coronary Artery Disease".to_string(),
            prevalence: 0.07,
            by_age_bucket: AHashMap::new(),
            by_gender: AHashMap::new(),
            by_race: AHashMap::new(),
            chronic: true,
            mean_onset_age: 62.0,
        },
        ConditionStats {
            code: "35489007".to_string(),
            display: "Depression".to_string(),
            prevalence: 0.08,
            by_age_bucket: AHashMap::new(),
            by_gender: AHashMap::new(),
            by_race: AHashMap::new(),
            chronic: true,
            mean_onset_age: 30.0,
        },
    ];

    // Add more conditions to simulate realistic workload (total ~50 conditions)
    let mut all_conditions = conditions;
    for i in 0..42 {
        all_conditions.push(ConditionStats {
            code: format!("COND{:03}", i),
            display: format!("Condition {}", i),
            prevalence: 0.01 + (i as f64 * 0.002),
            by_age_bucket: AHashMap::new(),
            by_gender: AHashMap::new(),
            by_race: AHashMap::new(),
            chronic: i % 3 == 0,
            mean_onset_age: 30.0 + (i as f64),
        });
    }

    MssFingerprint {
        version: "1.0".to_string(),
        source: "benchmark".to_string(),
        total_patients: 100000,
        total_encounters: 1000000,
        joint_demographics: JointDemographics {
            buckets,
            total_patients: 100000,
        },
        conditions: all_conditions,
        medications: vec![],
        observations: vec![],
        procedures: vec![],
        cooccurrence: AHashMap::new(),
        encounter_stats: EncounterStats {
            mean_by_age: AHashMap::new(),
            type_distribution: AHashMap::new(),
            mean_events_per_encounter: 5.0,
        },
    }
}

fn bench_stats_only(c: &mut Criterion) {
    let fp = create_realistic_fingerprint();
    let config = BatchConfig::default();
    let generator = BatchGenerator::new(fp, config);

    let mut group = c.benchmark_group("mss_stats_only");

    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let stats = generator.generate_stats_only(black_box(size));
                black_box(stats)
            });
        });
    }

    group.finish();
}

fn bench_compact_generation(c: &mut Criterion) {
    let fp = create_realistic_fingerprint();
    let config = BatchConfig::default();
    let generator = BatchGenerator::new(fp, config);

    let mut group = c.benchmark_group("mss_compact");

    for size in [1_000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let patients = generator.generate_compact(black_box(size));
                black_box(patients)
            });
        });
    }

    group.finish();
}

fn bench_archetype_sampling(c: &mut Criterion) {
    use chronosynthea_mss::archetype::ArchetypeRegistry;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    let fp = create_realistic_fingerprint();
    let registry = ArchetypeRegistry::from_fingerprint(&fp);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

    c.bench_function("archetype_sample", |b| {
        b.iter(|| {
            let arch = registry.sample(&mut rng);
            black_box(arch.id)
        });
    });
}

fn bench_condition_sampling(c: &mut Criterion) {
    use chronosynthea_mss::archetype::ArchetypeRegistry;
    use chronosynthea_mss::sampler::SimdSampler;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use smallvec::SmallVec;

    let fp = create_realistic_fingerprint();
    let registry = ArchetypeRegistry::from_fingerprint(&fp);
    let mut sampler = SimdSampler::from_registry(&registry);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut buffer: SmallVec<[u16; 8]> = SmallVec::new();

    // Get a sample archetype
    let archetype = registry.sample(&mut rng);
    let thresholds = registry.condition_thresholds(archetype.id);

    c.bench_function("condition_sample_simd", |b| {
        b.iter(|| {
            sampler.sample_conditions(thresholds, &mut rng, &mut buffer);
            black_box(buffer.len())
        });
    });
}

fn bench_million_patients(c: &mut Criterion) {
    let fp = create_realistic_fingerprint();
    let config = BatchConfig::default();
    let generator = BatchGenerator::new(fp, config);

    let mut group = c.benchmark_group("million_patients");
    group.sample_size(10); // Fewer samples for long-running benchmark
    group.measurement_time(std::time::Duration::from_secs(30));

    group.throughput(Throughput::Elements(1_000_000));
    group.bench_function("1M_patients", |b| {
        b.iter(|| {
            let stats = generator.generate_stats_only(black_box(1_000_000));
            black_box(stats)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_stats_only,
    bench_compact_generation,
    bench_archetype_sampling,
    bench_condition_sampling,
    bench_million_patients,
);

criterion_main!(benches);
