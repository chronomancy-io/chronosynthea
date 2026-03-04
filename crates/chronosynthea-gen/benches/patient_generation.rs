//! Benchmarks for patient generation performance.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use ahash::AHashMap;
use chronosynthea_gen::{
    CuratedRegistry, DemographicProfile, Generator, GeneratorConfig, OptimizedRegistry,
    ParallelGenerator,
};

fn create_test_registry() -> OptimizedRegistry {
    let mut age_dist = AHashMap::new();
    age_dist.insert("18-44".to_string(), 0.5);
    age_dist.insert("45-64".to_string(), 0.3);
    age_dist.insert("65+".to_string(), 0.2);

    let mut gender_dist = AHashMap::new();
    gender_dist.insert("M".to_string(), 0.5);
    gender_dist.insert("F".to_string(), 0.5);

    let mut race_dist = AHashMap::new();
    race_dist.insert("white".to_string(), 0.6);
    race_dist.insert("black".to_string(), 0.2);
    race_dist.insert("asian".to_string(), 0.1);
    race_dist.insert("other".to_string(), 0.1);

    OptimizedRegistry::new(CuratedRegistry {
        version: "1.0".to_string(),
        conditions: vec![],
        medications: vec![],
        observations: vec![],
        procedures: vec![],
        demographics: DemographicProfile {
            age_distribution: age_dist,
            gender_distribution: gender_dist,
            race_distribution: race_dist,
            ..Default::default()
        },
    })
}

fn bench_sequential_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_generation");

    for size in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let config = GeneratorConfig::with_patients(size).with_seed(42);
            let registry = create_test_registry();
            let generator = Generator::new(config, registry);

            b.iter(|| black_box(generator.generate(size).unwrap()));
        });
    }

    group.finish();
}

fn bench_parallel_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_generation");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let config = GeneratorConfig::with_patients(size).with_seed(42);
            let registry = create_test_registry();
            let generator = ParallelGenerator::new(config, registry);

            b.iter(|| black_box(generator.generate(size).unwrap()));
        });
    }

    group.finish();
}

fn bench_sequential_vs_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_vs_parallel");
    let size = 1000;

    group.throughput(Throughput::Elements(size as u64));

    group.bench_function("sequential_1000", |b| {
        let config = GeneratorConfig::with_patients(size).with_seed(42);
        let registry = create_test_registry();
        let generator = Generator::new(config, registry);

        b.iter(|| black_box(generator.generate(size).unwrap()))
    });

    group.bench_function("parallel_1000", |b| {
        let config = GeneratorConfig::with_patients(size).with_seed(42);
        let registry = create_test_registry();
        let generator = ParallelGenerator::new(config, registry);

        b.iter(|| black_box(generator.generate(size).unwrap()))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sequential_generation,
    bench_parallel_generation,
    bench_sequential_vs_parallel
);
criterion_main!(benches);
