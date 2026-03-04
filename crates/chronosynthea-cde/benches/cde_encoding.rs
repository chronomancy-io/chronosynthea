//! Benchmarks for CDE encoding performance.

use chronosynthea_cde::{
    compute_structural_features, encode_module, encode_module_structural, extract_state_features,
    AxisModel, WeightedLinearAxisModel,
};
use chronosynthea_core::load_module_from_str;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn generate_module(num_states: usize) -> String {
    let mut states = Vec::with_capacity(num_states);

    states.push(r#""Initial": {"type": "Initial", "direct_transition": "State0"}"#.to_string());

    for i in 0..num_states - 2 {
        states.push(format!(
            r#""State{}": {{"type": "Simple", "direct_transition": "State{}"}}"#,
            i,
            i + 1
        ));
    }

    states.push(format!(
        r#""State{}": {{"type": "Simple", "direct_transition": "Terminal"}}"#,
        num_states - 2
    ));
    states.push(r#""Terminal": {"type": "Terminal"}"#.to_string());

    format!(
        r#"{{"name": "Benchmark Module", "states": {{{}}}}}"#,
        states.join(",")
    )
}

fn bench_cde_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("cde_encode");

    // Small module (10 states)
    let small_json = generate_module(10);
    let small_module = load_module_from_str(&small_json).unwrap();

    group.bench_function("structural_10_states", |b| {
        b.iter(|| encode_module_structural(black_box(&small_module)).unwrap())
    });

    // Medium module (50 states)
    let medium_json = generate_module(50);
    let medium_module = load_module_from_str(&medium_json).unwrap();

    group.bench_function("structural_50_states", |b| {
        b.iter(|| encode_module_structural(black_box(&medium_module)).unwrap())
    });

    group.bench_function("full_semantic_50_states", |b| {
        b.iter(|| encode_module(black_box(&medium_module)).unwrap())
    });

    // Large module (200 states)
    let large_json = generate_module(200);
    let large_module = load_module_from_str(&large_json).unwrap();

    group.bench_function("structural_200_states", |b| {
        b.iter(|| encode_module_structural(black_box(&large_module)).unwrap())
    });

    group.finish();
}

fn bench_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extraction");

    let medium_json = generate_module(50);
    let medium_module = load_module_from_str(&medium_json).unwrap();

    group.bench_function("structural_features_50_states", |b| {
        b.iter(|| compute_structural_features(black_box(&medium_module)))
    });

    group.bench_function("semantic_features_50_states", |b| {
        b.iter(|| extract_state_features(black_box(&medium_module)))
    });

    group.finish();
}

fn bench_axis_model(c: &mut Criterion) {
    let mut group = c.benchmark_group("axis_model");

    let model = WeightedLinearAxisModel::default_model();

    let medium_json = generate_module(50);
    let medium_module = load_module_from_str(&medium_json).unwrap();
    let features = compute_structural_features(&medium_module);

    group.bench_function("linear_model_encode", |b| {
        b.iter(|| {
            for fv in features.values() {
                black_box(model.encode(fv).unwrap());
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_cde_encode,
    bench_feature_extraction,
    bench_axis_model
);
criterion_main!(benches);
