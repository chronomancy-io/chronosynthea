//! Benchmarks for module loading performance.

use chronosynthea_core::load_module_from_str;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

const SMALL_MODULE: &str = r#"{
    "name": "Small Test Module",
    "states": {
        "Initial": {"type": "Initial", "direct_transition": "State1"},
        "State1": {"type": "Simple", "direct_transition": "State2"},
        "State2": {"type": "Simple", "direct_transition": "State3"},
        "State3": {"type": "Simple", "direct_transition": "State4"},
        "State4": {"type": "Simple", "direct_transition": "State5"},
        "State5": {"type": "Simple", "direct_transition": "State6"},
        "State6": {"type": "Simple", "direct_transition": "State7"},
        "State7": {"type": "Simple", "direct_transition": "State8"},
        "State8": {"type": "Simple", "direct_transition": "State9"},
        "State9": {"type": "Simple", "direct_transition": "Terminal"},
        "Terminal": {"type": "Terminal"}
    }
}"#;

fn generate_large_module(num_states: usize) -> String {
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
        r#"{{"name": "Large Module", "states": {{{}}}}}"#,
        states.join(",")
    )
}

fn bench_module_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("module_loading");

    // Small module (10 states)
    group.bench_function("small_10_states", |b| {
        b.iter(|| load_module_from_str(black_box(SMALL_MODULE)).unwrap())
    });

    // Medium module (50 states)
    let medium_module = generate_large_module(50);
    group.bench_function("medium_50_states", |b| {
        b.iter(|| load_module_from_str(black_box(&medium_module)).unwrap())
    });

    // Large module (200 states)
    let large_module = generate_large_module(200);
    group.bench_function("large_200_states", |b| {
        b.iter(|| load_module_from_str(black_box(&large_module)).unwrap())
    });

    group.finish();
}

fn bench_edge_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_computation");

    let medium_module = generate_large_module(50);
    let module = load_module_from_str(&medium_module).unwrap();

    group.bench_function("edges_50_states", |b| b.iter(|| black_box(module.edges())));

    let large_module = generate_large_module(200);
    let module = load_module_from_str(&large_module).unwrap();

    group.bench_function("edges_200_states", |b| b.iter(|| black_box(module.edges())));

    group.finish();
}

criterion_group!(benches, bench_module_loading, bench_edge_computation);
criterion_main!(benches);
