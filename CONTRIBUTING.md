# Contributing to ChronoSynthea

Thank you for your interest in contributing! This document outlines the process and requirements for contributing to ChronoSynthea.

## Getting Started

### Prerequisites

- Rust 1.75+ (install via [rustup](https://rustup.rs/))
- Git

### Setup

```bash
git clone https://github.com/your-org/chronosynthea
cd chronosynthea
cargo build
cargo test
```

## Code Review Process

All PRs must satisfy code quality requirements and pass automated checks before merge.

### Quality Checklist

- [ ] Code follows Rust idioms and best practices
- [ ] All tests pass (`cargo test --workspace`)
- [ ] No clippy warnings (`cargo clippy --workspace -- -D warnings`)
- [ ] Code is formatted (`cargo fmt --all`)
- [ ] Documentation is updated if needed
- [ ] Performance benchmarks show no regression (if applicable)

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_function_name() {
    let result = function_under_test(input);
    assert_eq!(result, expected);
}
```

### Integration Tests

Integration tests live in `crates/*/tests/`:

```rust
#[test]
fn test_java_validation() {
    let registry = CalibratedRegistry::load("path/to/registry.json").unwrap();
    let fingerprint = registry.to_fingerprint();
    // ...
}
```

### Performance Tests

```rust
#[bench]
fn bench_generation(b: &mut Bencher) {
    b.iter(|| {
        generator.generate_stats_only(1000);
    });
}
```

**Commands:**

```bash
# Run all tests
cargo test --workspace

# Run tests in release mode (for accurate performance)
cargo test --workspace --release

# Run specific test
cargo test --package chronosynthea-mss test_name

# Run benchmarks
cargo bench --package chronosynthea-mss
```

## Performance Requirements

ChronoSynthea has strict performance requirements:

| Metric | Requirement |
|--------|-------------|
| Throughput | ≥ 1M patients/second |
| Statistical deviation | < 1% max deviation |
| Memory | < 100 MB for 1M patients (stats-only) |

### Running Performance Validation

```bash
cargo test --package chronosynthea-mss --test java_validation --release \
    -- test_full_generation_performance --nocapture
```

Expected output should show `Projected time for 1M patients: < 1000ms`.

## Complexity Analysis

For any algorithm contribution:

1. **State the complexity:** Big-O notation (e.g., O(n log n))
2. **Justify it:** Explain the reasoning
3. **Benchmark it:** Include criterion benchmarks
4. **Document trade-offs:** Space, readability, maintenance

See [PERFORMANCE.md](PERFORMANCE.md) for examples.

## Submission Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Implement with tests
4. Run the full test suite:
   ```bash
   cargo fmt --all
   cargo clippy --workspace -- -D warnings
   cargo test --workspace --release
   ```
5. Push and open a Pull Request
6. All CI checks must pass before merge

## Commit Message Format

Use conventional commits:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting (no code change)
- `refactor`: Code restructuring (no behavior change)
- `perf`: Performance improvement
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

**Examples:**

```
feat(mss): add SIMD sampling for medications

Implements f32x8 vectorized sampling for medication thresholds,
reducing per-patient sampling time by 8x.

perf(batch): optimize atomic statistics recording

Replace bounds-checked indexing with unchecked access in hot loop.
Improves throughput from 1.2M to 1.6M patients/sec.
```

## Architecture Guidelines

### Crate Organization

- `chronosynthea-mss`: Core MSS implementation (primary development focus)
- `chronosynthea-cde`: CDE encoding (stable, minimal changes)
- `chronosynthea-core`: Shared types (stable)
- `chronosynthea-io`: I/O utilities (stable)

### Key Principles

1. **Zero-allocation hot paths**: Use arena allocation or pre-allocated buffers
2. **Lock-free parallelism**: Prefer atomics over mutexes
3. **SIMD where possible**: Use `wide` crate for vectorized operations
4. **Statistical correctness**: All changes must maintain < 1% deviation

## Questions?

- See [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- See [PERFORMANCE.md](PERFORMANCE.md) for benchmark methodology
- Open an issue for clarification

---

*Thank you for contributing to ChronoSynthea!*
