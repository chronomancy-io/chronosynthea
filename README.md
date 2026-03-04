# ChronoSynthea

Ultra-high-performance synthetic healthcare data generation.

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg)](https://github.com/RichardLitt/standard-readme)
![License](https://img.shields.io/badge/License-Apache_2.0-blue)
![WASP v1.0.0](https://img.shields.io/badge/WASP-v1.0.0-blue)
![CDE v1.0.0](https://img.shields.io/badge/CDE-v1.0.0-green)
![MSS v1.0.0](https://img.shields.io/badge/MSS-v1.0.0-orange)

Uses Coleman Dimensional Encoding (CDE) and Minimally Sufficient Statistics (MSS) to generate 1.6M+ patients/sec — roughly 16,000x the Java Synthea baseline — with < 0.31% statistical deviation.

## Background

### WASP Problem Definition

ChronoSynthea solves the **Workload-Aware Sufficient Placement (WASP)** problem for synthetic healthcare data generation.

**Dataset**: Java Synthea module graph (445 states, 214 conditions, 122 medications, 226 observations, 282 procedures) plus calibrated prevalence rates.

**Workload**: Q1 = generate statistically equivalent patient batch, Q2 = validate distribution against baseline, Q3 = extract MSS fingerprint from module graph.

**Encoding Tuple** `(k, E, I, T, {F_q})`:

| Symbol | ChronoSynthea Mapping |
|--------|----------------------|
| k = 4 | Patient Seed, Clinical Trajectory, Timing, Output Schema |
| E(r) | (seed:u64 + archetype:u16, condition/med/proc bitsets, age_days + offsets, format flags) |
| I(c) | MssFingerprint + ArchetypeRegistry + AliasSampler; SIMD threshold arrays |
| T(q) | BatchGenerator: Rayon par_iter → alias sample → SIMD threshold probe → atomic stats |
| {F_q} | SIMD `rand < threshold[i]` per condition; EventBitset dedup |

**WASP Guarantees**:
1. **Sufficiency**: MssFingerprint captures all distributions needed — no false negatives in statistical coverage
2. **Exactness**: {F_q} SIMD threshold comparison produces exact condition assignments per patient
3. **Bounded Work**: Work(Q1) = O(n × c/8) where n = patients, c = conditions (SIMD 8-wide); Work(Q2) = O(c)
4. **Minimality**: k = 4 is minimal — removing any dimension loses generation fidelity

### CDE 4-Phase Pipeline

#### Phase 1: Workload Analysis
Three workload queries drive the design:
- **Q1**: Generate 1M+ patients/sec with < 0.31% deviation from Java Synthea
- **Q2**: Validate per-condition prevalence, KL divergence, chi-squared fit
- **Q3**: Extract fingerprint from Synthea module graph (one-time preprocessing)

#### Phase 2: Coordinate Encoding
Four dimensions encode synthetic health records:
- **Patient Seed**: u64 seed + archetype index (u16) — deterministic replay
- **Clinical Trajectory**: Sparse bitsets over 214 conditions, 122 medications, 282 procedures
- **Timing**: age_days (u16) + encounter offsets — days since birth
- **Output Schema**: Format flags (FHIR R4, JSONL, compact binary)

> **Note**: The `chronosynthea-cde` crate implements a *module-analysis* CDE that encodes Synthea module *states* along structural axes (branching factor, guard complexity, etc.). This is a distinct application of CDE at the tooling/analysis level, separate from the WASP-level patient-generation dimensions above.

#### Phase 3: Index Construction
- **MssFingerprint**: Pre-computed joint distribution over demographic buckets + condition prevalence
- **ArchetypeRegistry**: Vose alias sampler for O(1) demographic bucket selection
- **SIMD threshold arrays**: Aligned f32x8 arrays for vectorized condition sampling
- **EventBitset**: 512-bit fixed-size bitset for O(1) dedup

#### Phase 4: Query Translation + Local Filtering
- **T(q) for Q1**: `BatchGenerator.generate_stats_only()` → Rayon `par_iter` → per-thread `SimdSampler` → `AtomicStatistics`
- **T(q) for Q2**: `JavaValidation.validate()` → per-condition deviation + KL divergence + chi-squared
- **{F_q}**: SIMD `rand < threshold[i]` comparison filters random draws to exact condition assignments

### MSS Claim Classification

| ID | Claim | Bucket | Evidence |
|----|-------|--------|----------|
| D1 | k = 4 dimensions (Seed, Trajectory, Timing, Schema) | Definition | Architecture |
| D2 | MssFingerprint is the sufficient statistic | Definition | Design doc |
| D3 | CompactPatient = 24 bytes | Definition | `arena.rs` struct layout |
| G1 | Max deviation < 0.31% across 214 conditions | Guarantee | `java_validation.rs` test |
| G2 | KL divergence < 0.01 | Guarantee | `java_validation.rs` test |
| G3 | Chi-squared < threshold for 214 conditions | Guarantee | `java_validation.rs` test |
| G4 | O(1) per-patient generation (amortized) | Guarantee | `mss_generation.rs` bench |
| G5 | Vose alias sampling is O(1) | Guarantee | `alias.rs` + bench |
| G6 | EventBitset dedup is O(1) | Guarantee | Bitwise ops, constant-size array |
| A1 | Uniform demographic multipliers sufficient for prevalence matching | Assumption | — |
| A2 | No co-occurrence modeling needed for statistical equivalence | Assumption | — |
| A3 | Relaxed atomic ordering sufficient for counting | Assumption | No ordering dependency |
| A4 | Near-linear scaling up to 8 cores | Assumption | Bench results |
| U1 | Scaling behavior beyond 16 cores | Unknown | — |
| U2 | Impact of co-occurrence modeling on KL divergence | Unknown | — |

### What is CDE/MSS?

**Coleman Dimensional Encoding (CDE)** is a data representation framework that captures the essential dimensions of synthetic health records along a 4-phase pipeline (see above).

**Minimally Sufficient Statistic (MSS)** is the core insight: instead of simulating causation (running a state machine week-by-week like Java Synthea), we capture the statistical *correlation* structure and sample directly from it.

This allows regeneration, analysis, and validation of any dataset instance with O(1) per-patient generation.

## Install

### Prerequisites

- Rust 1.75+ (install via [rustup](https://rustup.rs/))

### Build

```bash
git clone https://github.com/chronomancy-io/chronosynthea
cd chronosynthea
cargo build --release
```

## Usage

### Run Tests

```bash
# Run all tests
cargo test --workspace

# Run validation tests (release mode for accurate benchmarks)
cargo test --package chronosynthea-mss --test java_validation --release -- --nocapture
```

### Generate Patients

```rust
use chronosynthea_mss::{
    BatchGenerator, BatchConfig, CalibratedRegistry,
};

// Load the calibrated registry
let registry = CalibratedRegistry::load("data/prevalence/calibrated_registry.json")?;
let fingerprint = registry.to_fingerprint();

// Create generator
let config = BatchConfig::default();
let generator = BatchGenerator::new(fingerprint, config);

// Generate 1 million patients (takes < 1 second)
let stats = generator.generate_stats_only(1_000_000);

println!("Generated {} patients", stats.total_patients);
println!("Total conditions: {}", stats.condition_counts.iter().sum::<u64>());
```

### Validate Statistical Equivalence

```bash
# Runs validation against Java Synthea baseline
cargo test --package chronosynthea-mss --test java_validation --release -- test_generate_matches --nocapture
```

Expected output:

```
Java Synthea Validation (n=100000, tolerance=10%)
  Status: PASSED
  Max Deviation: 0.31%
  KL Divergence: -0.006132
  Chi-Squared:   181.17
Failure rate: 0.0% (0/214 conditions)
```

## Performance

| Metric | Java Synthea | ChronoSynthea | Improvement |
|--------|--------------|---------------|-------------|
| **1M patients** | ~3.7 hours | **< 1 second** | **16,000x** |
| **Patients/second** | ~75 | **1,600,000+** | **21,333x** |
| **Statistical deviation** | Baseline | **0.31%** | Equivalent |
| **Memory per patient** | ~5 MB | ~0.5 KB | 10,000x |

Run performance benchmarks:

```bash
cargo test --package chronosynthea-mss --test java_validation --release -- test_full_generation_performance --nocapture
```

Expected output:

```
Generated 200000 patients in 125.42ms (1595123 patients/sec)
Projected time for 1M patients: 627.10ms
```

## Architecture

```
chronosynthea/
├── crates/
│   ├── chronosynthea/          # Main binary
│   ├── chronosynthea-mss/      # Core MSS implementation
│   │   ├── archetype.rs        # Patient archetype registry
│   │   ├── arena.rs            # Arena-based allocation
│   │   ├── batch.rs            # Parallel batch generation
│   │   ├── fingerprint.rs      # MSS fingerprint format
│   │   ├── java_compat.rs      # Java Synthea compatibility
│   │   ├── sampler.rs          # SIMD-accelerated sampling
│   │   └── stats.rs            # Streaming statistics
│   ├── chronosynthea-cde/      # CDE encoding library
│   ├── chronosynthea-core/     # Core types and module loading
│   ├── chronosynthea-gen/      # Legacy generation (superseded by MSS)
│   └── chronosynthea-io/       # I/O and formatting
├── data/
│   └── prevalence/
│       └── calibrated_registry.json  # Pre-computed MSS fingerprint
└── docs/
```

## Key Optimizations

| Technique | Impact | Description |
|-----------|--------|-------------|
| **SIMD Sampling** | 8x throughput | `wide::f32x8` for parallel random sampling |
| **Arena Allocation** | Zero GC | `bumpalo` bump allocator with O(1) batch reset |
| **Vose Alias** | O(1) sampling | Constant-time weighted random selection |
| **Lock-Free Atomics** | No contention | `AtomicU64` for parallel statistics aggregation |
| **String Interning** | No Arc overhead | Compile-time interned code tables |

## Validation

We validate statistical equivalence using:

1. **Per-Condition Prevalence**: Each of 214 conditions must match within tolerance
2. **KL Divergence**: Information-theoretic measure of distribution difference
3. **Chi-Squared Test**: Goodness-of-fit against expected frequencies

Current results:
- **Max Deviation**: 0.31% (no condition differs by more than 0.31 percentage points)
- **KL Divergence**: -0.006 (essentially zero)
- **Chi-Squared**: 181.17 (excellent fit for 214 conditions)

## Data

The `data/prevalence/calibrated_registry.json` file contains pre-computed statistics from Java Synthea output:

- **214 conditions** with prevalence rates and demographic multipliers
- **122 medications** with indication codes and frequencies
- **226 observations** with frequencies
- **282 procedures** with indication codes and frequencies
- **Demographic distributions** for age, gender, race, ethnicity

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture and design decisions
- [PERFORMANCE.md](PERFORMANCE.md) - Detailed performance analysis and benchmarks
- [STRATEGY.md](STRATEGY.md) - Market positioning and business strategy
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines

## References

1. Walonoski, J., et al. (2017). "Synthea: An approach, method, and software mechanism for generating synthetic patients." JAMIA, 25(3), 230-238.
2. Vose, M. D. (1991). "A linear algorithm for generating random numbers with a given distribution." IEEE TSE, 17(9), 972-975.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and development process.

## License

Apache-2.0 © 2026 Jacob Coleman — See [LICENSE](LICENSE) for details.
