# ChronoSynthea

Ultra-high-performance synthetic healthcare data generation.

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg)](https://github.com/RichardLitt/standard-readme)
![License](https://img.shields.io/badge/License-Apache_2.0-blue)
![WASP v1.0.0](https://img.shields.io/badge/WASP-v1.0.0-blue)
![CDE v1.0.0](https://img.shields.io/badge/CDE-v1.0.0-green)
![MSS v1.0.0](https://img.shields.io/badge/MSS-v1.0.0-orange)

Uses Coleman Dimensional Encoding (CDE) and Minimally Sufficient Statistics (MSS) to generate synthetic patient *statistics* at millions of patients/sec on a multi-core CPU. The fastest path counts conditions/medications/etc. into atomic counters without materializing patient records; see [Performance](#performance) for the exact paths, machine, and measured numbers.

> **Honesty note.** ChronoSynthea samples each condition independently at a precomputed base rate. It does **not** reproduce Java Synthea's causal week-by-week state machine, and the statistical-deviation figure below is sampling noise against ChronoSynthea's *own* base rates — it is **not** a validated equivalence to Java Synthea output. See [Validation](#validation) and the MSS claim table for the precise scope.

## Background

### WASP Problem Definition

ChronoSynthea solves the **Workload-Aware Sufficient Placement (WASP)** problem for synthetic healthcare data generation.

**Dataset**: A calibrated prevalence registry (`data/prevalence/calibrated_registry.json`) containing 214 conditions, 122 medications, 226 observations, and 282 procedures, each with a base rate. The registry's `description` field claims these rates match Java Synthea occurrence rates; that match is **Assumed**, not verified in this repo (no Java Synthea run, version, or source dataset is committed here).

**Workload**: Q1 = generate statistically equivalent patient batch, Q2 = validate distribution against baseline, Q3 = extract MSS fingerprint from module graph.

**Encoding Tuple** `(k, E, I, T, {F_q})`:

| Symbol | ChronoSynthea Mapping |
|--------|----------------------|
| k = 4 | Patient Seed, Clinical Trajectory, Timing, Output Schema |
| E(r) | (seed:u64 + archetype:u16, condition/med/proc bitsets, age_days + offsets, format flags) |
| I(c) | MssFingerprint + ArchetypeRegistry (Vose alias table for archetype selection) + per-archetype f32 threshold arrays |
| T(q) | BatchGenerator: Rayon par_iter → Vose-alias archetype sample (O(1) scalar) → per-condition threshold draws → atomic stats |
| {F_q} | per-element `rand < threshold[i]` over conditions/meds/obs/procs (SIMD f32x8 in the `sampler.rs` paths; scalar in the default flat path); EventBitset dedup |

**WASP claims** (each labeled per MSS — see the claim table below for evidence):
1. **Sufficiency** (Assumption): MssFingerprint captures the marginal distributions sampled from. Joint/co-occurrence structure is *not* captured (co-occurrence is disabled in the default fingerprint), so this is sufficiency for marginals only.
2. **Exactness** (Guarantee): the threshold comparison `rand < threshold[i]` deterministically assigns condition *i* for a given RNG draw.
3. **Bounded Work** (Guarantee): Work(Q1) = O(n × c) per the per-condition loop (the SIMD path processes 8 lanes per step but still touches all c conditions); Work(Q2) = O(c).
4. **Minimality** (Definition): k = 4 dimensions is the chosen encoding; "removing any dimension loses fidelity" is asserted by design, not proven here (Assumption).

> **Note on the two sampling primitives** (these were previously conflated): the **Vose alias table** gives O(1) *scalar* selection of one demographic-bucket archetype per patient; the **SIMD f32x8** code performs the per-condition / per-medication / per-event threshold draws (`rand < threshold`). They are distinct mechanisms. The default `generate_stats_only` flat path samples conditions with a *scalar* loop (`sample_conditions_flat`); the SIMD `f32x8` routines in `sampler.rs` are used for the full-stats medication/observation/procedure draws.

### CDE 4-Phase Pipeline

#### Phase 1: Workload Analysis
Three workload queries drive the design:
- **Q1**: Generate patient statistics in parallel (measured at millions/sec — see Performance)
- **Q2**: Validate per-condition prevalence, KL divergence, chi-squared fit against the fingerprint's base rates
- **Q3**: Extract a fingerprint from a calibrated registry (one-time preprocessing)

#### Phase 2: Coordinate Encoding
Four dimensions encode synthetic health records:
- **Patient Seed**: u64 seed + archetype index (u16) — deterministic replay
- **Clinical Trajectory**: Sparse `u16` index lists (`SmallVec`) over 214 conditions, 122 medications, 226 observations, 282 procedures (a 512-bit `EventBitset` is used only for O(1) dedup during accumulation, not as the primary representation)
- **Timing**: age_days (u16) + encounter offsets — days since birth
- **Output Schema**: Format flags (FHIR R4, JSONL, compact binary)

> **Note**: The `chronosynthea-cde` crate implements a *module-analysis* CDE that encodes Synthea module *states* along structural axes (branching factor, guard complexity, etc.). This is a distinct application of CDE at the tooling/analysis level, separate from the WASP-level patient-generation dimensions above.

#### Phase 3: Index Construction
- **MssFingerprint**: Pre-computed distribution over demographic buckets + per-bucket condition prevalence. (Co-occurrence is a field in the fingerprint but is left empty by `CalibratedRegistry::to_fingerprint`, so generation samples conditions independently.)
- **ArchetypeRegistry**: Vose alias table for O(1) selection of one demographic-bucket archetype per patient
- **Threshold arrays**: Per-archetype `f32` arrays of condition/medication probabilities, padded to a multiple of 8 so the `f32x8` SIMD routines can read 8 lanes at a time
- **EventBitset**: 512-bit fixed-size bitset (`[u64; 8]`) for O(1) dedup of accumulated events

#### Phase 4: Query Translation + Local Filtering
- **T(q) for Q1**: `BatchGenerator.generate_stats_only()` → Rayon `par_iter` → per-thread RNG + scalar `sample_conditions_flat` → `AtomicStatistics` (the full-event path `generate_full_stats_only` additionally uses the `f32x8` SIMD routines for medications/observations/procedures)
- **T(q) for Q2**: `JavaValidation.validate()` → per-condition deviation + KL divergence + chi-squared, computed against the fingerprint's own base rates
- **{F_q}**: `rand < threshold[i]` comparison maps random draws to condition assignments

### MSS Claim Classification

| ID | Claim | Bucket | Evidence |
|----|-------|--------|----------|
| D1 | k = 4 dimensions (Seed, Trajectory, Timing, Schema) | Definition | This README / ARCHITECTURE.md |
| D2 | MssFingerprint is the sufficient statistic for the *marginal* distributions sampled | Definition | `fingerprint.rs` (joint/co-occurrence not populated by default) |
| D3 | `CompactPatient` is a small fixed struct plus an inline `SmallVec` of condition indices | Definition | `arena.rs` (`CompactPatient`); see note below — the literal "24 bytes" claim is **not** accurate for the current struct |
| G1 | Per-condition prevalence of generated stats stays within ~0.31% of the fingerprint's own base rates at n=100K | Guarantee (self-consistency, **not** Java equivalence) | `tests/java_validation.rs::test_show_top_deviations` / `test_generate_matches_java_baseline` |
| G2 | KL divergence vs the fingerprint's base rates ≈ 0 (measured -0.006132 at n=100K) | Guarantee (self-consistency) | same test |
| G3 | Chi-squared vs base rates = 181.17 at n=100K (214 conditions) | Guarantee (self-consistency) | same test |
| G4 | Per-patient work is bounded (O(c+m+e), independent of patient lifespan) | Guarantee | per-condition loops in `archetype.rs` / `sampler.rs`; throughput in `benches/mss_generation.rs` |
| G5 | Vose alias *archetype* sampling is O(1) | Guarantee | `archetype.rs::AliasTable::sample` + `benches/mss_generation.rs` (`archetype_sample` ≈ 8 ns) |
| G6 | EventBitset `test_and_set` / `clear` are O(1) | Guarantee | `sampler.rs` (bitwise ops on a constant-size `[u64; 8]`) |
| A1 | Uniform (all-1.0) demographic multipliers are used; per-demographic variation is *not* modeled | Assumption | `java_compat.rs::fine_tune_age_multipliers` |
| A2 | Independent per-condition sampling (no co-occurrence) is adequate for the intended use | Assumption | `java_compat.rs::to_fingerprint` sets `cooccurrence = AHashMap::new()` |
| A3 | Relaxed atomic ordering is sufficient for counting | Assumption | `batch.rs` uses `Ordering::Relaxed`; counts are order-independent |
| A4 | The base rates in `calibrated_registry.json` match Java Synthea occurrence rates | Assumption | registry `description` field only — no Java run/source in repo |
| U1 | Throughput vs a real, measured Java Synthea baseline | Unknown | no Java baseline measured in this repo |
| U2 | Statistical equivalence to actual Java Synthea output (joint structure, comorbidities, timing) | Unknown | not measured |
| U3 | Scaling behavior beyond the 8-core / 16-thread machine measured here | Unknown | only one machine measured |

> **D3 detail**: `CompactPatient` (in `arena.rs`) holds `id: u64`, `birth_date_days: i32`, `sex/race/ethnicity/encounter_count/condition_count: u8`, `archetype_id: u16`, and an inline `conditions: SmallVec<[u16; 8]>`. It is not 24 bytes; the older "24 bytes" figure described a different/earlier layout and has been removed.

### What is CDE/MSS?

**Coleman Dimensional Encoding (CDE)** is a data representation framework that captures the essential dimensions of synthetic health records along a 4-phase pipeline (see above).

**Minimally Sufficient Statistic (MSS)** is the core insight: instead of simulating causation (running a state machine week-by-week like Java Synthea), we capture marginal prevalence rates and sample directly from them. Note that the current fingerprint captures *marginal* rates only — it does **not** capture the joint/correlation structure between conditions (co-occurrence is disabled by default), so it is a sufficient statistic for marginals, not for comorbidity patterns.

Per-patient generation work is bounded and independent of patient lifespan: O(c + m + e) where c, m, e are the condition/medication/event counts. The archetype draw is O(1) (Vose alias); the condition/event draws iterate over all c/m/e thresholds (8 lanes at a time in the SIMD paths), so the per-patient cost is a fixed constant in n, not literally O(1).

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

// Generate aggregate statistics for 1 million patients (no per-patient records
// are materialized in this path). Measured at ~0.11-0.12 s on the machine in
// PERFORMANCE.md; your time depends on core count and the registry size.
let stats = generator.generate_stats_only(1_000_000);

println!("Generated {} patients", stats.total_patients);
println!("Total conditions: {}", stats.condition_counts.iter().sum::<u64>());
```

### Validate Statistical Equivalence

```bash
# Compares generated stats against the fingerprint's own base rates
# (the test is named ..._java_baseline, but the reference IS the fingerprint;
#  see the Validation section for why this is self-consistency, not Java equivalence)
cargo test --package chronosynthea-mss --test java_validation --release -- test_generate_matches_java_baseline --nocapture
```

Measured output (AMD Ryzen 7 5800X, 16 threads, Rust 1.94.0, release; reproduced 2026-06-04):

```
Java Synthea Validation (n=100000, tolerance=10%)
  Status: PASSED
  Max Deviation: 0.31%
  KL Divergence: -0.006132
  Chi-Squared:   181.17
Failure rate: 0.0% (0/214 conditions)
```

These figures are stable run-to-run because the per-patient seed is derived deterministically from the base seed. They measure how closely 100K sampled patients reproduce the registry's base rates — i.e. sampling noise — **not** agreement with Java Synthea.

## Performance

All numbers below were measured on **AMD Ryzen 7 5800X (8 cores / 16 threads), 32 GB RAM, Rust 1.94.0, `--release` (thin LTO), default Rayon thread pool (16 threads)**, using the real 214-condition `calibrated_registry.json`. They were reproduced on 2026-06-04. There is **no Java Synthea baseline measured in this repo**, so no speedup multiplier is claimed — see Unknown U1.

| Path | What it produces | 1M patients | Throughput | Source |
|------|------------------|-------------|------------|--------|
| `generate_stats_only` | atomic counts only, no patient records | ~0.11–0.12 s | ~8.4–9.0 M pts/sec | `tests/java_validation.rs::test_high_volume_generation_with_validation` |
| `generate_full_stats_only` | counts incl. meds/obs/procs, no records | ~0.25–0.27 s | ~3.7–4.0 M pts/sec | `tests/java_validation.rs::test_full_generation_performance` (200K × 5 avg) |

Single-thread (`RAYON_NUM_THREADS=1`) for reference: `generate_stats_only` ≈ 2.0 M pts/sec, `generate_full_stats_only` ≈ 0.62 M pts/sec. So the 16-thread speedup over 1 thread is ~4x (stats-only) to ~6x (full-stats) — see the scaling notes in PERFORMANCE.md.

> Memory per patient: the fastest path materializes **no** patient records (only atomic counters), so "~0.5 KB/patient" does not apply there. `CompactPatient`/`FullPatient` sizes only matter for `generate_compact`/`generate_full`; see `arena.rs`.

Reproduce:

```bash
# stats-only (fastest path)
cargo test --package chronosynthea-mss --test java_validation --release \
    -- test_high_volume_generation_with_validation --nocapture

# full-stats path
cargo test --package chronosynthea-mss --test java_validation --release \
    -- test_full_generation_performance --nocapture
```

Example measured output (full-stats path):

```
Full stats generation (avg of 5 runs): 200K patients in 52.26ms (3827.01K patients/sec)
Projected time for 1M patients: 261.30ms
```

A Criterion benchmark (`cargo bench --package chronosynthea-mss`) also exists, but its `create_realistic_fingerprint` uses a small synthetic fingerprint (far fewer than 214 conditions), so its `mss_stats_only/1000000` number (~30 ms / ~33 M pts/sec here) is **not** comparable to the real-registry numbers above and should not be quoted as headline throughput.

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
└── data/
    └── prevalence/
        └── calibrated_registry.json  # Pre-computed base-rate registry
```

## Key Optimizations

Impact column shows the *mechanism*, not a measured per-technique speedup (those were never isolated in this repo).

| Technique | Where used | Description |
|-----------|------------|-------------|
| **SIMD threshold draws** | full-stats meds/obs/procs (`sampler.rs`) | `wide::f32x8` compares 8 thresholds per step. The default `generate_stats_only` condition path is a *scalar* loop (`sample_conditions_flat`), not SIMD. |
| **Vose alias archetype draw** | `archetype.rs::AliasTable` | O(1) scalar selection of one demographic-bucket archetype per patient (≈8 ns in the `archetype_sample` bench) |
| **Lock-free atomics** | `batch.rs::AtomicStatistics` | `AtomicU64::fetch_add(.., Relaxed)` for parallel counting, no mutex |
| **u16 code indices** | hot path / `tables.rs` | Conditions/events are sampled as `u16` indices instead of `Arc<str>`, avoiding atomic refcounting. The code *strings* are owned `String`s in `CodeTable`, populated at load time — not compile-time-interned constants. |
| **Arena allocation** (`bumpalo`) | `arena.rs::WorkerArena` only | Defined but **not** on the headline throughput paths — `generate_stats_only` / `generate_full_stats_only` use thread-local `SmallVec` + atomics, no arena. |

## Validation

**Scope (important):** `JavaValidation` compares the *generated* statistics against the *same MssFingerprint* they were generated from (`java_compat.rs::JavaValidation::validate` reads `self.reference.conditions[i].prevalence`). It therefore measures **self-consistency / sampling noise**, i.e. how closely the sampler reproduces its own base rates. It does **not** compare against actual Java Synthea output — there is no Java Synthea dataset, run, or version committed in this repo. Treat all three metrics as "sampler reproduces its base rates," not "matches Java Synthea."

We compute:

1. **Per-Condition Prevalence**: observed vs the fingerprint's base rate for each of 214 conditions, within tolerance
2. **KL Divergence**: divergence of observed vs base-rate distribution
3. **Chi-Squared Test**: goodness-of-fit of observed counts vs base-rate expected frequencies

Measured results (n=100,000, AMD Ryzen 7 5800X, reproduced 2026-06-04, from `tests/java_validation.rs::test_show_top_deviations`):
- **Max Deviation**: 0.31% (largest |observed − base rate| across the 214 conditions)
- **KL Divergence**: -0.006132 (essentially zero vs base rates)
- **Chi-Squared**: 181.17 (vs the test's pass threshold; not compared to Java)

## Data

The `data/prevalence/calibrated_registry.json` file contains a pre-computed base-rate registry. Its `description` field states the rates are calibrated to match Java Synthea occurrence rates; that calibration is **Assumed** (the upstream Java run/source is not in this repo). Contents:

- **214 conditions** with base rates and (currently unused, uniform-1.0) demographic multipliers
- **122 medications** with indication codes and frequencies
- **226 observations** with frequencies
- **282 procedures** with indication codes and frequencies
- A `demographics` block with marginal distributions over 4 age buckets, 2 genders, 5 races, and 2 ethnicities. `java_compat.rs::build_demographics` multiplies these marginals into an 80-bucket joint distribution (4 × 2 × 5 × 2); the joint is thus an *independence approximation* of the marginals, not a measured joint demographic distribution.

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
