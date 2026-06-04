# ChronoSynthea Architecture

## Overview

ChronoSynthea is a high-performance synthetic healthcare data generator built on the **WASP/CDE/MSS** framework:

1. **WASP** (Workload-Aware Sufficient Placement): Defines the problem — encode healthcare data along k=4 dimensions with bounded-work query guarantees
2. **CDE** (Coleman Dimensional Encoding): The 4-phase constructive solution — workload analysis, coordinate encoding, index construction, query translation
3. **MSS** (Minimally Sufficient Statistics): Documentation discipline — every claim is classified as Definition, Guarantee, Assumption, or Unknown

> **WASP Encoding Tuple** `(k, E, I, T, {F_q})`:
> - **k = 4**: Patient Seed, Clinical Trajectory, Timing, Output Schema
> - **E(r)**: (seed:u64 + archetype:u16, condition/med/proc index lists, age_days + offsets, format flags)
> - **I(c)**: MssFingerprint + ArchetypeRegistry (Vose alias for archetype selection) + per-archetype f32 threshold arrays
> - **T(q)**: BatchGenerator (Q1), JavaValidation (Q2), module extractor (Q3)
> - **{F_q}**: `rand < threshold[i]` comparison (scalar in the default condition path, SIMD f32x8 for meds/obs/procs) + EventBitset dedup

> **Honesty scope.** The generator samples each condition independently at a precomputed base rate; it does **not** run Java Synthea's causal state machine, and co-occurrence is disabled in the default fingerprint. "Statistical validation" in §7 is self-consistency against the fingerprint's own base rates, not equivalence to Java Synthea. Bytes-per-patient and cache figures that were never measured have been removed.

This document describes the technical architecture, design decisions, and implementation details.

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Crate Structure](#crate-structure)
3. [Data Flow](#data-flow)
4. [Key Algorithms](#key-algorithms)
5. [Memory Model](#memory-model)
6. [Parallelization Strategy](#parallelization-strategy)
7. [Statistical Validation](#statistical-validation)

---

## Core Concepts

### CDE Phase 2: Coordinate Encoding

CDE represents synthetic health records along four dimensions:

| Dimension | Description | Representation | WASP Role |
|-----------|-------------|----------------|-----------|
| **Patient Seed** | Random seed + demographic parameters | `u64` seed + archetype index | E(r) dim 1 |
| **Clinical Trajectory** | Conditions, medications, procedures | Sparse bitsets of code indices | E(r) dim 2 |
| **Timing** | Event timestamps | Days since birth (u16) | E(r) dim 3 |
| **Output Schema** | Export format | FHIR, JSONL, compact | E(r) dim 4 |

### CDE Phase 3 + Phase 4: MSS Fingerprint as Index + Query

The key insight: **to generate statistically equivalent data, we only need the sufficient statistics, not the generative process.**

The MssFingerprint (Phase 3 index) and BatchGenerator (Phase 4 query translator) together implement the WASP I(c) → T(q) → {F_q} pipeline.

Traditional simulation (Java Synthea):
```
Patient → [Week 1] → [Week 2] → ... → [Week 4000] → Record
         State machine simulation over entire lifespan
```

MSS approach:
```
Fingerprint → Sample(archetype) → Sample(conditions) → Record
              O(1)                 O(c) where c << weeks
```

The MSS fingerprint contains:
- A demographic-bucket distribution (built as the product of 4 marginal distributions: age × gender × race × ethnicity = 80 buckets — an independence approximation, not a measured joint)
- Per-condition prevalence (the default registry uses uniform 1.0 demographic multipliers, so prevalence does not actually vary by bucket — see `java_compat.rs::fine_tune_age_multipliers`)
- Medication/observation/procedure frequencies
- A co-occurrence map — present as a field but left **empty** by `CalibratedRegistry::to_fingerprint`, so generation samples conditions independently

> The diagram below ("O(1)") is shorthand: the archetype draw is O(1), but the condition draw is O(c) over all conditions.

---

## Crate Structure

```
crates/
├── chronosynthea/              # Main binary entrypoint
│   └── src/main.rs
│
├── chronosynthea-mss/          # Core MSS implementation (primary)
│   ├── src/
│   │   ├── archetype.rs        # PatientArchetype, ArchetypeRegistry
│   │   ├── arena.rs            # CompactPatient, FullPatient, arena types
│   │   ├── batch.rs            # BatchGenerator, AtomicStatistics
│   │   ├── error.rs            # Error types
│   │   ├── extractor.rs        # FHIR bundle extraction
│   │   ├── fingerprint.rs      # MssFingerprint, ConditionStats
│   │   ├── java_compat.rs      # CalibratedRegistry, Java compatibility
│   │   ├── lib.rs              # Public API exports
│   │   ├── sampler.rs          # SimdSampler, EventSampler, EventBitset
│   │   ├── stats.rs            # StreamingStatistics, validation
│   │   └── tables.rs           # Interned string tables
│   ├── benches/
│   │   └── mss_generation.rs   # Criterion benchmarks
│   └── tests/
│       ├── java_validation.rs  # Statistical validation tests
│       └── validation.rs       # Unit validation tests
│
├── chronosynthea-cde/          # CDE encoding library
│   └── src/
│       ├── axis.rs             # Axis definitions
│       ├── config.rs           # Axis configuration
│       ├── encode.rs           # Encoding logic
│       ├── features.rs         # Feature extraction
│       ├── metrics.rs          # Quality metrics
│       └── signature.rs        # Deterministic signatures
│
├── chronosynthea-core/         # Core types and module loading
│   └── src/
│       ├── module/             # Synthea module types
│       │   ├── edge.rs
│       │   ├── loader.rs
│       │   ├── state.rs
│       │   └── types.rs
│       ├── module.rs
│       └── patient.rs          # Patient types
│
├── chronosynthea-gen/          # Legacy generation (superseded by MSS)
│   └── src/
│       ├── alias.rs            # Vose alias method
│       ├── buffer.rs           # Buffer management
│       ├── generator.rs        # Patient generator
│       └── parallel.rs         # Parallel generation
│
└── chronosynthea-io/           # I/O and formatting
    └── src/
        ├── format.rs           # Output formats
        └── stream.rs           # Streaming output
```

### Dependency Graph

```
chronosynthea (binary)
    └── chronosynthea-mss
            ├── chronosynthea-core (types)
            ├── chronosynthea-cde (encoding)
            └── chronosynthea-io (output)
```

---

## Data Flow

### Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INITIALIZATION (once)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  calibrated_registry.json                                            │
│           │                                                           │
│           ▼                                                           │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │ CalibratedReg.  │───▶│ MssFingerprint  │───▶│ ArchetypeReg.   │  │
│  │ (JSON loader)   │    │ (statistics)    │    │ (sampling ready)│  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│                                                           │          │
│                                                           ▼          │
│                                                  ┌─────────────────┐ │
│                                                  │ BatchGenerator  │ │
│                                                  │ (ready to gen)  │ │
│                                                  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    GENERATION (per batch)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  BatchGenerator.generate_stats_only(1_000_000)                       │
│           │                                                           │
│           ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    Rayon parallel_for                            │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │ │
│  │  │  Worker 0   │  │  Worker 1   │  │  Worker N   │   ...        │ │
│  │  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────┐  │              │ │
│  │  │  │ RNG   │  │  │  │ RNG   │  │  │  │ RNG   │  │              │ │
│  │  │  │SimdS. │  │  │  │SimdS. │  │  │  │SimdS. │  │              │ │
│  │  │  │EventS.│  │  │  │EventS.│  │  │  │EventS.│  │              │ │
│  │  │  └───────┘  │  │  └───────┘  │  │  └───────┘  │              │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │ │
│  │         │                │                │                      │ │
│  │         └────────────────┴────────────────┘                      │ │
│  │                          │                                        │ │
│  │                          ▼                                        │ │
│  │               ┌─────────────────────┐                            │ │
│  │               │  AtomicStatistics   │  (lock-free aggregation)   │ │
│  │               │  condition_counts[] │                            │ │
│  │               │  medication_counts[]│                            │ │
│  │               └─────────────────────┘                            │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                          │                                            │
│                          ▼                                            │
│               ┌─────────────────────┐                                │
│               │ StreamingStatistics │  (final output)                │
│               └─────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Per-Patient Flow

```
1. Sample archetype (Vose alias, O(1) scalar)
   └── Determines: age bucket, gender, race, condition thresholds

2. Sample conditions (per-condition `rand < threshold`)
   └── Default `generate_stats_only` path: SCALAR loop over c thresholds
       (`sample_conditions_flat`). The f32x8 SIMD routines in sampler.rs
       are used for the medication/observation/procedure draws, not the
       default condition path.
   └── Output: SmallVec<[u16; 8]> of condition indices

3. Estimate encounters (deterministic from age/conditions)
   └── Base + age factor + condition modifiers

4. Sample medications (SIMD with pre-computed thresholds)
   └── Archetype-level P(med) = Σ P(cond) × P(med|cond)

5. Sample observations/procedures (batch SIMD)
   └── Scaled probability: min(p × N_encounters, 1.0)
   └── Single pass for all encounters

6. Record statistics (atomic increment)
   └── condition_counts[idx].fetch_add(1, Relaxed)
```

---

## Key Algorithms

### 1. Vose Alias Method

**Purpose**: O(1) weighted random sampling from discrete distributions

**Used for**: Demographic bucket selection, archetype selection

```rust
pub struct AliasSampler {
    prob: Vec<f32>,   // Probability table
    alias: Vec<u32>,  // Alias table
}

impl AliasSampler {
    // O(n) preprocessing, O(1) sampling
    pub fn sample<R: Rng>(&self, rng: &mut R) -> usize {
        let i = rng.gen_range(0..self.prob.len());
        if rng.gen::<f32>() < self.prob[i] {
            i
        } else {
            self.alias[i] as usize
        }
    }
}
```

**Reference**: Vose, M.D. (1991). "A linear algorithm for generating random numbers with a given distribution."

### 2. SIMD Threshold Sampling

**Purpose**: Sample multiple events in parallel using CPU vector instructions

**Used for**: medication, observation, and procedure sampling (`sampler.rs`). The default `generate_stats_only` condition path is a **scalar** loop (`archetype.rs::sample_conditions_flat`), not this routine.

Shape of the real code in `sampler.rs` (e.g. `sample_medications_simd`):

```rust
let chunks = thresholds.len() / 8;
for chunk in 0..chunks {
    // fill an [f32; 8] scratch buffer with rng.gen() draws
    for i in 0..8 { self.rand_buffer[i] = rng.gen(); }

    let base = chunk * 8;
    let thresh = f32x8::from(&thresholds[base..base + 8]);
    let rands  = f32x8::new(self.rand_buffer);

    let mask = rands.cmp_lt(thresh);   // wide::CmpLt
    let mask_bits = mask.move_mask();  // bitmask of the 8 lanes
    if mask_bits != 0 {
        for bit in 0..8 {
            if (mask_bits & (1 << bit)) != 0 && thresholds[base + bit] > 0.0 {
                self.medication_buffer.push((base + bit) as u16);
            }
        }
    }
}
// plus a scalar remainder loop for thresholds.len() % 8
```

### 3. Event Bitset Deduplication

**Purpose**: O(1) deduplication for events across encounters

**Used for**: Observation/procedure accumulation

```rust
pub struct EventBitset {
    words: [u64; 8],  // 512 bits for up to 512 event types
}

impl EventBitset {
    #[inline]
    pub fn test_and_set(&mut self, idx: u16) -> bool {
        let word = (idx / 64) as usize;
        let bit = idx % 64;
        let mask = 1u64 << bit;
        let was_set = (self.words[word] & mask) != 0;
        self.words[word] |= mask;
        !was_set  // Returns true if newly set
    }
    
    #[inline]
    pub fn clear(&mut self) {
        self.words = [0; 8];  // O(1) reset
    }
}
```

---

## Memory Model

### Arena Allocation (`bumpalo`) — defined, but NOT on the throughput path

`WorkerArena` (in `arena.rs`) wraps a `bumpalo::Bump`. **The measured throughput paths do not use it.** `generate_stats_only` and `generate_full_stats_only` materialize no patient records — they update shared atomic counters and reuse a per-thread `SmallVec` scratch buffer. The arena type exists and is re-exported but is not exercised by the `*_stats_only` paths, so do not attribute their speed to arena allocation.

### Compact Data Structures (actual layout from `arena.rs`)

```rust
// CompactPatient — a small fixed header PLUS an inline SmallVec of condition
// indices. It is NOT 24 bytes (the SmallVec adds inline/heap storage).
pub struct CompactPatient {
    pub id: u64,                       // 8 bytes
    pub birth_date_days: i32,          // 4 bytes
    pub sex: u8,
    pub race: u8,
    pub ethnicity: u8,
    pub encounter_count: u8,
    pub condition_count: u8,
    pub archetype_id: u16,
    pub conditions: SmallVec<[u16; 8]>, // inline up to 8, else heap
}

// CompactEvent — exactly 8 bytes (enforced by a const assert in arena.rs):
//   const _: () = assert!(size_of::<CompactEvent>() == 8);
#[repr(C, align(8))]
pub struct CompactEvent {
    pub event_type: u8,        // diagnosis=0, medication=1, procedure=2, observation=3, immunization=4
    pub system_idx: u8,        // SNOMED=0, RxNorm=1, LOINC=2, CPT=3
    pub code_idx: u16,         // index into code table
    pub display_idx: u16,      // index into display table
    pub timestamp_offset: u16, // offset from encounter
}
```

### Memory Usage

The fastest paths store **0 bytes of per-patient records** (counters only). Exact byte sizes for `CompactPatient`/`FullPatient` and for the fingerprint/registry were **not benchmarked** in this repo. The previous comparison table ("Java ~5 MB / ChronoSynthea 24 bytes / ~5 GB / ~50 MB per million") was unmeasured — both the Java figures (no Java run here) and the ChronoSynthea figures — and has been removed.

**Verifiable** structural facts: `CompactEvent` is exactly 8 bytes (const-asserted); condition/event references on the hot path are `u16` indices, not `Arc<str>`.

---

## Parallelization Strategy

### Rayon Work-Stealing

We use Rayon's `par_iter` with `for_each_init` for per-thread state (simplified from `batch.rs`):

```rust
(0..count).into_par_iter().for_each_init(
    || {
        // Per-thread initialization (called once per thread)
        let thread_id = rayon::current_thread_index().unwrap_or(0);
        let rng = Xoshiro256PlusPlus::seed_from_u64(/* per-thread seed */);
        (rng, SmallVec::<[u16; 8]>::new(), EventSampler::new())
    },
    |(rng, condition_buffer, event_sampler), patient_id| {
        // Per-patient work (called count times total)
        // sample archetype, sample conditions into condition_buffer,
        // sample events into event_sampler, then record into atomics
    }
);
```

### Lock-Free Statistics

We avoid locks entirely by using atomic counters:

```rust
pub struct AtomicStatistics {
    pub patients: AtomicU64,
    pub encounters: AtomicU64,
    pub events: AtomicU64,
    pub condition_counts: Vec<AtomicU64>,  // One per condition
    pub medication_counts: Vec<AtomicU64>,
    // observation_counts, procedure_counts ...
}

// record() / record_full() in batch.rs do fetch_add(1, Ordering::Relaxed)
// per sampled index. Relaxed ordering is sufficient because the final counts
// are order-independent (Assumption A3 in the README claim table).
```

Note the per-condition counters are individual `AtomicU64`s in a `Vec`, so heavily-prevalent conditions can still cause cache-line contention across threads under high core counts — consistent with the sub-linear scaling measured above.

### Scaling Characteristics (measured)

On AMD Ryzen 7 5800X (8 cores / 16 threads), real 214-condition registry, `RAYON_NUM_THREADS` set explicitly:

| Path | 1 thread | 16 threads | Speedup |
|------|----------|------------|---------|
| `generate_stats_only` (1M) | 2.03 M pts/sec | 8.57 M pts/sec | ~4.2x |
| `generate_full_stats_only` (200K×5) | 0.62 M pts/sec | ~3.7–4.0 M pts/sec | ~6x |

Scaling is **sub-linear** across 16 threads (8 physical cores + SMT). The previous per-core efficiency table (1→200K … 16→2.4M, "near-linear, 75%+ efficiency") was unmeasured and has been replaced with these two endpoints. A full per-thread-count sweep has not been run.

---

## Statistical Validation

> **Scope:** despite the name `JavaValidation`, the reference distribution `self.fingerprint`/`self.reference` IS the same fingerprint the patients were generated from. So `validate()` measures **self-consistency** — how closely the sampler reproduces its own base rates — **not** agreement with actual Java Synthea output (no Java run exists in this repo). Read "expected" below as "the fingerprint's base rate."

### Validation Framework

```rust
pub struct JavaValidation {
    reference: MssFingerprint,  // == the generation fingerprint
    tolerance: f64,
}

impl JavaValidation {
    pub fn validate(&self, stats: &StreamingStatistics) -> ValidationResult {
        let mut max_deviation = 0.0;
        let mut failures = Vec::new();
        
        for (i, cond) in self.fingerprint.conditions.iter().enumerate() {
            let expected = cond.prevalence;
            let observed = stats.condition_counts[i] as f64 
                         / stats.total_patients as f64;
            let deviation = (observed - expected).abs();
            
            max_deviation = max_deviation.max(deviation);
            
            if deviation > self.tolerance {
                failures.push(ConditionFailure {
                    code: cond.code.clone(),
                    expected,
                    observed,
                    deviation,
                });
            }
        }
        
        ValidationResult {
            passed: failures.is_empty() && max_deviation < self.tolerance,
            max_deviation,
            failures,
            kl_divergence: self.compute_kl(stats),
            chi_squared: self.compute_chi_squared(stats),
        }
    }
}
```

### Validation Metrics (measured at n=100,000 against the fingerprint's own base rates)

1. **Max Deviation**: largest |observed − base rate| across the 214 conditions
   - Measured: **0.31%** (0.0031)

2. **KL Divergence**: D_KL(P || Q) = Σ P(x) log(P(x)/Q(x))
   - Measured: **-0.006132** (effectively zero; sign reflects floating-point summation, not a true negative divergence)

3. **Chi-Squared**: Σ (O − E)² / E
   - Measured: **181.17**
   - Pass threshold used in `stats.rs::compare` is the **ad-hoc** `sqrt(n) × num_conditions / 10`. At n=100000, num_conditions=214 that is `316.23 × 214 / 10 ≈ 6767` (not ~677 as a previous version stated). This threshold is a heuristic chosen in code, **not** a standard chi-squared critical value — treat "passed" as "within this project's chosen bound," an Assumption rather than a statistical Guarantee.

---

## Future Considerations

### Potential Enhancements (aspirational — not implemented or benchmarked)

1. **GPU Acceleration**: port the threshold draws to CUDA/Metal (any speedup is unmeasured/speculative)
2. **Streaming Output**: direct Arrow/Parquet output without materialization
3. **Co-occurrence / joint demographics**: actually populate the (currently empty) co-occurrence map and use non-uniform demographic multipliers, so generated data reflects comorbidity and demographic structure rather than independent marginals
4. **A real Java Synthea baseline**: run Java Synthea on the same machine to obtain a true throughput ratio and a true distributional comparison (currently Unknown)
5. **Temporal Modeling**: realistic event timing (currently approximate/deterministic)
6. **FHIR R4 Export**: native FHIR bundle generation

### Trade-offs Made

| Decision | Trade-off | Rationale |
|----------|-----------|-----------|
| Uniform (1.0) demographic multipliers | No age/gender/race variation in prevalence | Makes generated prevalence match the base rates exactly (so validation is self-consistent) |
| Co-occurrence disabled by default | No condition clustering / comorbidity | Keeps each condition's marginal exact; avoids the variance the co-occurrence pass would add |
| Pre-computed per-archetype thresholds | More memory | Avoids recomputing per-bucket probabilities per patient |
| u16 code indices | Max 65K codes | Sufficient for the 214/122/226/282 codes in the registry |

---

*Document Version: 3.0.0 (MSS-honesty pass)*
*Last Updated: 2026-06-04*
