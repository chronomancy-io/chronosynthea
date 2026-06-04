# Performance Analysis: ChronoSynthea MSS

## Executive Summary

ChronoSynthea generates synthetic patient *statistics* at millions of patients per second on a multi-core CPU. There are two distinct hot paths, with different throughput, and **neither materializes patient records**:

| Path | What it produces | Throughput (measured) | Time for 1M |
|------|------------------|-----------------------|-------------|
| `generate_stats_only` | atomic counts only | **~8.4–9.0 M pts/sec** | ~0.11–0.12 s |
| `generate_full_stats_only` | counts incl. meds/obs/procs | **~3.7–4.0 M pts/sec** | ~0.25–0.27 s |

There is **no Java Synthea baseline measured in this repo**, so no speedup multiplier is reported. The "statistical deviation" figures below are sampling noise against ChronoSynthea's *own* base rates, not a comparison to Java Synthea output (see §3).

---

## 0. Measurement Environment

All numbers in this document were reproduced on:

- **CPU**: AMD Ryzen 7 5800X — 8 cores / 16 threads
- **RAM**: 32 GB
- **Rust**: 1.94.0 (`cargo 1.94.0`)
- **Build**: `--release`, `lto = "thin"`, `codegen-units = 1`, `opt-level = 3`
- **Parallelism**: default Rayon thread pool (16 threads) unless noted
- **Registry**: the real `data/prevalence/calibrated_registry.json` (214 conditions, 122 meds, 226 obs, 282 procs)
- **Date reproduced**: 2026-06-04

Older versions of this document quoted an Apple M1 Pro and Rust 1.75 with throughput of ~1.6 M/sec; those numbers were not reproducible here and have been replaced.

---

## 1. Benchmark Results

### 1.1 Throughput (integration tests, real registry)

Reproduce with:

```bash
cargo test --package chronosynthea-mss --test java_validation --release \
    -- test_high_volume_generation_with_validation --nocapture     # stats-only
cargo test --package chronosynthea-mss --test java_validation --release \
    -- test_full_generation_performance --nocapture                # full-stats
```

`generate_stats_only`, 1M patients, repeated runs (16 threads):

| Run | Time | Throughput |
|-----|------|------------|
| 1 | 118.80 ms | 8.42 M pts/sec |
| 2 | 110.79 ms | 9.03 M pts/sec |
| 3 | 115.12 ms | 8.69 M pts/sec |
| 4 | 112.05 ms | 8.92 M pts/sec |

`generate_full_stats_only`, 200K × 5-run average (16 threads): ~52–53 ms per 200K ≈ **3.74–3.83 M pts/sec**, projecting ~261–267 ms for 1M.

### 1.2 Thread scaling (measured)

`RAYON_NUM_THREADS` set explicitly, same machine and registry:

| Path | 1 thread | 16 threads | Ratio |
|------|----------|------------|-------|
| `generate_stats_only` (1M) | 2.03 M pts/sec (492 ms) | 8.57 M pts/sec (117 ms) | ~4.2x |
| `generate_full_stats_only` (200K×5) | 0.62 M pts/sec | ~3.7–4.0 M pts/sec | ~6x |

This is well short of linear scaling across 16 threads (8 physical cores + SMT). The likely limiters are memory traffic on the per-condition threshold arrays and atomic increments into shared counters; this was observed, not separately profiled.

### 1.3 Criterion micro-benchmarks (synthetic fingerprint — not headline)

`cargo bench --package chronosynthea-mss` uses `create_realistic_fingerprint`, a **small synthetic** fingerprint with far fewer than 214 conditions. Its numbers are therefore **not comparable** to §1.1 and must not be quoted as overall throughput. Measured here:

| Bench | Time | Note |
|-------|------|------|
| `archetype_sample` | ~8.2 ns | Vose alias O(1) draw |
| `condition_sample_simd` | ~103 ns | one SIMD condition draw |
| `mss_stats_only/1000000` | ~30.4 ms | synthetic fp only — **not** the real registry |
| `mss_compact/100000` | ~2.04 ms | synthetic fp only |

### 1.4 Statistical self-consistency (n=100,000)

From `tests/java_validation.rs::test_show_top_deviations`:

```
Statistical Comparison (n=100000)
  KL Divergence: -0.006132
  Max Deviation: 0.0031
  Chi-Squared:   181.17
  Passed:        true

  Top Deviations:
    18718003: observed=0.6215, expected=0.6184, deviation=0.0031
    473461003: observed=0.4135, expected=0.4105, deviation=0.0030
    157141000119108: observed=0.0977, expected=0.0947, deviation=0.0030
```

"expected" here means **the fingerprint's own base rate**, not Java Synthea. The figures show that 100K sampled patients reproduce the registry's base rates to within ~0.31 percentage points — i.e. sampling noise. See §3 and §7.

---

## 2. Complexity Analysis

### 2.1 Time Complexity (Guarantee, by code inspection)

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Archetype sampling | O(1) | Vose alias (`archetype.rs::AliasTable::sample`) |
| Condition sampling | O(c) | scalar loop over c thresholds in the default path (`sample_conditions_flat`); the SIMD path still touches all c, 8 lanes per step |
| Medication sampling | O(m) | SIMD path, 8 lanes per step over m thresholds |
| Event sampling | O(e) | SIMD path, 8 lanes per step over e frequencies |
| Statistics recording | O(c + m + e) | atomic increments, one per sampled item |
| **Per-patient total** | **O(c + m + e)** | independent of patient lifespan/age |

The important property is that per-patient work is a **fixed bound independent of n and of simulated time** — unlike a week-by-week state machine. It is not literally O(1): each patient touches all c (and, on the full path, m and e) thresholds. SIMD reduces the constant factor (8 lanes/step) but not the asymptotic class.

### 2.2 Space Complexity

The fastest path stores **no per-patient records** — only shared atomic counters (one `AtomicU64` per condition/med/obs/proc) plus per-thread RNG and `SmallVec` scratch buffers. The exact byte sizes of the fingerprint and archetype registry were **not measured** in this repo; the previously listed "~2 MB / ~5 MB / ~0.5 KB" figures were unmeasured and have been removed.

| Component | Storage | Measured? |
|-----------|---------|-----------|
| Per-patient (stats-only / full-stats paths) | 0 bytes of records — counters only | yes (by construction) |
| Per-patient (`generate_compact`) | `CompactPatient` + inline `SmallVec<[u16;8]>` of conditions | size not benchmarked |
| Per-patient (`generate_full`) | `FullPatient` with encounters/events | size not benchmarked |
| Fingerprint + archetype registry | loaded once, shared via `Arc` | size not benchmarked |

---

## 3. Optimization Techniques

### 3.1 SIMD threshold draws

The full-event path (`sampler.rs`) compares 8 thresholds per step with `wide::f32x8`. **Note:** the default `generate_stats_only` condition path is the *scalar* loop shown first below (`sample_conditions_flat` in `archetype.rs`); SIMD is used for the medication/observation/procedure draws in the full-stats path.

Scalar form (default condition path):
```rust
for (idx, &threshold) in thresholds.iter().enumerate() {
    if threshold > 0.0 && rng.gen::<f32>() < threshold {
        buffer.push(idx as u16);
    }
}
```

SIMD form (`sampler.rs`, used for meds/obs/procs):
```rust
for chunk in 0..(thresholds.len() / 8) {
    // fill rand_buffer with 8 draws
    let thresh = f32x8::from(&thresholds[base..base + 8]);
    let rands = f32x8::new(self.rand_buffer);
    let mask = rands.cmp_lt(thresh);
    // ... process the 8-lane mask ...
}
```

**Impact**: processes 8 comparisons per vector op. A per-technique speedup was **not** isolated, so no "8x" multiplier is claimed.

### 3.2 Arena Allocation (`bumpalo`) — NOT on the throughput path

`WorkerArena` (in `arena.rs`) wraps a `bumpalo::Bump` allocator. **It is not used by the headline throughput paths.** `generate_stats_only` and `generate_full_stats_only` store no per-patient records at all (only atomic counters and per-thread `SmallVec` scratch), so there is nothing to arena-allocate. The arena type is defined and exported but is not exercised by the measured `*_stats_only` paths; do not attribute the throughput in §1 to it.

### 3.3 Vose Alias Method (O(1) sampling)

**Before**: Linear CDF search for weighted sampling
```rust
let r = rng.gen::<f64>();
let mut cumulative = 0.0;
for (i, &prob) in probabilities.iter().enumerate() {
    cumulative += prob;
    if r < cumulative {
        return i;  // O(n) average
    }
}
```

**After**: Alias table lookup
```rust
let i = rng.gen_range(0..n);
if rng.gen::<f32>() < prob[i] {
    return i;
} else {
    return alias[i];  // O(1) always
}
```

**Impact**: O(1) per draw instead of O(n) for a linear CDF scan. The `archetype_sample` bench measures ~8.2 ns per draw on this machine. No "5–10x" figure is claimed (no CDF baseline was benchmarked).

### 3.4 Lock-Free Atomic Statistics

**Before**: Mutex-protected counters
```rust
{
    let mut stats = stats.lock().unwrap();
    stats.condition_counts[idx] += 1;
}  // Lock released
```

**After**: Atomic fetch-add
```rust
stats.condition_counts[idx].fetch_add(1, Ordering::Relaxed);
// No lock, no contention, no cache line bouncing
```

**Impact**: avoids a mutex. Measured thread scaling is in §1.2 (~4.2x stats-only at 16 threads over 1 thread) — far from "near-perfect," so that earlier wording has been removed.

### 3.5 u16 code indices (avoid `Arc<str>` on the hot path)

On the sampling hot path, conditions and events are referred to by `u16` index, not by a reference-counted `Arc<str>`, so there is no atomic refcount traffic per item. **However**, the underlying code/display *strings* are stored as owned `String`s in `CodeTable` (`tables.rs`), populated at load time from the fingerprint — they are **not** compile-time-interned constants. The only `&'static` string tables are the small fixed enumerations (`SYSTEMS`, `RACES`, `ENCOUNTER_TYPES`, `EVENT_TYPES`, `ETHNICITIES`, `AGE_BUCKETS`). The previously shown `const DIABETES_T2: u16 = 42;` example did not exist in the code and has been removed.

### 3.6 Batch Event Sampling

**Before**: Per-encounter event sampling
```rust
for encounter in 0..num_encounters {
    for (i, &freq) in observation_freqs.iter().enumerate() {
        if rng.gen::<f32>() < freq {
            observations.push(i);
        }
    }
}
// O(encounters × events) random calls
```

**After**: Probability scaling for batch sampling
```rust
// P(at least once in N encounters) ≈ min(freq × N, 1.0)
let scaled_freq = (freq * num_encounters as f32).min(1.0);
// Sample once with scaled probability
if rng.gen::<f32>() < scaled_freq {
    observations.push(i);
}
// O(events) random calls regardless of encounter count
```

**Impact**: turns O(encounters × events) draws into O(events) draws (this is what `sample_events_batch` in `sampler.rs` does). This is an approximation — it models "at least one occurrence across N encounters" rather than per-encounter occurrence, trading per-encounter fidelity for fewer RNG calls. No measured multiplier is claimed.

---

## 4. Profiling Results

**Not measured.** This repo does not contain a profiler harness, and no `perf` run was performed for this document. The previous version of this section presented a CPU-time breakdown (35%/25%/20%/...), a fabricated `perf report` disassembly, and cache hit-rate figures (L1 98.5%, etc.) that were never measured. They have been removed rather than kept as "illustrative." If you want a real breakdown, run e.g.:

```bash
perf record -g target/release/deps/<test-binary>   # then: perf report
```

and record the actual output here.

---

## 5. Scaling Characteristics

The only scaling numbers that are real are the 1-thread vs 16-thread measurements in §1.2. The previous per-core efficiency table (1→200K, 2→390K, ... 16→2.1M pts/sec) and the memory-bandwidth estimates were unmeasured and have been removed.

Observed (this machine): going from 1 to 16 Rayon threads gives roughly a 4.2x speedup on the stats-only path and ~6x on the full-stats path — sub-linear, consistent with 8 physical cores plus SMT and contention on the shared atomic counters. A proper per-thread-count sweep and a bandwidth measurement remain **Unknown** until run.

---

## 6. Reproducibility

```bash
# Clone and build
git clone https://github.com/chronomancy-io/chronosynthea
cd chronosynthea
cargo build --release

# stats-only path (1M patients)
cargo test --package chronosynthea-mss --test java_validation --release \
    -- test_high_volume_generation_with_validation --nocapture

# full-stats path (200K x 5-run average)
cargo test --package chronosynthea-mss --test java_validation --release \
    -- test_full_generation_performance --nocapture

# self-consistency validation (n=100K)
cargo test --package chronosynthea-mss --test java_validation --release \
    -- test_show_top_deviations --nocapture

# control the thread count for scaling tests
RAYON_NUM_THREADS=1 cargo test --package chronosynthea-mss --test java_validation --release \
    -- test_high_volume_generation_with_validation --nocapture
```

### 6.1 Measured Output (this machine, 2026-06-04)

```
# test_full_generation_performance (16 threads)
Full stats generation (avg of 5 runs): 200K patients in 52.26ms (3827.01K patients/sec)
Projected time for 1M patients: 261.30ms

# test_high_volume_generation_with_validation (16 threads)
Generated 1M patients in 110.79ms (9.03M patients/sec)
```

(The `test_high_volume_*` test prints a "Debug build" note in its body; that text is misleading — the throughput shown is from the `--release` build. The text is in the test source, not this document.)

---

## 7. Comparison with Java Synthea

ChronoSynthea takes a fundamentally different approach from Java Synthea, but **this repo contains no measured Java Synthea baseline**, so the table below describes *approach* differences only — no speedup multiplier is claimed (see Unknown U1 in the README).

| Aspect | Java Synthea | ChronoSynthea |
|--------|--------------|---------------|
| **Approach** | Causal week-by-week state machine | Independent per-condition sampling at precomputed base rates |
| **Per-patient work** | Simulates the full lifespan | Bounded O(c + m + e), independent of lifespan |
| **Memory model** | JVM heap + GC | Native; fastest paths keep only atomic counters |
| **Parallelism** | JVM threads | Rayon work-stealing |
| **Random source** | java.util.Random | Xoshiro256++ |
| **String handling** | Java String objects | u16 indices on the hot path |
| **Output fidelity** | Causal model incl. comorbidities & timing | Marginal prevalence only; no co-occurrence, uniform demographic multipliers |

### Why is it fast?

1. **No week-by-week simulation**: outcomes are sampled directly from marginal rates rather than simulated over time. This is also the main *fidelity* trade-off — joint structure and timing are not reproduced.
2. **No JVM / GC overhead**: native code; fastest paths allocate no per-patient records.
3. **SIMD on the event path**: 8 threshold comparisons per vector op (full-stats path).
4. **Lock-free counting**: atomic `fetch_add` instead of a mutex.

A genuine speedup figure would require running Java Synthea on the same machine and dividing measured throughputs; that has not been done here.

---

## References

1. Vose, M.D. (1991). "A linear algorithm for generating random numbers with a given distribution." IEEE TSE.

2. Blackman, D. & Vigna, S. (2021). "Scrambled Linear Pseudorandom Number Generators." ACM TOMS. (Xoshiro256++)

3. Walonoski, J., et al. (2017). "Synthea: An approach, method, and software mechanism for generating synthetic patients." JAMIA.

---

*Document Version: 3.0.0 (MSS-honesty pass)*
*Last Updated: 2026-06-04*
*Benchmark Hardware: AMD Ryzen 7 5800X (8 cores / 16 threads), 32 GB RAM, Rust 1.94.0, release (thin LTO)*
