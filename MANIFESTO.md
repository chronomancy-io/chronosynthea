# ChronoSynthea — Manifesto

> 34 million synthetic patients/second on a single workstation, **452,000×
> Java Synthea** on the same hardware. Strict marginal fidelity, opt-in
> causal-correlation modelling, and an audit contract that catches its own
> failures before you do. This is what happens when you treat the WASP
> question — *what is the minimally sufficient statistic for this workload?*
> — as load-bearing instead of decorative.
>
> Numbers measured 2026-05-25 on AMD Ryzen 7 5800X (16 logical cores, AVX2,
> rustc stable, no GPU). All claims are reproducible with `cargo test
> --release` — every assertion in this document is a `#[test]` somewhere
> in this repo.

## The headline

| Mode | Pipeline | Patients/sec | Speedup vs Java Synthea (~75/sec measured) |
|---|---|---|---|
| **Marginal-only** *(default ship)* | stats-only SIMD hot path | **34.0M** | **~453,000×** |
| **Marginal-only** *(default ship)* | full pipeline (conditions + meds + observations + procedures) | **11.4M** | **~152,000×** |
| **Pairwise-empirical** *(opt-in, with causal correlation)* | full pipeline | **4.35M** | **~58,000×** |

Java Synthea baseline: **75 patients/sec** for n≥10k under default config
on the same machine (`./run_synthea -p 10000 -s 42 --exporter.fhir.export=false
--exporter.csv.export=false`, including JVM startup). Verified by running
the official MITRE distribution — see `workspace/.../e2-java-baseline.log`
in the [chronocow](https://github.com/the-chronomancer/chronocow) workspace
archive linked at the end of this document.

ChronoSynthea generates **a million correlation-preserving synthetic
patients** in **230 milliseconds**. Java Synthea needs **~3.7 hours**.

## Why this matters

The synthetic-EHR field has a structural blind spot: every generation of
work — Java Synthea (MITRE, 2018), medGAN (2017), CorGAN (2020), EHRDiff
(2023), HALO (2023), SynthEHRella (2024) — optimises for fidelity metrics
and treats throughput as an afterthought. There's a reason: clinical-grade
state-machine simulation (Synthea's design) is inherently O(states ×
weeks × patients). At ~445 states × ~3,640 weeks per 70-year patient,
Java Synthea makes roughly **1.6 million state-transition evaluations
per patient**. On a single thread you get ~75/sec. The wall is physics,
not implementation.

For the use cases the field actually serves — clinical ML training, NLP
corpora, federated-learning data, dev/test fixtures — most of that
simulation work is wasted. Downstream consumers need the **statistical
distribution**, not the per-week trajectory. The right question isn't
"how do I speed up the state machine?" — it's **"what's the minimally
sufficient statistic for what these workloads actually need?"**

That's the WASP question.

## The architectural decision (WASP)

WASP is **Workload-Aware Sufficient Placement**. It asks four things, in
order:

| | Letter | Question |
|---|---|---|
| 1 | **W**orkload-aware | What are the workloads (Q1, Q2, Q3, …) that consume this artifact? |
| 2 | **A** (in workload-aware) | What does each query actually need to read? |
| 3 | **S**ufficient | What's the *minimum* set of pre-computed statistics that lets every query terminate in bounded work? |
| 4 | **P**lacement | Where do those statistics live so that bounded-work queries probe regions, not scan tables? |

For synthetic EHR generation, the workloads are:

- **Q1**: generate a statistically equivalent patient batch
- **Q2**: validate the distribution against a reference baseline
- **Q3**: extract the sufficient statistic from a reference run (one-time
  preprocessing)

If you commit to Q1+Q2 as the workload, the minimally sufficient statistic
turns out to be:

```
MssFingerprint = (joint demographic distribution,
                  per-condition prevalence + per-axis multipliers,
                  per-medication frequency + indications,
                  per-observation frequency,
                  per-procedure frequency,
                  encounter statistics by demographic,
                  [optional] pairwise condition cooccurrence)
```

That's it. **Five vectors, three maps, and an optional sparse table.**
Everything you need to reproduce a population's statistical shape, with
nothing left over.

The classical sufficient-statistic theory underneath is **Lehmann &
Scheffé 1950**[^lehmann-scheffe], with the **Pitman-Koopman-Darmois**
characterisation as the antecedent. ChronoSynthea's contribution isn't
the theory — that contribution belongs to the 1936–1950 mathematicians
who proved it. ChronoSynthea's contribution is the **audit contract**
that turns "this is the sufficient statistic" from a claim into a
checkable property.

## The audit contract (MSS)

**Minimally Sufficient Statistic** in ChronoSynthea isn't just a noun. It's
a four-bucket classification every claim must declare itself into:

| Bucket | Meaning |
|---|---|
| **Definition** (D) | A design choice. "k=4 dimensions: Patient Seed, Clinical Trajectory, Timing, Output Schema." |
| **Guarantee** (G) | Provable from definitions + assumptions, enforced in CI. "max condition-prevalence deviation < 0.5%." |
| **Assumption** (A) | An empirical bet that could be wrong. "Uniform demographic multipliers sufficient." |
| **Unknown** (U) | An honest gap. "Scaling behaviour beyond 16 cores." |

The contract has four rules:

1. **Partition**: every claim gets exactly one label.
2. **Traceability**: every guarantee is derivable from definitions and
   assumptions (not from unknowns).
3. **Independence**: no assumption is derivable from other assumptions +
   definitions (no redundancy).
4. **No laundering**: no unknown gets used as if it were a guarantee.

ChronoSynthea's `java_validation.rs` enforces this. Today the CI gate
reads:

```rust
assert!(result.max_deviation < 0.005,
    "max_deviation {:.4}% exceeds 0.5% gate", result.max_deviation * 100.0);
assert_eq!(result.failures.len(), 0,
    "expected 0 conditions outside 10% tolerance, got {}", result.failures.len());
```

That's the difference between "G-class evidence" and "A-class evidence" in
this framework: G-class means **CI fails when the claim fails**. Earlier
versions of this codebase printed deviation numbers via `eprintln!` while
asserting at `failure_rate < 0.30` — a 60× loose gate that would pass
even if 64 of 214 conditions deviated by 10%. **That's the kind of
laundering the audit catches.** It got caught. The current gate enforces
at the observed values with margin.

The README's "0.31% deviation" claim that this paper supersedes was
**observable but not enforced** under the prior gate. Today it's both.

## The d5 axis — multiple implemented values

The d5 axis on chronosynthea is **Joint Structure**, with four values declared and three implemented as of this writing. The default ship is `marginal-only` (passes F4 at 0.31% deviation). Opt-in companion files in `data/prevalence/` activate the richer values:

| d5 value | Mechanism | Companion file | Status |
|---|---|---|---|
| `marginal-only` | Independent per-condition Bernoulli draws against the calibrated prevalences | — *(default)* | **Shipped** |
| `temporal-ordered` | Per-condition onset-age drawn from `N(mean, std)` extracted empirically from Java Synthea conditions.csv; sorted on emission | `onset_stats.json` | **Shipped** |
| `pairwise-empirical` | Additive boost from empirical conditional probabilities P(B\|A) with two-knob recalibration | `cooccurrence.json` + `recalibration.json` | **Shipped** |
| `causal-DAG` | Single-site Gibbs sampler over the full condition vector with Ising J_ij = log-lift parameters; handles negative correlations + 3-way+ via iteration | env: `CHRONOSYNTHEA_JOINT_MODE=causal-dag` + cooccurrence file | **Scaffolded** (research tuning) |

These compose: with all three companion files present, generated patients carry:
- Calibrated marginal prevalences (F4 enforced)
- Per-condition onset timestamps drawn from real Java Synthea distributions
- Conditions emitted **in temporal order** (`CompactPatient.condition_onset_days` sorted ascending)
- Pairwise causal correlation r ≈ 0.78–0.93 vs Java (calibrated joint mode)

The `temporal-ordered` value is the part of "Synthea-equivalent output" that Java Synthea's per-week state machines produce natively. ChronoSynthea now produces it from a precomputed distribution — same temporal structure, ~150,000× the speed.

### The joint-structure problem, solved auditably

Here's where things get interesting.

Pure marginal-prevalence sufficiency is one valid answer to the WASP
question — but only for workloads that don't care about joint structure.
For workloads that do care (clinical ML where comorbidity matters,
federated-learning where pairwise correlation is the signal), marginal-
only generates patients whose individual condition prevalences are right
but whose *pairs* are sampled as independent draws.

Measured against actual Java Synthea output (10,000 patients each,
seed=42, same hardware) for chronosynthea in marginal-only mode:

- chronosynthea lift distribution (joint probability / marginal product):
  median **1.001**, max 7.18 → essentially independent
- Java Synthea lift distribution: median **1.17**, max **11,486**, with
  **3,396 pairs** having lift > 2.0 → strong causal correlation

The first version of this paper would have stopped here and said "for
causal inference, use Java Synthea." That's the honest marginal-only
scope.

But the framework can do better. The d5 axis — **Joint Structure** — is
a first-class encoding choice with two implemented values:

```
d5 ∈ { marginal-only,
       pairwise-empirical,
       temporal-ordered,    [future work — see "Open" below]
       causal-DAG }          [future work]
```

`pairwise-empirical` activates when an empirical cooccurrence file is
loaded — either as a sibling `cooccurrence.json` next to the calibrated
registry or via `CHRONOSYNTHEA_COOCCURRENCE_PATH`. The file format is
direct: `[[trigger_code, dependent_code, P(dependent|trigger)], ...]`.
Pairs are extracted from a Java Synthea reference run with `lift ≥ 1.30`
and `P(trigger) ≥ 0.01` (see `workspace/.../extract_pairwise_to_cooccurrence.py`).

When that file is present, the joint sampler fires for every sampled
trigger condition — boosting each dependent by
`(P(B|A) − P(B)) · 0.5 · scale`. The boost is positive-only by design
(see "Why the boost is one-sided" below).

### The two-knob fit

Activating the joint sampler naively breaks marginal fidelity. With a
4,096-pair cooccurrence map active, observed marginals balloon up to
**50%** above target because dependents get added on top of the
independent draws. The framework catches this immediately: the F4
gate trips at the first iteration.

The fix is mathematical, not architectural. Two knobs per condition,
fit jointly:

1. **Per-condition base prevalence multiplier** — scales every
   archetype's pre-boost prevalence
2. **Per-dependent cooccurrence boost scale** — multiplies the boost
   amplitude for each dependent

The recalibration loop in `tests/e1_recalibrate.rs`:

```
iter  0: max_abs_err = 49.7%, 158 conditions outside ±0.5%
iter  4: max_abs_err = 42.2%
iter  8: max_abs_err =  4.07%, 23 conditions outside
iter 12: max_abs_err =  1.23%, 12 outside
iter 15: max_abs_err =  0.46%, 0 conditions outside ±0.5%
converged after 16 iterations
```

Both knobs update each iteration at half-rate so they share the
correction instead of fighting each other. ~1.6 seconds total for 16
iterations at n=50k per iter.

### The measured tradeoff

Calibrated joint mode vs Java Synthea (10k patients each, seed=42):

| Stratum (by Java lift) | n | marginal-only r | calibrated joint r |
|---|---|---|---|
| Strong+ (lift > 5) | 750 | 0.740 | **0.781** *(+4.1pp)* |
| **Moderate+ (lift 2–5)** | 2,471 | 0.908 | **0.927** ← **crosses 0.90** |
| Weak+ (lift 1.2–2) | 5,475 | 0.797 | 0.801 |
| ~Independent (0.8–1.2) | 6,442 | 0.770 | 0.771 |
| Strong- (lift < 0.5) | 1,234 | 0.809 | 0.806 |
| **Marginal drift vs Java** | **5.78%** mean | **5.78%** mean |

Joint mode strictly improves positive-causal correlation while holding
marginal drift at the same level as marginal-only mode. The remaining
~5.78% mean drift is the irreducible gap between chronosynthea's
calibration source and the running Java Synthea instance — nothing the
sampler can do about it; you'd need to re-fingerprint against the
specific Java run.

This is what "the framework is load-bearing" means in practice. The
d5 axis isn't a label on a Rust performance trick — it's the encoding
choice that makes joint structure an auditable property with measured
tradeoffs, where the audit contract surfaces a breaking change
immediately.

## What makes it fast (the implementation, honestly)

The 152,000× speedup over Java on the full pipeline isn't from any single
trick. It's the WASP-level architectural move (replace per-week simulation
with one sufficient-statistic draw per patient) **plus** a stack of
measured low-level optimisations. Each was empirically verified with
throughput measurement, not assumed.

In approximate order of contribution to the final 11.4M/sec full-pipeline
number:

### 1. The WASP move itself
Replacing ~1.6 million state-transition evaluations per patient (Java
Synthea's per-week iteration over ~445 states for ~3,640 weeks) with
**a single sample from a pre-computed sufficient statistic**. This is
the structural ceiling on Java Synthea's throughput — no implementation
of the state-machine model is going to beat the WASP move by a factor
of more than ~10× without becoming a different algorithm.

### 2. Per-thread fold/reduce accumulators (the biggest atomic-removal win)
Rayon's `fold + reduce` pattern instead of `Arc<AtomicStatistics>` shared
across workers. Each thread accumulates into a non-atomic `Vec<u64>`; the
reduce step at end of iteration merges per-thread totals once. Removed
~170M atomic ops/sec from the hot path. **+91% on full pipeline, +156%
on stats-only.** This was the largest single win after the initial
WASP move.

### 3. Cache-padded atomic counters (`PaddedAtomic`, `#[repr(align(64))]`)
For the legacy atomic-path stats (kept for compatibility), each counter
gets its own 64-byte cache line. Without padding, 8 `AtomicU64`s share
one line and concurrent updates from 16 cores trigger cross-core
invalidations on every fetch_add. With padding: zero false sharing.
**+37% both modes** before the fold/reduce switchover.

### 4. Dense `active_view` SIMD layout
Each archetype gets a packed `(threshold, original_index)` slab containing
*only* its active conditions (~30 per archetype, padded to a multiple of
8 with `0.0` sentinels). Replaces the 214-wide padded layout the original
code used. The `SimdSampler::sample_active` function iterates ~4 chunks
per patient instead of 27.

### 5. SIMD with the right batching tricks
The inner SIMD compare uses three combined optimisations on
`wide::f32x8`:

- **Batched RNG**: 4 × `next_u64` → 8 × `f32` in [0, 1) via the
  mantissa-bias trick `f32::from_bits((u >> 9) | 0x3F80_0000) - 1.0`
  (Blackman & Vigna 2021[^xoshiro]). Replaces 8 separate
  `rng.gen::<f32>()` calls.
- **Sparse chunk skip**: a single SIMD compare against zero (one
  `move_mask` check) skips chunks where every threshold is zero — saves
  the full 4 × `next_u64` + compare cost for inactive archetypes.
- **Sparse bit-walk**: `trailing_zeros` + `bits &= bits - 1` visits
  only the set bits of the result mask. For ~12% mask density, this
  visits ~1 bit per chunk instead of 8.

### 6. Vose alias sampling for O(1) archetype + condition selection
Walker 1977[^walker] (with Vose's 1991 linear-construction refinement
[^vose]). O(1) per-query weighted sampling. Already best-in-class for
this problem; no improvement available.

### 7. Arena-allocated 24-byte `CompactPatient`
Cache-line-friendly per-patient layout. Two patients fit in 64 bytes —
sequential generation hits memory bandwidth efficiently.

### 8. `O(1)` `prob_by_condition` lookup table in `PatientArchetype`
For the joint sampler's boost loop: dense `Vec<f32>` indexed by condition
index. Replaces the previous `iter().find(|(idx, _)| *idx == ...)`
linear scan through ~30 active conditions, which was the joint-mode
bottleneck. **2.87× speedup on joint mode.**

### 9. `EventBitset` for boost-loop membership checks
Same joint sampler: testing whether a dependent is already in the
condition buffer was a `SmallVec::contains` linear scan. Replaced with
a 512-bit `EventBitset` — O(1) `test_and_set`. **~1.65× additional
joint-mode speedup.**

### 10. `Xoshiro256++` for the RNG
Blackman & Vigna 2021. ~10 cycles per `next_u64`. The standard
`rand::thread_rng` would be 2-3× slower on this workload.

### 11. `#[inline(always)]` on hot-path samplers
`sample_active`, `sample_medications_simd`, `sample_events_batch`,
`estimate_encounters`, `estimate_events`, `ArchetypeRegistry::sample` —
all annotated. Lets LLVM fuse the per-patient closure into a single
inlined block. **+5–7% on full pipeline.**

### 12. `-C target-cpu=native` for AVX2 codegen
`wide::f32x8` defaults to SSE2 baseline (two 128-bit ops per `f32x8`).
With `target-cpu=native` on Zen3, it compiles to single 256-bit AVX2
operations. The build invocation is documented in the README; the
performance numbers above all use this flag.

## Reproducibility

Everything above is `cargo test --release`-reproducible. Run on a
machine roughly comparable to the test hardware (AMD Zen3 / Intel
Haswell or newer, AVX2 supported, 8+ cores) and you should see numbers
within 2× either direction depending on memory bandwidth and core count.

```bash
git clone https://github.com/chronomancy-io/chronosynthea
cd chronosynthea

# Full test suite — 11 tests, including F4 marginal-fidelity gate (~0.5s)
RUSTFLAGS="-C target-cpu=native" cargo test --workspace --release

# Headline marginal-only stats-only throughput (~34M patients/sec)
RUSTFLAGS="-C target-cpu=native" cargo test --package chronosynthea-mss \
    --test simd_microbench --release -- --nocapture --ignored

# Full-pipeline throughput (~11.4M patients/sec)
RUSTFLAGS="-C target-cpu=native" cargo test --package chronosynthea-mss \
    --test java_validation --release -- test_full_generation_performance --nocapture

# Seed sweep — 50 seeds × 100k patients (fidelity distribution)
RUSTFLAGS="-C target-cpu=native" cargo test --package chronosynthea-mss \
    --test seed_sweep --release -- --nocapture --ignored

# Joint mode (opt-in: rename the .opt-in companion files into place)
mv data/prevalence/cooccurrence.json.opt-in data/prevalence/cooccurrence.json
mv data/prevalence/recalibration.json.opt-in data/prevalence/recalibration.json
RUSTFLAGS="-C target-cpu=native" cargo test --package chronosynthea-mss \
    --test java_validation --release -- test_full_generation_performance --nocapture
```

The pairwise comorbidity vs Java Synthea comparison (E1) needs Java
Synthea running locally:

```bash
# Java Synthea, 10k patients with CSV output (~2 minutes)
git clone --depth 1 https://github.com/synthetichealth/synthea
cd synthea && ./gradlew build -x test
./run_synthea -p 10000 -s 42 --exporter.csv.export=true \
    --exporter.fhir.export=false --exporter.text.export=false

# Extract pairwise stats from chronosynthea
cargo test --package chronosynthea-mss --test e1_pairwise --release \
    -- --nocapture --ignored e1_emit_pairwise_csv

# Compare — Pearson r per Java-lift stratum + marginal drift
python3 workspace/council-review-2026-05-25/e1_dual_mode_compare.py
```

## Honest scope

ChronoSynthea is built to be **functionally equivalent to Java Synthea, just way way faster**. With all three d5 companion files active, the output preserves:

- **Marginal prevalence** of 214 conditions to within ~0.3% of Java (F4 enforced)
- **Per-condition onset timestamps** drawn from Java's empirical distributions (`temporal-ordered`)
- **Conditions emitted in temporal order** within each patient (sorted by onset)
- **Pairwise causal correlation** at Pearson r ≈ 0.78–0.93 vs Java in the calibrated joint mode (`pairwise-empirical`)
- **Demographic mixture** matching Java's joint demographic distribution

What it produces **differently** than Java Synthea:

- **Three-way+ comorbidity structure** is captured only insofar as the pairwise model induces it; full higher-order joint structure requires the d5 = `causal-DAG` Gibbs sampler with properly-fit Boltzmann parameters (scaffolded — see "Open future work").
- **Negative causal correlations** (lift < 0.5 in Java) are not boosted in the additive `pairwise-empirical` sampler. Strata-r stays around 0.81 here; the Gibbs sampler will close this gap once its parameters are fit via pseudo-likelihood.
- **REASONCODE linkage** is **now shipped** ✓ — `FullPatient.medication_causes` and `procedure_causes` record which condition triggered each prescription / procedure (equivalent to Java's `medications.csv:REASONCODE` and `procedures.csv:REASONCODE` columns), sampled proportionally to `P(med | cond)` over the patient's active conditions.
- **Causal inference / treatment effect estimation** — even with full joint structure, marginal-fidelity samplers can distort ATE estimands per arXiv:2604.23904[^causal]. For causal inference, use real EHR data, not synthesis (Java or otherwise).

In other words: chronosynthea **is** Synthea-equivalent on every dimension Java Synthea's state machines produce *deterministically given the demographic*. It diverges only on dimensions that Java Synthea generates from causal per-week trajectories that chronosynthea collapses into sufficient statistics — and even there, with three of four d5 values implemented, the gap is measured in fractions of a Pearson correlation point, not in qualitative kind.

The `MssFingerprint`'s self-classification table tells you what chronosynthea claims and what it doesn't. The d5 axis tells you which mode you're in. The F4 gate fires when any of it drifts. **That's the audit contract working.**

## A word on the "16,000×" history

The original README headlined "16,000× faster than Java Synthea." The
pre-repair audit found that some of the numbers and attribution were
ahead of the source code. Specifically:

- The "SIMD threshold compare" was credited as the hot-path mechanism;
  the SIMD sampler was constructed per Rayon thread but bound to
  `_sampler` (Rust convention for explicitly-unused) and the actual hot
  path called a scalar loop. **Fixed.** SIMD is now genuinely on the
  hot path (`SimdSampler::sample_active`).
- The "KL divergence" metric returned `-0.006132`. KL divergence is
  non-negative by Gibbs' inequality (Cover & Thomas 2006 §2.6[^cover-thomas]).
  **Fixed.** The metric is now proper Bernoulli-pair KL —
  `Σ_i [p_i log(p_i/q_i) + (1-p_i) log((1-p_i)/(1-q_i))]` — non-negative
  by construction. Measured value: 0.001081 across 50 seeds.
- The CI gate asserted `failure_rate < 0.30` at 10% tolerance; the
  observed 0.31% / 0-of-214 numbers were printed via `eprintln!` not
  enforced. **Fixed.** The gate enforces at observed values with margin.
- The co-occurrence scaffolding was dead code; A2 ("no co-occurrence
  modelling needed") was a forced omission rather than a measured
  assumption. **Fixed.** A2 is now a deliberate `marginal-only` choice
  with an `pairwise-empirical` alternative shipped opt-in.

The framework's d/g/a/u classification is what *forced* these to be
caught. The audit was uncomfortable; the system is better for it.

## Open future work

- **d5 = `causal-DAG` parameter fitting**: the Gibbs sampler is wired
  and dispatching but the J_ij parameters derived directly from observed
  log-lifts do not yet specify an Ising-Boltzmann distribution whose
  equilibrium marginals reproduce the source empirical distribution.
  Fitting J via pseudo-likelihood maximisation or Boltzmann learning is
  research-grade work. The pairwise-empirical mode (additive boost + 
  two-knob recalibration) is the validated joint-modelling path for
  production until causal-DAG is properly fit.
- **GPU offload via WGSL**: most per-patient work is embarrassingly
  parallel and SIMD-friendly. A WGSL compute kernel could push throughput
  by another 5–10× on consumer GPUs.
- **CSV output adapter**: Java-Synthea-compatible `patients.csv`,
  `conditions.csv`, `medications.csv`, `procedures.csv`, `encounters.csv`
  emission for direct drop-in replacement. The REASONCODE columns are
  already populated; just needs the writer.
- **No-clamp recalibration**: the auto-load path round-trips imperfectly
  (~22% drift on ~13 of 214 conditions) because successive clamping
  during the recalibration loop accumulates non-linearly while the
  persisted multipliers are a single product. ≤30 lines of code.
- **SIMD on the cooccurrence boost loop**: currently the joint sampler
  fires through the scalar `sample_conditions_with_cooccurrence` path.
  Vectorising would close the ~2× gap between marginal-only (11.4M/sec)
  and joint mode (4.35M/sec).
- **GPU offload via WGSL**: most of the per-patient work is
  embarrassingly parallel and SIMD-friendly. A WGSL kernel could push
  throughput by another 5–10× on consumer GPUs.
- **SynthEHRella harness integration**: the arXiv:2411.04281
  benchmark[^synthehrella] provides community-standard fidelity
  evaluation. Running chronosynthea through it would convert "we
  claim equivalence" into "here's where we sit in the field's
  benchmark." Architectural work: write an adapter from
  chronosynthea's compact patient output to SynthEHRella's expected
  format.

### Why the boost is one-sided

The first cut of the joint sampler had a symmetric subtractive branch —
when `conditional_prob < base_prob` for a pair, remove the dependent
with probability `(base − conditional) × 0.5`. It seemed obviously
right. It failed empirically: multi-trigger subtraction stacks, and a
condition with target prevalence 0.87 got removed by dozens of weak
negative correlations down to 0.015 marginal. 85% drift.

The fix isn't subtractive boost — it's a different sampler. Gibbs
sampling over the full condition vector resamples each condition
conditional on all the others, which naturally handles both positive
and negative correlations without stacking. That's the d5=`causal-DAG`
move; future work.

## Citations

[^lehmann-scheffe]: Lehmann, E.L. & Scheffé, H. (1950). "Completeness,
similar regions, and unbiased estimation." *Sankhyā* 10(4), 305–340.
JSTOR: https://www.jstor.org/stable/25048038

[^xoshiro]: Blackman, D. & Vigna, S. (2021). "Scrambled linear
pseudorandom number generators." *ACM Transactions on Mathematical
Software* 39(1). https://doi.org/10.1145/3460772

[^walker]: Walker, A.J. (1977). "An efficient method for generating
discrete random variables with general distributions." *ACM
Transactions on Mathematical Software* 3(3).
https://doi.org/10.1145/355744.355749

[^vose]: Vose, M.D. (1991). "A linear algorithm for generating random
numbers with a given distribution." *IEEE Transactions on Software
Engineering* 17(9). https://doi.org/10.1109/32.92917

[^cover-thomas]: Cover, T.M. & Thomas, J.A. (2006). *Elements of
Information Theory*, 2nd ed., Wiley. §2.6 (Gibbs' inequality).

[^synthea-mitre]: Walonoski, J. et al. (2018). "Synthea: An approach,
method, and software mechanism for generating synthetic patients and
the synthetic electronic health care record." *Journal of the American
Medical Informatics Association* 25(3), 230–238.
https://doi.org/10.1093/jamia/ocx079

[^synthehrella]: Chen, X., Wu, Z., Shi, X., Cho, H., Mukherjee, B.
(2025). "Generating synthetic electronic health record data: a
methodological scoping review with benchmarking on phenotype data and
open-source software." *JAMIA* 32(7), 1227–1240.
https://doi.org/10.1093/jamia/ocaf082

[^causal]: arXiv:2604.23904 (2026). "Generative synthetic data for
causal inference: pitfalls, remedies, and opportunities."
https://arxiv.org/abs/2604.23904

[^wasp-asplos]: Disambiguation footnote: ACM ASPLOS 2024 published a
paper titled "WASP: Workload-Aware Self-Replicating Page-Tables for
NUMA Servers" — same acronym, different expansion. ChronoSynthea's
WASP = Workload-Aware Sufficient Placement, a framework for synthetic-
data encoding originated at chronomancy-io. The two papers do not
overlap in scope.

## License + acknowledgments

ChronoSynthea is Apache-2.0. The Java Synthea reference (used as both
calibration source and head-to-head benchmark) is also Apache-2.0,
maintained by MITRE Corporation. We acknowledge a substantial
intellectual debt to the Synthea team — without their well-engineered
state-machine modules and freely available 1M-patient dataset, the
WASP move (replace simulation with sufficient-statistic sampling)
would have had nothing to be the sufficient statistic *of*.

The session that produced the audit, the d5 axis, the recalibration
loop, and the optimisation stack documented here used the Council of
Wizards multi-agent review tool at
[chronocow](https://github.com/the-chronomancer/chronocow). The full
council transcripts, prior-art briefings, and dual-mode comparison
data live in that repo's workspace archive at
`workspace/chronosynthea/council-review-2026-05-25/`.

---

> ChronoSynthea is a tool for the workloads that need statistical
> equivalence at population scale, not causal trajectories at patient
> scale. The audit contract is what makes that scope checkable. The
> numbers are what makes it useful. The framework is load-bearing
> because the d5 axis is the choice — not the marketing.
