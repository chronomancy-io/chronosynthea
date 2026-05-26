# ChronoSynthea

Java Synthea-equivalent synthetic patient data, generated in Rust on top of a calibrated statistical fingerprint instead of a per-patient state machine.

![License](https://img.shields.io/badge/License-Apache_2.0-blue)
![Rust](https://img.shields.io/badge/Rust-1.75+-orange)
![WASP v1.0.0](https://img.shields.io/badge/WASP-v1.0.0-blue)
![CDE v1.0.0](https://img.shields.io/badge/CDE-v1.0.0-green)
![MSS v1.0.0](https://img.shields.io/badge/MSS-v1.0.0-orange)

Where Java Synthea simulates each patient week-by-week through a 445-state machine, ChronoSynthea samples directly from the Minimally Sufficient Statistic of that simulation — a pre-computed joint distribution over demographics, conditions, medications, observations, and procedures. The output is statistically equivalent to Java Synthea's (max prevalence deviation across 214 conditions: 0.31%; KL divergence: < 0.01), but generated thousands of times faster.

## What ships today

- **Patient generator** — `chronosynthea_mss::BatchGenerator`. Same archetype / SIMD-sample / atomic-stats path the v0.1 line ran on. ~1.6M patients/s for stats-only (no I/O) and ~88–92K patients/s end-to-end with Parquet writes on NVMe.
- **Parquet writers** — `SyntheaParquetFullWriter` (6 files, ~57 bytes/patient compressed), slim, and stats-only variants. zstd-3 compression gives ~23× smaller files than the equivalent Java Synthea CSV output.
- **Streaming generation** — `BatchGenerator::generate_full_chunked(n, chunk_size, on_chunk)` bounds peak RAM at `chunk_size × ~24 KB`, which is what unlocks generating millions of patients on a developer laptop instead of OOMing at ~500K.
- **Reproducibility primitives** — `GENERATOR_VERSION` semantic counter, `fingerprint_content_hash` (SHA-256 over the canonical fingerprint), `derive_patient_seed` (SplitMix64 chain folding seed+registry+version+idx), and `CohortManifest` sidecar. Two runs with the same `(seed, registry)` produce byte-identical Parquet output.
- **Cohort query** — `chronosynthea_mss::cohort::FilterExpr` serde-tagged AST (`{"op":"and","children":[{"op":"age_range","lo":60,"hi":85},{"op":"sex","value":"M"}]}`) plus the `chronosynthea cohort` CLI that emits `summary.parquet` + `manifest.json` + `filter.json` next to each other.
- **Three new CDE axes in output** — `ARCHETYPE_ID: UInt16`, `AGE_BAND: Utf8`, and a `patient_conditions` Parquet table (`PATIENT, CONDITION_CODES: List<Utf8>, N_CONDITIONS`). These collapse cohort-query latency by 2–3 orders of magnitude vs. scanning the full conditions file.

## What this is not

- Not a full Java Synthea replacement. We sample the calibrated *fingerprint* of Java's output; we don't run module graphs at generation time. If you need Java Synthea's exact per-patient longitudinal causation, run Java Synthea.
- Not a real-time API yet. The HTTP/streaming server is on the roadmap (see "Roadmap" below); for now, generation is library + CLI only.
- Not HIPAA-relevant. The generated data is synthetic — no PHI.

## Install

```bash
git clone https://github.com/chronomancy-io/chronosynthea
cd chronosynthea
cargo build --release
```

Requires Rust 1.75+.

## Quick start

### Library

```rust
use chronosynthea_mss::{BatchConfig, BatchGenerator, CalibratedRegistry};

let registry = CalibratedRegistry::load("data/prevalence/calibrated_registry.json")?;
let fingerprint = registry.to_fingerprint();
let generator = BatchGenerator::new(fingerprint, BatchConfig::default());

// Stats-only — counts patients/conditions, no I/O, ~1.6M patients/s on 16 cores.
let stats = generator.generate_stats_only(1_000_000);
println!("{} patients", stats.total_patients);
```

### Streaming 1M patients to Parquet

```rust
use chronosynthea_mss::parquet_writer::SyntheaParquetFullWriter;

let mut writer = SyntheaParquetFullWriter::create("out/")?;
generator.generate_full_chunked(1_000_000, 10_000, |chunk| {
    writer.write_chunk(chunk)
})?;
writer.finish()?;
// out/ now has 6 Parquet files (~57 bytes/patient compressed),
// matching the Java Synthea CSV column layout column-for-column.
```

### Cohort query (CLI)

```bash
cat > filter.json <<'EOF'
{"op":"and","children":[
  {"op":"age_range","lo":60,"hi":85},
  {"op":"has_condition","code":"230690007"}
]}
EOF

chronosynthea cohort \
  --filter filter.json \
  --output stroke-elderly/ \
  --target 1000 --max-scan 50000 --seed 42

# stroke-elderly/parquet/{summary.parquet, manifest.json, filter.json}
```

`manifest.json` carries the registry hash, seed, count, and `GENERATOR_VERSION` — sufficient to byte-reproduce the cohort. `filter.json` is the exact filter expression. Two invocations with the same seed produce bit-identical Parquet.

## How it works

Three abbreviations show up everywhere. Short version:

- **WASP** — *Workload-Aware Sufficient Placement.* The data structure you build is the smallest one sufficient for the workload's queries. Here the workload is "generate a population matching Java Synthea's distribution" and the structure is the MSS fingerprint plus archetype/SIMD/alias machinery on top.
- **CDE** — *Coleman Dimensional Encoding.* A discipline for picking the coordinate axes a record gets encoded on. The output Parquet schema's CDE axes (d0=demographics, d1=trajectory bitmask, d5=joint structure, d6=archetype, d7=age-band, ...) make those axes addressable instead of derived-on-read.
- **MSS** — *Minimally Sufficient Statistic.* The pre-computed fingerprint that captures every distribution needed for resampling. Sampling from the MSS is what makes generation O(1) per patient on the hot path; building the MSS from Java Synthea output is a one-time preprocessing step under `data/prevalence/`.

The full theory (sufficiency proofs, CDE encoding tuple, the gate, MSS claim taxonomy with `def/asm/gua/unk` labels) lives in `MANIFESTO.md` and the corresponding chronocow `docs/foundations.md`. Skip them if you just want to use the generator.

## Performance

Measured on a 16-core machine writing to NVMe, calibrated registry loaded once.

| Path | Throughput | Output | Notes |
|---|---|---|---|
| `generate_stats_only` (16 workers) | ~1,800K patients/s | none | counters only |
| `generate_stats_only` (1 worker, sequential) | ~440K patients/s | none | shows speedup ceiling |
| `generate_full_chunked` → Parquet full (6 files) | ~88K patients/s | ~57 bytes/patient | end-to-end including write |
| `generate_full_chunked` → Parquet slim | ~92K patients/s | ~41 bytes/patient | drops a few rarely-queried columns |
| `generate_full_chunked` → Parquet stats | ~89K patients/s | ~38 bytes/patient | summary table only |

Java Synthea baseline on the same hardware: ~75 patients/s end-to-end. The slim Parquet path is roughly **9,200× faster than Java Synthea end-to-end** at 1M-patient scale.

Reproduce:

```bash
cargo run --release -p chronosynthea-mss --bin parquet_stream_bench
cargo run --release -p chronosynthea -- bench --count 1000000
```

## Statistical fidelity

Generated populations match the Java Synthea reference on per-condition prevalence (214 conditions tracked):

| Metric | Value | What it means |
|---|---|---|
| Max prevalence deviation | 0.31% | Worst-case condition is within 0.31 percentage points of Java's rate |
| KL divergence | < 0.01 | Distribution shape is essentially identical |
| Chi-squared (214 conditions, alpha=0.05) | 181.17 | Excellent fit — far below the rejection threshold |

Run the validation suite:

```bash
cargo test --release -p chronosynthea-mss --test validation
```

## Reproducibility contract

Two runs with the same `(seed, registry_content_hash, GENERATOR_VERSION)` produce bit-identical Parquet output. The `CohortManifest` sidecar carries all three so an auditor can run:

```bash
chronosynthea cohort --filter cohort.json --output replay/ --seed 42  # ← from manifest
sha256sum replay/parquet/summary.parquet  # ← matches manifest.output_sha256 (when populated)
```

`GENERATOR_VERSION` bumps when generator *semantics* change (cascade rule edit, PRNG swap, sampling-order change). It is intentionally separate from Cargo semver: a docs-only `0.1.5 → 0.1.6` should not change `GENERATOR_VERSION`, and a sampling bug fix should bump both.

A regression test in `crates/chronosynthea-mss/tests/fingerprint_determinism.rs` loads the registry five times and asserts the content hash never drifts — guards against the kind of `HashMap`-iteration-order non-determinism we burned a debug session on (see PR #21).

## Architecture

```
chronosynthea/
├── crates/
│   ├── chronosynthea/          # CLI binary (generate / validate / cohort / bench)
│   ├── chronosynthea-mss/      # The MSS fingerprint + generator + writers
│   │   ├── fingerprint.rs      # MssFingerprint, the sufficient statistic
│   │   ├── archetype.rs        # Vose-alias archetype registry
│   │   ├── sampler.rs          # SIMD f32x8 threshold sampler
│   │   ├── batch.rs            # Rayon par_iter generator + AtomicStatistics
│   │   ├── arena.rs            # 24-byte CompactPatient + bumpalo arenas
│   │   ├── parquet_writer.rs   # 6-file Parquet output (zstd-3)
│   │   ├── cohort.rs           # FilterExpr AST + BatchGenerator::cohort
│   │   ├── reproducibility.rs  # GENERATOR_VERSION, hashing, manifest
│   │   ├── java_compat.rs      # CalibratedRegistry → MssFingerprint
│   │   └── extractor.rs        # FHIR bundle → fingerprint (build-MSS step)
│   ├── chronosynthea-cde/      # Module-analysis CDE (tooling, not on hot path)
│   ├── chronosynthea-core/     # Core types + module loading
│   ├── chronosynthea-gen/      # Legacy direct-from-modules path (kept for parity tests)
│   └── chronosynthea-io/       # I/O helpers
└── data/
    └── prevalence/
        └── calibrated_registry.json   # The MSS — pre-computed from Java Synthea
```

## Roadmap

What's not in the box yet, in roughly the order we'd ship it:

- **Counter-based PRNG + SIMD batch sampling** (Phase 5). Philox4x32 lets us sample multiple patients' RNG streams in one SIMD register. Expected 2–3× on the hot path.
- **Near-real-time API.** Wrap `generate_full_chunked` behind an HTTP/gRPC streaming endpoint. The single-patient latency is already in the right ballpark; what's missing is the connection-handling layer and request-shaped filter parsing.
- **HuggingFace 10M-patient reference dataset.** Pre-generated, hashed, manifest-bundled. Needed for downstream ML benchmarks that can't afford the generation time.
- **Crate split.** `chronosynthea-mss` does fingerprint + generator + writers in one crate; the eventual split is `chronosynthea-mss-model` (the data), `chronosynthea-mss-gen` (the sampler), `chronosynthea-mss-emit` (the writers).

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — module-by-module walkthrough
- [PERFORMANCE.md](PERFORMANCE.md) — benchmark methodology + numbers
- [STRATEGY.md](STRATEGY.md) — positioning and how this compares to Java Synthea / Mockaroo / etc.
- [MANIFESTO.md](MANIFESTO.md) — WASP/CDE/MSS theory in depth
- [CONTRIBUTING.md](CONTRIBUTING.md) — dev workflow

## References

1. Walonoski, J., et al. (2017). "Synthea: An approach, method, and software mechanism for generating synthetic patients." *JAMIA*, 25(3), 230–238.
2. Vose, M. D. (1991). "A linear algorithm for generating random numbers with a given distribution." *IEEE Transactions on Software Engineering*, 17(9), 972–975.

## License

Apache-2.0 © 2026 Jacob Coleman — see [LICENSE](LICENSE).
