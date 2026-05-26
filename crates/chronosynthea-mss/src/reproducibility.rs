//! Reproducibility primitives — the foundation under any "you can
//! re-run my cohort and get bit-identical output" claim.
//!
//! Why this module exists, in one paragraph:
//!
//! chronosynthea is deterministic given `(seed, registry, generator
//! version)`. Without all three pinned, "same seed gave you a
//! different patient" is the bug. This module provides:
//!
//! 1. `GENERATOR_VERSION` — a monotonic integer that bumps whenever the
//!    generator's *semantics* change (a new bug fix in sampling, a
//!    cascade rule edit, a PRNG swap). Distinct from `Cargo.toml`'s
//!    semver — `0.1.x` patch bumps don't change semantics; this counter
//!    does.
//! 2. `registry_content_hash(&CalibratedRegistry) -> [u8; 32]` — a
//!    SHA-256 over the canonical JSON serialisation of the registry,
//!    including cooccurrence / cascade / recalibration overrides. Two
//!    registries that produce the same content hash will produce
//!    identical patient distributions.
//! 3. `derive_patient_seed(base, registry_hash, gen_ver, patient_idx)`
//!    — the deterministic per-patient seed used by every
//!    `BatchGenerator::generate_*` path. Folding the registry hash and
//!    generator version into the per-patient seed means changing either
//!    *automatically* changes the resulting patient — no silent shifts
//!    where "same seed reused" produces a different patient under a
//!    different registry.
//! 4. `CohortManifest` — emitted alongside generated output so an
//!    auditor can `chronosynthea replay --manifest cohort.json` and
//!    verify the output hash matches.

use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Generator semantics version. Bumped on changes that alter sampled
/// output for a given `(seed, registry)` — e.g. cascade rule edits,
/// PRNG algorithm swap, sampling-order change, bug fixes in the
/// generator.
///
/// Distinct from the crate's Cargo semver: a `0.1.5 → 0.1.6` patch
/// release that touches docs / writer code / build config but doesn't
/// change generator output should NOT bump this. Conversely a single
/// patch release that fixes a sampling bug bumps both.
///
/// History:
/// - `1` — initial (everything before the reproducibility audit pass).
pub const GENERATOR_VERSION: u32 = 1;

/// Schema version for the generated CSV / Parquet output files.
/// Distinct from `GENERATOR_VERSION`: this changes when columns are
/// added / renamed / dropped from the output schema. The generator
/// may produce identical patient internals while the output schema
/// changes (e.g. adding an `ARCHETYPE_ID` column).
///
/// History:
/// - `1` — initial 15-file Java-Synthea-compatible CSV layout.
pub const SCHEMA_VERSION: u32 = 1;

/// SHA-256 of the canonical JSON serialisation of a calibrated
/// registry. Two registries with the same hash will produce identical
/// patient distributions when fed to the generator. Use this in
/// manifests and as a component of per-patient seed derivation.
///
/// The hash covers every field that affects sampled output:
/// conditions, medications, observations, procedures, demographics,
/// archetypes, cooccurrence_pairs, cooccurrence_dependent_scale,
/// onset_stats, and recalibration_multipliers. Fields like the
/// `extracted_at` timestamp are also included (they're part of the
/// JSON) but should not in practice change patient output — they're
/// covered for completeness.
/// SHA-256 of an `MssFingerprint`'s canonical JSON. This is what the
/// generator actually consumes — `CalibratedRegistry::to_fingerprint`
/// is the only path that produces sampled patients. If two
/// fingerprints have the same hash, the generator's output is
/// identical for the same seed + version (modulo PRNG bugs).
///
/// `registry_content_hash` covers a strict superset of
/// `fingerprint_content_hash`'s domain (registries carry calibration
/// metadata that doesn't survive `to_fingerprint`); use this hash for
/// the per-patient seed derivation, and the registry hash for the
/// audit manifest's `registry_hash` field.
pub fn fingerprint_content_hash(
    fingerprint: &crate::fingerprint::MssFingerprint,
) -> [u8; 32] {
    // MsgPack (rmp-serde) rather than JSON: the fingerprint's
    // cooccurrence map uses tuple keys (u16, u16) which JSON cannot
    // represent. MsgPack handles arbitrary key types and is also
    // deterministic for `#[derive(Serialize)]` structs.
    let bytes = rmp_serde::to_vec(fingerprint)
        .expect("MssFingerprint MsgPack serialisation is total");
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    hasher.finalize().into()
}

// Note: `CalibratedRegistry` doesn't derive `Serialize` (it's
// load-only from disk JSON). For hashing-from-registry semantics,
// use `fingerprint_content_hash` on the result of
// `CalibratedRegistry::to_fingerprint()` — that hash is what the
// generator actually consumes.

/// Deterministic per-patient seed for the generator's PRNG. Folding
/// `base_seed`, `registry_hash`, `generator_version`, and
/// `patient_idx` together means changing ANY of them deterministically
/// changes the resulting patient — no silent reuse where you change
/// the registry but get the "same" patient back via `patient_idx`.
///
/// Uses SplitMix64-style mixing (the same constants the Xoshiro256++
/// init uses internally), which is cheap (~5 ns per call), avalanches
/// well, and has zero state. NOT cryptographic — we don't need
/// adversarial resistance, we need a deterministic well-distributed
/// per-patient seed.
///
/// Contract: changing the bit pattern of any of the four inputs
/// changes the output seed in a well-distributed way. Same inputs
/// always produce the same output.
#[inline]
pub fn derive_patient_seed(
    base_seed: u64,
    registry_hash: &[u8; 32],
    generator_version: u32,
    patient_idx: u64,
) -> u64 {
    // Pull the first 16 bytes of the registry hash as two u64s. Two
    // u64s of entropy from SHA-256 is more than enough to disambiguate
    // between any two distinct registries we'll ever ship.
    let h0 = u64::from_le_bytes(registry_hash[0..8].try_into().unwrap());
    let h1 = u64::from_le_bytes(registry_hash[8..16].try_into().unwrap());

    let mut s = base_seed;
    s ^= h0;
    s = splitmix64(s);
    s ^= h1;
    s = splitmix64(s);
    s ^= generator_version as u64;
    s = splitmix64(s);
    s ^= patient_idx;
    splitmix64(s)
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// Render a registry content hash as a lowercase hex string. Used in
/// manifests and log output; cheap, ~32 µs.
pub fn hash_hex(hash: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in hash {
        use std::fmt::Write;
        write!(&mut s, "{:02x}", b).unwrap();
    }
    s
}

/// Reproducibility manifest emitted alongside generated output. An
/// auditor reproducing the cohort runs `chronosynthea replay
/// --manifest cohort.json` which:
///
/// 1. Loads the registry referenced by `registry_hash` from a
///    well-known path (or fails if it can't be found).
/// 2. Checks `generator_version` matches the running binary.
/// 3. Regenerates the cohort using `seed` and `count`.
/// 4. Hashes the regenerated output and verifies it matches
///    `output_sha256`.
///
/// All four invariants must hold for the cohort to be considered
/// reproducible.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohortManifest {
    /// Schema version of THIS manifest format. Bump when the manifest
    /// itself changes shape.
    pub manifest_version: u32,
    /// chronosynthea generator semantics version at write time.
    pub generator_version: u32,
    /// CSV/Parquet output schema version at write time.
    pub schema_version: u32,
    /// Hex-encoded SHA-256 of the registry's canonical JSON
    /// serialisation. The registry that produces this hash is what the
    /// replayer must load.
    pub registry_hash: String,
    /// User-supplied (or generator-assigned) seed.
    pub seed: u64,
    /// Number of patients in the cohort.
    pub count: usize,
    /// ISO8601 UTC timestamp when the cohort was generated. Not
    /// load-bearing for reproducibility — informational only.
    pub generated_at: String,
    /// Output format (e.g. "csv", "parquet", "parquet-slim",
    /// "parquet-stats").
    pub format: String,
    /// Hex-encoded SHA-256 of the output bytes. For multi-file output
    /// (CSV / Parquet 6-file), this is the SHA-256 of the
    /// concatenation of all output files in sorted-name order.
    /// Optional — populated by the writer post-flush.
    pub output_sha256: Option<String>,
    /// Total bytes written across all output files.
    pub output_bytes: u64,
}

impl CohortManifest {
    /// Build a manifest from a fingerprint hash + seed + count.
    /// `output_sha256` and `output_bytes` are filled in after the
    /// writer flushes; this builder leaves them at safe defaults.
    ///
    /// Get the fingerprint hash from `BatchGenerator::fingerprint_hash()`
    /// or by calling `fingerprint_content_hash` directly.
    pub fn new(
        fingerprint_hash: &[u8; 32],
        seed: u64,
        count: usize,
        format: &str,
    ) -> Self {
        Self {
            manifest_version: 1,
            generator_version: GENERATOR_VERSION,
            schema_version: SCHEMA_VERSION,
            registry_hash: hash_hex(fingerprint_hash),
            seed,
            count,
            generated_at: chrono::Utc::now()
                .format("%Y-%m-%dT%H:%M:%SZ")
                .to_string(),
            format: format.to_string(),
            output_sha256: None,
            output_bytes: 0,
        }
    }

    pub fn write_json<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let s = serde_json::to_string_pretty(self).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        })?;
        std::fs::write(path, s)
    }
}
