//! Regression test for fingerprint content-hash determinism.
//!
//! `fingerprint_content_hash` is the load-bearing primitive for the
//! reproducibility manifest: same registry + seed must always produce
//! the same patient cohort, and the manifest's `registry_hash` is what
//! a replayer verifies against. If the hash drifts across loads of the
//! same registry file, every claim about reproducibility is broken.
//!
//! Historical regression: `CalibratedRegistry::to_fingerprint` used
//! `AHashMap` internally for the demographic-bucket join and called
//! `buckets.values().sum()` to normalise. Float addition is
//! non-associative, so summing in AHashMap iteration order (whose
//! hasher is randomly seeded per instance) produced subtly different
//! totals across runs, and the `*prob /= total` step propagated that
//! drift into every probability — and therefore into the fingerprint
//! hash. Fixed by switching the demographics builders to `BTreeMap`.

use chronosynthea_mss::{reproducibility::fingerprint_content_hash, CalibratedRegistry};

const REGISTRY_PATH: &str = "../../data/prevalence/calibrated_registry.json";

#[test]
fn fingerprint_hash_stable_across_loads() {
    if !std::path::Path::new(REGISTRY_PATH).exists() {
        eprintln!("skipping: registry not present at {}", REGISTRY_PATH);
        return;
    }

    let mut hashes = Vec::new();
    for _ in 0..5 {
        let registry = CalibratedRegistry::load(REGISTRY_PATH)
            .expect("registry loads from on-disk JSON");
        let fp = registry.to_fingerprint();
        hashes.push(fingerprint_content_hash(&fp));
    }

    let first = hashes[0];
    for (i, h) in hashes.iter().enumerate() {
        assert_eq!(
            h, &first,
            "fingerprint hash drifted between loads (run {}). \
             This means `to_fingerprint()` has a non-deterministic \
             code path — most likely a HashMap iteration order \
             affecting float arithmetic.",
            i
        );
    }
}
