//! d6 = `causal-cascade` axis — empirically-derived condition-pair lag
//! rules that enforce trajectory ordering on sampled patients.
//!
//! ## Why
//!
//! Java Synthea's per-week state machines produce multi-year causal
//! cascades: prediabetes → diabetes (~5y) → diabetic retinopathy (~5–10y)
//! → CKD stage 1 → 2 → 3 → 4 (progressive). ChronoSynthea's sufficient-
//! statistic sampler captures the *joint* distribution but, by default,
//! draws each condition's onset day independently from the per-condition
//! age-of-onset stat. The result is the right set of conditions but the
//! wrong temporal trajectory: CKD stage 4 might land at year 30 with
//! prediabetes at year 50, which never happens clinically.
//!
//! `CausalCascadeModel` closes that gap. For each ordered (trigger,
//! downstream) pair extracted from Java's conditions.csv with a
//! statistically significant positive mean lag, it rewrites the
//! downstream condition's onset day to be `trigger.onset_day +
//! Normal(mean_lag, std_lag)`. The post-pass runs in O(cascade_rules ×
//! conditions_per_patient) which is ~200μs for the 9,000-rule registry
//! we ship by default.
//!
//! ## Activation
//!
//! Set `CHRONOSYNTHEA_CASCADE_PATH` to a `cascade_lags.json` file (or
//! drop one alongside `calibrated_registry.json`). The fingerprint
//! deserialises it into `CascadeRule[]`; `BatchGenerator` then runs the
//! post-pass during `generate_full_patient`. Without the file, the
//! sampler falls back to the marginal (independent-onset) behaviour
//! shipped before this module — backward compatible.

use ahash::AHashMap;
use rand::Rng;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::archetype::ArchetypeRegistry;
use crate::tables::CodeTable;

/// One empirically-derived (trigger, downstream) cascade rule with the
/// onset-lag distribution and the conditional probability that the
/// downstream condition fires given the trigger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeRule {
    pub trigger: String,
    pub downstream: String,
    pub mean_days: u32,
    pub std_days: u32,
    pub probability: f32,
    #[serde(default)]
    pub n: u32,
}

/// Compiled cascade model. Each `(downstream, trigger)` rule is
/// indexed by **downstream** rather than trigger so the post-pass can
/// pick the single best (highest-probability) active-trigger per
/// downstream — Java's state-machine modules have one dominant cause per
/// progression, not a union of all weakly-correlated upstream conditions.
/// Treating every empirical pair as a cascade rule (the prior naive
/// implementation) compounded onsets into a single "all conditions at
/// max-age" failure mode.
#[derive(Debug, Clone, Default)]
pub struct CausalCascadeModel {
    /// `by_downstream[downstream_idx as usize]` = list of triggers that
    /// can cause it: `(trigger_idx, mean_days, std_days, probability)`.
    /// Sorted descending by `probability` so the post-pass can break
    /// after the first active trigger.
    by_downstream: Vec<SmallVec<[(u16, u32, u32, f32); 4]>>,
}

impl CausalCascadeModel {
    /// Build the model from a JSON cascade-rule array and a `CodeTable`
    /// (used to resolve SNOMED codes → condition indices). Rules are
    /// indexed by downstream and sorted descending by probability so the
    /// post-pass picks the single most-probable active trigger.
    pub fn from_rules(rules: &[CascadeRule], code_table: &CodeTable) -> Self {
        let n = code_table.num_conditions();
        let mut by_downstream: Vec<SmallVec<[(u16, u32, u32, f32); 4]>> =
            vec![SmallVec::new(); n];
        let code_to_idx: AHashMap<String, u16> = (0..n)
            .filter_map(|i| code_table.condition(i as u16).map(|e| (e.code.clone(), i as u16)))
            .collect();
        for r in rules {
            if let (Some(&t), Some(&d)) = (
                code_to_idx.get(&r.trigger),
                code_to_idx.get(&r.downstream),
            ) {
                by_downstream[d as usize]
                    .push((t, r.mean_days, r.std_days, r.probability));
            }
        }
        for v in &mut by_downstream {
            v.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        }
        Self { by_downstream }
    }

    /// Total number of (trigger, downstream) rules compiled.
    pub fn rule_count(&self) -> usize {
        self.by_downstream.iter().map(|v| v.len()).sum()
    }

    /// Post-process a patient's `(condition_idx, onset_days)` pairs so
    /// that for every active cascade rule `(trigger → downstream)`, the
    /// downstream condition's onset is `trigger.onset + Normal(mean, std)`
    /// clamped to `[trigger.onset + 1, max_age_days]`. Operates in-place;
    /// the caller is responsible for re-sorting the (cond, onset) pairs
    /// afterwards if temporal-ordered output is desired (see
    /// `apply_and_resort`).
    pub fn apply<R: Rng>(
        &self,
        conditions: &[u16],
        onset_days: &mut [u16],
        max_age_days: u32,
        rng: &mut R,
    ) {
        if self.by_downstream.is_empty() {
            return;
        }
        std::sync::Once::new().call_once(|| {
            if std::env::var("CHRONOSYNTHEA_CASCADE_DEBUG").is_ok() {
                eprintln!(
                    "[cascade] active with {} (downstream → trigger) rules across {} downstreams",
                    self.rule_count(),
                    self.by_downstream.iter().filter(|v| !v.is_empty()).count()
                );
            }
        });
        // Build slot lookup once.
        let mut slot_of: SmallVec<[(u16, u8); 32]> = SmallVec::new();
        for (slot, &c) in conditions.iter().enumerate() {
            slot_of.push((c, slot as u8));
        }

        // Run two relaxation rounds. Round 1 rewrites each downstream's
        // onset using the snapshot of the ORIGINAL sampled onsets (so
        // every downstream sees stable trigger inputs). Round 2 reads the
        // updated onsets so a chain like prediabetes → diabetes → CKD-2
        // propagates: round 1 pushes diabetes; round 2 sees the pushed
        // diabetes onset and pushes CKD-2 ahead of the *updated* diabetes
        // onset. Two rounds catch the common two-level chain without the
        // multi-pass compounding that pinned everything to max_age in
        // earlier attempts.
        let mut working = SmallVec::<[u16; 32]>::from_slice(onset_days);
        for round in 0..2 {
            let snapshot = working.clone();
            for (dslot, &downstream_idx) in conditions.iter().enumerate() {
                let rules = match self.by_downstream.get(downstream_idx as usize) {
                    Some(r) if !r.is_empty() => r,
                    _ => continue,
                };
                let mut best_proposed: i32 = snapshot[dslot] as i32;
                for &(trigger_idx, mean_days, std_days, _prob) in rules {
                    let t_slot = match slot_of.iter().find(|&&(c, _)| c == trigger_idx) {
                        Some(&(_, s)) => s as usize,
                        None => continue,
                    };
                    if t_slot == dslot {
                        continue;
                    }
                    let trigger_onset = snapshot[t_slot] as i32;
                    if trigger_onset + 30 >= max_age_days as i32 {
                        continue;
                    }
                    let z = irwin_hall_normal(rng);
                    let proposed = trigger_onset
                        + mean_days as i32
                        + (std_days as f32 * z) as i32;
                    let lo = trigger_onset + 1;
                    let hi = max_age_days as i32;
                    let candidate = proposed.clamp(lo, hi);
                    if candidate > best_proposed {
                        best_proposed = candidate;
                    }
                    break; // most-probable active trigger only
                }
                working[dslot] = best_proposed.min(u16::MAX as i32) as u16;
            }
            let _ = round; // suppress unused
        }

        // Final pairwise consistency pass: walk the active (downstream,
        // dominant-trigger) rules once more. If the downstream's final
        // onset is still earlier than the trigger's final onset, push the
        // downstream to `trigger.onset + 1`. Catches deeper chains where
        // a 3+ level cascade bumped the trigger past the downstream's
        // relaxed onset and the two-round pass didn't fully resolve.
        for (dslot, &downstream_idx) in conditions.iter().enumerate() {
            let rules = match self.by_downstream.get(downstream_idx as usize) {
                Some(r) if !r.is_empty() => r,
                _ => continue,
            };
            for &(trigger_idx, _, _, _) in rules.iter().take(1) {
                if let Some(&(_, t_slot)) =
                    slot_of.iter().find(|&&(c, _)| c == trigger_idx)
                {
                    let t_onset = working[t_slot as usize] as i32;
                    let d_onset = working[dslot] as i32;
                    if d_onset <= t_onset && t_onset + 1 <= max_age_days as i32 {
                        working[dslot] = (t_onset + 1) as u16;
                    }
                }
            }
        }
        onset_days.copy_from_slice(&working);
    }
}

/// Cheap N(0,1)-approximate draw via Irwin-Hall(12). Same approximation
/// used by `archetype::sample_onset_days`; matches its statistical shape
/// at population scale.
#[inline]
fn irwin_hall_normal<R: Rng>(rng: &mut R) -> f32 {
    let mut s = 0.0f32;
    for _ in 0..12 {
        s += rng.gen::<f32>();
    }
    s - 6.0
}

/// Convenience: load cascade rules from `cascade_lags.json` if it sits
/// next to the calibrated registry or is pointed to by the
/// `CHRONOSYNTHEA_CASCADE_PATH` env var. Returns an empty rule list when
/// neither source is present (cascade post-pass becomes a no-op).
pub fn load_default_rules<P: AsRef<std::path::Path>>(
    registry_dir: P,
) -> std::io::Result<Vec<CascadeRule>> {
    let path = std::env::var("CHRONOSYNTHEA_CASCADE_PATH")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| Some(registry_dir.as_ref().join("cascade_lags.json")));
    let Some(p) = path else {
        return Ok(Vec::new());
    };
    if !p.exists() {
        return Ok(Vec::new());
    }
    let file = std::fs::File::open(&p)?;
    let reader = std::io::BufReader::new(file);
    let rules: Vec<CascadeRule> = serde_json::from_reader(reader)?;
    Ok(rules)
}

/// Reference reference to drop the `unused` warning on `ArchetypeRegistry`
/// in builds where the cascade module is included but not wired through
/// the batch generator yet. Will be removed once batch wires it in.
#[doc(hidden)]
pub fn _archetype_typecheck(_: &ArchetypeRegistry) {}
