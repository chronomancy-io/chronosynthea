//! Patient archetype system for O(1) demographic and condition sampling.
//!
//! Instead of sampling demographics separately and then computing condition
//! probabilities, we pre-compute archetypes that bundle demographics with
//! their associated condition probabilities.

use ahash::AHashMap;
use rand::Rng;
use smallvec::SmallVec;

use crate::fingerprint::{DemographicBucket, MssFingerprint};

/// Maximum number of conditions per archetype.
/// Cap on the number of distinct conditions a fingerprint may carry.
/// Currently 256 (one byte's worth of indices), well above the 214 we ship.
/// Reserved for future bounds-checking; unused on the current hot path.
#[allow(dead_code)]
const MAX_CONDITIONS: usize = 256;

/// A patient archetype with pre-computed condition probabilities.
#[derive(Debug, Clone)]
pub struct PatientArchetype {
    /// Archetype ID (for deterministic generation). Typed as `ArchetypeId`
    /// so the compiler refuses to confuse it with a `ConditionIndex` or
    /// other bare `u16` index at API boundaries.
    pub id: crate::types::ArchetypeId,

    /// Dense per-archetype condition→base-probability lookup table.
    /// `prob_by_condition[cond_idx]` is the base prevalence of condition
    /// `cond_idx` in this archetype, or `0.0` if the condition is inactive.
    ///
    /// Replaces the previous linear scan through `self.conditions` in the
    /// joint sampler's boost loop. With ~25 triggers per patient × ~3
    /// dependents per trigger × ~30 active conditions, the scan cost was
    /// ~2,250 ops per patient just for base_prob lookups; this table makes
    /// each lookup O(1) (one `Vec` read). Populated by `ArchetypeBuilder::build`
    /// and `from_fingerprint` after `conditions` is finalised.
    pub prob_by_condition: Vec<f32>,

    /// Demographic bucket.
    pub demographics: DemographicBucket,

    /// Age range (min, max) for this archetype.
    pub age_range: (u32, u32),

    /// Pre-computed condition probabilities (indexed by condition ID).
    /// Stored as (condition_idx, probability) pairs, sorted by probability descending.
    pub conditions: SmallVec<[(u16, f32); 32]>,

    /// Probability weight for this archetype (for weighted sampling).
    pub weight: f32,

    /// Mean number of encounters for patients with this archetype.
    pub mean_encounters: f32,

    /// Mean number of events per encounter.
    pub mean_events_per_encounter: f32,
}

impl PatientArchetype {
    /// Samples conditions for a patient with this archetype (basic version).
    #[inline]
    pub fn sample_conditions<R: Rng>(&self, rng: &mut R, buffer: &mut SmallVec<[u16; 8]>) {
        buffer.clear();
        for &(cond_idx, prob) in &self.conditions {
            if rng.gen::<f32>() < prob {
                buffer.push(cond_idx);
            }
        }
    }

    /// Samples conditions with co-occurrence modeling.
    /// After initial sampling, applies conditional probabilities for related conditions.
    #[inline]
    pub fn sample_conditions_with_cooccurrence<R: Rng>(
        &self,
        rng: &mut R,
        buffer: &mut SmallVec<[u16; 8]>,
        cooccurrence: &CooccurrenceModel,
    ) {
        // First pass: sample independent conditions
        buffer.clear();
        for &(cond_idx, prob) in &self.conditions {
            if rng.gen::<f32>() < prob {
                buffer.push(cond_idx);
            }
        }

        // Second pass: additive co-occurrence boost. For each sampled trigger,
        // walk its (positive-correlation) dependents and add not-yet-sampled
        // ones with probability `(conditional - base) * 0.5 * dependent_scale`.
        //
        // We tried adding symmetric subtractive moves (`conditional < base`
        // → remove) to capture negative correlations. Result: multi-trigger
        // subtraction stacks and annihilates high-marginal conditions. The
        // right architecture for negatives is a joint sampler (Gibbs/Ising)
        // that resamples conditional on the full vector rather than
        // independently per pair. Documented future work (d5 = `causal-DAG`).
        //
        // Snapshot triggers before iterating — keeps the loop deterministic
        // even if we later restore a subtractive branch. Use a 512-bit
        // `EventBitset` for O(1) membership checks instead of the
        // `buffer.contains` linear scan (was ~25 ops per check × many checks).
        const DAMPING: f32 = 0.5;
        let triggers: SmallVec<[u16; 8]> = buffer.iter().copied().collect();
        let mut present = crate::sampler::EventBitset::default();
        for &c in &triggers {
            present.test_and_set(c);
        }
        // Hoist slice views once so the inner loop has no method dispatch.
        let scales = cooccurrence.dependent_scale();
        let prob_by_cond: &[f32] = &self.prob_by_condition;
        for &trigger_cond in &triggers {
            let dependents = cooccurrence.get_dependents(trigger_cond);
            for &(dependent_idx, conditional_prob) in dependents {
                // Bit test first — early exit before any arithmetic.
                if present.test(dependent_idx) {
                    continue;
                }
                let d = dependent_idx as usize;
                // `prob_by_condition` is sized to `num_conditions` and
                // `dependent_idx` is sourced from the same condition space,
                // so the bounds check is unreachable in practice. Both
                // `prob_by_condition` and `dependent_scale` are dense arrays
                // — direct indexing is the SIMD-friendly access pattern.
                let base_prob = *prob_by_cond.get(d).unwrap_or(&0.0);
                if conditional_prob <= base_prob {
                    continue;
                }
                let scale = *scales.get(d).unwrap_or(&1.0);
                let boost = (conditional_prob - base_prob) * DAMPING * scale;
                if boost > 0.0 && rng.gen::<f32>() < boost {
                    buffer.push(dependent_idx);
                    present.test_and_set(dependent_idx);
                }
            }
        }
    }

    /// Samples a specific age within this archetype's range.
    #[inline]
    pub fn sample_age<R: Rng>(&self, rng: &mut R) -> u32 {
        rng.gen_range(self.age_range.0..=self.age_range.1)
    }

    /// Returns the number of conditions with non-zero probability.
    pub fn active_conditions(&self) -> usize {
        self.conditions.len()
    }
}

/// Registry of patient archetypes for fast generation.
#[derive(Clone)]
pub struct ArchetypeRegistry {
    /// All archetypes, indexed by ID.
    archetypes: Vec<PatientArchetype>,

    /// Alias table for O(1) archetype sampling.
    alias_table: AliasTable,

    /// Number of conditions tracked.
    num_conditions: usize,

    /// Number of medications tracked.
    num_medications: usize,

    /// Number of observations tracked.
    num_observations: usize,

    /// Number of procedures tracked.
    num_procedures: usize,

    /// Condition thresholds for SIMD sampling (flat array for cache efficiency).
    /// Layout: [archetype_0_cond_0, archetype_0_cond_1, ..., archetype_1_cond_0, ...]
    condition_thresholds: Vec<f32>,

    /// Stride for condition_thresholds (= num_conditions padded to 8).
    threshold_stride: usize,

    /// Per-condition onset-age distribution in days since birth.
    /// `onset_mean_days[c]` is the empirical mean and `onset_std_days[c]` the
    /// standard deviation extracted from Java Synthea's conditions.csv (via
    /// `extract_temporal_stats.py` → `onset_stats.json` → MssFingerprint).
    /// Defaults: 40y mean, 10y std when no onset stats are loaded.
    ///
    /// Consumed by `sample_onset_days` to assign per-emitted-condition
    /// timestamps — the d5 `temporal-ordered` axis value's mechanism.
    onset_mean_days: Vec<f32>,
    onset_std_days: Vec<f32>,

    /// Dense per-archetype packed (threshold, idx) view for SIMD sampling.
    ///
    /// `condition_thresholds` above is a 214-wide padded layout in which
    /// most slots are zero (an archetype typically activates ~30/214
    /// conditions). The dense layout below packs only the active conditions
    /// (one f32 threshold + one u16 original-index per active condition,
    /// padded to a multiple of 8 with `threshold = 0.0` sentinels so the
    /// SIMD compare cannot set mask bits for padding).
    ///
    /// This is the hot-path layout used by `SimdSampler::sample_active` —
    /// the inner loop iterates ~4 SIMD chunks per archetype instead of 27.
    active_thresholds_flat: Vec<f32>,
    active_indices_flat: Vec<u16>,
    /// `active_offsets[arch_id]` = start offset into `active_thresholds_flat`
    /// and `active_indices_flat` for archetype `arch_id`.
    active_offsets: Vec<usize>,
    /// `active_padded_lens[arch_id]` = number of slots used (padded to 8) for
    /// archetype `arch_id`. The slice `[offset .. offset+padded_len]` is what
    /// the SIMD compare iterates.
    active_padded_lens: Vec<usize>,

    /// Pre-computed medication thresholds per archetype.
    /// Layout: [archetype_0_med_0, archetype_0_med_1, ..., archetype_1_med_0, ...]
    /// Thresholds are: sum of P(condition) * P(medication | condition) for each medication.
    medication_thresholds: Vec<f32>,

    /// Stride for medication_thresholds (= num_medications padded to 8).
    medication_stride: usize,

    /// Condition -> Medications lookup (indexed by condition_idx).
    /// Each entry contains medication indices and their frequencies.
    condition_to_medications: Vec<SmallVec<[(u16, f32); 4]>>,

    /// Condition -> Procedures lookup (indexed by condition_idx).
    /// Each entry contains procedure indices and their frequencies.
    condition_to_procedures: Vec<SmallVec<[(u16, f32); 4]>>,

    /// Medication -> Conditions inverse lookup (indexed by medication_idx).
    /// Each entry contains the condition indices that can cause this med
    /// and their P(med | cond) frequencies. Used for **REASONCODE linkage**:
    /// when a medication is prescribed to a patient, sample which of the
    /// patient's currently-active conditions caused it (weighted by P(med | cond)
    /// over active conditions).
    medication_to_conditions: Vec<SmallVec<[(u16, f32); 4]>>,

    /// Procedure -> Conditions inverse lookup (indexed by procedure_idx).
    /// Same shape as `medication_to_conditions` but for procedures' REASONCODE.
    procedure_to_conditions: Vec<SmallVec<[(u16, f32); 4]>>,

    /// Medication frequencies (indexed by medication_idx).
    medication_frequencies: Vec<f32>,

    /// Observation frequencies (indexed by observation_idx).
    observation_frequencies: Vec<f32>,

    /// Procedure frequencies (indexed by procedure_idx).
    procedure_frequencies: Vec<f32>,
}

/// Vose's alias method for O(1) weighted sampling.
#[derive(Debug, Clone)]
struct AliasTable {
    prob: Vec<f32>,
    alias: Vec<u16>,
}

impl AliasTable {
    /// Builds an alias table from weights.
    fn new(weights: &[f32]) -> Self {
        let n = weights.len();
        if n == 0 {
            return Self {
                prob: vec![],
                alias: vec![],
            };
        }

        let sum: f32 = weights.iter().sum();
        let scale = n as f32 / sum;

        let mut prob = vec![0.0f32; n];
        let mut alias = vec![0u16; n];

        let mut small: Vec<usize> = Vec::with_capacity(n);
        let mut large: Vec<usize> = Vec::with_capacity(n);

        let mut scaled: Vec<f32> = weights.iter().map(|&w| w * scale).collect();

        for (i, &s) in scaled.iter().enumerate() {
            if s < 1.0 {
                small.push(i);
            } else {
                large.push(i);
            }
        }

        while !small.is_empty() && !large.is_empty() {
            let l = small.pop().unwrap();
            let g = large.pop().unwrap();

            prob[l] = scaled[l];
            alias[l] = g as u16;

            scaled[g] = (scaled[g] + scaled[l]) - 1.0;

            if scaled[g] < 1.0 {
                small.push(g);
            } else {
                large.push(g);
            }
        }

        for &g in &large {
            prob[g] = 1.0;
        }
        for &l in &small {
            prob[l] = 1.0;
        }

        Self { prob, alias }
    }

    /// Samples from the alias table in O(1).
    #[inline]
    fn sample<R: Rng>(&self, rng: &mut R) -> u16 {
        if self.prob.is_empty() {
            return 0;
        }

        let i = rng.gen_range(0..self.prob.len());
        if rng.gen::<f32>() < self.prob[i] {
            i as u16
        } else {
            self.alias[i]
        }
    }
}

impl ArchetypeRegistry {
    /// Builds an archetype registry from an MSS fingerprint.
    pub fn from_fingerprint(fp: &MssFingerprint) -> Self {
        let mut archetypes = Vec::new();
        let num_conditions = fp.conditions.len();
        let num_medications = fp.medications.len();
        let num_observations = fp.observations.len();
        let num_procedures = fp.procedures.len();

        // Build condition code -> index lookup
        let condition_code_to_idx: AHashMap<&str, u16> = fp
            .conditions
            .iter()
            .enumerate()
            .map(|(i, c)| (c.code.as_str(), i as u16))
            .collect();

        // Create one archetype per demographic bucket
        for (bucket, &weight) in &fp.joint_demographics.buckets {
            if weight <= 0.0 {
                continue;
            }

            let age_range = match bucket.age_bucket.as_str() {
                "0-17" => (0, 17),
                "18-44" => (18, 44),
                "45-64" => (45, 64),
                "65+" => (65, 95),
                _ => (18, 44),
            };

            // Compute condition probabilities for this demographic
            let mut conditions: SmallVec<[(u16, f32); 32]> = SmallVec::new();
            for (idx, cond) in fp.conditions.iter().enumerate() {
                let prob = cond.prevalence_for(bucket) as f32;
                if prob > 0.001 {
                    conditions.push((idx as u16, prob));
                }
            }

            // Sort by probability descending for early termination
            conditions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Build the dense O(1) lookup table for the joint sampler.
            let mut prob_by_condition = vec![0.0f32; num_conditions];
            for &(cidx, p) in &conditions {
                if (cidx as usize) < prob_by_condition.len() {
                    prob_by_condition[cidx as usize] = p;
                }
            }

            let archetype = PatientArchetype {
                id: crate::types::ArchetypeId(archetypes.len() as u16),
                demographics: bucket.clone(),
                age_range,
                conditions,
                prob_by_condition,
                weight: weight as f32,
                mean_encounters: fp
                    .encounter_stats
                    .mean_by_age
                    .get(&bucket.age_bucket)
                    .copied()
                    .unwrap_or(10.0) as f32,
                mean_events_per_encounter: fp.encounter_stats.mean_events_per_encounter as f32,
            };

            archetypes.push(archetype);
        }

        // Build alias table
        let weights: Vec<f32> = archetypes.iter().map(|a| a.weight).collect();
        let alias_table = AliasTable::new(&weights);

        // Build flat condition threshold array for SIMD (legacy 214-wide padded layout)
        let threshold_stride = (num_conditions + 7) & !7; // Pad to multiple of 8
        let mut condition_thresholds = vec![0.0f32; archetypes.len() * threshold_stride];

        for (arch_idx, archetype) in archetypes.iter().enumerate() {
            let base = arch_idx * threshold_stride;
            for &(cond_idx, prob) in &archetype.conditions {
                condition_thresholds[base + cond_idx as usize] = prob;
            }
        }

        // Build per-condition onset distributions. Default: 40y ± 10y in days.
        // Override with empirical Java Synthea values when present in fingerprint.
        let mut onset_mean_days: Vec<f32> = vec![14_610.0; num_conditions]; // 40 * 365.25
        let mut onset_std_days: Vec<f32> = vec![3_653.0; num_conditions]; // 10 * 365.25
        if !fp.onset_stats.is_empty() {
            for (code, mean_years, std_years) in &fp.onset_stats {
                if let Some(&idx) = condition_code_to_idx.get(code.as_str()) {
                    if (idx as usize) < num_conditions {
                        onset_mean_days[idx as usize] = (*mean_years as f32) * 365.25;
                        onset_std_days[idx as usize] =
                            ((*std_years as f32) * 365.25).max(30.0); // floor at 1 month
                    }
                }
            }
        }

        // Build the dense active-only layout used by the SIMD hot path.
        // For each archetype, pack its `conditions` list (already filtered to
        // prob > 0.001 and sorted desc) into a multiple-of-8 slab, padding
        // with `threshold = 0.0` sentinels so the SIMD compare never sets
        // mask bits for padding slots.
        let mut active_thresholds_flat: Vec<f32> = Vec::new();
        let mut active_indices_flat: Vec<u16> = Vec::new();
        let mut active_offsets: Vec<usize> = Vec::with_capacity(archetypes.len());
        let mut active_padded_lens: Vec<usize> = Vec::with_capacity(archetypes.len());

        for archetype in archetypes.iter() {
            active_offsets.push(active_thresholds_flat.len());
            let raw_len = archetype.conditions.len();
            let padded = (raw_len + 7) & !7;
            for &(cond_idx, prob) in &archetype.conditions {
                active_thresholds_flat.push(prob);
                active_indices_flat.push(cond_idx);
            }
            for _ in raw_len..padded {
                active_thresholds_flat.push(0.0);
                active_indices_flat.push(0);
            }
            active_padded_lens.push(padded);
        }

        // Build condition -> medications lookup
        let mut condition_to_medications: Vec<SmallVec<[(u16, f32); 4]>> =
            vec![SmallVec::new(); num_conditions];
        for (med_idx, med) in fp.medications.iter().enumerate() {
            for indication in &med.indications {
                if let Some(&cond_idx) = condition_code_to_idx.get(indication.as_str()) {
                    condition_to_medications[cond_idx as usize]
                        .push((med_idx as u16, med.frequency as f32));
                }
            }
        }

        // Build condition -> procedures lookup
        let mut condition_to_procedures: Vec<SmallVec<[(u16, f32); 4]>> =
            vec![SmallVec::new(); num_conditions];
        for (proc_idx, proc) in fp.procedures.iter().enumerate() {
            for indication in &proc.indications {
                if let Some(&cond_idx) = condition_code_to_idx.get(indication.as_str()) {
                    condition_to_procedures[cond_idx as usize]
                        .push((proc_idx as u16, proc.frequency as f32));
                }
            }
        }

        // Inverse lookups for REASONCODE linkage: medication_idx → conditions
        // that can cause it. The weight is `P(reason | medication)` when the
        // fingerprint provides per-indication weights (extracted from Java's
        // empirical `REASONCODE` distribution), otherwise we fall back to
        // `med.frequency` — which gives uniform sampling among active causes,
        // matching the pre-v2 single-cause behaviour but generalised to the
        // multi-cause case.
        let mut medication_to_conditions: Vec<SmallVec<[(u16, f32); 4]>> =
            vec![SmallVec::new(); num_medications];
        for (med_idx, med) in fp.medications.iter().enumerate() {
            let has_weights = med.indication_weights.len() == med.indications.len()
                && !med.indication_weights.is_empty();
            for (i, indication) in med.indications.iter().enumerate() {
                if let Some(&cond_idx) = condition_code_to_idx.get(indication.as_str()) {
                    let w = if has_weights {
                        med.indication_weights[i] as f32
                    } else {
                        med.frequency as f32
                    };
                    medication_to_conditions[med_idx].push((cond_idx, w));
                }
            }
        }
        let mut procedure_to_conditions: Vec<SmallVec<[(u16, f32); 4]>> =
            vec![SmallVec::new(); num_procedures];
        for (proc_idx, proc) in fp.procedures.iter().enumerate() {
            let has_weights = proc.indication_weights.len() == proc.indications.len()
                && !proc.indication_weights.is_empty();
            for (i, indication) in proc.indications.iter().enumerate() {
                if let Some(&cond_idx) = condition_code_to_idx.get(indication.as_str()) {
                    let w = if has_weights {
                        proc.indication_weights[i] as f32
                    } else {
                        proc.frequency as f32
                    };
                    procedure_to_conditions[proc_idx].push((cond_idx, w));
                }
            }
        }

        // Build frequency arrays
        let medication_frequencies: Vec<f32> =
            fp.medications.iter().map(|m| m.frequency as f32).collect();

        // Build observation frequencies with non-zero index list for faster iteration
        let observation_frequencies: Vec<f32> =
            fp.observations.iter().map(|o| o.frequency as f32).collect();

        let procedure_frequencies: Vec<f32> =
            fp.procedures.iter().map(|p| p.frequency as f32).collect();

        // Pre-compute medication thresholds per archetype
        // For each archetype, compute P(medication) = sum over conditions of P(cond) * P(med | cond)
        let medication_stride = (num_medications + 7) & !7;
        let mut medication_thresholds = vec![0.0f32; archetypes.len() * medication_stride];

        for (arch_idx, archetype) in archetypes.iter().enumerate() {
            let base = arch_idx * medication_stride;

            // For each condition in this archetype, add medication probabilities
            for &(cond_idx, cond_prob) in &archetype.conditions {
                if let Some(meds) = condition_to_medications.get(cond_idx as usize) {
                    for &(med_idx, med_freq) in meds {
                        // P(med) += P(cond) * P(med | cond)
                        // Cap at 1.0 to avoid over-counting
                        let idx = base + med_idx as usize;
                        medication_thresholds[idx] =
                            (medication_thresholds[idx] + cond_prob * med_freq).min(1.0);
                    }
                }
            }
        }

        Self {
            archetypes,
            alias_table,
            num_conditions,
            num_medications,
            num_observations,
            num_procedures,
            condition_thresholds,
            threshold_stride,
            onset_mean_days,
            onset_std_days,
            active_thresholds_flat,
            active_indices_flat,
            active_offsets,
            active_padded_lens,
            medication_thresholds,
            medication_stride,
            condition_to_medications,
            condition_to_procedures,
            medication_to_conditions,
            procedure_to_conditions,
            medication_frequencies,
            observation_frequencies,
            procedure_frequencies,
        }
    }

    /// Dense per-archetype active-condition view for the SIMD hot path.
    /// Returns `(thresholds, original_indices)` slices of equal length, padded
    /// to a multiple of 8 with `0.0` thresholds so the SIMD compare cannot
    /// set mask bits for padding slots.
    #[inline]
    pub fn active_view(&self, archetype_id: crate::types::ArchetypeId) -> (&[f32], &[u16]) {
        let base = self.active_offsets[archetype_id.as_index()];
        let len = self.active_padded_lens[archetype_id.as_index()];
        (
            &self.active_thresholds_flat[base..base + len],
            &self.active_indices_flat[base..base + len],
        )
    }

    /// REASONCODE inverse: returns the conditions that can cause this
    /// medication and their `P(med | cond)` weights. Used by event sampling
    /// to link each prescribed medication to a triggering condition (the
    /// equivalent of Java Synthea's `medications.csv:REASONCODE` column).
    #[inline]
    pub fn medication_to_conditions(&self, med_idx: u16) -> &[(u16, f32)] {
        self.medication_to_conditions
            .get(med_idx as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// REASONCODE inverse for procedures (matches the Java Synthea
    /// `procedures.csv:REASONCODE` column).
    #[inline]
    pub fn procedure_to_conditions(&self, proc_idx: u16) -> &[(u16, f32)] {
        self.procedure_to_conditions
            .get(proc_idx as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Sample which condition caused a particular medication for a patient
    /// with the given active condition set, weighted by `P(med | cond)`
    /// over the intersection of `active_conditions` and the medication's
    /// indication list. Returns `u16::MAX` as a sentinel when no active
    /// condition is an indication for this medication (matches Java's
    /// empty-REASONCODE rows for prophylactic or routine prescriptions).
    #[inline]
    pub fn sample_medication_cause<R: Rng>(
        &self,
        med_idx: u16,
        active_conditions: &[u16],
        rng: &mut R,
    ) -> u16 {
        let candidates = self.medication_to_conditions(med_idx);
        let mut total = 0.0f32;
        for &(cond_idx, freq) in candidates {
            if active_conditions.contains(&cond_idx) {
                total += freq;
            }
        }
        if total <= 0.0 {
            return u16::MAX;
        }
        let mut pick = rng.gen::<f32>() * total;
        for &(cond_idx, freq) in candidates {
            if !active_conditions.contains(&cond_idx) {
                continue;
            }
            pick -= freq;
            if pick <= 0.0 {
                return cond_idx;
            }
        }
        // Fallback: return the last-considered candidate (shouldn't usually fire)
        candidates
            .iter()
            .rev()
            .find(|(c, _)| active_conditions.contains(c))
            .map(|(c, _)| *c)
            .unwrap_or(u16::MAX)
    }

    /// REASONCODE-for-procedure equivalent of `sample_medication_cause`.
    #[inline]
    pub fn sample_procedure_cause<R: Rng>(
        &self,
        proc_idx: u16,
        active_conditions: &[u16],
        rng: &mut R,
    ) -> u16 {
        let candidates = self.procedure_to_conditions(proc_idx);
        let mut total = 0.0f32;
        for &(cond_idx, freq) in candidates {
            if active_conditions.contains(&cond_idx) {
                total += freq;
            }
        }
        if total <= 0.0 {
            return u16::MAX;
        }
        let mut pick = rng.gen::<f32>() * total;
        for &(cond_idx, freq) in candidates {
            if !active_conditions.contains(&cond_idx) {
                continue;
            }
            pick -= freq;
            if pick <= 0.0 {
                return cond_idx;
            }
        }
        candidates
            .iter()
            .rev()
            .find(|(c, _)| active_conditions.contains(c))
            .map(|(c, _)| *c)
            .unwrap_or(u16::MAX)
    }

    /// Sample an onset age (days since birth) for the given condition.
    /// Truncated normal: clipped to `[0, max_age_days]` so the timestamp
    /// can't precede birth or exceed the patient's current age.
    ///
    /// Uses a 12-sum Irwin-Hall approximation to N(0, 1) — cheap (12 RNG
    /// draws) and approximate but indistinguishable from Java's empirical
    /// distribution at the population scale we generate.
    #[inline(always)]
    pub fn sample_onset_days<R: Rng>(
        &self,
        cond_idx: u16,
        max_age_days: u32,
        rng: &mut R,
    ) -> u16 {
        let i = cond_idx as usize;
        let mean = self.onset_mean_days.get(i).copied().unwrap_or(14_610.0);
        let std = self.onset_std_days.get(i).copied().unwrap_or(3_653.0);

        // Irwin-Hall(12): sum of 12 uniform [0,1] minus 6 ≈ N(0,1).
        let mut z: f32 = 0.0;
        for _ in 0..12 {
            z += rng.gen::<f32>();
        }
        z -= 6.0;

        let days = (mean + z * std).clamp(0.0, max_age_days as f32);
        days as u16
    }

    /// Apply a per-condition multiplicative scale to every threshold layout the
    /// registry holds. Used by the marginal-recalibration loop (closes the F4
    /// gate gap when the d5 `pairwise-empirical` mode is active — the joint
    /// boost adds dependents on top of the independent draws, inflating their
    /// marginals; this method scales every per-archetype prevalence to
    /// counter-balance until the aggregate marginal matches the calibrated
    /// target).
    ///
    /// `multipliers` is indexed by condition (length = num_conditions).
    ///
    /// **No-clamp during sweep** — successive in-loop application clamps
    /// non-linearly relative to the persisted multiplier product (which the
    /// auto-loader uses to reproduce calibrated state). To make `cargo test
    /// e1_recalibrate` round-trip exactly through the persisted JSON, we
    /// store unclamped intermediate state and clamp only on the final
    /// readback. The active-view and padded layouts always emit the
    /// clamped-to-[0, 1] view since the SIMD compare relies on that domain.
    pub fn scale_per_condition_prevalence(&mut self, multipliers: &[f32]) {
        assert_eq!(
            multipliers.len(),
            self.num_conditions,
            "multipliers length must equal num_conditions"
        );

        // 1. Update each archetype's packed `conditions` list. Do NOT clamp;
        // the SIMD threshold compare (rand < threshold) already returns
        // identical results for any threshold ≥ 1.0 (rand ∈ [0, 1) is always
        // below it), and for threshold ≤ 0.0 the compare can't fire, so the
        // unclamped intermediate state behaves identically to the clamped
        // view for sampling purposes — while letting subsequent multipliers
        // recover from overshoots that clamping would freeze.
        for arch in self.archetypes.iter_mut() {
            for (cond_idx, prob) in arch.conditions.iter_mut() {
                let m = multipliers[*cond_idx as usize];
                *prob = (*prob * m).max(0.0);
            }
        }

        // 2. Update the legacy 214-wide padded layout. Same logic.
        for arch_idx in 0..self.archetypes.len() {
            let base = arch_idx * self.threshold_stride;
            for cond_idx in 0..self.num_conditions {
                let m = multipliers[cond_idx];
                let p = self.condition_thresholds[base + cond_idx];
                self.condition_thresholds[base + cond_idx] = (p * m).max(0.0);
            }
        }

        // 3. Update the dense `active_thresholds_flat` similarly.
        for i in 0..self.active_thresholds_flat.len() {
            let cond_idx = self.active_indices_flat[i] as usize;
            // Padding slots have threshold == 0.0; leave them alone so the
            // SIMD compare doesn't accidentally fire on padding.
            if self.active_thresholds_flat[i] == 0.0 {
                continue;
            }
            let m = multipliers[cond_idx];
            self.active_thresholds_flat[i] =
                (self.active_thresholds_flat[i] * m).max(0.0);
        }

        // 4. Re-populate `prob_by_condition` from the (possibly-clamped)
        // per-archetype conditions list so downstream readers (Gibbs sampler,
        // boost lookups) see the calibrated values. The conditions list IS
        // the source of truth here; the parallel dense table is a cache.
        for arch in self.archetypes.iter_mut() {
            arch.prob_by_condition.fill(0.0);
            for &(cond_idx, prob) in &arch.conditions {
                if (cond_idx as usize) < arch.prob_by_condition.len() {
                    arch.prob_by_condition[cond_idx as usize] = prob.clamp(0.0, 1.0);
                }
            }
        }
    }

    /// Samples an archetype in O(1).
    #[inline(always)]
    pub fn sample<R: Rng>(&self, rng: &mut R) -> &PatientArchetype {
        let idx = self.alias_table.sample(rng);
        &self.archetypes[idx as usize]
    }

    /// Gets an archetype by ID.
    #[inline]
    pub fn get(&self, id: crate::types::ArchetypeId) -> Option<&PatientArchetype> {
        self.archetypes.get(id.as_index())
    }

    /// Returns the number of archetypes.
    pub fn len(&self) -> usize {
        self.archetypes.len()
    }

    /// Returns whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.archetypes.is_empty()
    }

    /// Returns the number of conditions tracked.
    pub fn num_conditions(&self) -> usize {
        self.num_conditions
    }

    /// Returns the condition thresholds for SIMD sampling.
    #[inline]
    pub fn condition_thresholds(&self, archetype_id: crate::types::ArchetypeId) -> &[f32] {
        let base = archetype_id.as_index() * self.threshold_stride;
        &self.condition_thresholds[base..base + self.num_conditions]
    }

    /// Returns the threshold stride.
    pub fn threshold_stride(&self) -> usize {
        self.threshold_stride
    }

    /// Returns the pre-computed medication thresholds for an archetype.
    /// These are P(medication) = sum of P(cond) * P(med | cond) for SIMD sampling.
    #[inline]
    pub fn medication_thresholds(&self, archetype_id: crate::types::ArchetypeId) -> &[f32] {
        let base = archetype_id.as_index() * self.medication_stride;
        &self.medication_thresholds[base..base + self.num_medications]
    }

    /// Returns the medication stride.
    pub fn medication_stride(&self) -> usize {
        self.medication_stride
    }

    /// Returns all archetypes.
    pub fn archetypes(&self) -> &[PatientArchetype] {
        &self.archetypes
    }

    /// Samples conditions using the flat threshold array (SIMD-friendly).
    #[inline]
    pub fn sample_conditions_flat<R: Rng>(
        &self,
        archetype_id: crate::types::ArchetypeId,
        rng: &mut R,
        buffer: &mut SmallVec<[u16; 8]>,
    ) {
        buffer.clear();
        let thresholds = self.condition_thresholds(archetype_id);

        for (idx, &threshold) in thresholds.iter().enumerate() {
            if threshold > 0.0 && rng.gen::<f32>() < threshold {
                buffer.push(idx as u16);
            }
        }
    }

    /// Returns medications associated with a condition.
    #[inline]
    pub fn medications_for_condition(&self, condition_idx: u16) -> &[(u16, f32)] {
        self.condition_to_medications
            .get(condition_idx as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Returns procedures associated with a condition.
    #[inline]
    pub fn procedures_for_condition(&self, condition_idx: u16) -> &[(u16, f32)] {
        self.condition_to_procedures
            .get(condition_idx as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Returns the number of medications.
    pub fn num_medications(&self) -> usize {
        self.num_medications
    }

    /// Returns the number of observations.
    pub fn num_observations(&self) -> usize {
        self.num_observations
    }

    /// Returns the number of procedures.
    pub fn num_procedures(&self) -> usize {
        self.num_procedures
    }

    /// Returns medication frequencies.
    pub fn medication_frequencies(&self) -> &[f32] {
        &self.medication_frequencies
    }

    /// Returns observation frequencies.
    pub fn observation_frequencies(&self) -> &[f32] {
        &self.observation_frequencies
    }

    /// Returns procedure frequencies.
    pub fn procedure_frequencies(&self) -> &[f32] {
        &self.procedure_frequencies
    }
}

/// Co-occurrence model for conditional sampling of related conditions.
#[derive(Debug, Clone, Default)]
pub struct CooccurrenceModel {
    /// Trigger-indexed direct lookup of dependents. Index is the trigger's
    /// condition index (`u16`). Empty `SmallVec` means "no dependents".
    /// Avoids the per-trigger hash lookup that an `AHashMap<u16, _>` would
    /// incur on the hot path. Sized to `num_conditions` after build.
    dependents_by_trigger: Vec<SmallVec<[(u16, f32); 4]>>,

    /// Per-dependent boost-scale knob set by the recalibration loop.
    /// `boost_contribution_to(c) = (conditional - base) * 0.5 * dependent_scale[c]`.
    /// Default `1.0` (no scaling).
    ///
    /// The recalibration loop adjusts these *and* the per-archetype base
    /// prevalences together so the joint sampler converges to the calibrated
    /// marginal targets — a two-knob fit that the additive-only sampler
    /// needs to close the marginal-vs-joint gap.
    dependent_scale: Vec<f32>,
}

impl CooccurrenceModel {
    /// Creates a new empty co-occurrence model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a co-occurrence model from a fingerprint.
    pub fn from_fingerprint(fp: &MssFingerprint) -> Self {
        let mut model = Self::new();
        let n = fp.conditions.len();
        model.dependent_scale = vec![1.0; n];
        model.dependents_by_trigger = vec![SmallVec::new(); n];

        // Build code -> index lookup
        let code_to_idx: AHashMap<&str, u16> = fp
            .conditions
            .iter()
            .enumerate()
            .map(|(i, c)| (c.code.as_str(), i as u16))
            .collect();

        // Populate from fingerprint cooccurrence
        for ((trigger_code, dependent_code), &prob) in &fp.cooccurrence {
            if let (Some(&trigger_idx), Some(&dependent_idx)) = (
                code_to_idx.get(trigger_code.as_str()),
                code_to_idx.get(dependent_code.as_str()),
            ) {
                model.add_dependency(trigger_idx, dependent_idx, prob as f32);
            }
        }

        // Apply recalibration boost multipliers if the fingerprint carries
        // them. This is the second knob of the two-knob fit that makes joint
        // mode converge to the calibrated marginal targets without losing the
        // joint-correlation signal.
        for (code, &mult) in &fp.cooccurrence_dependent_scale {
            if let Some(&idx) = code_to_idx.get(code.as_str()) {
                if (idx as usize) < model.dependent_scale.len() {
                    model.dependent_scale[idx as usize] = mult as f32;
                }
            }
        }

        model
    }

    /// Multiplies each entry of `dependent_scale` by the supplied factors.
    /// Lengths must match. Each scale is clamped into `[0.0, 4.0]` after
    /// multiplication. Used by the recalibration loop to adjust per-dependent
    /// boost magnitude.
    pub fn scale_dependent_boosts(&mut self, factors: &[f32]) {
        if self.dependent_scale.len() != factors.len() {
            // Resize to match (in case the model was freshly built without
            // a known num_conditions).
            self.dependent_scale.resize(factors.len(), 1.0);
        }
        for (s, f) in self.dependent_scale.iter_mut().zip(factors.iter()) {
            *s = (*s * *f).clamp(0.0, 4.0);
        }
    }

    /// Returns the current dependent-boost scale vector. `1.0` per entry by
    /// default.
    pub fn dependent_scale(&self) -> &[f32] {
        &self.dependent_scale
    }

    /// Adds a dependency relationship.
    pub fn add_dependency(&mut self, trigger: u16, dependent: u16, conditional_prob: f32) {
        let t = trigger as usize;
        if t >= self.dependents_by_trigger.len() {
            self.dependents_by_trigger.resize(t + 1, SmallVec::new());
        }
        self.dependents_by_trigger[t].push((dependent, conditional_prob));
    }

    /// Gets the dependents for a trigger condition. Empty slice when the
    /// trigger has no recorded dependents.
    #[inline(always)]
    pub fn get_dependents(&self, trigger: u16) -> &[(u16, f32)] {
        let t = trigger as usize;
        if t < self.dependents_by_trigger.len() {
            &self.dependents_by_trigger[t]
        } else {
            &[]
        }
    }

    /// Returns the number of trigger conditions with at least one dependent.
    pub fn len(&self) -> usize {
        self.dependents_by_trigger
            .iter()
            .filter(|v| !v.is_empty())
            .count()
    }

    /// Returns whether the model is empty.
    pub fn is_empty(&self) -> bool {
        self.dependents_by_trigger.iter().all(|v| v.is_empty())
    }

    /// Returns total number of dependency relationships.
    pub fn total_dependencies(&self) -> usize {
        self.dependents_by_trigger.iter().map(|v| v.len()).sum()
    }
}

/// Builder for creating archetypes with custom configurations.
pub struct ArchetypeBuilder {
    demographics: DemographicBucket,
    conditions: SmallVec<[(u16, f32); 32]>,
    weight: f32,
    mean_encounters: f32,
    mean_events_per_encounter: f32,
}

impl ArchetypeBuilder {
    /// Creates a new builder.
    pub fn new(demographics: DemographicBucket) -> Self {
        Self {
            demographics,
            conditions: SmallVec::new(),
            weight: 1.0,
            mean_encounters: 10.0,
            mean_events_per_encounter: 5.0,
        }
    }

    /// Adds a condition with its probability.
    pub fn with_condition(mut self, condition_idx: u16, probability: f32) -> Self {
        self.conditions.push((condition_idx, probability));
        self
    }

    /// Sets the weight for archetype sampling.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Sets the mean number of encounters.
    pub fn with_mean_encounters(mut self, mean: f32) -> Self {
        self.mean_encounters = mean;
        self
    }

    /// Builds the archetype.
    pub fn build(mut self, id: u16) -> PatientArchetype {
        let id = crate::types::ArchetypeId(id);
        self.conditions
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let age_range = match self.demographics.age_bucket.as_str() {
            "0-17" => (0, 17),
            "18-44" => (18, 44),
            "45-64" => (45, 64),
            "65+" => (65, 95),
            _ => (18, 44),
        };

        // Build the dense O(1) lookup table. Test fixtures use small
        // condition spaces (often single-digit `cond_idx` values); pick a
        // size at least big enough to cover the largest cond_idx the
        // builder has seen.
        let max_idx = self.conditions.iter().map(|(c, _)| *c as usize).max();
        let size = max_idx.map(|m| m + 1).unwrap_or(0);
        let mut prob_by_condition = vec![0.0f32; size];
        for &(cidx, p) in &self.conditions {
            if (cidx as usize) < prob_by_condition.len() {
                prob_by_condition[cidx as usize] = p;
            }
        }

        PatientArchetype {
            id,
            demographics: self.demographics,
            age_range,
            conditions: self.conditions,
            prob_by_condition,
            weight: self.weight,
            mean_encounters: self.mean_encounters,
            mean_events_per_encounter: self.mean_events_per_encounter,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_alias_table() {
        let weights = vec![0.5, 0.3, 0.2];
        let table = AliasTable::new(&weights);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut counts = [0u32; 3];

        for _ in 0..10000 {
            let idx = table.sample(&mut rng);
            counts[idx as usize] += 1;
        }

        // Should be roughly proportional to weights
        assert!(counts[0] > counts[1]);
        assert!(counts[1] > counts[2]);
    }

    #[test]
    fn test_archetype_builder() {
        let demo = DemographicBucket::new("45-64", "male", "white", "nonhispanic");
        let archetype = ArchetypeBuilder::new(demo)
            .with_condition(0, 0.3)
            .with_condition(1, 0.1)
            .with_weight(0.5)
            .build(0);

        assert_eq!(archetype.id, crate::types::ArchetypeId(0));
        assert_eq!(archetype.age_range, (45, 64));
        assert_eq!(archetype.conditions.len(), 2);
        // Sorted by probability descending
        assert_eq!(archetype.conditions[0].0, 0);
        assert_eq!(archetype.conditions[1].0, 1);
    }

    #[test]
    fn test_archetype_condition_sampling() {
        let demo = DemographicBucket::new("45-64", "male", "white", "nonhispanic");
        let archetype = ArchetypeBuilder::new(demo)
            .with_condition(0, 1.0) // Always
            .with_condition(1, 0.0) // Never
            .with_condition(2, 0.5) // Sometimes
            .build(0);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut buffer = SmallVec::new();

        archetype.sample_conditions(&mut rng, &mut buffer);

        // Condition 0 should always be present
        assert!(buffer.contains(&0));
        // Condition 1 should never be present
        assert!(!buffer.contains(&1));
    }

    /// Boundary test: build a fingerprint whose joint_demographics yields one
    /// well-populated archetype plus one archetype whose conditions are
    /// universally below the 0.001 prevalence floor (i.e. zero active
    /// conditions). Confirm that the dense `active_view` layout returns
    /// empty-but-valid slices for the zero-active archetype, that
    /// `sample_active` produces an empty output without panicking, and that
    /// the adjacent populated archetype's slab is not corrupted.
    ///
    /// Skeptic Wave 2 surfaced this as a structural risk: the old 214-wide
    /// padded layout had each archetype occupy a fixed 214-slot block, so
    /// off-by-one offsets were impossible. The dense layout uses variable-
    /// sized slabs with per-archetype `active_offsets` + `active_padded_lens`
    /// — a boundary miscalculation could silently cross-contaminate adjacent
    /// archetype thresholds. This test pins the boundary.
    #[test]
    fn test_active_view_zero_active_archetype() {
        use crate::fingerprint::{
            ConditionStats, DemographicBucket as FpDemo, EncounterStats, JointDemographics,
            MssFingerprint,
        };
        use std::collections::BTreeMap;
        use crate::sampler::SimdSampler;

        // Two demographic buckets: a populated one and a "ghost" one.
        let populated = FpDemo::new("45-64", "male", "white", "nonhispanic");
        let ghost = FpDemo::new("18-44", "female", "asian", "hispanic");

        let mut buckets: BTreeMap<FpDemo, f64> = BTreeMap::new();
        buckets.insert(populated.clone(), 0.7);
        buckets.insert(ghost.clone(), 0.3);
        let joint_demographics = JointDemographics {
            buckets,
            total_patients: 1000,
        };

        // Three conditions. Each has a `by_gender` multiplier of 0.0 for the
        // ghost's gender ("female") and 1.0 for the populated bucket's gender
        // ("male"). `ConditionStats::prevalence_for` multiplies the base
        // prevalence by these multipliers, so the ghost archetype ends up with
        // prevalence 0 for every condition — filtered out by the
        // `if prob > 0.001` gate at line ~290 in `from_fingerprint`.
        let mut zero_for_female: BTreeMap<String, f64> = BTreeMap::new();
        zero_for_female.insert("male".to_string(), 1.0);
        zero_for_female.insert("female".to_string(), 0.0);

        let mk_cond = |code: &str, display: &str, prevalence: f64| ConditionStats {
            code: code.to_string(),
            display: display.to_string(),
            prevalence,
            by_age_bucket: BTreeMap::new(),
            by_gender: zero_for_female.clone(),
            by_race: BTreeMap::new(),
            chronic: true,
            mean_onset_age: 40.0,
        };
        let conditions = vec![
            mk_cond("COND_A", "A", 0.40),
            mk_cond("COND_B", "B", 0.25),
            mk_cond("COND_C", "C", 0.10),
        ];

        let fp = MssFingerprint {
            version: "1.0".to_string(),
            source: "test-zero-active".to_string(),
            total_patients: 1000,
            total_encounters: 5000,
            joint_demographics,
            conditions,
            medications: vec![],
            observations: vec![],
            procedures: vec![],
            cooccurrence: BTreeMap::new(),
            cooccurrence_dependent_scale: BTreeMap::new(),
            onset_stats: Vec::new(),
            encounter_stats: EncounterStats::default(),
        };

        let registry = ArchetypeRegistry::from_fingerprint(&fp);
        assert_eq!(registry.len(), 2, "expected two archetypes");

        // Identify the populated vs ghost archetype by inspecting the
        // condition list each one carries (populated has 3, ghost has 0).
        let (pop_id, ghost_id) = if registry.archetypes[0].conditions.is_empty() {
            (
                crate::types::ArchetypeId(1),
                crate::types::ArchetypeId(0),
            )
        } else {
            (
                crate::types::ArchetypeId(0),
                crate::types::ArchetypeId(1),
            )
        };
        assert!(
            registry.archetypes[ghost_id.as_index()].conditions.is_empty(),
            "ghost archetype should have zero active conditions"
        );
        assert_eq!(
            registry.archetypes[pop_id.as_index()].conditions.len(),
            3,
            "populated archetype should keep all three conditions"
        );

        // active_view on the ghost: empty slices, no panic.
        let (ghost_thr, ghost_idx) = registry.active_view(ghost_id);
        assert_eq!(ghost_thr.len(), 0);
        assert_eq!(ghost_idx.len(), 0);

        // active_view on the populated: 3 actives padded to 8.
        let (pop_thr, pop_idx) = registry.active_view(pop_id);
        assert_eq!(pop_thr.len(), 8, "padded to multiple of 8");
        assert_eq!(pop_idx.len(), 8);
        // First three slots are the real thresholds (sorted desc by prevalence).
        assert!(pop_thr[0] > 0.0);
        assert!(pop_thr[1] > 0.0);
        assert!(pop_thr[2] > 0.0);
        // Padding slots are sentinel zeros.
        for k in 3..8 {
            assert_eq!(pop_thr[k], 0.0, "padding slot {} must be 0.0 sentinel", k);
        }

        // Sample on the ghost: empty output, no panic.
        let mut sampler = SimdSampler::from_registry(&registry);
        let mut buf: SmallVec<[u16; 8]> = SmallVec::new();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        sampler.sample_active(ghost_thr, ghost_idx, &mut rng, &mut buf);
        assert_eq!(buf.len(), 0, "ghost archetype must sample zero conditions");

        // Sample on the populated: produces some conditions from the active set.
        for seed in 0..10u64 {
            buf.clear();
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
            sampler.sample_active(pop_thr, pop_idx, &mut rng, &mut buf);
            for &emitted in buf.iter() {
                // Every emitted index must be one of the populated archetype's
                // original condition indices — never a ghost index, never a
                // bogus sentinel index, never an out-of-range value.
                assert!(
                    pop_idx[..3].contains(&emitted),
                    "sample_active leaked index {emitted}; expected one of {:?}",
                    &pop_idx[..3]
                );
            }
        }
    }
}
