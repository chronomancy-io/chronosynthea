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
const MAX_CONDITIONS: usize = 256;

/// A patient archetype with pre-computed condition probabilities.
#[derive(Debug, Clone)]
pub struct PatientArchetype {
    /// Archetype ID (for deterministic generation).
    pub id: u16,

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

        // Second pass: apply co-occurrence boosts
        // Check sampled conditions and potentially add co-occurring ones
        // We only boost conditions that were NOT already sampled
        let initial_len = buffer.len();
        for i in 0..initial_len {
            let trigger_cond = buffer[i];
            if let Some(dependents) = cooccurrence.get_dependents(trigger_cond) {
                for &(dependent_idx, conditional_prob) in dependents {
                    // Skip if already in buffer
                    if buffer.contains(&dependent_idx) {
                        continue;
                    }

                    // Find the base probability for this condition
                    let base_prob = self
                        .conditions
                        .iter()
                        .find(|(idx, _)| *idx == dependent_idx)
                        .map(|(_, p)| *p)
                        .unwrap_or(0.0);

                    // If base prob is already high, less room for boost
                    // Calculate the additional probability to add
                    // boost = conditional_prob - base_prob, clamped to positive
                    let remaining = 1.0 - base_prob;
                    if remaining <= 0.0 {
                        continue;
                    }

                    // Scale the conditional prob by how much room is left
                    // This prevents double-counting
                    let boost = (conditional_prob - base_prob).max(0.0) * 0.5;

                    if rng.gen::<f32>() < boost {
                        buffer.push(dependent_idx);
                    }
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

            let archetype = PatientArchetype {
                id: archetypes.len() as u16,
                demographics: bucket.clone(),
                age_range,
                conditions,
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

        // Build flat condition threshold array for SIMD
        let threshold_stride = (num_conditions + 7) & !7; // Pad to multiple of 8
        let mut condition_thresholds = vec![0.0f32; archetypes.len() * threshold_stride];

        for (arch_idx, archetype) in archetypes.iter().enumerate() {
            let base = arch_idx * threshold_stride;
            for &(cond_idx, prob) in &archetype.conditions {
                condition_thresholds[base + cond_idx as usize] = prob;
            }
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
            medication_thresholds,
            medication_stride,
            condition_to_medications,
            condition_to_procedures,
            medication_frequencies,
            observation_frequencies,
            procedure_frequencies,
        }
    }

    /// Samples an archetype in O(1).
    #[inline]
    pub fn sample<R: Rng>(&self, rng: &mut R) -> &PatientArchetype {
        let idx = self.alias_table.sample(rng);
        &self.archetypes[idx as usize]
    }

    /// Gets an archetype by ID.
    #[inline]
    pub fn get(&self, id: u16) -> Option<&PatientArchetype> {
        self.archetypes.get(id as usize)
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
    pub fn condition_thresholds(&self, archetype_id: u16) -> &[f32] {
        let base = archetype_id as usize * self.threshold_stride;
        &self.condition_thresholds[base..base + self.num_conditions]
    }

    /// Returns the threshold stride.
    pub fn threshold_stride(&self) -> usize {
        self.threshold_stride
    }

    /// Returns the pre-computed medication thresholds for an archetype.
    /// These are P(medication) = sum of P(cond) * P(med | cond) for SIMD sampling.
    #[inline]
    pub fn medication_thresholds(&self, archetype_id: u16) -> &[f32] {
        let base = archetype_id as usize * self.medication_stride;
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
        archetype_id: u16,
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
    /// Maps condition index -> list of (dependent_condition_idx, conditional_probability).
    /// P(dependent | trigger) = conditional_probability
    dependents: AHashMap<u16, SmallVec<[(u16, f32); 4]>>,
}

impl CooccurrenceModel {
    /// Creates a new empty co-occurrence model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a co-occurrence model from a fingerprint.
    pub fn from_fingerprint(fp: &MssFingerprint) -> Self {
        let mut model = Self::new();

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

        model
    }

    /// Adds a dependency relationship.
    pub fn add_dependency(&mut self, trigger: u16, dependent: u16, conditional_prob: f32) {
        self.dependents
            .entry(trigger)
            .or_default()
            .push((dependent, conditional_prob));
    }

    /// Gets the dependents for a trigger condition.
    #[inline]
    pub fn get_dependents(&self, trigger: u16) -> Option<&SmallVec<[(u16, f32); 4]>> {
        self.dependents.get(&trigger)
    }

    /// Returns the number of trigger conditions with dependencies.
    pub fn len(&self) -> usize {
        self.dependents.len()
    }

    /// Returns whether the model is empty.
    pub fn is_empty(&self) -> bool {
        self.dependents.is_empty()
    }

    /// Returns total number of dependency relationships.
    pub fn total_dependencies(&self) -> usize {
        self.dependents.values().map(|v| v.len()).sum()
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
        self.conditions
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let age_range = match self.demographics.age_bucket.as_str() {
            "0-17" => (0, 17),
            "18-44" => (18, 44),
            "45-64" => (45, 64),
            "65+" => (65, 95),
            _ => (18, 44),
        };

        PatientArchetype {
            id,
            demographics: self.demographics,
            age_range,
            conditions: self.conditions,
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

        assert_eq!(archetype.id, 0);
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
}
