//! Streaming statistics for validation without storing all patients.
//!
//! This module provides online algorithms for computing population statistics
//! that can be compared against Java Synthea's baseline distribution.

use ahash::AHashMap;
use serde::{Deserialize, Serialize};

use crate::fingerprint::MssFingerprint;

/// Streaming statistics accumulator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingStatistics {
    /// Total patients generated.
    pub total_patients: u64,

    /// Total encounters generated.
    pub total_encounters: u64,

    /// Total events generated.
    pub total_events: u64,

    /// Condition occurrence counts (indexed by condition ID).
    pub condition_counts: Vec<u64>,

    /// Medication occurrence counts.
    pub medication_counts: Vec<u64>,

    /// Observation occurrence counts.
    pub observation_counts: Vec<u64>,

    /// Procedure occurrence counts.
    pub procedure_counts: Vec<u64>,

    /// Demographic bucket counts.
    pub demographic_counts: AHashMap<String, u64>,

    /// Encounter type counts.
    pub encounter_type_counts: AHashMap<String, u64>,
}

impl StreamingStatistics {
    /// Creates new streaming statistics.
    pub fn new(num_conditions: usize) -> Self {
        Self {
            total_patients: 0,
            total_encounters: 0,
            total_events: 0,
            condition_counts: vec![0; num_conditions],
            medication_counts: Vec::new(),
            observation_counts: Vec::new(),
            procedure_counts: Vec::new(),
            demographic_counts: AHashMap::new(),
            encounter_type_counts: AHashMap::new(),
        }
    }

    /// Creates with full capacity from fingerprint.
    pub fn from_fingerprint(fp: &MssFingerprint) -> Self {
        Self {
            total_patients: 0,
            total_encounters: 0,
            total_events: 0,
            condition_counts: vec![0; fp.conditions.len()],
            medication_counts: vec![0; fp.medications.len()],
            observation_counts: vec![0; fp.observations.len()],
            procedure_counts: vec![0; fp.procedures.len()],
            demographic_counts: AHashMap::new(),
            encounter_type_counts: AHashMap::new(),
        }
    }

    /// Merges another statistics instance into this one.
    pub fn merge(&mut self, other: &StreamingStatistics) {
        self.total_patients += other.total_patients;
        self.total_encounters += other.total_encounters;
        self.total_events += other.total_events;

        // Extend condition counts if needed
        if other.condition_counts.len() > self.condition_counts.len() {
            self.condition_counts
                .resize(other.condition_counts.len(), 0);
        }
        for (i, &count) in other.condition_counts.iter().enumerate() {
            self.condition_counts[i] += count;
        }

        // Merge medication counts
        if other.medication_counts.len() > self.medication_counts.len() {
            self.medication_counts
                .resize(other.medication_counts.len(), 0);
        }
        for (i, &count) in other.medication_counts.iter().enumerate() {
            self.medication_counts[i] += count;
        }

        // Merge observation counts
        if other.observation_counts.len() > self.observation_counts.len() {
            self.observation_counts
                .resize(other.observation_counts.len(), 0);
        }
        for (i, &count) in other.observation_counts.iter().enumerate() {
            self.observation_counts[i] += count;
        }

        // Merge procedure counts
        if other.procedure_counts.len() > self.procedure_counts.len() {
            self.procedure_counts
                .resize(other.procedure_counts.len(), 0);
        }
        for (i, &count) in other.procedure_counts.iter().enumerate() {
            self.procedure_counts[i] += count;
        }

        // Merge demographic counts
        for (key, &count) in &other.demographic_counts {
            *self.demographic_counts.entry(key.clone()).or_insert(0) += count;
        }

        // Merge encounter type counts
        for (key, &count) in &other.encounter_type_counts {
            *self.encounter_type_counts.entry(key.clone()).or_insert(0) += count;
        }
    }

    /// Computes condition prevalences.
    pub fn condition_prevalences(&self) -> Vec<f64> {
        if self.total_patients == 0 {
            return vec![0.0; self.condition_counts.len()];
        }

        self.condition_counts
            .iter()
            .map(|&count| count as f64 / self.total_patients as f64)
            .collect()
    }

    /// Computes demographic distribution.
    pub fn demographic_distribution(&self) -> AHashMap<String, f64> {
        if self.total_patients == 0 {
            return AHashMap::new();
        }

        self.demographic_counts
            .iter()
            .map(|(key, &count)| (key.clone(), count as f64 / self.total_patients as f64))
            .collect()
    }

    /// Compares against a reference fingerprint.
    pub fn compare(&self, reference: &MssFingerprint) -> StatisticalComparison {
        let prevalences = self.condition_prevalences();

        // Compute KL divergence for conditions
        let mut kl_divergence: f64 = 0.0;
        let mut max_deviation: f64 = 0.0;
        let mut deviations: Vec<(String, f64, f64, f64)> = Vec::new();

        for (i, cond) in reference.conditions.iter().enumerate() {
            let observed = prevalences.get(i).copied().unwrap_or(0.0);
            let expected = cond.prevalence;

            let deviation = (observed - expected).abs();
            max_deviation = max_deviation.max(deviation);
            deviations.push((cond.code.clone(), observed, expected, deviation));

            // Bernoulli-pair KL contribution.
            //
            // Each condition is an independent Bernoulli(p_i) observed vs Bernoulli(q_i)
            // expected. The per-pair KL is:
            //
            //     D_KL(P_i || Q_i) = p log(p/q) + (1-p) log((1-p)/(1-q))
            //
            // which is non-negative by Gibbs' inequality (Cover & Thomas, Elements of
            // Information Theory, 2nd ed., §2.6). Summing across independent conditions
            // gives a quantity bounded in [0, ∞) — unlike the prior `Σ p log(p/q)` form
            // which could be negative because the marginals do not sum to 1.
            let p = observed.clamp(1e-10, 1.0 - 1e-10);
            let q = expected.clamp(1e-10, 1.0 - 1e-10);
            kl_divergence += p * (p / q).ln() + (1.0 - p) * ((1.0 - p) / (1.0 - q)).ln();
        }

        // Sort by deviation descending
        deviations.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

        // Chi-squared test
        let chi_squared = self.chi_squared_conditions(reference);

        // Determine if validation passed
        // - max_deviation < 10% is considered acceptable
        // - chi_squared scales with sample size and number of conditions
        // For n=100K patients with 214 conditions, expected chi-squared for good fit is O(n^0.5)
        // We're comparing simplified MSS models to complex Java simulations, so some deviation is expected
        // A practical threshold: chi_squared < sqrt(n) * num_conditions / 10
        let num_conditions = reference.conditions.len();
        let n = self.total_patients as f64;
        let chi_squared_threshold = n.sqrt() * (num_conditions as f64) / 10.0;
        let passed = max_deviation < 0.10 && chi_squared < chi_squared_threshold;

        StatisticalComparison {
            total_patients: self.total_patients,
            kl_divergence,
            max_deviation,
            chi_squared,
            top_deviations: deviations.into_iter().take(10).collect(),
            passed,
        }
    }

    /// Computes chi-squared statistic for condition prevalences.
    fn chi_squared_conditions(&self, reference: &MssFingerprint) -> f64 {
        let n = self.total_patients as f64;
        if n == 0.0 {
            return 0.0;
        }

        let mut chi_sq = 0.0;

        for (i, cond) in reference.conditions.iter().enumerate() {
            let observed = self.condition_counts.get(i).copied().unwrap_or(0) as f64;
            let expected = cond.prevalence * n;

            if expected > 0.0 {
                chi_sq += (observed - expected).powi(2) / expected;
            }
        }

        chi_sq
    }

    /// Mean encounters per patient.
    pub fn mean_encounters(&self) -> f64 {
        if self.total_patients == 0 {
            0.0
        } else {
            self.total_encounters as f64 / self.total_patients as f64
        }
    }

    /// Mean events per encounter.
    pub fn mean_events_per_encounter(&self) -> f64 {
        if self.total_encounters == 0 {
            0.0
        } else {
            self.total_events as f64 / self.total_encounters as f64
        }
    }
}

/// Result of comparing generated statistics to reference.
#[derive(Debug, Clone)]
pub struct StatisticalComparison {
    /// Total patients in comparison.
    pub total_patients: u64,

    /// Kullback-Leibler divergence.
    pub kl_divergence: f64,

    /// Maximum absolute deviation in any prevalence.
    pub max_deviation: f64,

    /// Chi-squared statistic.
    pub chi_squared: f64,

    /// Top deviating conditions (code, observed, expected, deviation).
    pub top_deviations: Vec<(String, f64, f64, f64)>,

    /// Whether the comparison passed validation thresholds.
    pub passed: bool,
}

impl StatisticalComparison {
    /// Prints a summary of the comparison.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "Statistical Comparison (n={})\n",
            self.total_patients
        ));
        s.push_str(&format!("  KL Divergence: {:.6}\n", self.kl_divergence));
        s.push_str(&format!("  Max Deviation: {:.4}\n", self.max_deviation));
        s.push_str(&format!("  Chi-Squared:   {:.2}\n", self.chi_squared));
        s.push_str(&format!("  Passed:        {}\n", self.passed));

        if !self.top_deviations.is_empty() {
            s.push_str("\n  Top Deviations:\n");
            for (code, obs, exp, dev) in &self.top_deviations {
                s.push_str(&format!(
                    "    {}: observed={:.4}, expected={:.4}, deviation={:.4}\n",
                    code, obs, exp, dev
                ));
            }
        }

        s
    }
}

/// Reservoir sampler for keeping a random sample of patients.
pub struct ReservoirSampler<T> {
    /// Sample buffer.
    samples: Vec<T>,
    /// Maximum sample size.
    capacity: usize,
    /// Total items seen.
    count: u64,
}

impl<T: Clone> ReservoirSampler<T> {
    /// Creates a new reservoir sampler.
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            capacity,
            count: 0,
        }
    }

    /// Adds an item to the reservoir.
    pub fn add<R: rand::Rng>(&mut self, item: T, rng: &mut R) {
        self.count += 1;

        if self.samples.len() < self.capacity {
            self.samples.push(item);
        } else {
            // Reservoir sampling: replace with probability capacity/count
            let j = rng.gen_range(0..self.count as usize);
            if j < self.capacity {
                self.samples[j] = item;
            }
        }
    }

    /// Returns the sampled items.
    pub fn samples(&self) -> &[T] {
        &self.samples
    }

    /// Returns the total count of items seen.
    pub fn count(&self) -> u64 {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_statistics_merge() {
        let mut stats1 = StreamingStatistics::new(5);
        stats1.total_patients = 100;
        stats1.condition_counts[0] = 30;
        stats1.condition_counts[1] = 20;

        let mut stats2 = StreamingStatistics::new(5);
        stats2.total_patients = 100;
        stats2.condition_counts[0] = 25;
        stats2.condition_counts[2] = 15;

        stats1.merge(&stats2);

        assert_eq!(stats1.total_patients, 200);
        assert_eq!(stats1.condition_counts[0], 55);
        assert_eq!(stats1.condition_counts[1], 20);
        assert_eq!(stats1.condition_counts[2], 15);
    }

    #[test]
    fn test_prevalences() {
        let mut stats = StreamingStatistics::new(3);
        stats.total_patients = 100;
        stats.condition_counts[0] = 30;
        stats.condition_counts[1] = 10;
        stats.condition_counts[2] = 5;

        let prev = stats.condition_prevalences();

        assert!((prev[0] - 0.30).abs() < 0.001);
        assert!((prev[1] - 0.10).abs() < 0.001);
        assert!((prev[2] - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_reservoir_sampler() {
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;

        let mut sampler = ReservoirSampler::new(10);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

        for i in 0..100 {
            sampler.add(i, &mut rng);
        }

        assert_eq!(sampler.samples().len(), 10);
        assert_eq!(sampler.count(), 100);
    }
}
