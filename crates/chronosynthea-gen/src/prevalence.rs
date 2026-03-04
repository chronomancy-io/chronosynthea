//! Prevalence registry for condition, medication, and procedure data.

use ahash::AHashMap;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use crate::alias::AliasSampler;
use crate::error::GeneratorResult;

/// Pre-interned string cache for zero-copy event creation.
#[derive(Debug, Clone)]
pub struct InternedEntry {
    /// Interned code string.
    pub code: Arc<str>,
    /// Interned system string.
    pub system: Arc<str>,
    /// Interned display string.
    pub display: Arc<str>,
    /// Per-encounter frequency.
    pub frequency: f64,
}

/// Curated prevalence registry loaded from JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuratedRegistry {
    /// Registry version.
    pub version: String,

    /// Condition prevalence data.
    #[serde(default)]
    pub conditions: Vec<ConditionPrevalence>,

    /// Medication data.
    #[serde(default)]
    pub medications: Vec<MedicationEntry>,

    /// Observation data.
    #[serde(default)]
    pub observations: Vec<ObservationEntry>,

    /// Procedure data.
    #[serde(default)]
    pub procedures: Vec<ProcedureEntry>,

    /// Demographic profile.
    #[serde(default)]
    pub demographics: DemographicProfile,
}

/// Condition prevalence data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionPrevalence {
    /// SNOMED code.
    pub code: String,

    /// Human-readable display name.
    pub display: String,

    /// Source module name.
    #[serde(default)]
    pub module: String,

    /// Base prevalence rate (0.0 - 1.0).
    #[serde(default)]
    pub base_rate: f64,

    /// Per-encounter frequency.
    #[serde(default)]
    pub per_encounter_freq: f64,

    /// Calibrated Java per-patient rate.
    #[serde(default)]
    pub java_per_patient: f64,

    /// Prevalence by age bucket.
    #[serde(default)]
    pub by_age: AHashMap<String, f64>,

    /// Prevalence by gender.
    #[serde(default)]
    pub by_gender: AHashMap<String, f64>,

    /// Prevalence by race.
    #[serde(default)]
    pub by_race: AHashMap<String, f64>,

    /// Onset age distribution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub onset_age: Option<Distribution>,

    /// Risk factors.
    #[serde(default)]
    pub risk_factors: AHashMap<String, f64>,

    /// Whether the condition is chronic.
    #[serde(default)]
    pub chronic: bool,
}

/// Probability distribution specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Distribution {
    /// Distribution type (normal, uniform, exact).
    pub kind: String,

    /// Mean value.
    #[serde(default)]
    pub mean: f64,

    /// Standard deviation.
    #[serde(default)]
    pub std_dev: f64,

    /// Minimum value.
    #[serde(default)]
    pub min: f64,

    /// Maximum value.
    #[serde(default)]
    pub max: f64,

    /// Unit.
    #[serde(default)]
    pub unit: String,
}

/// Medication entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedicationEntry {
    /// RxNorm code.
    pub code: String,

    /// Human-readable display name.
    pub display: String,

    /// Indication condition code.
    #[serde(default)]
    pub indication_code: String,

    /// Whether the medication is for a chronic condition.
    #[serde(default)]
    pub chronic: bool,

    /// Per-encounter frequency.
    #[serde(default)]
    pub frequency: f64,

    /// Calibrated Java per-patient rate.
    #[serde(default)]
    pub java_per_patient: f64,
}

/// Observation entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationEntry {
    /// LOINC or other code.
    pub code: String,

    /// Coding system.
    #[serde(default)]
    pub system: String,

    /// Human-readable display name.
    pub display: String,

    /// Observation category.
    #[serde(default)]
    pub category: String,

    /// Source module name.
    #[serde(default)]
    pub module: String,

    /// Unit of measurement.
    #[serde(default)]
    pub unit: String,

    /// Per-encounter frequency.
    #[serde(default)]
    pub frequency: f64,

    /// Calibrated Java per-patient rate.
    #[serde(default)]
    pub java_per_patient: f64,
}

/// Procedure entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureEntry {
    /// SNOMED code.
    pub code: String,

    /// Coding system.
    #[serde(default)]
    pub system: String,

    /// Human-readable display name.
    pub display: String,

    /// Source module name.
    #[serde(default)]
    pub module: String,

    /// Indication condition code.
    #[serde(default)]
    pub indication_code: String,

    /// Per-encounter frequency.
    #[serde(default)]
    pub frequency: f64,

    /// Calibrated Java per-patient rate.
    #[serde(default)]
    pub java_per_patient: f64,
}

/// Demographic distribution profile.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DemographicProfile {
    /// Age bucket distribution.
    #[serde(default)]
    pub age_distribution: AHashMap<String, f64>,

    /// Gender distribution.
    #[serde(default)]
    pub gender_distribution: AHashMap<String, f64>,

    /// Race distribution.
    #[serde(default)]
    pub race_distribution: AHashMap<String, f64>,

    /// Ethnicity distribution.
    #[serde(default)]
    pub ethnicity_distribution: AHashMap<String, f64>,
}

impl CuratedRegistry {
    /// Loads a curated registry from a JSON file.
    pub fn load<P: AsRef<Path>>(path: P) -> GeneratorResult<Self> {
        let data = fs::read(path)?;
        let registry: Self = serde_json::from_slice(&data)?;
        Ok(registry)
    }
}

/// Optimized registry with pre-computed data structures for O(1) sampling.
#[derive(Debug, Clone)]
pub struct OptimizedRegistry {
    /// Underlying curated registry.
    pub registry: CuratedRegistry,

    /// Pre-computed alias sampler for age distribution.
    pub age_sampler: Option<AliasSampler>,

    /// Pre-computed alias sampler for gender distribution.
    pub gender_sampler: Option<AliasSampler>,

    /// Pre-computed alias sampler for race distribution.
    pub race_sampler: Option<AliasSampler>,

    /// Pre-computed alias sampler for ethnicity distribution.
    pub ethnicity_sampler: Option<AliasSampler>,

    /// Pre-indexed medications by indication code.
    pub meds_by_indication: AHashMap<String, Vec<MedicationEntry>>,

    /// Pre-indexed procedures by indication code.
    pub procs_by_indication: AHashMap<String, Vec<ProcedureEntry>>,

    /// Pre-computed condition codes for fast iteration.
    pub condition_codes: Vec<String>,

    /// Pre-computed condition base rates.
    pub condition_rates: Vec<f64>,

    /// Pre-interned condition entries for zero-copy event creation.
    pub interned_conditions: Vec<InternedEntry>,

    /// Pre-interned medication entries for zero-copy event creation.
    pub interned_medications: Vec<InternedEntry>,

    /// Pre-interned observation entries for zero-copy event creation.
    pub interned_observations: Vec<InternedEntry>,

    /// Pre-interned procedure entries for zero-copy event creation.
    pub interned_procedures: Vec<InternedEntry>,
}

impl OptimizedRegistry {
    /// Creates an optimized registry from a curated registry.
    pub fn new(registry: CuratedRegistry) -> Self {
        // Build alias samplers for demographics
        let age_sampler = AliasSampler::new(&registry.demographics.age_distribution);
        let gender_sampler = AliasSampler::new(&registry.demographics.gender_distribution);
        let race_sampler = AliasSampler::new(&registry.demographics.race_distribution);
        let ethnicity_sampler = AliasSampler::new(&registry.demographics.ethnicity_distribution);

        // Build medication index by indication
        let mut meds_by_indication: AHashMap<String, Vec<MedicationEntry>> = AHashMap::new();
        for med in &registry.medications {
            if !med.indication_code.is_empty() {
                meds_by_indication
                    .entry(med.indication_code.clone())
                    .or_default()
                    .push(med.clone());
            }
        }

        // Build procedure index by indication
        let mut procs_by_indication: AHashMap<String, Vec<ProcedureEntry>> = AHashMap::new();
        for proc in &registry.procedures {
            if !proc.indication_code.is_empty() {
                procs_by_indication
                    .entry(proc.indication_code.clone())
                    .or_default()
                    .push(proc.clone());
            }
        }

        // Pre-compute condition codes and rates
        let condition_codes: Vec<String> =
            registry.conditions.iter().map(|c| c.code.clone()).collect();
        let condition_rates: Vec<f64> = registry.conditions.iter().map(|c| c.base_rate).collect();

        // Pre-intern all strings for zero-copy event creation
        let snomed_ct: Arc<str> = Arc::from("SNOMED-CT");
        let rxnorm: Arc<str> = Arc::from("RxNorm");

        let interned_conditions: Vec<InternedEntry> = registry
            .conditions
            .iter()
            .map(|c| InternedEntry {
                code: Arc::from(c.code.as_str()),
                system: Arc::clone(&snomed_ct),
                display: Arc::from(c.display.as_str()),
                frequency: c.per_encounter_freq.max(0.005),
            })
            .collect();

        let interned_medications: Vec<InternedEntry> = registry
            .medications
            .iter()
            .map(|m| InternedEntry {
                code: Arc::from(m.code.as_str()),
                system: Arc::clone(&rxnorm),
                display: Arc::from(m.display.as_str()),
                frequency: m.frequency.max(0.01),
            })
            .collect();

        let interned_observations: Vec<InternedEntry> = registry
            .observations
            .iter()
            .map(|o| InternedEntry {
                code: Arc::from(o.code.as_str()),
                system: Arc::from(if o.system.is_empty() {
                    "LOINC"
                } else {
                    o.system.as_str()
                }),
                display: Arc::from(o.display.as_str()),
                frequency: o.frequency.max(0.01),
            })
            .collect();

        let interned_procedures: Vec<InternedEntry> = registry
            .procedures
            .iter()
            .map(|p| InternedEntry {
                code: Arc::from(p.code.as_str()),
                system: Arc::from(if p.system.is_empty() {
                    "SNOMED-CT"
                } else {
                    p.system.as_str()
                }),
                display: Arc::from(p.display.as_str()),
                frequency: p.frequency.max(0.01),
            })
            .collect();

        Self {
            registry,
            age_sampler,
            gender_sampler,
            race_sampler,
            ethnicity_sampler,
            meds_by_indication,
            procs_by_indication,
            condition_codes,
            condition_rates,
            interned_conditions,
            interned_medications,
            interned_observations,
            interned_procedures,
        }
    }

    /// Loads and optimizes a registry from a file.
    pub fn load<P: AsRef<Path>>(path: P) -> GeneratorResult<Self> {
        let registry = CuratedRegistry::load(path)?;
        Ok(Self::new(registry))
    }

    /// Samples demographics using O(1) alias method.
    pub fn sample_demographics_fast<R: Rng>(&self, rng: &mut R) -> (u32, String, String, String) {
        // Sample age bucket, then age within bucket
        let age_bucket = self
            .age_sampler
            .as_ref()
            .map(|s| s.sample(rng).to_string())
            .unwrap_or_else(|| "18-44".to_string());
        let age = sample_age_in_bucket(&age_bucket, rng);

        // Sample gender
        let gender = self
            .gender_sampler
            .as_ref()
            .map(|s| s.sample(rng).to_string())
            .unwrap_or_else(|| "M".to_string());

        // Sample race
        let race = self
            .race_sampler
            .as_ref()
            .map(|s| s.sample(rng).to_string())
            .unwrap_or_else(|| "white".to_string());

        // Sample ethnicity
        let ethnicity = self
            .ethnicity_sampler
            .as_ref()
            .map(|s| s.sample(rng).to_string())
            .unwrap_or_else(|| "nonhispanic".to_string());

        (age, gender, race, ethnicity)
    }

    /// Returns medications for a condition code in O(1).
    #[inline]
    pub fn get_medications_for_condition(&self, condition_code: &str) -> &[MedicationEntry] {
        self.meds_by_indication
            .get(condition_code)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Returns procedures for a condition code in O(1).
    #[inline]
    pub fn get_procedures_for_condition(&self, condition_code: &str) -> &[ProcedureEntry] {
        self.procs_by_indication
            .get(condition_code)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

/// Samples a specific age within an age bucket.
fn sample_age_in_bucket<R: Rng>(bucket: &str, rng: &mut R) -> u32 {
    match bucket {
        "0-17" => rng.gen_range(0..18),
        "18-44" => 18 + rng.gen_range(0..27),
        "45-64" => 45 + rng.gen_range(0..20),
        "65+" => 65 + rng.gen_range(0..30),
        _ => 40, // Default middle age
    }
}

/// Returns the age bucket for a given age.
pub fn get_age_bucket(age: u32) -> &'static str {
    match age {
        0..=17 => "0-17",
        18..=44 => "18-44",
        45..=64 => "45-64",
        _ => "65+",
    }
}

/// A condition sampled for a patient.
#[derive(Debug, Clone)]
pub struct SampledCondition {
    /// Condition code.
    pub code: String,
    /// Display name.
    pub display: String,
    /// Onset age.
    pub onset_age: u32,
    /// Whether the condition is chronic.
    pub chronic: bool,
    /// Associated medications.
    pub medications: Vec<MedicationEntry>,
}

/// Samples conditions for a patient based on demographics.
pub fn sample_conditions_for_patient<R: Rng>(
    registry: &CuratedRegistry,
    rng: &mut R,
    age: u32,
    gender: &str,
    race: &str,
) -> Vec<SampledCondition> {
    let mut conditions = Vec::new();
    let age_bucket = get_age_bucket(age);

    for cond in &registry.conditions {
        // Calculate adjusted prevalence
        let mut rate = cond.base_rate;

        // Apply age multiplier
        if let Some(&mult) = cond.by_age.get(age_bucket) {
            rate *= mult;
        }

        // Apply gender multiplier
        if let Some(&mult) = cond.by_gender.get(gender) {
            rate *= mult;
        }

        // Apply race multiplier
        if let Some(&mult) = cond.by_race.get(race) {
            rate *= mult;
        }

        // Clamp rate to valid range
        rate = rate.clamp(0.0, 1.0);

        // Sample with adjusted rate
        if rng.gen::<f64>() < rate {
            // Sample onset age
            let onset_age = sample_onset_age(cond.onset_age.as_ref(), rng, age);

            // Only include if onset has occurred
            if onset_age <= age {
                // Find medications for this condition
                let medications: Vec<MedicationEntry> = registry
                    .medications
                    .iter()
                    .filter(|med| med.indication_code == cond.code)
                    .cloned()
                    .collect();

                conditions.push(SampledCondition {
                    code: cond.code.clone(),
                    display: cond.display.clone(),
                    onset_age,
                    chronic: cond.chronic,
                    medications,
                });
            }
        }
    }

    conditions
}

/// Samples an onset age from a distribution.
fn sample_onset_age<R: Rng>(dist: Option<&Distribution>, rng: &mut R, max_age: u32) -> u32 {
    let onset = match dist {
        Some(d) => match d.kind.as_str() {
            "normal" => {
                let normal = rand_distr::Normal::new(d.mean, d.std_dev)
                    .unwrap_or_else(|_| rand_distr::Normal::new(d.mean, 1.0).unwrap());
                rng.sample(normal)
            }
            "uniform" => d.min + rng.gen::<f64>() * (d.max - d.min),
            "exact" => d.mean,
            _ => rng.gen_range(0..=max_age) as f64,
        },
        None => rng.gen_range(0..=max_age) as f64,
    };

    onset.clamp(0.0, max_age as f64).round() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_registry() -> CuratedRegistry {
        let mut age_dist = AHashMap::new();
        age_dist.insert("18-44".to_string(), 0.5);
        age_dist.insert("45-64".to_string(), 0.3);
        age_dist.insert("65+".to_string(), 0.2);

        let mut gender_dist = AHashMap::new();
        gender_dist.insert("M".to_string(), 0.5);
        gender_dist.insert("F".to_string(), 0.5);

        CuratedRegistry {
            version: "1.0".to_string(),
            conditions: vec![ConditionPrevalence {
                code: "38341003".to_string(),
                display: "Hypertension".to_string(),
                base_rate: 0.3,
                ..Default::default()
            }],
            medications: vec![],
            observations: vec![],
            procedures: vec![],
            demographics: DemographicProfile {
                age_distribution: age_dist,
                gender_distribution: gender_dist,
                ..Default::default()
            },
        }
    }

    impl Default for ConditionPrevalence {
        fn default() -> Self {
            Self {
                code: String::new(),
                display: String::new(),
                module: String::new(),
                base_rate: 0.0,
                per_encounter_freq: 0.0,
                java_per_patient: 0.0,
                by_age: AHashMap::new(),
                by_gender: AHashMap::new(),
                by_race: AHashMap::new(),
                onset_age: None,
                risk_factors: AHashMap::new(),
                chronic: false,
            }
        }
    }

    #[test]
    fn test_optimized_registry() {
        let registry = create_test_registry();
        let opt = OptimizedRegistry::new(registry);

        assert!(opt.age_sampler.is_some());
        assert!(opt.gender_sampler.is_some());
        assert_eq!(opt.condition_codes.len(), 1);
    }

    #[test]
    fn test_age_bucket() {
        assert_eq!(get_age_bucket(10), "0-17");
        assert_eq!(get_age_bucket(30), "18-44");
        assert_eq!(get_age_bucket(50), "45-64");
        assert_eq!(get_age_bucket(70), "65+");
    }
}
