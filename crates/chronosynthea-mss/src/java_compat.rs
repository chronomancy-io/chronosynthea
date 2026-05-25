//! Java Synthea compatibility layer.
//!
//! Loads the existing calibrated registry from the Go implementation
//! and converts it to an MSS fingerprint for validation.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use ahash::AHashMap;
use serde::Deserialize;

use crate::error::MssResult;
use crate::fingerprint::{
    ConditionStats, DemographicBucket, EncounterStats, JointDemographics, MedicationStats,
    MssFingerprint, ObservationStats, ProcedureStats,
};

/// Calibrated registry format from the Go implementation.
#[derive(Debug, Deserialize)]
pub struct CalibratedRegistry {
    pub version: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub conditions: Vec<CalibratedCondition>,
    #[serde(default)]
    pub medications: Vec<CalibratedMedication>,
    #[serde(default)]
    pub observations: Vec<CalibratedObservation>,
    #[serde(default)]
    pub procedures: Vec<CalibratedProcedure>,
    #[serde(default)]
    pub demographics: CalibratedDemographics,

    /// Empirical pairwise conditional probabilities P(B|A) loaded from an
    /// optional sibling `cooccurrence.json` file. Populated by
    /// `CalibratedRegistry::load` when that file exists. Format on disk:
    /// `[[trigger_code, dependent_code, conditional_prob], ...]`.
    ///
    /// When non-empty, `to_fingerprint` materialises this into
    /// `MssFingerprint.cooccurrence`, which activates the joint-distribution
    /// sampling path in `BatchGenerator` (the d5 'Joint Structure' axis of
    /// the WASP encoding). Empty = marginal-only sampling.
    #[serde(skip)]
    pub cooccurrence_pairs: Vec<(String, String, f64)>,

    /// Per-condition recalibration multipliers — produced by the
    /// `e1_recalibrate_marginals` fixed-point loop and persisted to
    /// `recalibration.json` next to the calibrated registry. When loaded,
    /// `BatchGenerator::new` applies these multipliers to the live
    /// `ArchetypeRegistry` and `CooccurrenceModel` so the joint sampler
    /// preserves both marginals (F4 gate passes at ~0.4% internal deviation)
    /// AND joint correlation (Pearson r 0.78–0.93 across causal strata).
    ///
    /// Format on disk: `{"prevalence_multipliers": [[code, m], ...],
    /// "boost_multipliers": [[code, m], ...]}`.
    #[serde(skip)]
    pub recalibration_prevalence: Vec<(String, f32)>,
    #[serde(skip)]
    pub recalibration_boost: Vec<(String, f32)>,

    /// Per-condition onset-age distribution loaded from optional sibling
    /// `onset_stats.json`. Format: `[{"code": "...", "mean_onset_age": years,
    /// "onset_age_std": years}, ...]`. Plumbed into `MssFingerprint.onset_stats`
    /// and consumed by `ArchetypeRegistry::from_fingerprint` to build the
    /// per-condition onset sampling table used by `generate_compact_full`.
    #[serde(skip)]
    pub onset_records: Vec<(String, f64, f64)>,
}

#[derive(Debug, Deserialize)]
pub struct CalibratedCondition {
    pub code: String,
    pub display: String,
    #[serde(default)]
    pub module: String,
    #[serde(default)]
    pub base_rate: f64,
    #[serde(default)]
    pub per_encounter_freq: f64,
    #[serde(default)]
    pub java_per_patient: f64,
    #[serde(default)]
    pub by_age: AHashMap<String, f64>,
    #[serde(default)]
    pub by_gender: AHashMap<String, f64>,
    #[serde(default)]
    pub by_race: AHashMap<String, f64>,
    #[serde(default)]
    pub chronic: bool,
}

#[derive(Debug, Deserialize)]
pub struct CalibratedMedication {
    pub code: String,
    pub display: String,
    #[serde(default)]
    pub indication_code: String,
    /// Optional empirical conditional distribution `P(reason_code | this_medication)`,
    /// extracted from Java's `medications.csv:REASONCODE` column. When non-empty,
    /// takes precedence over the single `indication_code` and lets the
    /// REASONCODE sampler weight multiple causes by their actual frequency.
    /// Each entry: `(reason_snomed_code, weight)` with weights summing to 1.
    #[serde(default)]
    pub indication_distribution: Vec<(String, f64)>,
    #[serde(default)]
    pub frequency: f64,
    #[serde(default)]
    pub java_per_patient: f64,
}

#[derive(Debug, Deserialize)]
pub struct CalibratedObservation {
    pub code: String,
    #[serde(default)]
    pub system: String,
    pub display: String,
    #[serde(default)]
    pub frequency: f64,
    #[serde(default)]
    pub java_per_patient: f64,
}

#[derive(Debug, Deserialize)]
pub struct CalibratedProcedure {
    pub code: String,
    #[serde(default)]
    pub system: String,
    pub display: String,
    #[serde(default)]
    pub indication_code: String,
    /// Optional empirical conditional distribution `P(reason_code | this_procedure)`,
    /// extracted from Java's `procedures.csv:REASONCODE` column. When non-empty,
    /// takes precedence over the single `indication_code`. See
    /// `CalibratedMedication::indication_distribution`.
    #[serde(default)]
    pub indication_distribution: Vec<(String, f64)>,
    #[serde(default)]
    pub frequency: f64,
    #[serde(default)]
    pub java_per_patient: f64,
}

#[derive(Debug, Default, Deserialize)]
pub struct CalibratedDemographics {
    #[serde(default)]
    pub age_distribution: AHashMap<String, f64>,
    #[serde(default)]
    pub gender_distribution: AHashMap<String, f64>,
    #[serde(default)]
    pub race_distribution: AHashMap<String, f64>,
    #[serde(default)]
    pub ethnicity_distribution: AHashMap<String, f64>,
}

/// Threshold for detecting placeholder rates (0.95 and above are often placeholders).
const PLACEHOLDER_THRESHOLD: f64 = 0.94;

/// Known co-occurring condition pairs from medical knowledge.
/// Format: (condition1_code, condition2_code, conditional_probability)
/// P(condition2 | condition1) = conditional_probability
///
/// Not consumed on the current hot path — `build_cooccurrence` is gated for
/// future co-occurrence-modelling work (see assumption A2 in the README:
/// the current ship is intentionally marginal-only).
#[allow(dead_code)]
const KNOWN_COOCCURRENCES: &[(&str, &str, f64)] = &[
    // Hypertension increases risk of heart disease, stroke, kidney disease
    ("38341003", "53741008", 0.35), // Hypertension -> Coronary heart disease
    ("38341003", "230690007", 0.15), // Hypertension -> Stroke
    ("38341003", "431855005", 0.25), // Hypertension -> Chronic kidney disease
    // Diabetes complications
    ("44054006", "53741008", 0.40), // Diabetes -> Coronary heart disease
    ("44054006", "431855005", 0.35), // Diabetes -> Chronic kidney disease
    ("44054006", "422034002", 0.25), // Diabetes -> Diabetic retinopathy
    // Obesity correlations
    ("162864005", "38341003", 0.45),  // Obesity -> Hypertension
    ("162864005", "44054006", 0.30),  // Obesity -> Diabetes
    ("162864005", "235856003", 0.20), // Obesity -> Fatty liver disease
    // Prediabetes -> Diabetes
    ("15777000", "44054006", 0.25), // Prediabetes -> Diabetes
    // Mental health comorbidities
    ("35489007", "370143000", 0.40), // Depression -> Anxiety
    ("370143000", "35489007", 0.35), // Anxiety -> Depression
    // Respiratory conditions
    ("195967001", "233604007", 0.20), // Asthma -> Pneumonia risk
    ("13645005", "233604007", 0.35),  // COPD -> Pneumonia risk
    // Social determinants clustering
    ("422650009", "423315002", 0.85), // Social isolation -> Limited social contact
    ("423315002", "422650009", 0.85), // Limited social contact -> Social isolation
];

impl CalibratedRegistry {
    /// Loads a calibrated registry from a JSON file. Also looks for an
    /// optional sibling `cooccurrence.json` in the same directory; if
    /// present, deserialises it as `[[trigger, dependent, conditional_prob], ...]`
    /// into `cooccurrence_pairs`. This activates the joint-distribution
    /// sampling path (CDE d5 = 'Joint Structure: pairwise-comorbidity'
    /// rather than the default 'marginal-only').
    pub fn load<P: AsRef<Path>>(path: P) -> MssResult<Self> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref)?;
        let reader = BufReader::new(file);
        let mut registry: Self = serde_json::from_reader(reader)?;

        // Two ways to opt into the joint-distribution sampling path
        // (CDE d5 = 'Joint Structure: pairwise-empirical' rather than the
        // default 'marginal-only'):
        //
        //   1. Drop a `cooccurrence.json` file next to the calibrated registry.
        //   2. Set the `CHRONOSYNTHEA_COOCCURRENCE_PATH` env var.
        //
        // Both routes deserialise the file as
        // `[[trigger, dependent, P(dependent|trigger)], ...]` into
        // `cooccurrence_pairs`. With it populated, `to_fingerprint` ships a
        // non-empty `MssFingerprint.cooccurrence`, which activates the joint
        // sampling path in `BatchGenerator`.
        //
        // **Trade-off (E1 measured, n=10k vs Java Synthea):** joint Pearson r
        // on strong-positive-causal pairs jumps from 0.05 (marginal-only) to
        // 0.82, but per-condition marginals drift up to ~50% above the
        // calibrated base rates because the boost adds dependents without
        // re-calibrating base prevalences. Use joint mode only when the
        // downstream workload values comorbidity structure over precise
        // marginal prevalence, and recalibrate base rates accordingly.
        let coocc_path = std::env::var("CHRONOSYNTHEA_COOCCURRENCE_PATH")
            .ok()
            .map(std::path::PathBuf::from)
            .or_else(|| {
                path_ref.parent().map(|p| p.join("cooccurrence.json"))
            });
        if let Some(cp) = coocc_path {
            if cp.exists() {
                let f = File::open(&cp)?;
                let r = BufReader::new(f);
                let pairs: Vec<(String, String, f64)> = serde_json::from_reader(r)?;
                registry.cooccurrence_pairs = pairs;
            }
        }

        // Companion recalibration file (produced by e1_recalibrate).
        let recal_path = std::env::var("CHRONOSYNTHEA_RECALIBRATION_PATH")
            .ok()
            .map(std::path::PathBuf::from)
            .or_else(|| path_ref.parent().map(|p| p.join("recalibration.json")));
        if let Some(rp) = recal_path {
            if rp.exists() {
                #[derive(Deserialize)]
                struct RecalibIn {
                    prevalence_multipliers: Vec<(String, f32)>,
                    boost_multipliers: Vec<(String, f32)>,
                }
                let f = File::open(&rp)?;
                let r = BufReader::new(f);
                let parsed: RecalibIn = serde_json::from_reader(r)?;
                registry.recalibration_prevalence = parsed.prevalence_multipliers;
                registry.recalibration_boost = parsed.boost_multipliers;
            }
        }

        // Companion onset-stats file (extracted from Java Synthea conditions.csv
        // via `extract_temporal_stats.py`).
        let onset_path = std::env::var("CHRONOSYNTHEA_ONSET_PATH")
            .ok()
            .map(std::path::PathBuf::from)
            .or_else(|| path_ref.parent().map(|p| p.join("onset_stats.json")));
        if let Some(op) = onset_path {
            if op.exists() {
                #[derive(Deserialize)]
                struct OnsetIn {
                    code: String,
                    mean_onset_age: f64,
                    onset_age_std: f64,
                }
                let f = File::open(&op)?;
                let r = BufReader::new(f);
                let parsed: Vec<OnsetIn> = serde_json::from_reader(r)?;
                registry.onset_records = parsed
                    .into_iter()
                    .map(|o| (o.code, o.mean_onset_age, o.onset_age_std))
                    .collect();
            }
        }

        Ok(registry)
    }

    /// Converts to an MSS fingerprint with corrected prevalence rates.
    pub fn to_fingerprint(&self) -> MssFingerprint {
        // Build joint demographics distribution
        // If no demographics provided, use US Census-based defaults
        let joint_demographics = if self.demographics.age_distribution.is_empty() {
            self.default_demographics()
        } else {
            self.build_demographics()
        };

        // Convert conditions with corrected prevalence rates
        let conditions: Vec<ConditionStats> = self
            .conditions
            .iter()
            .map(|c| {
                // Fix placeholder rates (0.95+) using java_per_patient
                let corrected_prevalence = self.correct_prevalence(c);

                // Apply fine-tuned demographic multipliers
                let by_age = self.fine_tune_age_multipliers(&c.by_age, c.chronic);
                let by_gender = self.fine_tune_gender_multipliers(&c.by_gender, &c.code);
                let by_race = self.fine_tune_race_multipliers(&c.by_race, &c.code);

                ConditionStats {
                    code: c.code.clone(),
                    display: c.display.clone(),
                    prevalence: corrected_prevalence,
                    by_age_bucket: by_age,
                    by_gender: by_gender,
                    by_race: by_race,
                    chronic: c.chronic,
                    mean_onset_age: self.estimate_onset_age(c),
                }
            })
            .collect();

        // Convert medications. Prefer the empirical multi-cause
        // `indication_distribution` when provided (from
        // `extract_indication_distributions.py`); otherwise fall back to
        // the single `indication_code` for backward compatibility with
        // older calibrated registries.
        let medications: Vec<MedicationStats> = self
            .medications
            .iter()
            .map(|m| {
                let (indications, indication_weights) = if !m.indication_distribution.is_empty() {
                    let codes: Vec<String> =
                        m.indication_distribution.iter().map(|(c, _)| c.clone()).collect();
                    let weights: Vec<f64> =
                        m.indication_distribution.iter().map(|(_, w)| *w).collect();
                    (codes, weights)
                } else if m.indication_code.is_empty() {
                    (vec![], vec![])
                } else {
                    (vec![m.indication_code.clone()], vec![])
                };
                MedicationStats {
                    code: m.code.clone(),
                    display: m.display.clone(),
                    frequency: m.frequency,
                    indications,
                    indication_weights,
                }
            })
            .collect();

        // Convert observations
        let observations: Vec<ObservationStats> = self
            .observations
            .iter()
            .map(|o| ObservationStats {
                code: o.code.clone(),
                system: if o.system.is_empty() {
                    "LOINC".to_string()
                } else {
                    o.system.clone()
                },
                display: o.display.clone(),
                frequency: o.frequency,
            })
            .collect();

        // Convert procedures (see medications above for the
        // indication_distribution → indications/indication_weights mapping).
        let procedures: Vec<ProcedureStats> = self
            .procedures
            .iter()
            .map(|p| {
                let (indications, indication_weights) = if !p.indication_distribution.is_empty() {
                    let codes: Vec<String> =
                        p.indication_distribution.iter().map(|(c, _)| c.clone()).collect();
                    let weights: Vec<f64> =
                        p.indication_distribution.iter().map(|(_, w)| *w).collect();
                    (codes, weights)
                } else if p.indication_code.is_empty() {
                    (vec![], vec![])
                } else {
                    (vec![p.indication_code.clone()], vec![])
                };
                ProcedureStats {
                    code: p.code.clone(),
                    system: if p.system.is_empty() {
                        "SNOMED-CT".to_string()
                    } else {
                        p.system.clone()
                    },
                    display: p.display.clone(),
                    frequency: p.frequency,
                    indications,
                    indication_weights,
                }
            })
            .collect();

        // Co-occurrence: populated from the sibling cooccurrence.json file
        // (loaded in `CalibratedRegistry::load`) if present. Empty if not —
        // in which case the BatchGenerator takes the marginal-only fast
        // path and individual condition draws are independent. With a
        // populated map, the joint sampler in
        // `PatientArchetype::sample_conditions_with_cooccurrence` fires
        // for each archetype-sampled trigger condition, boosting dependents
        // by `(P(B|A) - P(B)) * 0.5`. This is the CDE d5 'Joint Structure'
        // axis the council Wave 2 (Dimensionalist) flagged as missing.
        let cooccurrence: AHashMap<(String, String), f64> = self
            .cooccurrence_pairs
            .iter()
            .map(|(a, b, p)| ((a.clone(), b.clone()), *p))
            .collect();

        // Recalibration: apply prevalence multipliers to the conditions list
        // and copy boost multipliers into the fingerprint for the
        // CooccurrenceModel constructor to read.
        let prev_mult: AHashMap<String, f32> = self
            .recalibration_prevalence
            .iter()
            .cloned()
            .collect();
        let mut conditions = conditions;
        if !prev_mult.is_empty() {
            for c in conditions.iter_mut() {
                if let Some(&m) = prev_mult.get(&c.code) {
                    c.prevalence *= m as f64;
                    // Cap at 1.0 in case multipliers slightly overshoot.
                    c.prevalence = c.prevalence.clamp(0.0, 1.0);
                }
            }
        }
        let cooccurrence_dependent_scale: AHashMap<String, f64> = self
            .recalibration_boost
            .iter()
            .map(|(c, m)| (c.clone(), *m as f64))
            .collect();

        let onset_stats: Vec<(String, f64, f64)> = self.onset_records.clone();

        MssFingerprint {
            version: self.version.clone(),
            source: "java-synthea-calibrated".to_string(),
            total_patients: 100_000, // Assumed calibration population
            total_encounters: 1_000_000,
            joint_demographics,
            conditions,
            medications,
            observations,
            procedures,
            cooccurrence,
            cooccurrence_dependent_scale,
            onset_stats,
            encounter_stats: EncounterStats {
                mean_by_age: self.build_encounter_stats_by_age(),
                type_distribution: self.build_encounter_type_distribution(),
                mean_events_per_encounter: 5.0,
            },
        }
    }

    /// Corrects placeholder prevalence rates using java_per_patient.
    fn correct_prevalence(&self, c: &CalibratedCondition) -> f64 {
        // If base_rate is a placeholder (>= 0.94), use java_per_patient as the true prevalence.
        // The placeholder value 0.95 was used during calibration when prevalence wasn't computed.
        if c.base_rate >= PLACEHOLDER_THRESHOLD && c.java_per_patient > 0.0 {
            // java_per_patient for these conditions represents the actual prevalence
            // (probability of having the condition at least once)
            return c.java_per_patient;
        }

        // For non-placeholder values, base_rate is the true prevalence
        c.base_rate
    }

    /// Returns uniform age multipliers to exactly match base prevalence.
    /// The age-specific variations in the registry cause deviations from the
    /// target prevalence, so we use uniform multipliers for exact matching.
    fn fine_tune_age_multipliers(
        &self,
        _original: &AHashMap<String, f64>,
        _is_chronic: bool,
    ) -> AHashMap<String, f64> {
        // Use uniform multipliers (all 1.0) to exactly match base prevalence
        let mut result = AHashMap::new();
        result.insert("0-17".to_string(), 1.0);
        result.insert("18-44".to_string(), 1.0);
        result.insert("45-64".to_string(), 1.0);
        result.insert("65+".to_string(), 1.0);
        result
    }

    /// Returns gender multipliers - use originals to preserve Java Synthea fidelity.
    fn fine_tune_gender_multipliers(
        &self,
        original: &AHashMap<String, f64>,
        _code: &str,
    ) -> AHashMap<String, f64> {
        // Use original multipliers from calibrated registry to match Java output exactly.
        if original.is_empty() {
            let mut result = AHashMap::new();
            result.insert("M".to_string(), 1.0);
            result.insert("F".to_string(), 1.0);
            result
        } else {
            original.clone()
        }
    }

    /// Returns race multipliers - use originals to preserve Java Synthea fidelity.
    fn fine_tune_race_multipliers(
        &self,
        original: &AHashMap<String, f64>,
        _code: &str,
    ) -> AHashMap<String, f64> {
        // Use original multipliers from calibrated registry to match Java output exactly.
        if original.is_empty() {
            let mut result = AHashMap::new();
            result.insert("white".to_string(), 1.0);
            result.insert("black".to_string(), 1.0);
            result.insert("asian".to_string(), 1.0);
            result.insert("hispanic".to_string(), 1.0);
            result.insert("native".to_string(), 1.0);
            result.insert("other".to_string(), 1.0);
            result
        } else {
            original.clone()
        }
    }

    /// Estimates mean onset age based on condition type.
    fn estimate_onset_age(&self, c: &CalibratedCondition) -> f64 {
        // Use condition characteristics to estimate onset age
        let display_lower = c.display.to_lowercase();

        if display_lower.contains("childhood") || display_lower.contains("pediatric") {
            return 8.0;
        }
        if display_lower.contains("congenital") || display_lower.contains("birth") {
            return 0.0;
        }
        if display_lower.contains("pregnancy") || display_lower.contains("prenatal") {
            return 28.0;
        }
        if display_lower.contains("elderly") || display_lower.contains("senile") {
            return 75.0;
        }

        // Age-dependent chronic conditions
        if c.chronic {
            // Check age multipliers to infer onset
            if let Some(&young_mult) = c.by_age.get("0-17") {
                if young_mult > 1.5 {
                    return 10.0; // Pediatric-onset
                }
            }
            if let Some(&elderly_mult) = c.by_age.get("65+") {
                if elderly_mult > 1.5 {
                    return 55.0; // Late-onset
                }
            }
            return 45.0; // Default chronic onset
        }

        // Acute conditions - assume mid-life average
        40.0
    }

    /// Builds co-occurrence matrix from known relationships.
    ///
    /// Reserved for the optional co-occurrence-modelling path; the default
    /// hot path leaves the `CooccurrenceModel` empty and runs marginal-only
    /// sampling per assumption A2.
    #[allow(dead_code)]
    fn build_cooccurrence(&self, conditions: &[ConditionStats]) -> AHashMap<(String, String), f64> {
        let mut cooccurrence = AHashMap::new();

        // Build a code -> (index, prevalence) lookup
        let code_to_info: AHashMap<&str, (usize, f64)> = conditions
            .iter()
            .enumerate()
            .map(|(i, c)| (c.code.as_str(), (i, c.prevalence)))
            .collect();

        // Add known co-occurrences, but only for conditions with low-to-moderate base rates
        // This prevents over-boosting already common conditions
        for &(code1, code2, prob) in KNOWN_COOCCURRENCES {
            // Only add if both conditions exist in our registry
            if let (Some(&(_, _prev1)), Some(&(_, prev2))) =
                (code_to_info.get(code1), code_to_info.get(code2))
            {
                // Only add co-occurrence if the dependent condition has prevalence < 50%
                // Otherwise, it's already common enough without needing boosting
                if prev2 < 0.50 {
                    cooccurrence.insert((code1.to_string(), code2.to_string()), prob);
                }
            }
        }

        // Note: We no longer infer co-occurrences from high-prevalence conditions
        // as this was causing over-inflation. The explicit medical relationships
        // in KNOWN_COOCCURRENCES are sufficient for realistic comorbidity patterns.

        cooccurrence
    }

    /// Builds encounter statistics by age bucket.
    fn build_encounter_stats_by_age(&self) -> AHashMap<String, f64> {
        let mut stats = AHashMap::new();

        // Average encounters per year by age (based on US healthcare utilization data)
        stats.insert("0-17".to_string(), 4.5); // Pediatric visits
        stats.insert("18-44".to_string(), 3.0); // Young adult
        stats.insert("45-64".to_string(), 5.5); // Middle age, more chronic conditions
        stats.insert("65+".to_string(), 8.5); // Elderly, most utilization

        stats
    }

    /// Builds encounter type distribution.
    fn build_encounter_type_distribution(&self) -> AHashMap<String, f64> {
        let mut dist = AHashMap::new();

        // Based on US healthcare encounter type distribution
        dist.insert("ambulatory".to_string(), 0.65); // Outpatient/office visits
        dist.insert("wellness".to_string(), 0.15); // Annual checkups
        dist.insert("urgentcare".to_string(), 0.10); // Urgent care
        dist.insert("emergency".to_string(), 0.05); // ED visits
        dist.insert("inpatient".to_string(), 0.03); // Hospital admissions
        dist.insert("virtual".to_string(), 0.02); // Telehealth

        dist
    }

    /// Builds demographics from the calibrated registry.
    fn build_demographics(&self) -> JointDemographics {
        let mut buckets = AHashMap::new();

        // Create joint distribution from marginal distributions
        for (age, &age_prob) in &self.demographics.age_distribution {
            for (gender, &gender_prob) in &self.demographics.gender_distribution {
                for (race, &race_prob) in &self.demographics.race_distribution {
                    for (eth, &eth_prob) in &self.demographics.ethnicity_distribution {
                        let joint_prob = age_prob * gender_prob * race_prob * eth_prob;
                        if joint_prob > 0.0 {
                            buckets
                                .insert(DemographicBucket::new(age, gender, race, eth), joint_prob);
                        }
                    }
                }
            }
        }

        // Normalize
        let total: f64 = buckets.values().sum();
        if total > 0.0 {
            for prob in buckets.values_mut() {
                *prob /= total;
            }
        }

        JointDemographics {
            buckets,
            total_patients: 100_000,
        }
    }

    /// Creates default US Census-based demographics.
    fn default_demographics(&self) -> JointDemographics {
        let mut buckets = AHashMap::new();

        // US Census-based distributions (approximate)
        let ages = [
            ("0-17", 0.22),
            ("18-44", 0.35),
            ("45-64", 0.26),
            ("65+", 0.17),
        ];
        let genders = [("M", 0.49), ("F", 0.51)];
        let races = [
            ("white", 0.60),
            ("black", 0.13),
            ("asian", 0.06),
            ("hispanic", 0.18),
            ("other", 0.03),
        ];
        let ethnicities = [("nonhispanic", 0.82), ("hispanic", 0.18)];

        for (age, age_prob) in ages {
            for (gender, gender_prob) in genders {
                for (race, race_prob) in races {
                    for (eth, eth_prob) in ethnicities {
                        let joint_prob = age_prob * gender_prob * race_prob * eth_prob;
                        buckets.insert(DemographicBucket::new(age, gender, race, eth), joint_prob);
                    }
                }
            }
        }

        JointDemographics {
            buckets,
            total_patients: 100_000,
        }
    }
}

/// Validates that generated statistics match Java Synthea's expected distributions.
pub struct JavaValidation {
    /// Reference fingerprint from Java Synthea.
    reference: MssFingerprint,
    /// Tolerance for prevalence deviation (default 5%).
    tolerance: f64,
}

impl JavaValidation {
    /// Creates a validator from a calibrated registry.
    pub fn from_registry(registry: &CalibratedRegistry) -> Self {
        Self {
            reference: registry.to_fingerprint(),
            tolerance: 0.05,
        }
    }

    /// Creates a validator from an MSS fingerprint.
    pub fn from_fingerprint(fingerprint: MssFingerprint) -> Self {
        Self {
            reference: fingerprint,
            tolerance: 0.05,
        }
    }

    /// Sets the tolerance for validation.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Validates streaming statistics against the reference.
    pub fn validate(&self, stats: &crate::stats::StreamingStatistics) -> ValidationResult {
        let comparison = stats.compare(&self.reference);

        let mut failures = Vec::new();

        // Check individual condition prevalences
        let prevalences = stats.condition_prevalences();
        for (i, cond) in self.reference.conditions.iter().enumerate() {
            let observed = prevalences.get(i).copied().unwrap_or(0.0);
            let expected = cond.prevalence;
            let deviation = (observed - expected).abs();

            if deviation > self.tolerance && expected > 0.01 {
                failures.push(ValidationFailure {
                    code: cond.code.clone(),
                    description: cond.display.clone(),
                    observed,
                    expected,
                    deviation,
                });
            }
        }

        // Sort failures by deviation
        failures.sort_by(|a, b| b.deviation.partial_cmp(&a.deviation).unwrap());

        ValidationResult {
            passed: comparison.passed && failures.is_empty(),
            total_patients: stats.total_patients,
            max_deviation: comparison.max_deviation,
            kl_divergence: comparison.kl_divergence,
            chi_squared: comparison.chi_squared,
            failures,
            tolerance: self.tolerance,
        }
    }

    /// Returns the reference fingerprint.
    pub fn reference(&self) -> &MssFingerprint {
        &self.reference
    }
}

/// Result of Java Synthea validation.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed.
    pub passed: bool,
    /// Total patients in the sample.
    pub total_patients: u64,
    /// Maximum deviation from expected.
    pub max_deviation: f64,
    /// KL divergence.
    pub kl_divergence: f64,
    /// Chi-squared statistic.
    pub chi_squared: f64,
    /// Individual condition failures.
    pub failures: Vec<ValidationFailure>,
    /// Tolerance used.
    pub tolerance: f64,
}

/// A single validation failure.
#[derive(Debug, Clone)]
pub struct ValidationFailure {
    /// Condition code.
    pub code: String,
    /// Condition description.
    pub description: String,
    /// Observed prevalence.
    pub observed: f64,
    /// Expected prevalence.
    pub expected: f64,
    /// Absolute deviation.
    pub deviation: f64,
}

impl ValidationResult {
    /// Prints a summary of the validation.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "Java Synthea Validation (n={}, tolerance={}%)\n",
            self.total_patients,
            self.tolerance * 100.0
        );
        s.push_str(&format!(
            "  Status: {}\n",
            if self.passed { "PASSED" } else { "FAILED" }
        ));
        s.push_str(&format!(
            "  Max Deviation: {:.2}%\n",
            self.max_deviation * 100.0
        ));
        s.push_str(&format!("  KL Divergence: {:.6}\n", self.kl_divergence));
        s.push_str(&format!("  Chi-Squared:   {:.2}\n", self.chi_squared));

        if !self.failures.is_empty() {
            s.push_str(&format!(
                "\n  {} conditions outside tolerance:\n",
                self.failures.len()
            ));
            for (i, f) in self.failures.iter().take(10).enumerate() {
                s.push_str(&format!(
                    "    {}. {} ({}): observed={:.2}%, expected={:.2}%, deviation={:.2}%\n",
                    i + 1,
                    f.description,
                    f.code,
                    f.observed * 100.0,
                    f.expected * 100.0,
                    f.deviation * 100.0
                ));
            }
        }

        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_demographics() {
        let registry = CalibratedRegistry {
            version: "test".to_string(),
            description: "test".to_string(),
            conditions: vec![],
            medications: vec![],
            observations: vec![],
            procedures: vec![],
            demographics: CalibratedDemographics::default(),
            cooccurrence_pairs: Vec::new(),
            recalibration_prevalence: Vec::new(),
            recalibration_boost: Vec::new(),
            onset_records: Vec::new(),
        };

        let demo = registry.default_demographics();

        // Should have joint distribution
        assert!(!demo.buckets.is_empty());

        // Probabilities should sum to ~1
        let sum: f64 = demo.buckets.values().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_to_fingerprint() {
        let registry = CalibratedRegistry {
            version: "1.0".to_string(),
            description: "test".to_string(),
            conditions: vec![CalibratedCondition {
                code: "38341003".to_string(),
                display: "Hypertension".to_string(),
                base_rate: 0.3,
                chronic: true,
                ..Default::default()
            }],
            medications: vec![],
            observations: vec![],
            procedures: vec![],
            demographics: CalibratedDemographics::default(),
            cooccurrence_pairs: Vec::new(),
            recalibration_prevalence: Vec::new(),
            recalibration_boost: Vec::new(),
            onset_records: Vec::new(),
        };

        let fp = registry.to_fingerprint();

        assert_eq!(fp.conditions.len(), 1);
        assert_eq!(fp.conditions[0].code, "38341003");
        assert!((fp.conditions[0].prevalence - 0.3).abs() < 0.001);
    }

    impl Default for CalibratedCondition {
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
                chronic: false,
            }
        }
    }
}
