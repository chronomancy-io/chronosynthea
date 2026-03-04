//! MSS Fingerprint - Binary format for the Minimally Sufficient Statistic.
//!
//! WASP Role: I(c) — Primary index structure for the Clinical Trajectory dimension.
//! CDE Phase: Phase 3 (Index Construction) — pre-computed from Java Synthea output.
//!
//! The fingerprint captures all statistical properties needed to generate
//! patients that are statistically equivalent to Java Synthea output.
//!
//! WASP Guarantee: Sufficiency — the fingerprint is the MSS for patient generation;
//! validated by `java_validation.rs` (max deviation < 0.31%, KL < 0.01).

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use ahash::AHashMap;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};

use crate::error::{MssError, MssResult};

/// Magic bytes for MSS fingerprint files.
const MSS_MAGIC: &[u8; 4] = b"MSS1";

/// Minimally Sufficient Statistic fingerprint.
///
/// Contains all statistical distributions needed to generate synthetic patients
/// that match Java Synthea's output distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MssFingerprint {
    /// Format version.
    pub version: String,

    /// Source identifier (e.g., "java-synthea-3.2.0").
    pub source: String,

    /// Total patients in the reference population.
    pub total_patients: u64,

    /// Total encounters in the reference population.
    pub total_encounters: u64,

    /// Joint demographic distribution.
    pub joint_demographics: JointDemographics,

    /// Condition statistics.
    pub conditions: Vec<ConditionStats>,

    /// Medication statistics.
    pub medications: Vec<MedicationStats>,

    /// Observation statistics.
    pub observations: Vec<ObservationStats>,

    /// Procedure statistics.
    pub procedures: Vec<ProcedureStats>,

    /// Code co-occurrence probabilities (sparse).
    pub cooccurrence: AHashMap<(String, String), f64>,

    /// Encounter statistics.
    pub encounter_stats: EncounterStats,
}

/// Joint distribution over demographic buckets.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JointDemographics {
    /// Probability for each demographic bucket.
    pub buckets: AHashMap<DemographicBucket, f64>,

    /// Total patients used to compute this distribution.
    pub total_patients: u64,
}

/// A demographic bucket key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DemographicBucket {
    pub age_bucket: String,
    pub gender: String,
    pub race: String,
    pub ethnicity: String,
}

impl DemographicBucket {
    /// Creates a new demographic bucket.
    pub fn new(age_bucket: &str, gender: &str, race: &str, ethnicity: &str) -> Self {
        Self {
            age_bucket: age_bucket.to_string(),
            gender: gender.to_string(),
            race: race.to_string(),
            ethnicity: ethnicity.to_string(),
        }
    }

    /// Computes a compact index for this bucket (for archetype lookup).
    pub fn to_index(&self) -> u16 {
        let age_idx: u16 = match self.age_bucket.as_str() {
            "0-17" => 0,
            "18-44" => 1,
            "45-64" => 2,
            "65+" => 3,
            _ => 1,
        };

        let gender_idx: u16 = match self.gender.as_str() {
            "male" | "M" => 0,
            "female" | "F" => 1,
            _ => 0,
        };

        let race_idx: u16 = match self.race.as_str() {
            "white" => 0,
            "black" => 1,
            "asian" => 2,
            "hispanic" => 3,
            "native" => 4,
            _ => 5,
        };

        let eth_idx: u16 = match self.ethnicity.as_str() {
            "hispanic" => 1,
            _ => 0,
        };

        // Encode as: age(2 bits) | gender(1 bit) | race(3 bits) | ethnicity(1 bit)
        (age_idx << 5) | (gender_idx << 4) | (race_idx << 1) | eth_idx
    }

    /// Creates from a compact index.
    pub fn from_index(idx: u16) -> Self {
        let age_bucket = match (idx >> 5) & 0x3 {
            0 => "0-17",
            1 => "18-44",
            2 => "45-64",
            _ => "65+",
        }
        .to_string();

        let gender = if (idx >> 4) & 0x1 == 0 {
            "male"
        } else {
            "female"
        }
        .to_string();

        let race = match (idx >> 1) & 0x7 {
            0 => "white",
            1 => "black",
            2 => "asian",
            3 => "hispanic",
            4 => "native",
            _ => "other",
        }
        .to_string();

        let ethnicity = if idx & 0x1 == 1 {
            "hispanic"
        } else {
            "nonhispanic"
        }
        .to_string();

        Self {
            age_bucket,
            gender,
            race,
            ethnicity,
        }
    }
}

/// Statistics for a condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionStats {
    /// SNOMED code.
    pub code: String,

    /// Display name.
    pub display: String,

    /// Base prevalence (probability per patient).
    pub prevalence: f64,

    /// Prevalence multiplier by age bucket.
    pub by_age_bucket: AHashMap<String, f64>,

    /// Prevalence multiplier by gender.
    pub by_gender: AHashMap<String, f64>,

    /// Prevalence multiplier by race.
    pub by_race: AHashMap<String, f64>,

    /// Whether the condition is chronic.
    pub chronic: bool,

    /// Mean onset age.
    pub mean_onset_age: f64,
}

impl ConditionStats {
    /// Computes the adjusted prevalence for a demographic bucket.
    pub fn prevalence_for(&self, bucket: &DemographicBucket) -> f64 {
        let mut rate = self.prevalence;

        if let Some(&mult) = self.by_age_bucket.get(&bucket.age_bucket) {
            rate *= mult;
        }
        if let Some(&mult) = self.by_gender.get(&bucket.gender) {
            rate *= mult;
        }
        if let Some(&mult) = self.by_race.get(&bucket.race) {
            rate *= mult;
        }

        rate.clamp(0.0, 1.0)
    }
}

/// Statistics for a medication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedicationStats {
    /// RxNorm code.
    pub code: String,

    /// Display name.
    pub display: String,

    /// Per-encounter frequency.
    pub frequency: f64,

    /// Indication condition codes.
    pub indications: Vec<String>,
}

/// Statistics for an observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationStats {
    /// LOINC or other code.
    pub code: String,

    /// Coding system.
    pub system: String,

    /// Display name.
    pub display: String,

    /// Per-encounter frequency.
    pub frequency: f64,
}

/// Statistics for a procedure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureStats {
    /// SNOMED code.
    pub code: String,

    /// Coding system.
    pub system: String,

    /// Display name.
    pub display: String,

    /// Per-encounter frequency.
    pub frequency: f64,

    /// Indication condition codes.
    pub indications: Vec<String>,
}

/// Encounter distribution statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EncounterStats {
    /// Mean encounters per patient by age bucket.
    pub mean_by_age: AHashMap<String, f64>,

    /// Distribution of encounter types.
    pub type_distribution: AHashMap<String, f64>,

    /// Mean events per encounter.
    pub mean_events_per_encounter: f64,
}

impl MssFingerprint {
    /// Saves the fingerprint to a binary file (MessagePack + gzip).
    pub fn save<P: AsRef<Path>>(&self, path: P) -> MssResult<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write magic bytes
        writer.write_all(MSS_MAGIC)?;

        // Compress and serialize
        let encoder = GzEncoder::new(writer, Compression::fast());
        rmp_serde::encode::write(&mut BufWriter::new(encoder), self)?;

        Ok(())
    }

    /// Loads a fingerprint from a binary file.
    pub fn load<P: AsRef<Path>>(path: P) -> MssResult<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Verify magic bytes
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;

        if &magic != MSS_MAGIC {
            return Err(MssError::InvalidFormat(
                "Invalid MSS fingerprint magic bytes".to_string(),
            ));
        }

        // Decompress and deserialize
        let decoder = GzDecoder::new(reader);
        let fingerprint: Self = rmp_serde::decode::from_read(BufReader::new(decoder))?;

        Ok(fingerprint)
    }

    /// Saves as JSON (for debugging/inspection).
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> MssResult<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Loads from JSON.
    pub fn load_json<P: AsRef<Path>>(path: P) -> MssResult<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let fingerprint: Self = serde_json::from_reader(reader)?;
        Ok(fingerprint)
    }

    /// Returns the number of unique condition codes.
    pub fn condition_count(&self) -> usize {
        self.conditions.len()
    }

    /// Returns the number of unique medication codes.
    pub fn medication_count(&self) -> usize {
        self.medications.len()
    }

    /// Returns the number of unique observation codes.
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Returns the number of unique procedure codes.
    pub fn procedure_count(&self) -> usize {
        self.procedures.len()
    }

    /// Builds a lookup map from condition code to index.
    pub fn condition_index(&self) -> AHashMap<String, u16> {
        self.conditions
            .iter()
            .enumerate()
            .map(|(i, c)| (c.code.clone(), i as u16))
            .collect()
    }

    /// Validates the fingerprint for consistency.
    pub fn validate(&self) -> MssResult<()> {
        // Check demographic probabilities sum to ~1.0
        let demo_sum: f64 = self.joint_demographics.buckets.values().sum();
        if (demo_sum - 1.0).abs() > 0.01 && demo_sum > 0.0 {
            return Err(MssError::ValidationFailed(format!(
                "Demographic probabilities sum to {}, expected 1.0",
                demo_sum
            )));
        }

        // Check all prevalences are in [0, 1]
        for cond in &self.conditions {
            if cond.prevalence < 0.0 || cond.prevalence > 1.0 {
                return Err(MssError::ValidationFailed(format!(
                    "Condition {} has invalid prevalence: {}",
                    cond.code, cond.prevalence
                )));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_demographic_bucket_index_roundtrip() {
        let bucket = DemographicBucket::new("45-64", "female", "asian", "nonhispanic");
        let idx = bucket.to_index();
        let recovered = DemographicBucket::from_index(idx);

        assert_eq!(bucket.age_bucket, recovered.age_bucket);
        assert_eq!(recovered.gender, "female");
        assert_eq!(recovered.race, "asian");
        assert_eq!(recovered.ethnicity, "nonhispanic");
    }

    #[test]
    fn test_fingerprint_json_roundtrip() {
        let fp = MssFingerprint {
            version: "1.0".to_string(),
            source: "test".to_string(),
            total_patients: 1000,
            total_encounters: 5000,
            joint_demographics: JointDemographics::default(),
            conditions: vec![],
            medications: vec![],
            observations: vec![],
            procedures: vec![],
            cooccurrence: AHashMap::new(),
            encounter_stats: EncounterStats::default(),
        };

        let json = serde_json::to_string(&fp).unwrap();
        let recovered: MssFingerprint = serde_json::from_str(&json).unwrap();

        assert_eq!(fp.version, recovered.version);
        assert_eq!(fp.total_patients, recovered.total_patients);
    }
}
