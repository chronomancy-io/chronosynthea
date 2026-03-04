//! Patient record types for synthetic healthcare data.

use ahash::AHashSet;
use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A synthetic patient with demographic information and medical history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Patient {
    /// Unique patient identifier
    pub id: String,

    /// Patient birth date
    pub birth_date: NaiveDate,

    /// Biological sex ("M" or "F")
    pub sex: Sex,

    /// Race category
    pub race: Race,

    /// Ethnicity category
    pub ethnicity: Ethnicity,

    /// Healthcare encounters
    pub encounters: Vec<Encounter>,
}

/// Biological sex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sex {
    #[serde(rename = "M")]
    Male,
    #[serde(rename = "F")]
    Female,
}

impl Sex {
    /// Returns the short code ("M" or "F").
    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            Sex::Male => "M",
            Sex::Female => "F",
        }
    }
}

/// Race category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Race {
    White,
    Black,
    Asian,
    Hispanic,
    Native,
    Other,
}

impl Race {
    /// Returns the string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Race::White => "white",
            Race::Black => "black",
            Race::Asian => "asian",
            Race::Hispanic => "hispanic",
            Race::Native => "native",
            Race::Other => "other",
        }
    }
}

/// Ethnicity category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Ethnicity {
    Hispanic,
    #[serde(rename = "nonhispanic")]
    NonHispanic,
    Unknown,
}

impl Ethnicity {
    /// Returns the string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Ethnicity::Hispanic => "hispanic",
            Ethnicity::NonHispanic => "nonhispanic",
            Ethnicity::Unknown => "unknown",
        }
    }
}

/// A healthcare encounter for a patient.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Encounter {
    /// Encounter timestamp
    pub timestamp: DateTime<Utc>,

    /// Encounter type (ambulatory, emergency, inpatient, wellness)
    pub encounter_type: EncounterType,

    /// SNOMED encounter class code
    pub class: String,

    /// Events within this encounter
    pub events: Vec<Event>,
}

/// Type of healthcare encounter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EncounterType {
    Wellness,
    Outpatient,
    Ambulatory,
    Emergency,
    Inpatient,
    Urgentcare,
}

impl EncounterType {
    /// Returns the short code for compact format.
    #[inline]
    pub fn short_code(&self) -> char {
        match self {
            EncounterType::Wellness => 'w',
            EncounterType::Outpatient => 'o',
            EncounterType::Ambulatory => 'a',
            EncounterType::Emergency => 'e',
            EncounterType::Inpatient => 'i',
            EncounterType::Urgentcare => 'u',
        }
    }

    /// Returns the string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            EncounterType::Wellness => "wellness",
            EncounterType::Outpatient => "outpatient",
            EncounterType::Ambulatory => "ambulatory",
            EncounterType::Emergency => "emergency",
            EncounterType::Inpatient => "inpatient",
            EncounterType::Urgentcare => "urgentcare",
        }
    }
}

/// A medical event within an encounter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Event type
    pub event_type: EventType,

    /// Medical code (SNOMED, RxNorm, LOINC, etc.)
    #[serde(
        serialize_with = "serialize_arc_str",
        deserialize_with = "deserialize_arc_str"
    )]
    pub code: Arc<str>,

    /// Coding system
    #[serde(
        serialize_with = "serialize_arc_str",
        deserialize_with = "deserialize_arc_str"
    )]
    pub system: Arc<str>,

    /// Human-readable description
    #[serde(
        serialize_with = "serialize_arc_str",
        deserialize_with = "deserialize_arc_str"
    )]
    pub description: Arc<str>,

    /// Event timestamp
    pub timestamp: DateTime<Utc>,
}

fn serialize_arc_str<S>(s: &Arc<str>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(s)
}

fn deserialize_arc_str<'de, D>(deserializer: D) -> Result<Arc<str>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    Ok(Arc::from(s))
}

/// Static system strings for zero-allocation event creation.
pub mod systems {
    use std::sync::Arc;
    use std::sync::LazyLock;

    pub static SNOMED_CT: LazyLock<Arc<str>> = LazyLock::new(|| Arc::from("SNOMED-CT"));
    pub static RXNORM: LazyLock<Arc<str>> = LazyLock::new(|| Arc::from("RxNorm"));
    pub static LOINC: LazyLock<Arc<str>> = LazyLock::new(|| Arc::from("LOINC"));
    pub static CPT: LazyLock<Arc<str>> = LazyLock::new(|| Arc::from("CPT"));
}

/// Type of medical event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EventType {
    Diagnosis,
    Procedure,
    Medication,
    Observation,
    Immunization,
}

impl EventType {
    /// Returns the short code for compact format.
    #[inline]
    pub fn short_code(&self) -> char {
        match self {
            EventType::Diagnosis => 'd',
            EventType::Procedure => 'p',
            EventType::Medication => 'm',
            EventType::Observation => 'o',
            EventType::Immunization => 'i',
        }
    }

    /// Returns the string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            EventType::Diagnosis => "diagnosis",
            EventType::Procedure => "procedure",
            EventType::Medication => "medication",
            EventType::Observation => "observation",
            EventType::Immunization => "immunization",
        }
    }
}

// === Compact Output Formats ===

/// Compact patient structure with minimal fields (~2x size reduction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactPatient {
    /// Patient ID
    pub id: String,
    /// Birth date as YYYY-MM-DD
    pub bd: String,
    /// Sex (M/F)
    pub s: String,
    /// Race
    pub r: String,
    /// Encounters
    pub e: Vec<CompactEncounter>,
}

/// Compact encounter structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactEncounter {
    /// Timestamp as ISO date
    pub t: String,
    /// Type (w=wellness, o=outpatient, e=emergency, i=inpatient)
    pub ty: String,
    /// Events
    pub ev: Vec<CompactEvent>,
}

/// Compact event structure (code only).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactEvent {
    /// Type (d=diagnosis, m=medication, o=observation, p=procedure)
    pub t: String,
    /// Code
    pub c: String,
}

/// Most minimal representation for analytics (~15x size reduction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodesOnlyPatient {
    /// Patient ID
    pub id: String,
    /// Diagnosis codes
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub dx: Vec<String>,
    /// Medication codes
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub rx: Vec<String>,
    /// Observation codes
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub ob: Vec<String>,
    /// Procedure codes
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub px: Vec<String>,
}

impl Patient {
    /// Converts to compact format (~2x size reduction).
    pub fn to_compact(&self) -> CompactPatient {
        CompactPatient {
            id: self.id.clone(),
            bd: self.birth_date.format("%Y-%m-%d").to_string(),
            s: self.sex.as_str().to_string(),
            r: self.race.as_str().to_string(),
            e: self
                .encounters
                .iter()
                .map(|enc| CompactEncounter {
                    t: enc.timestamp.format("%Y-%m-%d").to_string(),
                    ty: enc.encounter_type.short_code().to_string(),
                    ev: enc
                        .events
                        .iter()
                        .map(|ev| CompactEvent {
                            t: ev.event_type.short_code().to_string(),
                            c: ev.code.to_string(),
                        })
                        .collect(),
                })
                .collect(),
        }
    }

    /// Converts to codes-only format (~15x size reduction).
    ///
    /// Collects unique codes by type across all encounters.
    /// Convert to codes-only format (optimized for minimal allocations).
    pub fn to_codes_only(&self) -> CodesOnlyPatient {
        // Use &str for comparison, only allocate String at the end
        let mut dx_set: AHashSet<&str> = AHashSet::with_capacity(32);
        let mut rx_set: AHashSet<&str> = AHashSet::with_capacity(32);
        let mut ob_set: AHashSet<&str> = AHashSet::with_capacity(32);
        let mut px_set: AHashSet<&str> = AHashSet::with_capacity(32);

        for enc in &self.encounters {
            for ev in &enc.events {
                match ev.event_type {
                    EventType::Diagnosis => {
                        dx_set.insert(&ev.code);
                    }
                    EventType::Medication => {
                        rx_set.insert(&ev.code);
                    }
                    EventType::Observation => {
                        ob_set.insert(&ev.code);
                    }
                    EventType::Procedure | EventType::Immunization => {
                        px_set.insert(&ev.code);
                    }
                }
            }
        }

        CodesOnlyPatient {
            id: self.id.clone(),
            dx: dx_set.into_iter().map(String::from).collect(),
            rx: rx_set.into_iter().map(String::from).collect(),
            ob: ob_set.into_iter().map(String::from).collect(),
            px: px_set.into_iter().map(String::from).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn create_test_patient() -> Patient {
        Patient {
            id: "test-001".to_string(),
            birth_date: NaiveDate::from_ymd_opt(1990, 5, 15).unwrap(),
            sex: Sex::Male,
            race: Race::White,
            ethnicity: Ethnicity::NonHispanic,
            encounters: vec![Encounter {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 15, 10, 30, 0).unwrap(),
                encounter_type: EncounterType::Wellness,
                class: "AMB".to_string(),
                events: vec![
                    Event {
                        event_type: EventType::Diagnosis,
                        code: Arc::from("38341003"),
                        system: Arc::from("SNOMED-CT"),
                        description: Arc::from("Hypertension"),
                        timestamp: Utc.with_ymd_and_hms(2024, 1, 15, 10, 35, 0).unwrap(),
                    },
                    Event {
                        event_type: EventType::Medication,
                        code: Arc::from("197361"),
                        system: Arc::from("RxNorm"),
                        description: Arc::from("Lisinopril 10 MG"),
                        timestamp: Utc.with_ymd_and_hms(2024, 1, 15, 10, 40, 0).unwrap(),
                    },
                ],
            }],
        }
    }

    #[test]
    fn test_patient_to_compact() {
        let patient = create_test_patient();
        let compact = patient.to_compact();

        assert_eq!(compact.id, "test-001");
        assert_eq!(compact.bd, "1990-05-15");
        assert_eq!(compact.s, "M");
        assert_eq!(compact.r, "white");
        assert_eq!(compact.e.len(), 1);
        assert_eq!(compact.e[0].ev.len(), 2);
    }

    #[test]
    fn test_patient_to_codes_only() {
        let patient = create_test_patient();
        let codes = patient.to_codes_only();

        assert_eq!(codes.id, "test-001");
        assert_eq!(codes.dx.len(), 1);
        assert_eq!(codes.rx.len(), 1);
        assert!(codes.ob.is_empty());
        assert!(codes.px.is_empty());
    }
}
