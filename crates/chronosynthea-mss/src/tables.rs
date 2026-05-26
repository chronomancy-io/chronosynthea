//! Static lookup tables for zero-allocation code resolution.
//!
//! These tables are built from the MSS fingerprint at compile time (via build.rs)
//! or loaded at runtime. Using u16 indices instead of Arc<str> eliminates
//! atomic reference counting overhead.

use std::sync::OnceLock;

use ahash::AHashMap;

/// Pre-escape a display string for embedding inside CSV's `"…"` quoted
/// fields: every internal `"` becomes `\"`. Computed once at table-load
/// time so per-event writeln!() sites can emit it verbatim.
fn escape_csv_display(s: &str) -> String {
    // Fast path: most display strings have no embedded quote and the
    // escaped form is byte-identical to the original.
    if !s.contains('"') {
        return s.to_owned();
    }
    s.replace('"', "\\\"")
}

/// Imaging-keyword detector for procedure displays. Java Synthea's
/// state machines flag a procedure as imaging when its display contains
/// any of these substrings; we mirror that once at load so the
/// CSV-write loop doesn't have to `to_ascii_lowercase()` per event.
fn contains_imaging_keyword(display: &str) -> bool {
    let lower = display.to_ascii_lowercase();
    const IMAGING_KEYWORDS: &[&str] = &[
        "x-ray",
        "radiograph",
        "ct ",
        "ct scan",
        "mri",
        "magnetic resonance",
        "ultrasound",
        "ultrasonograph",
        "scan",
        "mammogr",
        "angiogra",
        "angiogram",
        "scintigraph",
        "ecg",
        "echocardio",
        "electrocardio",
        "dexa",
        "imaging",
    ];
    IMAGING_KEYWORDS.iter().any(|kw| lower.contains(kw))
}

/// System codes (indexed by u8).
pub const SYSTEMS: &[&str] = &[
    "SNOMED-CT", // 0
    "RxNorm",    // 1
    "LOINC",     // 2
    "CPT",       // 3
    "ICD-10-CM", // 4
    "CVX",       // 5
];

/// Maps system string to index.
#[inline]
pub fn system_to_idx(system: &str) -> u8 {
    match system {
        "SNOMED-CT" | "http://snomed.info/sct" => 0,
        "RxNorm" | "http://www.nlm.nih.gov/research/umls/rxnorm" => 1,
        "LOINC" | "http://loinc.org" => 2,
        "CPT" | "http://www.ama-assn.org/go/cpt" => 3,
        "ICD-10-CM" | "http://hl7.org/fhir/sid/icd-10-cm" => 4,
        "CVX" | "http://hl7.org/fhir/sid/cvx" => 5,
        _ => 0, // Default to SNOMED-CT
    }
}

/// Encounter type indices.
pub const ENCOUNTER_TYPES: &[&str] = &[
    "wellness",   // 0
    "ambulatory", // 1
    "outpatient", // 2
    "emergency",  // 3
    "inpatient",  // 4
    "urgentcare", // 5
];

/// Maps encounter type to index.
#[inline]
pub fn encounter_type_to_idx(enc_type: &str) -> u8 {
    match enc_type {
        "wellness" => 0,
        "ambulatory" => 1,
        "outpatient" => 2,
        "emergency" => 3,
        "inpatient" => 4,
        "urgentcare" => 5,
        _ => 1, // Default to ambulatory
    }
}

/// Event type indices.
pub const EVENT_TYPES: &[&str] = &[
    "diagnosis",    // 0
    "medication",   // 1
    "procedure",    // 2
    "observation",  // 3
    "immunization", // 4
];

/// Maps event type to index.
#[inline]
pub fn event_type_to_idx(event_type: &str) -> u8 {
    match event_type {
        "diagnosis" => 0,
        "medication" => 1,
        "procedure" => 2,
        "observation" => 3,
        "immunization" => 4,
        _ => 0,
    }
}

/// Race indices.
pub const RACES: &[&str] = &[
    "white",    // 0
    "black",    // 1
    "asian",    // 2
    "hispanic", // 3
    "native",   // 4
    "other",    // 5
];

/// Maps race to index.
#[inline]
pub fn race_to_idx(race: &str) -> u8 {
    match race {
        "white" => 0,
        "black" => 1,
        "asian" => 2,
        "hispanic" => 3,
        "native" => 4,
        _ => 5,
    }
}

/// Ethnicity indices.
pub const ETHNICITIES: &[&str] = &[
    "nonhispanic", // 0
    "hispanic",    // 1
    "unknown",     // 2
];

/// Maps ethnicity to index.
#[inline]
pub fn ethnicity_to_idx(ethnicity: &str) -> u8 {
    match ethnicity {
        "nonhispanic" => 0,
        "hispanic" => 1,
        _ => 2,
    }
}

/// Age bucket indices.
pub const AGE_BUCKETS: &[&str] = &[
    "0-17",  // 0
    "18-44", // 1
    "45-64", // 2
    "65+",   // 3
];

/// Maps age bucket to index.
#[inline]
pub fn age_bucket_to_idx(bucket: &str) -> u8 {
    match bucket {
        "0-17" => 0,
        "18-44" => 1,
        "45-64" => 2,
        "65+" => 3,
        _ => 1,
    }
}

/// Maps age to bucket index.
#[inline]
pub fn age_to_bucket_idx(age: u32) -> u8 {
    match age {
        0..=17 => 0,
        18..=44 => 1,
        45..=64 => 2,
        _ => 3,
    }
}

/// Dynamic code table (loaded from MSS fingerprint).
pub struct CodeTable {
    /// Condition codes.
    pub conditions: Vec<CodeEntry>,
    /// Medication codes.
    pub medications: Vec<CodeEntry>,
    /// Observation codes.
    pub observations: Vec<CodeEntry>,
    /// Procedure codes.
    pub procedures: Vec<CodeEntry>,

    /// Reverse lookup: code string -> index.
    pub condition_index: AHashMap<String, u16>,
    pub medication_index: AHashMap<String, u16>,
    pub observation_index: AHashMap<String, u16>,
    pub procedure_index: AHashMap<String, u16>,
}

/// A single code entry.
///
/// `display_escaped` and `is_imaging_hint` are precomputed at load time
/// so the CSV-write hot path can skip per-event `replace('"', "\\\"")`
/// (an allocation per event) and the per-event
/// `display.to_ascii_lowercase().contains(...)` imaging-keyword scan.
#[derive(Debug, Clone)]
pub struct CodeEntry {
    /// The code string (owned for table storage).
    pub code: String,
    /// The display string.
    pub display: String,
    /// `display` with embedded `"` characters replaced by `\"` —
    /// pre-baked once at load so the per-event writeln! sites can
    /// emit it verbatim instead of allocating a quote-escaped copy.
    pub display_escaped: String,
    /// True when `display` matches one of Java Synthea's imaging
    /// keywords (x-ray, ct scan, mri, ultrasound, etc.). Lets the
    /// procedure-loop skip the per-event `to_ascii_lowercase` + contains
    /// chain that used to fire on every procedure event.
    pub is_imaging_hint: bool,
    /// System index.
    pub system_idx: u8,
    /// Per-encounter frequency (for sampling).
    pub frequency: f32,
}

impl CodeTable {
    /// Creates a new empty code table.
    pub fn new() -> Self {
        Self {
            conditions: Vec::new(),
            medications: Vec::new(),
            observations: Vec::new(),
            procedures: Vec::new(),
            condition_index: AHashMap::new(),
            medication_index: AHashMap::new(),
            observation_index: AHashMap::new(),
            procedure_index: AHashMap::new(),
        }
    }

    /// Loads a code table from an MSS fingerprint.
    pub fn from_fingerprint(fp: &crate::fingerprint::MssFingerprint) -> Self {
        let mut table = Self::new();

        // Load conditions
        for (i, cond) in fp.conditions.iter().enumerate() {
            table.condition_index.insert(cond.code.clone(), i as u16);
            table.conditions.push(CodeEntry {
                display_escaped: escape_csv_display(&cond.display),
                is_imaging_hint: false, // not used for conditions
                code: cond.code.clone(),
                display: cond.display.clone(),
                system_idx: 0, // SNOMED-CT
                frequency: cond.prevalence as f32,
            });
        }

        // Load medications
        for (i, med) in fp.medications.iter().enumerate() {
            table.medication_index.insert(med.code.clone(), i as u16);
            table.medications.push(CodeEntry {
                display_escaped: escape_csv_display(&med.display),
                is_imaging_hint: false,
                code: med.code.clone(),
                display: med.display.clone(),
                system_idx: 1, // RxNorm
                frequency: med.frequency as f32,
            });
        }

        // Load observations
        for (i, obs) in fp.observations.iter().enumerate() {
            table.observation_index.insert(obs.code.clone(), i as u16);
            table.observations.push(CodeEntry {
                display_escaped: escape_csv_display(&obs.display),
                is_imaging_hint: false,
                code: obs.code.clone(),
                display: obs.display.clone(),
                system_idx: system_to_idx(&obs.system),
                frequency: obs.frequency as f32,
            });
        }

        // Load procedures (compute `is_imaging_hint` here — only used by
        // the procedure-loop's imaging-study emission).
        for (i, proc) in fp.procedures.iter().enumerate() {
            table.procedure_index.insert(proc.code.clone(), i as u16);
            table.procedures.push(CodeEntry {
                display_escaped: escape_csv_display(&proc.display),
                is_imaging_hint: contains_imaging_keyword(&proc.display),
                code: proc.code.clone(),
                display: proc.display.clone(),
                system_idx: system_to_idx(&proc.system),
                frequency: proc.frequency as f32,
            });
        }

        table
    }

    /// Looks up a condition by index.
    #[inline]
    pub fn condition(&self, idx: u16) -> Option<&CodeEntry> {
        self.conditions.get(idx as usize)
    }

    /// Looks up a medication by index.
    #[inline]
    pub fn medication(&self, idx: u16) -> Option<&CodeEntry> {
        self.medications.get(idx as usize)
    }

    /// Looks up an observation by index.
    #[inline]
    pub fn observation(&self, idx: u16) -> Option<&CodeEntry> {
        self.observations.get(idx as usize)
    }

    /// Looks up a procedure by index.
    #[inline]
    pub fn procedure(&self, idx: u16) -> Option<&CodeEntry> {
        self.procedures.get(idx as usize)
    }

    /// Returns total number of codes.
    pub fn total_codes(&self) -> usize {
        self.conditions.len()
            + self.medications.len()
            + self.observations.len()
            + self.procedures.len()
    }

    /// Number of condition codes.
    pub fn num_conditions(&self) -> usize {
        self.conditions.len()
    }

    /// Number of medication codes.
    pub fn num_medications(&self) -> usize {
        self.medications.len()
    }

    /// Number of observation codes.
    pub fn num_observations(&self) -> usize {
        self.observations.len()
    }

    /// Number of procedure codes.
    pub fn num_procedures(&self) -> usize {
        self.procedures.len()
    }
}

impl Default for CodeTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Global code table (initialized once from fingerprint).
static GLOBAL_CODE_TABLE: OnceLock<CodeTable> = OnceLock::new();

/// Initializes the global code table from a fingerprint.
pub fn init_global_table(fp: &crate::fingerprint::MssFingerprint) {
    let _ = GLOBAL_CODE_TABLE.get_or_init(|| CodeTable::from_fingerprint(fp));
}

/// Gets the global code table.
pub fn global_table() -> Option<&'static CodeTable> {
    GLOBAL_CODE_TABLE.get()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_mapping() {
        assert_eq!(system_to_idx("SNOMED-CT"), 0);
        assert_eq!(system_to_idx("RxNorm"), 1);
        assert_eq!(system_to_idx("LOINC"), 2);
        assert_eq!(system_to_idx("http://snomed.info/sct"), 0);
    }

    #[test]
    fn test_age_bucket_mapping() {
        assert_eq!(age_to_bucket_idx(5), 0);
        assert_eq!(age_to_bucket_idx(25), 1);
        assert_eq!(age_to_bucket_idx(50), 2);
        assert_eq!(age_to_bucket_idx(70), 3);
    }

    #[test]
    fn test_empty_code_table() {
        let table = CodeTable::new();
        assert_eq!(table.total_codes(), 0);
        assert!(table.condition(0).is_none());
    }
}
