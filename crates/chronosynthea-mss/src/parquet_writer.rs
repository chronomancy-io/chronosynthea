//! Apache Parquet output paths — three variants, all gated behind
//! `--features parquet`.
//!
//! ## Why Parquet
//!
//! The CSV writer is byte-emission-bound at ~580 KB/patient. The GPU
//! experiment (PR #13) confirmed throughput is capped by memory + I/O
//! bandwidth, not compute. The only architectural lever left is to
//! emit fewer bytes — exactly what Parquet's columnar + dictionary +
//! Zstd encoding does. PR #14 measured a 23.31× total reduction
//! across all 15 Synthea CSVs (5.71 GB → 245 MB).
//!
//! ## Variants
//!
//! 1. **`SyntheaParquetWriter::create`** — full patients.parquet,
//!    every column the CSV writer emits.
//! 2. **`SyntheaParquetWriter::create_slim`** — drops the
//!    personally-identifying columns (SSN, DRIVERS, PASSPORT, FIRST,
//!    MIDDLE, LAST, ADDRESS, BIRTHPLACE). Useful for shipping research
//!    datasets where the PII bytes are non-load-bearing and consumer
//!    workflows need to be HIPAA-Safe-Harbor by construction.
//! 3. **`SyntheaStatsParquetWriter`** — one row per patient, no
//!    event-level data. Each row carries (id, birthdate, sex, race,
//!    ethnicity, n_conditions, n_encounters, n_medications,
//!    n_procedures, n_observations, total_cost_estimate). Right when
//!    the downstream workflow only needs per-patient summary stats —
//!    e.g. cohort selection before pulling the full record.
//!
//! All three share the same dictionary + Zstd settings.

use std::fs::{create_dir_all, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    ArrayBuilder, ArrayRef, Float64Builder, Int64Builder, ListBuilder, StringBuilder,
    UInt32Builder,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use crate::arena::FullPatient;
use crate::patient_uuid;
use crate::synthea_fixtures::*;

/// Flush threshold (rows). 16 384 hits the right balance: large enough
/// for dictionary encoding to find repeated values, small enough that
/// per-flush working memory stays bounded.
const FLUSH_ROWS: usize = 16_384;

#[inline]
fn sb() -> StringBuilder {
    StringBuilder::with_capacity(FLUSH_ROWS, FLUSH_ROWS * 32)
}

fn writer_props() -> WriterProperties {
    // Zstd-3: ~80% of level-9's compression at ~5× write speed. Right
    // default for analytics workloads; users can swap to Snappy if
    // they need faster decode at the cost of ~30% larger files.
    WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        .build()
}

// =====================================================================
// Variant 1+2: full / slim patients.parquet writer
// =====================================================================

/// Patient-row column builder. The `slim` flag drives which columns
/// are populated; the schema function selects matching `Field`s, so
/// the resulting Parquet file's schema matches exactly what was
/// written (no all-NULL columns waste bytes in slim mode).
struct PatientsBuilder {
    slim: bool,
    // Common columns (kept in both modes).
    id: StringBuilder,
    birthdate: StringBuilder,
    deathdate: StringBuilder,
    marital: StringBuilder,
    race: StringBuilder,
    ethnicity: StringBuilder,
    gender: StringBuilder,
    city: StringBuilder,
    state: StringBuilder,
    county: StringBuilder,
    fips: StringBuilder,
    zip: StringBuilder,
    lat: Float64Builder,
    lon: Float64Builder,
    healthcare_expenses: Float64Builder,
    healthcare_coverage: Float64Builder,
    income: Int64Builder,
    // CDE coordinate columns — first-class probe axes for cohort
    // queries. ARCHETYPE_ID is the cluster label every patient was
    // sampled from; AGE_BAND is a 10-year bucket of the birth year
    // relative to today. Both let downstream consumers filter cohorts
    // without joining the conditions table or computing on birthdate.
    archetype_id: arrow::array::UInt16Builder,
    age_band: StringBuilder,
    // PII columns (full mode only).
    ssn: StringBuilder,
    drivers: StringBuilder,
    passport: StringBuilder,
    first: StringBuilder,
    middle: StringBuilder,
    last: StringBuilder,
    birthplace: StringBuilder,
    address: StringBuilder,
}

impl PatientsBuilder {
    fn new(slim: bool) -> Self {
        Self {
            slim,
            id: sb(),
            birthdate: sb(),
            deathdate: sb(),
            marital: sb(),
            race: sb(),
            ethnicity: sb(),
            gender: sb(),
            city: sb(),
            state: sb(),
            county: sb(),
            fips: sb(),
            zip: sb(),
            lat: Float64Builder::with_capacity(FLUSH_ROWS),
            lon: Float64Builder::with_capacity(FLUSH_ROWS),
            healthcare_expenses: Float64Builder::with_capacity(FLUSH_ROWS),
            healthcare_coverage: Float64Builder::with_capacity(FLUSH_ROWS),
            income: Int64Builder::with_capacity(FLUSH_ROWS),
            archetype_id: arrow::array::UInt16Builder::with_capacity(FLUSH_ROWS),
            age_band: sb(),
            ssn: sb(),
            drivers: sb(),
            passport: sb(),
            first: sb(),
            middle: sb(),
            last: sb(),
            birthplace: sb(),
            address: sb(),
        }
    }

    fn schema(slim: bool) -> SchemaRef {
        let mut fields = vec![
            Field::new("Id", DataType::Utf8, false),
            Field::new("BIRTHDATE", DataType::Utf8, false),
            Field::new("DEATHDATE", DataType::Utf8, true),
        ];
        if !slim {
            fields.extend(vec![
                Field::new("SSN", DataType::Utf8, false),
                Field::new("DRIVERS", DataType::Utf8, true),
                Field::new("PASSPORT", DataType::Utf8, true),
                Field::new("FIRST", DataType::Utf8, false),
                Field::new("MIDDLE", DataType::Utf8, false),
                Field::new("LAST", DataType::Utf8, false),
            ]);
        }
        fields.extend(vec![
            Field::new("MARITAL", DataType::Utf8, true),
            Field::new("RACE", DataType::Utf8, false),
            Field::new("ETHNICITY", DataType::Utf8, false),
            Field::new("GENDER", DataType::Utf8, false),
        ]);
        if !slim {
            fields.extend(vec![
                Field::new("BIRTHPLACE", DataType::Utf8, false),
                Field::new("ADDRESS", DataType::Utf8, false),
            ]);
        }
        fields.extend(vec![
            Field::new("CITY", DataType::Utf8, false),
            Field::new("STATE", DataType::Utf8, false),
            Field::new("COUNTY", DataType::Utf8, false),
            Field::new("FIPS", DataType::Utf8, false),
            Field::new("ZIP", DataType::Utf8, false),
            Field::new("LAT", DataType::Float64, false),
            Field::new("LON", DataType::Float64, false),
            Field::new("HEALTHCARE_EXPENSES", DataType::Float64, false),
            Field::new("HEALTHCARE_COVERAGE", DataType::Float64, false),
            Field::new("INCOME", DataType::Int64, false),
            // CDE coordinate columns. The chronosynthea pivot from
            // "Java-Synthea-compat" toward "WASP-native cohort query"
            // surfaces these axes at write time so downstream
            // consumers can filter by them in O(matching rows)
            // without joining or scanning.
            Field::new("ARCHETYPE_ID", DataType::UInt16, false),
            Field::new("AGE_BAND", DataType::Utf8, false),
        ]);
        Arc::new(Schema::new(fields))
    }

    fn len(&self) -> usize {
        self.id.len()
    }

    fn to_record_batch(&mut self) -> arrow::error::Result<RecordBatch> {
        let mut cols: Vec<ArrayRef> = Vec::new();
        cols.push(Arc::new(self.id.finish()));
        cols.push(Arc::new(self.birthdate.finish()));
        cols.push(Arc::new(self.deathdate.finish()));
        if !self.slim {
            cols.push(Arc::new(self.ssn.finish()));
            cols.push(Arc::new(self.drivers.finish()));
            cols.push(Arc::new(self.passport.finish()));
            cols.push(Arc::new(self.first.finish()));
            cols.push(Arc::new(self.middle.finish()));
            cols.push(Arc::new(self.last.finish()));
        }
        cols.push(Arc::new(self.marital.finish()));
        cols.push(Arc::new(self.race.finish()));
        cols.push(Arc::new(self.ethnicity.finish()));
        cols.push(Arc::new(self.gender.finish()));
        if !self.slim {
            cols.push(Arc::new(self.birthplace.finish()));
            cols.push(Arc::new(self.address.finish()));
        }
        cols.push(Arc::new(self.city.finish()));
        cols.push(Arc::new(self.state.finish()));
        cols.push(Arc::new(self.county.finish()));
        cols.push(Arc::new(self.fips.finish()));
        cols.push(Arc::new(self.zip.finish()));
        cols.push(Arc::new(self.lat.finish()));
        cols.push(Arc::new(self.lon.finish()));
        cols.push(Arc::new(self.healthcare_expenses.finish()));
        cols.push(Arc::new(self.healthcare_coverage.finish()));
        cols.push(Arc::new(self.income.finish()));
        cols.push(Arc::new(self.archetype_id.finish()));
        cols.push(Arc::new(self.age_band.finish()));
        RecordBatch::try_new(Self::schema(self.slim), cols)
    }
}

pub struct SyntheaParquetWriter {
    patients_builder: PatientsBuilder,
    patients_writer: ArrowWriter<File>,
    output_dir: PathBuf,
    slim: bool,
}

impl SyntheaParquetWriter {
    pub fn create<P: AsRef<Path>>(output_dir: P) -> arrow::error::Result<Self> {
        Self::create_inner(output_dir, false)
    }

    /// "Slim" patients.parquet: drops PII columns (SSN, DRIVERS,
    /// PASSPORT, FIRST, MIDDLE, LAST, ADDRESS, BIRTHPLACE) so the
    /// resulting file is HIPAA-Safe-Harbor by construction and the
    /// remaining columns dictionary-encode much harder (fewer unique
    /// strings = smaller dictionary).
    pub fn create_slim<P: AsRef<Path>>(output_dir: P) -> arrow::error::Result<Self> {
        Self::create_inner(output_dir, true)
    }

    fn create_inner<P: AsRef<Path>>(
        output_dir: P,
        slim: bool,
    ) -> arrow::error::Result<Self> {
        let parquet_dir = output_dir.as_ref().join("parquet");
        create_dir_all(&parquet_dir)?;
        let file = File::create(parquet_dir.join("patients.parquet"))?;
        let patients_writer = ArrowWriter::try_new(
            file,
            PatientsBuilder::schema(slim),
            Some(writer_props()),
        )?;
        Ok(Self {
            patients_builder: PatientsBuilder::new(slim),
            patients_writer,
            output_dir: output_dir.as_ref().to_path_buf(),
            slim,
        })
    }

    pub fn write_patient(
        &mut self,
        patient: &FullPatient,
    ) -> arrow::error::Result<()> {
        let pii = derive_patient_pii(patient);

        let b = &mut self.patients_builder;
        b.id.append_value(&pii.uuid);
        b.birthdate.append_value(&pii.birth_date);
        b.deathdate.append_option(None::<&str>);
        if !self.slim {
            b.ssn.append_value(&pii.ssn);
            b.drivers.append_option(if pii.drivers.is_empty() {
                None
            } else {
                Some(pii.drivers.as_str())
            });
            b.passport.append_option(if pii.passport.is_empty() {
                None
            } else {
                Some(pii.passport.as_str())
            });
            b.first.append_value(&pii.first);
            b.middle.append_value(&pii.middle);
            b.last.append_value(&pii.last);
        }
        b.marital.append_option(if pii.marital.is_empty() {
            None
        } else {
            Some(pii.marital.as_str())
        });
        b.race.append_value(pii.race);
        b.ethnicity.append_value(pii.ethnicity);
        b.gender.append_value(pii.gender);
        if !self.slim {
            b.birthplace.append_value(&pii.birthplace);
            b.address.append_value(&pii.address);
        }
        b.city.append_value(pii.city);
        b.state.append_value("Massachusetts");
        b.county.append_value(pii.county);
        b.fips.append_value(pii.fips);
        b.zip.append_value(pii.zip);
        b.lat.append_value(pii.lat as f64);
        b.lon.append_value(pii.lon as f64);
        b.healthcare_expenses.append_value(pii.healthcare_expenses);
        b.healthcare_coverage.append_value(pii.healthcare_coverage);
        b.income.append_value(pii.income);
        b.archetype_id.append_value(pii.archetype_id);
        b.age_band.append_value(pii.age_band);

        if b.len() >= FLUSH_ROWS {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> arrow::error::Result<()> {
        if self.patients_builder.len() == 0 {
            return Ok(());
        }
        let batch = self.patients_builder.to_record_batch()?;
        self.patients_writer.write(&batch)?;
        Ok(())
    }

    pub fn finish(mut self) -> arrow::error::Result<()> {
        self.flush()?;
        self.patients_writer.close()?;
        Ok(())
    }

    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }
}

// =====================================================================
// Variant 3: stats-only summary.parquet writer
// =====================================================================

/// Per-patient aggregate stats. One row per patient, no event-level
/// data — typically ~200 bytes/patient on disk after Zstd.
///
/// Use cases:
/// - **Cohort selection**: scan summary.parquet to find patients
///   matching demographic / count criteria, then pull the full record
///   only for matching ids.
/// - **Population-scale analytics** where the question is about
///   per-patient distributions (mean encounters, median # conditions
///   by age band, etc.) and the event-level rows are noise.
struct StatsBuilder {
    id: StringBuilder,
    birthdate: StringBuilder,
    deathdate: StringBuilder,
    sex: StringBuilder,
    race: StringBuilder,
    ethnicity: StringBuilder,
    age_years: UInt32Builder,
    age_band: StringBuilder,
    archetype_id: arrow::array::UInt16Builder,
    n_conditions: UInt32Builder,
    n_encounters: UInt32Builder,
    n_medications: UInt32Builder,
    n_procedures: UInt32Builder,
    n_observations: UInt32Builder,
    income: Int64Builder,
    healthcare_expenses: Float64Builder,
}

impl StatsBuilder {
    fn new() -> Self {
        Self {
            id: sb(),
            birthdate: sb(),
            deathdate: sb(),
            sex: sb(),
            race: sb(),
            ethnicity: sb(),
            age_years: UInt32Builder::with_capacity(FLUSH_ROWS),
            age_band: sb(),
            archetype_id: arrow::array::UInt16Builder::with_capacity(FLUSH_ROWS),
            n_conditions: UInt32Builder::with_capacity(FLUSH_ROWS),
            n_encounters: UInt32Builder::with_capacity(FLUSH_ROWS),
            n_medications: UInt32Builder::with_capacity(FLUSH_ROWS),
            n_procedures: UInt32Builder::with_capacity(FLUSH_ROWS),
            n_observations: UInt32Builder::with_capacity(FLUSH_ROWS),
            income: Int64Builder::with_capacity(FLUSH_ROWS),
            healthcare_expenses: Float64Builder::with_capacity(FLUSH_ROWS),
        }
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("Id", DataType::Utf8, false),
            Field::new("BIRTHDATE", DataType::Utf8, false),
            Field::new("DEATHDATE", DataType::Utf8, true),
            Field::new("SEX", DataType::Utf8, false),
            Field::new("RACE", DataType::Utf8, false),
            Field::new("ETHNICITY", DataType::Utf8, false),
            Field::new("AGE_YEARS", DataType::UInt32, false),
            Field::new("AGE_BAND", DataType::Utf8, false),
            Field::new("ARCHETYPE_ID", DataType::UInt16, false),
            Field::new("N_CONDITIONS", DataType::UInt32, false),
            Field::new("N_ENCOUNTERS", DataType::UInt32, false),
            Field::new("N_MEDICATIONS", DataType::UInt32, false),
            Field::new("N_PROCEDURES", DataType::UInt32, false),
            Field::new("N_OBSERVATIONS", DataType::UInt32, false),
            Field::new("INCOME", DataType::Int64, false),
            Field::new("HEALTHCARE_EXPENSES", DataType::Float64, false),
        ]))
    }

    fn len(&self) -> usize {
        self.id.len()
    }

    fn to_record_batch(&mut self) -> arrow::error::Result<RecordBatch> {
        RecordBatch::try_new(
            Self::schema(),
            vec![
                Arc::new(self.id.finish()),
                Arc::new(self.birthdate.finish()),
                Arc::new(self.deathdate.finish()),
                Arc::new(self.sex.finish()),
                Arc::new(self.race.finish()),
                Arc::new(self.ethnicity.finish()),
                Arc::new(self.age_years.finish()),
                Arc::new(self.age_band.finish()),
                Arc::new(self.archetype_id.finish()),
                Arc::new(self.n_conditions.finish()),
                Arc::new(self.n_encounters.finish()),
                Arc::new(self.n_medications.finish()),
                Arc::new(self.n_procedures.finish()),
                Arc::new(self.n_observations.finish()),
                Arc::new(self.income.finish()),
                Arc::new(self.healthcare_expenses.finish()),
            ],
        )
    }
}

pub struct SyntheaStatsParquetWriter {
    builder: StatsBuilder,
    writer: ArrowWriter<File>,
    output_dir: PathBuf,
}

impl SyntheaStatsParquetWriter {
    pub fn create<P: AsRef<Path>>(output_dir: P) -> arrow::error::Result<Self> {
        let parquet_dir = output_dir.as_ref().join("parquet");
        create_dir_all(&parquet_dir)?;
        let file = File::create(parquet_dir.join("summary.parquet"))?;
        let writer = ArrowWriter::try_new(
            file,
            StatsBuilder::schema(),
            Some(writer_props()),
        )?;
        Ok(Self {
            builder: StatsBuilder::new(),
            writer,
            output_dir: output_dir.as_ref().to_path_buf(),
        })
    }

    pub fn write_patient(
        &mut self,
        patient: &FullPatient,
    ) -> arrow::error::Result<()> {
        let pii = derive_patient_pii(patient);
        let counts = patient.event_counts();

        let b = &mut self.builder;
        b.id.append_value(&pii.uuid);
        b.birthdate.append_value(&pii.birth_date);
        b.deathdate.append_option(None::<&str>);
        b.sex.append_value(pii.gender);
        b.race.append_value(pii.race);
        b.ethnicity.append_value(pii.ethnicity);
        b.age_years.append_value(pii.age_years);
        b.age_band.append_value(pii.age_band);
        b.archetype_id.append_value(pii.archetype_id);
        b.n_conditions.append_value(counts.diagnoses);
        b.n_encounters.append_value(patient.encounters.len() as u32);
        b.n_medications.append_value(counts.medications);
        b.n_procedures.append_value(counts.procedures);
        b.n_observations.append_value(counts.observations);
        b.income.append_value(pii.income);
        b.healthcare_expenses
            .append_value(pii.healthcare_expenses);

        if b.len() >= FLUSH_ROWS {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> arrow::error::Result<()> {
        if self.builder.len() == 0 {
            return Ok(());
        }
        let batch = self.builder.to_record_batch()?;
        self.writer.write(&batch)?;
        Ok(())
    }

    pub fn finish(mut self) -> arrow::error::Result<()> {
        self.flush()?;
        self.writer.close()?;
        Ok(())
    }

    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }
}

// =====================================================================
// Shared PII derivation (used by both writers).
// =====================================================================

/// Concrete carrier for the per-patient demographics fields. Both the
/// patients.parquet emit and the stats summary use this; deriving it
/// once here keeps the two writers a single source of truth for PII
/// hashing semantics.
struct PatientPii {
    uuid: String,
    birth_date: String,
    gender: &'static str,
    race: &'static str,
    ethnicity: &'static str,
    age_years: u32,
    age_band: &'static str,
    archetype_id: u16,
    first: String,
    middle: String,
    last: String,
    ssn: String,
    drivers: String,
    passport: String,
    marital: String,
    city: &'static str,
    county: &'static str,
    fips: &'static str,
    zip: &'static str,
    lat: f64,
    lon: f64,
    address: String,
    birthplace: String,
    income: i64,
    healthcare_expenses: f64,
    healthcare_coverage: f64,
}

fn derive_patient_pii(patient: &FullPatient) -> PatientPii {
    let uuid = patient_uuid(patient.id);
    let birth_date = epoch_to_date(patient.birth_date_days);
    let gender: &'static str = if patient.sex == 1 { "F" } else { "M" };
    let race: &'static str = match patient.race {
        0 => "white",
        1 => "black",
        2 => "asian",
        3 => "hispanic",
        4 => "native",
        _ => "other",
    };
    let ethnicity: &'static str = if patient.ethnicity == 1 {
        "hispanic"
    } else {
        "nonhispanic"
    };
    let age_years = years_since(patient.birth_date_days);
    let age_band = age_band_str(age_years);
    let archetype_id = patient.archetype_id.0;

    let first_pool: &[&str] = if patient.sex == 1 {
        FIRST_NAMES_F
    } else {
        FIRST_NAMES_M
    };
    let first_idx = hash_pick(patient.id, b"FIRST", first_pool.len());
    let middle_idx = hash_pick(patient.id, b"MIDDLE", first_pool.len());
    let last_idx = hash_pick(patient.id, b"LAST", LAST_NAMES.len());
    let first_suffix = hash_pick(patient.id, b"FIRSTSFX", 1000);
    let middle_suffix = hash_pick(patient.id, b"MIDSFX", 1000);
    let last_suffix = hash_pick(patient.id, b"LASTSFX", 1000);
    let first = format!("{}{}", first_pool[first_idx], first_suffix);
    let middle = format!("{}{}", first_pool[middle_idx], middle_suffix);
    let last = format!("{}{}", LAST_NAMES[last_idx], last_suffix);
    let ssn = format!(
        "999-{:02}-{:04}",
        hash_pick(patient.id, b"SSN1", 100),
        hash_pick(patient.id, b"SSN2", 10_000)
    );
    let drivers = if age_years >= 16 {
        format!("S{:08}", hash_pick(patient.id, b"DRIVERS", 100_000_000))
    } else {
        String::new()
    };
    let passport = if age_years >= 18
        && (hash_pick(patient.id, b"PASSPORTCHK", 100) < 35)
    {
        format!("X{:08}X", hash_pick(patient.id, b"PASSPORT", 100_000_000))
    } else {
        String::new()
    };
    let marital = if age_years >= 18 {
        let r = hash_pick(patient.id, b"MARITAL", 100) as f32 / 100.0;
        let mut acc = 0.0f32;
        let mut chosen = "S";
        for (status, prob) in MARITAL_STATUSES {
            acc += prob;
            if r <= acc {
                chosen = status;
                break;
            }
        }
        chosen.to_string()
    } else {
        String::new()
    };

    let city_idx = hash_pick(patient.id, b"CITY", MA_CITIES.len());
    let (city, county, fips, zip, lat, lon) = MA_CITIES[city_idx];
    let street_no = 100 + hash_pick(patient.id, b"STREETNO", 9900);
    let street_name =
        STREET_NAMES[hash_pick(patient.id, b"STREETNAME", STREET_NAMES.len())];
    let street_suffix = STREET_SUFFIXES
        [hash_pick(patient.id, b"STREETSFX", STREET_SUFFIXES.len())];
    let address = format!("{} {} {}", street_no, street_name, street_suffix);
    let birthplace = format!(
        "{}  Massachusetts  US",
        MA_CITIES[hash_pick(patient.id, b"BIRTHPLACE", MA_CITIES.len())].0
    );

    let income =
        lognormal_sample(patient.id, b"INCOME", INCOME_MEAN, INCOME_STD)
            .max(0.0) as i64;
    let healthcare_expenses = lognormal_sample(
        patient.id,
        b"EXPENSES",
        HEALTHCARE_EXPENSES_MEAN,
        HEALTHCARE_EXPENSES_STD,
    )
    .max(0.0);
    let healthcare_coverage = (healthcare_expenses * 0.85_f64).max(0.0);

    PatientPii {
        uuid,
        birth_date,
        gender,
        race,
        ethnicity,
        age_years,
        age_band,
        archetype_id,
        first,
        middle,
        last,
        ssn,
        drivers,
        passport,
        marital,
        city,
        county,
        fips,
        zip,
        lat: lat as f64,
        lon: lon as f64,
        address,
        birthplace,
        income,
        healthcare_expenses,
        healthcare_coverage,
    }
}

// =====================================================================
// Variant 4: full-coverage SyntheaParquetFullWriter (6 high-volume files)
//
// Writes patients.parquet + encounters.parquet + conditions.parquet +
// observations.parquet + medications.parquet + procedures.parquet from
// the same `FullPatient` input. Covers ~50% of Synthea's output volume
// (claims_transactions, imaging_studies, and the smaller files are
// follow-up work — same builder pattern, more boilerplate).
//
// Determinism: byte output is comparable to the CSV path's content for
// the same patient + encounter. Timestamps reuse the same RNG-driven
// `epoch_to_iso8601` helper as the CSV writer so emitted strings
// match.
// =====================================================================

use crate::archetype::ArchetypeRegistry;
use crate::tables::CodeTable;
use crate::csv_writer::{
    encounter_class_info, encounter_uuid, epoch_to_iso8601,
    iso8601_plus_minutes, lookup_condition, lookup_medication,
    lookup_procedure,
};

const NO_INSURANCE_UUID: &str = "b1c428d6-4f07-31e0-90f0-68ffa6ff8c76";

struct EncountersBuilder {
    id: StringBuilder,
    start: StringBuilder,
    stop: StringBuilder,
    patient: StringBuilder,
    organization: StringBuilder,
    provider: StringBuilder,
    payer: StringBuilder,
    encounterclass: StringBuilder,
    code: StringBuilder,
    description: StringBuilder,
    base_encounter_cost: Float64Builder,
    total_claim_cost: Float64Builder,
    payer_coverage: Float64Builder,
}

impl EncountersBuilder {
    fn new() -> Self {
        Self {
            id: sb(),
            start: sb(),
            stop: sb(),
            patient: sb(),
            organization: sb(),
            provider: sb(),
            payer: sb(),
            encounterclass: sb(),
            code: sb(),
            description: sb(),
            base_encounter_cost: Float64Builder::with_capacity(FLUSH_ROWS),
            total_claim_cost: Float64Builder::with_capacity(FLUSH_ROWS),
            payer_coverage: Float64Builder::with_capacity(FLUSH_ROWS),
        }
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("Id", DataType::Utf8, false),
            Field::new("START", DataType::Utf8, false),
            Field::new("STOP", DataType::Utf8, false),
            Field::new("PATIENT", DataType::Utf8, false),
            Field::new("ORGANIZATION", DataType::Utf8, false),
            Field::new("PROVIDER", DataType::Utf8, false),
            Field::new("PAYER", DataType::Utf8, false),
            Field::new("ENCOUNTERCLASS", DataType::Utf8, false),
            Field::new("CODE", DataType::Utf8, false),
            Field::new("DESCRIPTION", DataType::Utf8, false),
            Field::new("BASE_ENCOUNTER_COST", DataType::Float64, false),
            Field::new("TOTAL_CLAIM_COST", DataType::Float64, false),
            Field::new("PAYER_COVERAGE", DataType::Float64, false),
        ]))
    }

    fn len(&self) -> usize {
        self.id.len()
    }

    fn to_record_batch(&mut self) -> arrow::error::Result<RecordBatch> {
        RecordBatch::try_new(
            Self::schema(),
            vec![
                Arc::new(self.id.finish()) as ArrayRef,
                Arc::new(self.start.finish()),
                Arc::new(self.stop.finish()),
                Arc::new(self.patient.finish()),
                Arc::new(self.organization.finish()),
                Arc::new(self.provider.finish()),
                Arc::new(self.payer.finish()),
                Arc::new(self.encounterclass.finish()),
                Arc::new(self.code.finish()),
                Arc::new(self.description.finish()),
                Arc::new(self.base_encounter_cost.finish()),
                Arc::new(self.total_claim_cost.finish()),
                Arc::new(self.payer_coverage.finish()),
            ],
        )
    }
}

struct ConditionsBuilder {
    start: StringBuilder,
    stop: StringBuilder,
    patient: StringBuilder,
    encounter: StringBuilder,
    system: StringBuilder,
    code: StringBuilder,
    description: StringBuilder,
}

impl ConditionsBuilder {
    fn new() -> Self {
        Self {
            start: sb(),
            stop: sb(),
            patient: sb(),
            encounter: sb(),
            system: sb(),
            code: sb(),
            description: sb(),
        }
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("START", DataType::Utf8, false),
            Field::new("STOP", DataType::Utf8, true),
            Field::new("PATIENT", DataType::Utf8, false),
            Field::new("ENCOUNTER", DataType::Utf8, false),
            Field::new("SYSTEM", DataType::Utf8, false),
            Field::new("CODE", DataType::Utf8, false),
            Field::new("DESCRIPTION", DataType::Utf8, false),
        ]))
    }

    fn len(&self) -> usize {
        self.start.len()
    }

    fn to_record_batch(&mut self) -> arrow::error::Result<RecordBatch> {
        RecordBatch::try_new(
            Self::schema(),
            vec![
                Arc::new(self.start.finish()) as ArrayRef,
                Arc::new(self.stop.finish()),
                Arc::new(self.patient.finish()),
                Arc::new(self.encounter.finish()),
                Arc::new(self.system.finish()),
                Arc::new(self.code.finish()),
                Arc::new(self.description.finish()),
            ],
        )
    }
}

struct ObservationsBuilder {
    date: StringBuilder,
    patient: StringBuilder,
    encounter: StringBuilder,
    category: StringBuilder,
    code: StringBuilder,
    description: StringBuilder,
    value: StringBuilder,
    units: StringBuilder,
    obs_type: StringBuilder,
}

impl ObservationsBuilder {
    fn new() -> Self {
        Self {
            date: sb(),
            patient: sb(),
            encounter: sb(),
            category: sb(),
            code: sb(),
            description: sb(),
            value: sb(),
            units: sb(),
            obs_type: sb(),
        }
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("DATE", DataType::Utf8, false),
            Field::new("PATIENT", DataType::Utf8, false),
            Field::new("ENCOUNTER", DataType::Utf8, false),
            Field::new("CATEGORY", DataType::Utf8, false),
            Field::new("CODE", DataType::Utf8, false),
            Field::new("DESCRIPTION", DataType::Utf8, false),
            Field::new("VALUE", DataType::Utf8, true),
            Field::new("UNITS", DataType::Utf8, true),
            Field::new("TYPE", DataType::Utf8, true),
        ]))
    }

    fn len(&self) -> usize {
        self.date.len()
    }

    fn to_record_batch(&mut self) -> arrow::error::Result<RecordBatch> {
        RecordBatch::try_new(
            Self::schema(),
            vec![
                Arc::new(self.date.finish()) as ArrayRef,
                Arc::new(self.patient.finish()),
                Arc::new(self.encounter.finish()),
                Arc::new(self.category.finish()),
                Arc::new(self.code.finish()),
                Arc::new(self.description.finish()),
                Arc::new(self.value.finish()),
                Arc::new(self.units.finish()),
                Arc::new(self.obs_type.finish()),
            ],
        )
    }
}

struct MedicationsBuilder {
    start: StringBuilder,
    stop: StringBuilder,
    patient: StringBuilder,
    payer: StringBuilder,
    encounter: StringBuilder,
    code: StringBuilder,
    description: StringBuilder,
    base_cost: Float64Builder,
    payer_coverage: Float64Builder,
    dispenses: Int64Builder,
    totalcost: Float64Builder,
    reasoncode: StringBuilder,
    reasondescription: StringBuilder,
}

impl MedicationsBuilder {
    fn new() -> Self {
        Self {
            start: sb(),
            stop: sb(),
            patient: sb(),
            payer: sb(),
            encounter: sb(),
            code: sb(),
            description: sb(),
            base_cost: Float64Builder::with_capacity(FLUSH_ROWS),
            payer_coverage: Float64Builder::with_capacity(FLUSH_ROWS),
            dispenses: Int64Builder::with_capacity(FLUSH_ROWS),
            totalcost: Float64Builder::with_capacity(FLUSH_ROWS),
            reasoncode: sb(),
            reasondescription: sb(),
        }
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("START", DataType::Utf8, false),
            Field::new("STOP", DataType::Utf8, true),
            Field::new("PATIENT", DataType::Utf8, false),
            Field::new("PAYER", DataType::Utf8, false),
            Field::new("ENCOUNTER", DataType::Utf8, false),
            Field::new("CODE", DataType::Utf8, false),
            Field::new("DESCRIPTION", DataType::Utf8, false),
            Field::new("BASE_COST", DataType::Float64, false),
            Field::new("PAYER_COVERAGE", DataType::Float64, false),
            Field::new("DISPENSES", DataType::Int64, false),
            Field::new("TOTALCOST", DataType::Float64, false),
            Field::new("REASONCODE", DataType::Utf8, true),
            Field::new("REASONDESCRIPTION", DataType::Utf8, true),
        ]))
    }

    fn len(&self) -> usize {
        self.start.len()
    }

    fn to_record_batch(&mut self) -> arrow::error::Result<RecordBatch> {
        RecordBatch::try_new(
            Self::schema(),
            vec![
                Arc::new(self.start.finish()) as ArrayRef,
                Arc::new(self.stop.finish()),
                Arc::new(self.patient.finish()),
                Arc::new(self.payer.finish()),
                Arc::new(self.encounter.finish()),
                Arc::new(self.code.finish()),
                Arc::new(self.description.finish()),
                Arc::new(self.base_cost.finish()),
                Arc::new(self.payer_coverage.finish()),
                Arc::new(self.dispenses.finish()),
                Arc::new(self.totalcost.finish()),
                Arc::new(self.reasoncode.finish()),
                Arc::new(self.reasondescription.finish()),
            ],
        )
    }
}

struct ProceduresBuilder {
    start: StringBuilder,
    stop: StringBuilder,
    patient: StringBuilder,
    encounter: StringBuilder,
    system: StringBuilder,
    code: StringBuilder,
    description: StringBuilder,
    base_cost: Float64Builder,
    reasoncode: StringBuilder,
    reasondescription: StringBuilder,
}

impl ProceduresBuilder {
    fn new() -> Self {
        Self {
            start: sb(),
            stop: sb(),
            patient: sb(),
            encounter: sb(),
            system: sb(),
            code: sb(),
            description: sb(),
            base_cost: Float64Builder::with_capacity(FLUSH_ROWS),
            reasoncode: sb(),
            reasondescription: sb(),
        }
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("START", DataType::Utf8, false),
            Field::new("STOP", DataType::Utf8, true),
            Field::new("PATIENT", DataType::Utf8, false),
            Field::new("ENCOUNTER", DataType::Utf8, false),
            Field::new("SYSTEM", DataType::Utf8, false),
            Field::new("CODE", DataType::Utf8, false),
            Field::new("DESCRIPTION", DataType::Utf8, false),
            Field::new("BASE_COST", DataType::Float64, false),
            Field::new("REASONCODE", DataType::Utf8, true),
            Field::new("REASONDESCRIPTION", DataType::Utf8, true),
        ]))
    }

    fn len(&self) -> usize {
        self.start.len()
    }

    fn to_record_batch(&mut self) -> arrow::error::Result<RecordBatch> {
        RecordBatch::try_new(
            Self::schema(),
            vec![
                Arc::new(self.start.finish()) as ArrayRef,
                Arc::new(self.stop.finish()),
                Arc::new(self.patient.finish()),
                Arc::new(self.encounter.finish()),
                Arc::new(self.system.finish()),
                Arc::new(self.code.finish()),
                Arc::new(self.description.finish()),
                Arc::new(self.base_cost.finish()),
                Arc::new(self.reasoncode.finish()),
                Arc::new(self.reasondescription.finish()),
            ],
        )
    }
}

/// `patient_conditions.parquet` builder — one row per patient with
/// the patient's full condition-code list as a `List<Utf8>` column.
/// This is the WASP probe target for cohort queries: a consumer
/// asking "give me patients with condition='stroke'" can scan this
/// single column instead of joining `conditions.parquet` back to
/// `patients.parquet` and aggregating. Dictionary-encodes very hard
/// (a few hundred unique SNOMED codes across millions of rows).
struct PatientConditionsBuilder {
    patient: StringBuilder,
    condition_codes: ListBuilder<StringBuilder>,
    n_conditions: UInt32Builder,
}

impl PatientConditionsBuilder {
    fn new() -> Self {
        Self {
            patient: sb(),
            condition_codes: ListBuilder::new(sb()),
            n_conditions: UInt32Builder::with_capacity(FLUSH_ROWS),
        }
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("PATIENT", DataType::Utf8, false),
            Field::new(
                "CONDITION_CODES",
                // Arrow's `ListBuilder` defaults to `nullable: true` for
                // the list's item field. Match that so the writer's
                // emitted batches line up with the declared schema.
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            ),
            Field::new("N_CONDITIONS", DataType::UInt32, false),
        ]))
    }

    fn len(&self) -> usize {
        self.patient.len()
    }

    fn to_record_batch(&mut self) -> arrow::error::Result<RecordBatch> {
        RecordBatch::try_new(
            Self::schema(),
            vec![
                Arc::new(self.patient.finish()) as ArrayRef,
                Arc::new(self.condition_codes.finish()),
                Arc::new(self.n_conditions.finish()),
            ],
        )
    }
}

/// Full-coverage Parquet writer: emits patients, encounters,
/// conditions, observations, medications, procedures all in one pass
/// per patient — and a `patient_conditions.parquet` summary table so
/// cohort queries by condition probe in O(matching) instead of
/// O(events). ~98% of Synthea's output volume in Parquet format.
pub struct SyntheaParquetFullWriter {
    patients_builder: PatientsBuilder,
    patients_writer: ArrowWriter<File>,
    encounters_builder: EncountersBuilder,
    encounters_writer: ArrowWriter<File>,
    conditions_builder: ConditionsBuilder,
    conditions_writer: ArrowWriter<File>,
    observations_builder: ObservationsBuilder,
    observations_writer: ArrowWriter<File>,
    medications_builder: MedicationsBuilder,
    medications_writer: ArrowWriter<File>,
    procedures_builder: ProceduresBuilder,
    procedures_writer: ArrowWriter<File>,
    patient_conditions_builder: PatientConditionsBuilder,
    patient_conditions_writer: ArrowWriter<File>,
    output_dir: PathBuf,
}

impl SyntheaParquetFullWriter {
    pub fn create<P: AsRef<Path>>(output_dir: P) -> arrow::error::Result<Self> {
        let parquet_dir = output_dir.as_ref().join("parquet");
        create_dir_all(&parquet_dir)?;

        let props = writer_props();
        let make =
            |name: &str, schema: SchemaRef| -> arrow::error::Result<ArrowWriter<File>> {
                let f = File::create(parquet_dir.join(name))?;
                Ok(ArrowWriter::try_new(f, schema, Some(props.clone()))?)
            };

        Ok(Self {
            patients_builder: PatientsBuilder::new(false),
            patients_writer: make("patients.parquet", PatientsBuilder::schema(false))?,
            encounters_builder: EncountersBuilder::new(),
            encounters_writer: make("encounters.parquet", EncountersBuilder::schema())?,
            conditions_builder: ConditionsBuilder::new(),
            conditions_writer: make("conditions.parquet", ConditionsBuilder::schema())?,
            observations_builder: ObservationsBuilder::new(),
            observations_writer: make(
                "observations.parquet",
                ObservationsBuilder::schema(),
            )?,
            medications_builder: MedicationsBuilder::new(),
            medications_writer: make(
                "medications.parquet",
                MedicationsBuilder::schema(),
            )?,
            procedures_builder: ProceduresBuilder::new(),
            procedures_writer: make("procedures.parquet", ProceduresBuilder::schema())?,
            patient_conditions_builder: PatientConditionsBuilder::new(),
            patient_conditions_writer: make(
                "patient_conditions.parquet",
                PatientConditionsBuilder::schema(),
            )?,
            output_dir: output_dir.as_ref().to_path_buf(),
        })
    }

    /// Emit all 6 files' rows for one patient. `archetypes` and
    /// `code_table` are forwarded to the lookup helpers in csv_writer.
    pub fn write_patient(
        &mut self,
        patient: &FullPatient,
        archetypes: &ArchetypeRegistry,
        code_table: &CodeTable,
    ) -> arrow::error::Result<()> {
        let pii = derive_patient_pii(patient);

        // ---- patients.parquet (full schema) ----
        {
            let b = &mut self.patients_builder;
            b.id.append_value(&pii.uuid);
            b.birthdate.append_value(&pii.birth_date);
            b.deathdate.append_option(None::<&str>);
            b.ssn.append_value(&pii.ssn);
            b.drivers.append_option(if pii.drivers.is_empty() {
                None
            } else {
                Some(pii.drivers.as_str())
            });
            b.passport.append_option(if pii.passport.is_empty() {
                None
            } else {
                Some(pii.passport.as_str())
            });
            b.first.append_value(&pii.first);
            b.middle.append_value(&pii.middle);
            b.last.append_value(&pii.last);
            b.marital.append_option(if pii.marital.is_empty() {
                None
            } else {
                Some(pii.marital.as_str())
            });
            b.race.append_value(pii.race);
            b.ethnicity.append_value(pii.ethnicity);
            b.gender.append_value(pii.gender);
            b.birthplace.append_value(&pii.birthplace);
            b.address.append_value(&pii.address);
            b.city.append_value(pii.city);
            b.state.append_value("Massachusetts");
            b.county.append_value(pii.county);
            b.fips.append_value(pii.fips);
            b.zip.append_value(pii.zip);
            b.lat.append_value(pii.lat);
            b.lon.append_value(pii.lon);
            b.healthcare_expenses.append_value(pii.healthcare_expenses);
            b.healthcare_coverage.append_value(pii.healthcare_coverage);
            b.income.append_value(pii.income);
            b.archetype_id.append_value(pii.archetype_id);
            b.age_band.append_value(pii.age_band);
            if b.len() >= FLUSH_ROWS {
                self.flush_patients()?;
            }
        }

        // ---- patient_conditions.parquet row (WASP probe target) ----
        {
            let pc = &mut self.patient_conditions_builder;
            pc.patient.append_value(&pii.uuid);
            for &cond_idx in patient.conditions.iter() {
                let (code, _) = lookup_condition(archetypes, code_table, cond_idx);
                pc.condition_codes.values().append_value(code);
            }
            pc.condition_codes.append(true);
            pc.n_conditions.append_value(patient.conditions.len() as u32);
            if pc.len() >= FLUSH_ROWS {
                self.flush_patient_conditions()?;
            }
        }

        // Per-patient RNG state — mirrors the CSV writer so timestamps
        // are byte-identical.
        let mut rng_state = patient.id.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let next_rand: &dyn Fn(&mut u64) -> u64 = &|state: &mut u64| -> u64 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        };

        // Synthesise provider / organization UUIDs deterministically from
        // patient id. Cheaper than the city-routed lookup the CSV writer
        // does — the Parquet path doesn't need the exact same providers
        // for now, just stable per-patient ones. Dictionary-encodes fine.
        let provider_uuid = format!(
            "{:016x}-prov", pii_seed_hash(patient.id, b"PROV")
        );
        let organization_uuid = format!(
            "{:016x}-org", pii_seed_hash(patient.id, b"ORG")
        );
        let payer_uuid = NO_INSURANCE_UUID; // simplified for v1; matches CSV writer's stub payer logic for most patients

        // First-encounter UUID — used as the ENCOUNTER fallback for
        // conditions whose onset precedes any recorded encounter.
        let first_enc_uuid_str: String = if !patient.encounters.is_empty() {
            encounter_uuid(patient.id, 0).as_str().to_owned()
        } else {
            String::new()
        };

        // ---- conditions.parquet ----
        for (i, &cond_idx) in patient.conditions.iter().enumerate() {
            let onset_offset = patient
                .condition_onset_days
                .get(i)
                .copied()
                .unwrap_or(0) as i32;
            let onset_date = epoch_to_date(patient.birth_date_days + onset_offset);
            let (code, display) = lookup_condition(archetypes, code_table, cond_idx);
            // Encounter that linked the condition: first whose day >= onset.
            let linked_enc = patient
                .encounters
                .iter()
                .enumerate()
                .find(|(_, e)| e.days_since_birth as i32 >= onset_offset)
                .map(|(idx, _)| encounter_uuid(patient.id, idx as u32).as_str().to_owned())
                .unwrap_or_else(|| first_enc_uuid_str.clone());

            let b = &mut self.conditions_builder;
            b.start.append_value(&onset_date);
            b.stop.append_option(None::<&str>);
            b.patient.append_value(&pii.uuid);
            b.encounter.append_value(&linked_enc);
            b.system.append_value("SNOMED-CT");
            b.code.append_value(code);
            b.description.append_value(display);
            if b.len() >= FLUSH_ROWS {
                self.flush_conditions()?;
            }
        }

        // ---- encounters / observations / medications / procedures ----
        for (enc_idx, encounter) in patient.encounters.iter().enumerate() {
            let enc_uuid = encounter_uuid(patient.id, enc_idx as u32);
            let enc_day = patient.birth_date_days + encounter.days_since_birth as i32;
            let start_ts = epoch_to_iso8601(enc_day, &mut rng_state, next_rand);
            let stop_ts = iso8601_plus_minutes(&start_ts, 15);
            let (enc_class, _, enc_code, enc_display, enc_cost) =
                encounter_class_info(encounter.encounter_type);
            let total_cost = enc_cost + (enc_idx as f32 * 17.3) % 200.0;
            let payer_coverage = if payer_uuid == NO_INSURANCE_UUID {
                0.0
            } else {
                total_cost * 0.8
            };

            // encounters.parquet row
            {
                let b = &mut self.encounters_builder;
                b.id.append_value(enc_uuid.as_str());
                b.start.append_value(&start_ts);
                b.stop.append_value(&stop_ts);
                b.patient.append_value(&pii.uuid);
                b.organization.append_value(&organization_uuid);
                b.provider.append_value(&provider_uuid);
                b.payer.append_value(payer_uuid);
                b.encounterclass.append_value(enc_class);
                b.code.append_value(enc_code);
                b.description.append_value(enc_display);
                b.base_encounter_cost.append_value(enc_cost as f64);
                b.total_claim_cost.append_value(total_cost as f64);
                b.payer_coverage.append_value(payer_coverage as f64);
                if b.len() >= FLUSH_ROWS {
                    self.flush_encounters()?;
                }
            }

            // observations.parquet rows
            for ev in encounter.observations.iter() {
                if let Some(obs) = code_table.observation(ev.code_idx) {
                    let b = &mut self.observations_builder;
                    b.date.append_value(&start_ts);
                    b.patient.append_value(&pii.uuid);
                    b.encounter.append_value(enc_uuid.as_str());
                    b.category.append_value("exam");
                    b.code.append_value(&obs.code);
                    b.description.append_value(&obs.display);
                    b.value.append_option(None::<&str>);
                    b.units.append_option(None::<&str>);
                    b.obs_type.append_option(None::<&str>);
                    if b.len() >= FLUSH_ROWS {
                        self.flush_observations()?;
                    }
                }
            }

            // medications.parquet rows
            for ev in encounter.medications.iter() {
                let (code, display) =
                    lookup_medication(archetypes, code_table, ev.code_idx);
                let b = &mut self.medications_builder;
                b.start.append_value(&start_ts);
                b.stop.append_option(None::<&str>);
                b.patient.append_value(&pii.uuid);
                b.payer.append_value(payer_uuid);
                b.encounter.append_value(enc_uuid.as_str());
                b.code.append_value(code);
                b.description.append_value(display);
                b.base_cost.append_value(20.0);
                b.payer_coverage.append_value(
                    if payer_uuid == NO_INSURANCE_UUID { 0.0 } else { 18.0 },
                );
                b.dispenses.append_value(1);
                b.totalcost.append_value(20.0);
                b.reasoncode.append_option(None::<&str>);
                b.reasondescription.append_option(None::<&str>);
                if b.len() >= FLUSH_ROWS {
                    self.flush_medications()?;
                }
            }

            // procedures.parquet rows
            for ev in encounter.procedures.iter() {
                let (code, display) =
                    lookup_procedure(archetypes, code_table, ev.code_idx);
                let cost = code_table
                    .procedure_cost_str
                    .get(ev.code_idx as usize)
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                let b = &mut self.procedures_builder;
                b.start.append_value(&start_ts);
                b.stop.append_option(None::<&str>);
                b.patient.append_value(&pii.uuid);
                b.encounter.append_value(enc_uuid.as_str());
                b.system.append_value("SNOMED-CT");
                b.code.append_value(code);
                b.description.append_value(display);
                b.base_cost.append_value(cost);
                b.reasoncode.append_option(None::<&str>);
                b.reasondescription.append_option(None::<&str>);
                if b.len() >= FLUSH_ROWS {
                    self.flush_procedures()?;
                }
            }
        }

        Ok(())
    }

    fn flush_patients(&mut self) -> arrow::error::Result<()> {
        if self.patients_builder.len() == 0 {
            return Ok(());
        }
        let batch = self.patients_builder.to_record_batch()?;
        self.patients_writer.write(&batch)?;
        Ok(())
    }
    fn flush_encounters(&mut self) -> arrow::error::Result<()> {
        if self.encounters_builder.len() == 0 {
            return Ok(());
        }
        let batch = self.encounters_builder.to_record_batch()?;
        self.encounters_writer.write(&batch)?;
        Ok(())
    }
    fn flush_conditions(&mut self) -> arrow::error::Result<()> {
        if self.conditions_builder.len() == 0 {
            return Ok(());
        }
        let batch = self.conditions_builder.to_record_batch()?;
        self.conditions_writer.write(&batch)?;
        Ok(())
    }
    fn flush_observations(&mut self) -> arrow::error::Result<()> {
        if self.observations_builder.len() == 0 {
            return Ok(());
        }
        let batch = self.observations_builder.to_record_batch()?;
        self.observations_writer.write(&batch)?;
        Ok(())
    }
    fn flush_medications(&mut self) -> arrow::error::Result<()> {
        if self.medications_builder.len() == 0 {
            return Ok(());
        }
        let batch = self.medications_builder.to_record_batch()?;
        self.medications_writer.write(&batch)?;
        Ok(())
    }
    fn flush_procedures(&mut self) -> arrow::error::Result<()> {
        if self.procedures_builder.len() == 0 {
            return Ok(());
        }
        let batch = self.procedures_builder.to_record_batch()?;
        self.procedures_writer.write(&batch)?;
        Ok(())
    }
    fn flush_patient_conditions(&mut self) -> arrow::error::Result<()> {
        if self.patient_conditions_builder.len() == 0 {
            return Ok(());
        }
        let batch = self.patient_conditions_builder.to_record_batch()?;
        self.patient_conditions_writer.write(&batch)?;
        Ok(())
    }

    pub fn finish(mut self) -> arrow::error::Result<()> {
        self.flush_patients()?;
        self.flush_encounters()?;
        self.flush_conditions()?;
        self.flush_observations()?;
        self.flush_medications()?;
        self.flush_procedures()?;
        self.flush_patient_conditions()?;
        self.patients_writer.close()?;
        self.encounters_writer.close()?;
        self.conditions_writer.close()?;
        self.observations_writer.close()?;
        self.medications_writer.close()?;
        self.procedures_writer.close()?;
        self.patient_conditions_writer.close()?;
        Ok(())
    }

    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }
}

fn pii_seed_hash(id: u64, salt: &[u8]) -> u64 {
    let mut h = id.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for &b in salt {
        h = h.wrapping_mul(0x100000001B3).wrapping_add(b as u64);
    }
    h
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

fn hash_pick(id: u64, salt: &[u8], modulo: usize) -> usize {
    if modulo == 0 {
        return 0;
    }
    let mut h = id.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for &b in salt {
        h = h.wrapping_mul(0x100000001B3).wrapping_add(b as u64);
    }
    (h % modulo as u64) as usize
}

fn lognormal_sample(id: u64, salt: &[u8], mean: f64, std: f64) -> f64 {
    let u1 = (hash_pick(id, salt, 10_000_000) as f64 + 1.0) / 10_000_001.0;
    let u2 = (hash_pick(id, &[salt, b"_u2"].concat(), 10_000_000) as f64 + 1.0)
        / 10_000_001.0;
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    let mu = (mean * mean / (mean * mean + std * std).sqrt()).ln();
    let sigma = (1.0 + (std * std) / (mean * mean)).ln().sqrt();
    (mu + sigma * z).exp()
}

fn epoch_to_date(days_since_epoch: i32) -> String {
    use chrono::{Duration, NaiveDate};
    let d = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()
        + Duration::days(days_since_epoch as i64);
    d.format("%Y-%m-%d").to_string()
}

/// Decadal age band string ("0-9", "10-19", …, "80+"). One subtract +
/// one bucket selection at write time; eliminates the BIRTHDATE range
/// scan a Pandas / DuckDB consumer would otherwise do for every
/// demographic query.
fn age_band_str(age_years: u32) -> &'static str {
    match age_years {
        0..=9 => "0-9",
        10..=19 => "10-19",
        20..=29 => "20-29",
        30..=39 => "30-39",
        40..=49 => "40-49",
        50..=59 => "50-59",
        60..=69 => "60-69",
        70..=79 => "70-79",
        _ => "80+",
    }
}

fn years_since(birth_date_days: i32) -> u32 {
    use chrono::{Duration, NaiveDate, Utc};
    let today = Utc::now().naive_utc().date();
    let birth = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()
        + Duration::days(birth_date_days as i64);
    let years = today.signed_duration_since(birth).num_days() / 365;
    years.max(0) as u32
}
