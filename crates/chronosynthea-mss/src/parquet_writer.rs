//! Apache Parquet output path — alternative to the CSV writer.
//!
//! ## Why Parquet
//!
//! The CSV writer is byte-emission-bound at ~580 KB/patient. The GPU
//! experiment (PR #13) confirmed throughput is capped by memory + I/O
//! bandwidth, not compute. The only architectural lever left is to
//! emit fewer bytes — which is exactly what Parquet's columnar +
//! dictionary + zstd encoding does. Typical reductions on tabular
//! healthcare data:
//!
//! - SNOMED / RxNorm / LOINC code columns dictionary-encode to ~2 bits
//!   per value (a few hundred unique codes; thousands of rows per code).
//! - UUID columns compress modestly (~2× via Zstd).
//! - Timestamps stored as Int32 days-since-epoch instead of ISO8601
//!   ASCII (~4 bytes vs ~20 bytes).
//!
//! Expected on-disk size: ~50–80 KB/patient versus CSV's ~580 KB.
//! That's a 7–12× output-volume reduction. Combined with the existing
//! parallel writer architecture, this is the path toward 10,000× Java
//! throughput on the same hardware: less to write = faster.
//!
//! ## Status — proof of concept
//!
//! This first cut implements **patients.parquet** only — the simplest
//! file (one row per patient) and the right place to validate the
//! Arrow/Parquet integration before expanding to the other 14 files.
//! Each follows the same builder + ArrowWriter pattern.
//!
//! See `bin/parquet_bench.rs` for the CSV-vs-Parquet size + throughput
//! comparison the proof-of-concept produces.

use std::fs::{create_dir_all, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    ArrayBuilder, ArrayRef, Float64Builder, Int64Builder, StringBuilder,
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

// ---------------------------------------------------------------------
// patients.parquet — column builder + schema.
// ---------------------------------------------------------------------

struct PatientsBuilder {
    id: StringBuilder,
    birthdate: StringBuilder,
    deathdate: StringBuilder,
    ssn: StringBuilder,
    drivers: StringBuilder,
    passport: StringBuilder,
    first: StringBuilder,
    middle: StringBuilder,
    last: StringBuilder,
    marital: StringBuilder,
    race: StringBuilder,
    ethnicity: StringBuilder,
    gender: StringBuilder,
    birthplace: StringBuilder,
    address: StringBuilder,
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
}

impl PatientsBuilder {
    fn new() -> Self {
        Self {
            id: sb(),
            birthdate: sb(),
            deathdate: sb(),
            ssn: sb(),
            drivers: sb(),
            passport: sb(),
            first: sb(),
            middle: sb(),
            last: sb(),
            marital: sb(),
            race: sb(),
            ethnicity: sb(),
            gender: sb(),
            birthplace: sb(),
            address: sb(),
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
        }
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("Id", DataType::Utf8, false),
            Field::new("BIRTHDATE", DataType::Utf8, false),
            Field::new("DEATHDATE", DataType::Utf8, true),
            Field::new("SSN", DataType::Utf8, false),
            Field::new("DRIVERS", DataType::Utf8, true),
            Field::new("PASSPORT", DataType::Utf8, true),
            Field::new("FIRST", DataType::Utf8, false),
            Field::new("MIDDLE", DataType::Utf8, false),
            Field::new("LAST", DataType::Utf8, false),
            Field::new("MARITAL", DataType::Utf8, true),
            Field::new("RACE", DataType::Utf8, false),
            Field::new("ETHNICITY", DataType::Utf8, false),
            Field::new("GENDER", DataType::Utf8, false),
            Field::new("BIRTHPLACE", DataType::Utf8, false),
            Field::new("ADDRESS", DataType::Utf8, false),
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
                Arc::new(self.birthdate.finish()),
                Arc::new(self.deathdate.finish()),
                Arc::new(self.ssn.finish()),
                Arc::new(self.drivers.finish()),
                Arc::new(self.passport.finish()),
                Arc::new(self.first.finish()),
                Arc::new(self.middle.finish()),
                Arc::new(self.last.finish()),
                Arc::new(self.marital.finish()),
                Arc::new(self.race.finish()),
                Arc::new(self.ethnicity.finish()),
                Arc::new(self.gender.finish()),
                Arc::new(self.birthplace.finish()),
                Arc::new(self.address.finish()),
                Arc::new(self.city.finish()),
                Arc::new(self.state.finish()),
                Arc::new(self.county.finish()),
                Arc::new(self.fips.finish()),
                Arc::new(self.zip.finish()),
                Arc::new(self.lat.finish()),
                Arc::new(self.lon.finish()),
                Arc::new(self.healthcare_expenses.finish()),
                Arc::new(self.healthcare_coverage.finish()),
                Arc::new(self.income.finish()),
            ],
        )
    }
}

// ---------------------------------------------------------------------
// Top-level writer.
// ---------------------------------------------------------------------

pub struct SyntheaParquetWriter {
    patients_builder: PatientsBuilder,
    patients_writer: ArrowWriter<File>,
    output_dir: PathBuf,
}

impl SyntheaParquetWriter {
    pub fn create<P: AsRef<Path>>(output_dir: P) -> arrow::error::Result<Self> {
        let parquet_dir = output_dir.as_ref().join("parquet");
        create_dir_all(&parquet_dir)?;

        // Zstd level 3 — ~80% of level 9's compression at ~5× write speed.
        // Snappy is faster but ~30% larger; for chronosynthea's analytics
        // audience Zstd is the right default.
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
            .build();

        let file = File::create(parquet_dir.join("patients.parquet"))?;
        let patients_writer =
            ArrowWriter::try_new(file, PatientsBuilder::schema(), Some(props))?;

        Ok(Self {
            patients_builder: PatientsBuilder::new(),
            patients_writer,
            output_dir: output_dir.as_ref().to_path_buf(),
        })
    }

    /// Emit one patient row to patients.parquet.
    pub fn write_patient(
        &mut self,
        patient: &FullPatient,
    ) -> arrow::error::Result<()> {
        // Inline PII derivation — same logic as the CSV writer's first
        // ~150 lines, but typed for Parquet instead of byte-emit.
        let uuid = patient_uuid(patient.id);
        let birth_date = epoch_to_date(patient.birth_date_days);
        let gender = if patient.sex == 1 { "F" } else { "M" };
        let race = match patient.race {
            0 => "white",
            1 => "black",
            2 => "asian",
            3 => "hispanic",
            4 => "native",
            _ => "other",
        };
        let ethnicity = if patient.ethnicity == 1 {
            "hispanic"
        } else {
            "nonhispanic"
        };
        let age_years = years_since(patient.birth_date_days);

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

        let b = &mut self.patients_builder;
        b.id.append_value(&uuid);
        b.birthdate.append_value(&birth_date);
        b.deathdate.append_option(None::<&str>);
        b.ssn.append_value(&ssn);
        b.drivers
            .append_option(if drivers.is_empty() { None } else { Some(drivers.as_str()) });
        b.passport
            .append_option(if passport.is_empty() { None } else { Some(passport.as_str()) });
        b.first.append_value(&first);
        b.middle.append_value(&middle);
        b.last.append_value(&last);
        b.marital
            .append_option(if marital.is_empty() { None } else { Some(marital.as_str()) });
        b.race.append_value(race);
        b.ethnicity.append_value(ethnicity);
        b.gender.append_value(gender);
        b.birthplace.append_value(&birthplace);
        b.address.append_value(&address);
        b.city.append_value(city);
        b.state.append_value("Massachusetts");
        b.county.append_value(county);
        b.fips.append_value(fips);
        b.zip.append_value(zip);
        b.lat.append_value(lat as f64);
        b.lon.append_value(lon as f64);
        b.healthcare_expenses.append_value(healthcare_expenses);
        b.healthcare_coverage.append_value(healthcare_coverage);
        b.income.append_value(income);

        if b.len() >= FLUSH_ROWS {
            self.flush_patients()?;
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

    /// Drain remaining builders + close the Parquet footers.
    pub fn finish(mut self) -> arrow::error::Result<()> {
        self.flush_patients()?;
        self.patients_writer.close()?;
        Ok(())
    }

    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }
}

// ---------------------------------------------------------------------
// Helpers — kept inline so this module is independent of csv_writer.
// `hash_pick` and `lognormal_sample` are intentionally duplicated; the
// CSV writer's copies are `fn` and not exposed across modules. If/when
// the Parquet writer covers more files, refactor those into a shared
// module rather than continuing to duplicate.
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

fn years_since(birth_date_days: i32) -> u32 {
    use chrono::{Duration, NaiveDate, Utc};
    let today = Utc::now().naive_utc().date();
    let birth = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()
        + Duration::days(birth_date_days as i64);
    let years = today.signed_duration_since(birth).num_days() / 365;
    years.max(0) as u32
}
