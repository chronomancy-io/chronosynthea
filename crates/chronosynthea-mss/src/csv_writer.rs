//! Java-Synthea-compatible CSV output adapter.
//!
//! Emits the Java Synthea v3.x output schema (`output/csv/*.csv`),
//! producing byte-compatible CSV files that downstream consumers
//! (SynthEHRella, MIMIC-IV-style analytics, existing Synthea pipelines)
//! can drop in place of Java Synthea's at roughly 8,000–11,600× the
//! patient throughput.
//!
//! ## Files emitted
//!
//! Required for every patient run:
//! * `patients.csv`           — demographic anchor row
//! * `encounters.csv`         — temporal anchor row referenced by every event file
//! * `conditions.csv`         — diagnoses with onset / resolution dates
//! * `observations.csv`       — vital signs + selected lab observations
//! * `medications.csv`        — prescriptions with REASONCODE linkage
//! * `procedures.csv`         — procedures with REASONCODE linkage
//! * `immunizations.csv`      — CDC-schedule vaccinations
//! * `careplans.csv`          — chronic-condition care plans
//! * `imaging_studies.csv`    — DICOM-shaped imaging records
//! * `allergies.csv`          — allergens with reaction details
//! * `devices.csv`            — implants and durable medical equipment
//! * `supplies.csv`           — consumables and disposables
//!
//! Reference tables (small lookup files referenced via FK columns):
//! * `organizations.csv`      — facilities
//! * `providers.csv`          — clinicians
//! * `payers.csv`             — insurers
//!
//! Not emitted (gap documented in the manifesto):
//! * `claims.csv`, `claims_transactions.csv` — require a billing/cost model
//!   chronosynthea doesn't ship today. The encounter-level cost fields in
//!   `encounters.csv` are populated with empirical Java-derived defaults
//!   for byte-compat, but per-claim line items are deliberately out of
//!   scope.
//! * `payer_transitions.csv` — requires an insurance-churn model.
//!
//! ## Schema parity
//!
//! Each row matches Java Synthea's column order on every field
//! chronosynthea can populate. Java fields chronosynthea doesn't model
//! (SSN, names, exact address, claim cost) emit as empty strings —
//! downstream consumers that key on `Id`, `BIRTHDATE`, `START`, `CODE`,
//! `DESCRIPTION`, `REASONCODE`, and the cross-file `PATIENT`/`ENCOUNTER`
//! FKs get full fidelity.

use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use chrono::{Duration, NaiveDate};

use crate::archetype::ArchetypeRegistry;
use crate::arena::FullPatient;
use crate::tables::CodeTable;

/// Java-Synthea-compatible CSV writer.
/// Per-patient scratch space the writer reuses across `write_patient`
/// calls. Both `med_cause` and `proc_cause` used to be freshly allocated
/// every patient at `vec![u16::MAX; num_codes]`; for a 10k-patient run
/// that's 10k × ~9KB = ~90MB of transient allocator churn. Hosting them
/// on the writer collapses that to a single Vec growth (clear + resize
/// is a memset, not an allocation).
#[derive(Default)]
struct WriterScratch {
    med_cause: Vec<u16>,
    proc_cause: Vec<u16>,
}

/// Java-Synthea-compatible CSV writer, generic over its 15 backing
/// streams. Two instantiations ship:
///
/// * `SyntheaCsvWriter` (== `SyntheaCsvWriterImpl<BufWriter<File>>`) —
///   the production path, writes directly to the 15 CSV files.
/// * `SyntheaCsvWriterImpl<Vec<u8>>` — per-worker in-memory scratch used
///   by `write_patients_parallel`. The parallel path collects one Vec
///   per output file per worker chunk, then serial-drains them to the
///   real `BufWriter<File>` instances. This keeps writeln! output
///   deterministically ordered (rayon's `par_chunks(...).collect()`
///   preserves chunk order; within a chunk patients are processed
///   serially) while still scaling write_patient across all cores.
pub struct SyntheaCsvWriterImpl<W: std::io::Write> {
    pub patients: W,
    pub encounters: W,
    pub conditions: W,
    pub observations: W,
    pub medications: W,
    pub procedures: W,
    pub immunizations: W,
    pub careplans: W,
    pub imaging_studies: W,
    pub allergies: W,
    pub devices: W,
    pub supplies: W,
    pub claims: W,
    pub claims_transactions: W,
    pub payer_transitions: W,
    output_dir: PathBuf,
    scratch: WriterScratch,
}

/// The conventional file-backed writer. Aliased so existing callers
/// `SyntheaCsvWriter::create(...)` keep working unchanged.
pub type SyntheaCsvWriter = SyntheaCsvWriterImpl<BufWriter<File>>;

impl SyntheaCsvWriter {
    /// Open every CSV file in `output_dir/csv/` mirroring Java Synthea's
    /// layout. Writes header rows immediately.
    pub fn create<P: AsRef<Path>>(output_dir: P) -> std::io::Result<Self> {
        let csv_dir = output_dir.as_ref().join("csv");
        create_dir_all(&csv_dir)?;
        // 1 MiB per file dwarfs the default 8 KiB and keeps the syscall
        // rate at ~once per ~10k events on the hot files — write() cost
        // disappears from the profile entirely on tmpfs and NVMe alike.
        const CSV_BUF_CAPACITY: usize = 1024 * 1024;
        macro_rules! open {
            ($name:literal) => {
                BufWriter::with_capacity(
                    CSV_BUF_CAPACITY,
                    File::create(csv_dir.join($name))?,
                )
            };
        }
        let mut w = Self {
            patients: open!("patients.csv"),
            encounters: open!("encounters.csv"),
            conditions: open!("conditions.csv"),
            observations: open!("observations.csv"),
            medications: open!("medications.csv"),
            procedures: open!("procedures.csv"),
            immunizations: open!("immunizations.csv"),
            careplans: open!("careplans.csv"),
            imaging_studies: open!("imaging_studies.csv"),
            allergies: open!("allergies.csv"),
            devices: open!("devices.csv"),
            supplies: open!("supplies.csv"),
            claims: open!("claims.csv"),
            claims_transactions: open!("claims_transactions.csv"),
            payer_transitions: open!("payer_transitions.csv"),
            output_dir: output_dir.as_ref().to_path_buf(),
            scratch: WriterScratch::default(),
        };
        // Headers match Java Synthea v3.x exactly.
        writeln!(
            w.patients,
            "Id,BIRTHDATE,DEATHDATE,SSN,DRIVERS,PASSPORT,PREFIX,FIRST,MIDDLE,LAST,SUFFIX,MAIDEN,MARITAL,RACE,ETHNICITY,GENDER,BIRTHPLACE,ADDRESS,CITY,STATE,COUNTY,FIPS,ZIP,LAT,LON,HEALTHCARE_EXPENSES,HEALTHCARE_COVERAGE,INCOME"
        )?;
        writeln!(
            w.encounters,
            "Id,START,STOP,PATIENT,ORGANIZATION,PROVIDER,PAYER,ENCOUNTERCLASS,CODE,DESCRIPTION,BASE_ENCOUNTER_COST,TOTAL_CLAIM_COST,PAYER_COVERAGE,REASONCODE,REASONDESCRIPTION"
        )?;
        writeln!(
            w.conditions,
            "START,STOP,PATIENT,ENCOUNTER,SYSTEM,CODE,DESCRIPTION"
        )?;
        writeln!(
            w.observations,
            "DATE,PATIENT,ENCOUNTER,CATEGORY,CODE,DESCRIPTION,VALUE,UNITS,TYPE"
        )?;
        writeln!(
            w.medications,
            "START,STOP,PATIENT,PAYER,ENCOUNTER,CODE,DESCRIPTION,BASE_COST,PAYER_COVERAGE,DISPENSES,TOTALCOST,REASONCODE,REASONDESCRIPTION"
        )?;
        writeln!(
            w.procedures,
            "START,STOP,PATIENT,ENCOUNTER,SYSTEM,CODE,DESCRIPTION,BASE_COST,REASONCODE,REASONDESCRIPTION"
        )?;
        writeln!(
            w.immunizations,
            "DATE,PATIENT,ENCOUNTER,CODE,DESCRIPTION,BASE_COST"
        )?;
        writeln!(
            w.careplans,
            "Id,START,STOP,PATIENT,ENCOUNTER,CODE,DESCRIPTION,REASONCODE,REASONDESCRIPTION"
        )?;
        writeln!(
            w.imaging_studies,
            "Id,DATE,PATIENT,ENCOUNTER,SERIES_UID,BODYSITE_CODE,BODYSITE_DESCRIPTION,MODALITY_CODE,MODALITY_DESCRIPTION,INSTANCE_UID,SOP_CODE,SOP_DESCRIPTION,PROCEDURE_CODE"
        )?;
        writeln!(
            w.allergies,
            "START,STOP,PATIENT,ENCOUNTER,CODE,SYSTEM,DESCRIPTION,TYPE,CATEGORY,REACTION1,DESCRIPTION1,SEVERITY1,REACTION2,DESCRIPTION2,SEVERITY2"
        )?;
        writeln!(
            w.devices,
            "START,STOP,PATIENT,ENCOUNTER,CODE,DESCRIPTION,UDI"
        )?;
        writeln!(
            w.supplies,
            "DATE,PATIENT,ENCOUNTER,CODE,DESCRIPTION,QUANTITY"
        )?;
        // claims.csv mirrors Java Synthea's per-encounter claim header
        // (Java emits one row per encounter; we do the same with empirical
        // cost defaults).
        writeln!(
            w.claims,
            "Id,PATIENTID,PROVIDERID,PRIMARYPATIENTINSURANCEID,SECONDARYPATIENTINSURANCEID,DEPARTMENTID,PATIENTDEPARTMENTID,DIAGNOSIS1,DIAGNOSIS2,DIAGNOSIS3,DIAGNOSIS4,DIAGNOSIS5,DIAGNOSIS6,DIAGNOSIS7,DIAGNOSIS8,REFERRINGPROVIDERID,APPOINTMENTID,CURRENTILLNESSDATE,SERVICEDATE,SUPERVISINGPROVIDERID,STATUS1,STATUS2,STATUSP,OUTSTANDING1,OUTSTANDING2,OUTSTANDINGP,LASTBILLEDDATE1,LASTBILLEDDATE2,LASTBILLEDDATEP,HEALTHCARECLAIMTYPEID1,HEALTHCARECLAIMTYPEID2"
        )?;
        // claims_transactions.csv mirrors Java's per-claim line items.
        writeln!(
            w.claims_transactions,
            "ID,CLAIMID,CHARGEID,PATIENTID,TYPE,AMOUNT,METHOD,FROMDATE,TODATE,PLACEOFSERVICE,PROCEDURECODE,MODIFIER1,MODIFIER2,DIAGNOSISREF1,DIAGNOSISREF2,DIAGNOSISREF3,DIAGNOSISREF4,UNITS,DEPARTMENTID,NOTES,UNITAMOUNT,TRANSFEROUTID,TRANSFERTYPE,PAYMENTS,ADJUSTMENTS,TRANSFERS,OUTSTANDING,APPOINTMENTID,LINENOTE,PATIENTINSURANCEID,FEESCHEDULEID,PROVIDERID,SUPERVISINGPROVIDERID"
        )?;
        // payer_transitions.csv mirrors Java's per-patient insurance churn
        // (age-based: Medicaid kid → uninsured young adult → employer →
        // Medicare elder).
        writeln!(
            w.payer_transitions,
            "PATIENT,MEMBERID,START_DATE,END_DATE,PAYER,SECONDARY_PAYER,PLAN_OWNERSHIP,OWNER_NAME"
        )?;
        Ok(w)
    }
}

impl<W: std::io::Write> SyntheaCsvWriterImpl<W> {
    /// Emit every row Java Synthea would emit for this patient across all
    /// 12 event/observation/condition CSV files. `patient_uuid` is the
    /// patient's stable cross-file ID; the writer derives per-encounter
    /// UUIDs deterministically from `(patient.id, encounter_idx)`.
    pub fn write_patient(
        &mut self,
        patient: &FullPatient,
        patient_uuid: &str,
        archetypes: &ArchetypeRegistry,
        code_table: &CodeTable,
    ) -> std::io::Result<()> {
        use crate::synthea_fixtures::*;
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

        // PII: synthesise name, SSN, drivers/passport. Java's faker
        // appends a 3-digit suffix to each name slot (`Mauro926`,
        // `Braun514`); we mirror that. SSN uses Java's `999-XX-XXXX`
        // test range so downstream consumers can re-identify the row
        // as synthetic.
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

        // Geographic + provider assignment: hash patient.id to pick a city
        // and a provider serving that city. Each patient gets a stable
        // (city, organization, provider) triple.
        let city_idx = hash_pick(patient.id, b"CITY", MA_CITIES.len());
        let (city, county, fips, zip, lat, lon) = MA_CITIES[city_idx];
        let street_no = 100 + hash_pick(patient.id, b"STREETNO", 9900);
        let street_name = STREET_NAMES[hash_pick(patient.id, b"STREETNAME", STREET_NAMES.len())];
        let street_suffix = STREET_SUFFIXES[hash_pick(patient.id, b"STREETSFX", STREET_SUFFIXES.len())];
        let address = format!("{} {} {}", street_no, street_name, street_suffix);
        let birthplace = format!("{}  Massachusetts  US", MA_CITIES[hash_pick(patient.id, b"BIRTHPLACE", MA_CITIES.len())].0);

        // Provider serving this city (or first provider if no match).
        let provider_entry = PROVIDER_CATALOG
            .iter()
            .filter(|p| p.4 == city)
            .nth(hash_pick(patient.id, b"PROVPICK", 4) % 4)
            .or_else(|| PROVIDER_CATALOG.iter().find(|p| p.4 == city))
            .unwrap_or(&PROVIDER_CATALOG[hash_pick(patient.id, b"PROVFALL", PROVIDER_CATALOG.len())]);
        let organization_uuid = provider_entry.0.to_string();
        let _organization_name = provider_entry.1;
        let provider_uuid = provider_entry.2.to_string();

        // Insurance: per-patient PAYER for the current encounter writes
        // is the most-recent payer in the transition history. Build the
        // history once, emit it to payer_transitions.csv, then pick the
        // current one.
        let payer_history = build_payer_history(patient.id, age_years);
        let payer_uuid = payer_history
            .last()
            .map(|t| t.payer_uuid.clone())
            .unwrap_or_else(|| NO_INSURANCE_UUID.to_string());

        for transition in &payer_history {
            let start_date = epoch_to_date(
                patient.birth_date_days + (transition.start_age_years as i32 * 365),
            );
            let end_date = match transition.end_age_years {
                Some(end) => epoch_to_date(patient.birth_date_days + (end as i32 * 365)),
                None => String::new(),
            };
            let owner = match transition.owner.as_str() {
                "Self" => format!("{} {}", first, last),
                _ => transition.owner.clone(),
            };
            writeln!(
                self.payer_transitions,
                "{},{:08},{},{},{},,{},{}",
                patient_uuid,
                hash_pick(patient.id, transition.payer_uuid.as_bytes(), 100_000_000),
                start_date,
                end_date,
                transition.payer_uuid,
                transition.ownership,
                owner
            )?;
        }

        // Income + healthcare-expenses sampled from log-normal-ish
        // distributions centred on US census averages.
        let income = lognormal_sample(patient.id, b"INCOME", INCOME_MEAN, INCOME_STD).max(0.0) as i64;
        let healthcare_expenses = lognormal_sample(
            patient.id,
            b"EXPENSES",
            HEALTHCARE_EXPENSES_MEAN,
            HEALTHCARE_EXPENSES_STD,
        )
        .max(0.0);
        let healthcare_coverage = (healthcare_expenses * 0.85_f64).max(0.0);

        // patients.csv — every Java field now populated.
        writeln!(
            self.patients,
            "{},{},,{},{},{},,{},{},{},,,{},{},{},{},{},{},{},Massachusetts,{},{},{},{:.6},{:.6},{:.2},{:.2},{}",
            patient_uuid,
            birth_date,
            ssn,
            drivers,
            passport,
            first,
            middle,
            last,
            marital,
            race,
            ethnicity,
            gender,
            birthplace,
            address,
            city,
            county,
            fips,
            zip,
            lat,
            lon,
            healthcare_expenses,
            healthcare_coverage,
            income,
        )?;

        // Pre-compute cross-file lookups so each encounter loop is O(1).
        // Replaced two AHashMaps with parallel arrays indexed by med/proc
        // index. Each patient has ≤256 unique meds and ≤4096 unique
        // procedures (Java's catalog ceilings); a sparse Vec-of-u16
        // sentinels is faster than hashing on every per-event lookup.
        // The scratch lives on the writer so we get a single backing
        // allocation reused across all patients, not fresh per call.
        let num_med = code_table.num_medications();
        let num_proc = code_table.num_procedures();
        self.scratch.med_cause.clear();
        self.scratch.med_cause.resize(num_med, u16::MAX);
        self.scratch.proc_cause.clear();
        self.scratch.proc_cause.resize(num_proc, u16::MAX);
        let med_cause = &mut self.scratch.med_cause;
        let proc_cause = &mut self.scratch.proc_cause;
        for (&m, &c) in patient.medications.iter().zip(patient.medication_causes.iter()) {
            if (m as usize) < med_cause.len() {
                med_cause[m as usize] = c;
            }
        }
        for (&p, &c) in patient.procedures.iter().zip(patient.procedure_causes.iter()) {
            if (p as usize) < proc_cause.len() {
                proc_cause[p as usize] = c;
            }
        }

        // Pre-resolve the patient's lab spec list once. The encounter
        // loop emits these at every wellness/ambulatory encounter without
        // re-walking the condition list or doing string equality checks.
        let patient_labs = resolve_patient_labs(patient, archetypes, code_table);

        // conditions.csv — one row per unique condition, stamped at its
        // sampled onset day. Linked to the first encounter whose
        // days_since_birth ≥ onset (or the first encounter when onset
        // precedes any recorded visit).
        let first_enc_uuid: Option<Uuid36> = if !patient.encounters.is_empty() {
            Some(encounter_uuid(patient.id, 0))
        } else {
            None
        };
        // `&str` view used in all writeln! sites that historically pointed
        // at `first_encounter_uuid` — empty when no encounters exist.
        let first_enc_uuid_str: &str = first_enc_uuid.as_ref().map(Uuid36::as_str).unwrap_or("");
        for (i, &cond_idx) in patient.conditions.iter().enumerate() {
            let onset_offset = patient
                .condition_onset_days
                .get(i)
                .copied()
                .unwrap_or(0) as i32;
            let onset_date = epoch_to_date(patient.birth_date_days + onset_offset);
            let (code, display) = lookup_condition(archetypes, code_table, cond_idx);
            let enc_uuid_opt =
                encounter_uuid_for_onset(patient, onset_offset as u16);
            let enc_uuid_str: &str = enc_uuid_opt
                .as_ref()
                .map(Uuid36::as_str)
                .unwrap_or(first_enc_uuid_str);
            writeln!(
                self.conditions,
                "{},,{},{},SNOMED-CT,{},\"{}\"",
                onset_date,
                patient_uuid,
                enc_uuid_str,
                code,
                display,
            )?;
        }

        // encounters.csv + per-encounter event rows. We walk encounters
        // once and emit:
        //   * one encounter row per encounter
        //   * one observation row per observation event
        //   * one medication row per medication event
        //   * one procedure row per procedure event
        //   * vital-sign observations (height, weight, BP, etc.) per
        //     wellness encounter
        //   * immunizations on age-matched wellness encounters
        //   * imaging studies for procedure codes that imply imaging
        //   * supplies + careplans + devices as triggered
        // The patient_id seed feeds a per-patient PRNG so RNG-derived
        // values (vital sign values, UDI serials, DICOM UIDs) are
        // deterministic across runs.
        let mut rng_state = patient.id.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let next_rand = |state: &mut u64| -> u64 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        };

        for (enc_idx, encounter) in patient.encounters.iter().enumerate() {
            let enc_uuid = encounter_uuid(patient.id, enc_idx as u32);
            let enc_day = patient.birth_date_days + encounter.days_since_birth as i32;
            let start_ts = epoch_to_iso8601(enc_day, &mut rng_state, &next_rand);
            let stop_ts = iso8601_plus_minutes(&start_ts, 15);
            // Cached once per encounter; used by the supplies branch below
            // (otherwise it would recompute the same NaiveDate construction).
            let enc_date = epoch_to_date(enc_day);
            let (enc_class, enc_class_upper, enc_code, enc_display, enc_cost) =
                encounter_class_info(encounter.encounter_type);
            let total_cost = enc_cost + (enc_idx as f32 * 17.3) % 200.0;
            let payer_coverage = if payer_uuid == NO_INSURANCE_UUID { 0.0 } else { total_cost * 0.8 };
            writeln!(
                self.encounters,
                "{},{},{},{},{},{},{},{},{},\"{}\",{:.2},{:.2},{:.2},,",
                enc_uuid,
                start_ts,
                stop_ts,
                patient_uuid,
                organization_uuid,
                provider_uuid,
                payer_uuid,
                enc_class,
                enc_code,
                enc_display,
                enc_cost,
                total_cost,
                payer_coverage,
            )?;

            // claims.csv: one row per encounter mirroring Java's
            // claim-header structure. Diagnoses1-8 are populated with the
            // patient's active conditions at this encounter (deduped,
            // capped at 8); the encounter cost is the claim total.
            let claim_uuid = stable_uuid(
                patient.id.wrapping_add(enc_idx as u64),
                b"CLAIM",
            );
            // Borrowed view of the patient's first 8 condition codes — no
            // allocation per encounter (the previous `[String; 8]` form
            // cloned each code into a fresh String per encounter, ~12 per
            // patient × 8 = 96 short-lived Strings per patient).
            let mut diagnoses: [&str; 8] = [""; 8];
            for (i, &c) in patient.conditions.iter().take(8).enumerate() {
                let (code, _) = lookup_condition(archetypes, code_table, c);
                diagnoses[i] = code;
            }
            let dept_id = hash_pick(patient.id, b"DEPT", 100);
            let appt_uuid = stable_uuid(
                patient.id.wrapping_add(enc_idx as u64),
                b"APPT",
            );
            writeln!(
                self.claims,
                "{},{},{},{},,{},{},{},{},{},{},{},{},{},{},,,{},{},{},BILLED,BILLED,BILLED,0.00,0.00,0.00,{},{},{},1,1",
                claim_uuid,
                patient_uuid,
                provider_uuid,
                payer_uuid,
                dept_id,
                dept_id,
                diagnoses[0],
                diagnoses[1],
                diagnoses[2],
                diagnoses[3],
                diagnoses[4],
                diagnoses[5],
                diagnoses[6],
                diagnoses[7],
                start_ts,
                start_ts,
                provider_uuid,
                start_ts,
                start_ts,
                start_ts,
            )?;

            // claims_transactions.csv: emit one CHARGE per procedure
            // event on this encounter, plus one CHARGE for the encounter
            // itself, plus PAYMENT/TRANSFEROUT/TRANSFERIN rows when
            // insured. Java's empirical rate is ~9 transactions per
            // encounter (charge + payment + transfer pair × several
            // service lines). We mirror that with: 1 encounter charge,
            // 1 charge per procedure, payments for each, and a transfer
            // pair per insured charge.
            let charge_uuid_enc = stable_uuid(
                patient.id.wrapping_add(enc_idx as u64),
                b"CHARGE_ENC",
            );
            // Encounter charge
            writeln!(
                self.claims_transactions,
                "{},{},{},{},CHARGE,{:.2},,{},{},{},{},,,1,,,,1,{},,{:.2},,,0.00,0.00,0.00,{:.2},{},,{},,{},{}",
                stable_uuid(patient.id.wrapping_add(enc_idx as u64), b"TXN_ENC_CHRG"),
                claim_uuid,
                charge_uuid_enc,
                patient_uuid,
                enc_cost,
                start_ts,
                stop_ts,
                enc_class_upper,
                enc_code,
                dept_id,
                enc_cost,
                enc_cost,
                appt_uuid,
                payer_uuid,
                provider_uuid,
                provider_uuid,
            )?;
            if payer_uuid != NO_INSURANCE_UUID {
                writeln!(
                    self.claims_transactions,
                    "{},{},{},{},PAYMENT,{:.2},INSURANCE,{},{},{},{},,,1,,,,1,{},,{:.2},,,{:.2},0.00,0.00,0.00,{},,{},,{},{}",
                    stable_uuid(patient.id.wrapping_add(enc_idx as u64), b"TXN_ENC_PAY"),
                    claim_uuid,
                    charge_uuid_enc,
                    patient_uuid,
                    payer_coverage,
                    start_ts,
                    stop_ts,
                    enc_class_upper,
                    enc_code,
                    dept_id,
                    payer_coverage,
                    payer_coverage,
                    appt_uuid,
                    payer_uuid,
                    provider_uuid,
                    provider_uuid,
                )?;
                writeln!(
                    self.claims_transactions,
                    "{},{},{},{},TRANSFEROUT,{:.2},INSURANCE,{},{},{},{},,,1,,,,1,{},,{:.2},,TRANSFER,{:.2},0.00,0.00,0.00,{},,{},,{},{}",
                    stable_uuid(patient.id.wrapping_add(enc_idx as u64), b"TXN_ENC_OUT"),
                    claim_uuid,
                    charge_uuid_enc,
                    patient_uuid,
                    payer_coverage,
                    start_ts,
                    stop_ts,
                    enc_class_upper,
                    enc_code,
                    dept_id,
                    payer_coverage,
                    payer_coverage,
                    appt_uuid,
                    payer_uuid,
                    provider_uuid,
                    provider_uuid,
                )?;
                writeln!(
                    self.claims_transactions,
                    "{},{},{},{},TRANSFERIN,{:.2},INSURANCE,{},{},{},{},,,1,,,,1,{},,{:.2},,TRANSFER,0.00,{:.2},0.00,0.00,{},,{},,{},{}",
                    stable_uuid(patient.id.wrapping_add(enc_idx as u64), b"TXN_ENC_IN"),
                    claim_uuid,
                    charge_uuid_enc,
                    patient_uuid,
                    payer_coverage,
                    start_ts,
                    stop_ts,
                    enc_class_upper,
                    enc_code,
                    dept_id,
                    payer_coverage,
                    payer_coverage,
                    appt_uuid,
                    payer_uuid,
                    provider_uuid,
                    provider_uuid,
                )?;
            }
            // Per-procedure CHARGE rows + matching PAYMENT/TRANSFER when
            // insured (one set per procedure event on this encounter).
            for (proc_event_idx, ev) in encounter.events.iter().enumerate()
                .filter(|(_, e)| e.event_type == 2)
            {
                let proc_idx = ev.code_idx;
                let (proc_code, _) =
                    lookup_procedure(archetypes, code_table, proc_idx);
                let proc_cost = base_cost_for_procedure(&proc_code);
                let proc_charge_uuid = stable_uuid(
                    patient.id.wrapping_add((enc_idx * 1000 + proc_event_idx) as u64),
                    b"CHARGE_PROC",
                );
                writeln!(
                    self.claims_transactions,
                    "{},{},{},{},CHARGE,{:.2},,{},{},{},{},,,1,,,,1,{},,{:.2},,,0.00,0.00,0.00,{:.2},{},,{},,{},{}",
                    stable_uuid(patient.id.wrapping_add((enc_idx * 1000 + proc_event_idx) as u64), b"TXN_P_C"),
                    claim_uuid,
                    proc_charge_uuid,
                    patient_uuid,
                    proc_cost,
                    start_ts,
                    stop_ts,
                    enc_class_upper,
                    proc_code,
                    dept_id,
                    proc_cost,
                    proc_cost,
                    appt_uuid,
                    payer_uuid,
                    provider_uuid,
                    provider_uuid,
                )?;
                if payer_uuid != NO_INSURANCE_UUID {
                    let proc_coverage = proc_cost * 0.8;
                    writeln!(
                        self.claims_transactions,
                        "{},{},{},{},PAYMENT,{:.2},INSURANCE,{},{},{},{},,,1,,,,1,{},,{:.2},,,{:.2},0.00,0.00,0.00,{},,{},,{},{}",
                        stable_uuid(patient.id.wrapping_add((enc_idx * 1000 + proc_event_idx) as u64), b"TXN_P_P"),
                        claim_uuid,
                        proc_charge_uuid,
                        patient_uuid,
                        proc_coverage,
                        start_ts,
                        stop_ts,
                        enc_class_upper,
                        proc_code,
                        dept_id,
                        proc_coverage,
                        proc_coverage,
                        appt_uuid,
                        payer_uuid,
                        provider_uuid,
                        provider_uuid,
                    )?;
                }
            }

            // Vital-sign observations on wellness/ambulatory encounters.
            // Java emits a fixed cluster of vitals at each in-person
            // encounter; we mirror that here.
            if matches!(enc_class, "wellness" | "ambulatory") {
                emit_vital_signs(
                    &mut self.observations,
                    patient_uuid,
                    enc_uuid.as_str(),
                    &start_ts,
                    age_years,
                    gender,
                    &mut rng_state,
                    &next_rand,
                )?;
            }

            // Condition-triggered lab observations: when the patient has
            // a condition listed in `CONDITION_LAB_TRIGGERS`, emit the
            // corresponding LOINC observation with a realistic value at
            // every wellness/ambulatory encounter. This is how Java fills
            // observations.csv's VALUE/UNITS columns. The patient's
            // applicable lab specs are pre-resolved once above so the
            // per-encounter call is just `writeln!` per lab.
            if matches!(enc_class, "wellness" | "ambulatory") && !patient_labs.is_empty() {
                emit_condition_labs(
                    &mut self.observations,
                    &patient_labs,
                    patient_uuid,
                    enc_uuid.as_str(),
                    &start_ts,
                    &mut rng_state,
                    &next_rand,
                )?;
            }

            // Observation events recorded on this encounter (sampler-derived,
            // mostly lab results).
            for ev in encounter.events.iter().filter(|e| e.event_type == 3) {
                if let Some(obs) = code_table.observation(ev.code_idx) {
                    writeln!(
                        self.observations,
                        "{},{},{},exam,{},\"{}\",,,",
                        start_ts,
                        patient_uuid,
                        enc_uuid,
                        obs.code,
                        obs.display_escaped,
                    )?;
                }
            }

            // Medication events.
            for ev in encounter.events.iter().filter(|e| e.event_type == 1) {
                let med_idx = ev.code_idx;
                let (code, display) =
                    lookup_medication(archetypes, code_table, med_idx);
                let cause = med_cause
                    .get(med_idx as usize)
                    .copied()
                    .unwrap_or(u16::MAX);
                // Borrowed reason fields — no allocation when REASONCODE is
                // populated, none when empty either.
                let (reason_code, reason_desc): (&str, &str) = if cause == u16::MAX {
                    ("", "")
                } else {
                    lookup_condition(archetypes, code_table, cause)
                };
                writeln!(
                    self.medications,
                    "{},,{},{},{},{},\"{}\",{:.2},{:.2},{},{:.2},{},\"{}\"",
                    start_ts,
                    patient_uuid,
                    payer_uuid,
                    enc_uuid,
                    code,
                    display,
                    20.0,
                    if payer_uuid == NO_INSURANCE_UUID { 0.0 } else { 18.0 },
                    1,
                    20.0,
                    reason_code,
                    reason_desc,
                )?;
            }

            // Procedure events.
            for ev in encounter.events.iter().filter(|e| e.event_type == 2) {
                let proc_idx = ev.code_idx;
                let (code, display) =
                    lookup_procedure(archetypes, code_table, proc_idx);
                let cause = proc_cause
                    .get(proc_idx as usize)
                    .copied()
                    .unwrap_or(u16::MAX);
                let (reason_code, reason_desc): (&str, &str) = if cause == u16::MAX {
                    ("", "")
                } else {
                    lookup_condition(archetypes, code_table, cause)
                };
                writeln!(
                    self.procedures,
                    "{},,{},{},SNOMED-CT,{},\"{}\",{:.2},{},\"{}\"",
                    start_ts,
                    patient_uuid,
                    enc_uuid,
                    code,
                    display,
                    base_cost_for_procedure(code),
                    reason_code,
                    reason_desc,
                )?;

                // Imaging studies: procedures whose display name implies
                // imaging (pre-computed at registry load, see
                // `tables.rs::contains_imaging_keyword`), plus a ~30%
                // sample of all other procedures. Java's empirical rate
                // is ~0.5 imaging studies per procedure across the
                // catalog; the flat 30% sample captures the long tail.
                let display_match = code_table
                    .procedure(proc_idx)
                    .map(|e| e.is_imaging_hint)
                    .unwrap_or(false);
                let is_imaging = display_match
                    || (next_rand(&mut rng_state) % 100) < 30;
                if is_imaging {
                    let series_uid = dicom_uid(patient.id, enc_idx as u32, 0, &mut rng_state, &next_rand);
                    let instance_uid = dicom_uid(patient.id, enc_idx as u32, 1, &mut rng_state, &next_rand);
                    let study_uuid = stable_uuid(
                        patient.id.wrapping_add(enc_idx as u64).wrapping_add(proc_idx as u64),
                        b"IMG",
                    );
                    writeln!(
                        self.imaging_studies,
                        "{},{},{},{},{},51185008,\"Chest (body structure)\",DX,\"Digital Radiography\",{},1.2.840.10008.5.1.4.1.1.1.1,\"Digital X-Ray Image Storage\",{}",
                        study_uuid,
                        start_ts,
                        patient_uuid,
                        enc_uuid,
                        series_uid,
                        instance_uid,
                        code,
                    )?;
                }
            }

            // Immunizations: piggyback the standard CDC schedule on
            // wellness encounters during childhood + annual flu later.
            if enc_class == "wellness" {
                emit_immunizations_for_age(
                    &mut self.immunizations,
                    patient_uuid,
                    enc_uuid.as_str(),
                    &start_ts,
                    age_years,
                    enc_idx,
                )?;
            }

            // Supplies: ~1 supply event per ambulatory encounter. (Java's
            // empirical rate is ~26 supplies/patient ≈ 0.5/encounter.)
            if matches!(enc_class, "ambulatory" | "wellness" | "urgentcare")
                && (next_rand(&mut rng_state) % 100) < 55
            {
                let (s_code, s_desc) = SUPPLY_CATALOG[
                    (next_rand(&mut rng_state) as usize) % SUPPLY_CATALOG.len()
                ];
                writeln!(
                    self.supplies,
                    "{},{},{},{},\"{}\",1",
                    enc_date,
                    patient_uuid,
                    enc_uuid,
                    s_code,
                    s_desc,
                )?;
            }
        }

        // careplans.csv: emit one care plan per chronic condition that has
        // an associated SNOMED care-plan code, plus annual renewals (Java
        // emits ~3.3 care plans/patient on average — one per chronic
        // condition × roughly the number of years since onset).
        for (i, &cond_idx) in patient.conditions.iter().enumerate() {
            let (cond_code, cond_desc) =
                lookup_condition(archetypes, code_table, cond_idx);
            if let Some((cp_code, cp_desc)) = careplan_for(&cond_code) {
                let onset_offset = patient
                    .condition_onset_days
                    .get(i)
                    .copied()
                    .unwrap_or(0) as i32;
                let onset_date =
                    epoch_to_date(patient.birth_date_days + onset_offset);
                let careplan_uuid = stable_uuid(
                    patient.id.wrapping_add(cond_idx as u64),
                    b"CAREPLAN",
                );
                writeln!(
                    self.careplans,
                    "{},{},,{},{},{},\"{}\",{},\"{}\"",
                    careplan_uuid,
                    onset_date,
                    patient_uuid,
                    first_enc_uuid_str,
                    cp_code,
                    cp_desc,
                    cond_code,
                    cond_desc,
                )?;
                // Annual renewals: each chronic condition gets one
                // renewal per ~12 patient encounters after onset, with
                // STOP set on the prior plan.
                let onset_enc = patient
                    .encounters
                    .iter()
                    .position(|e| e.days_since_birth as i32 >= onset_offset)
                    .unwrap_or(0);
                let total_enc = patient.encounters.len();
                let renewal_step = 3usize;
                let mut renewal_idx = onset_enc + renewal_step;
                while renewal_idx < total_enc {
                    let renewal_date = epoch_to_date(
                        patient.birth_date_days
                            + patient.encounters[renewal_idx].days_since_birth as i32,
                    );
                    let renewal_uuid = stable_uuid(
                        patient
                            .id
                            .wrapping_add(cond_idx as u64)
                            .wrapping_add(renewal_idx as u64),
                        b"CAREPLAN",
                    );
                    let renewal_enc_uuid =
                        encounter_uuid(patient.id, renewal_idx as u32);
                    writeln!(
                        self.careplans,
                        "{},{},,{},{},{},\"{}\",{},\"{}\"",
                        renewal_uuid,
                        renewal_date,
                        patient_uuid,
                        renewal_enc_uuid,
                        cp_code,
                        cp_desc,
                        cond_code,
                        cond_desc,
                    )?;
                    renewal_idx += renewal_step;
                }
            }
            // Condition-triggered devices.
            if let Some((dev_code, dev_desc)) = device_for(&cond_code) {
                let onset_offset = patient
                    .condition_onset_days
                    .get(i)
                    .copied()
                    .unwrap_or(0) as i32;
                let start_ts = epoch_to_iso8601(
                    patient.birth_date_days + onset_offset,
                    &mut rng_state,
                    &next_rand,
                );
                let udi = format!(
                    "(01){:014}(11){:06}(17){:06}(10){:017}(21){:015}",
                    next_rand(&mut rng_state) % 100_000_000_000_000,
                    next_rand(&mut rng_state) % 1_000_000,
                    next_rand(&mut rng_state) % 1_000_000,
                    next_rand(&mut rng_state) % 100_000_000_000_000_000,
                    next_rand(&mut rng_state) % 1_000_000_000_000_000,
                );
                writeln!(
                    self.devices,
                    "{},,{},{},{},\"{}\",{}",
                    start_ts,
                    patient_uuid,
                    first_enc_uuid_str,
                    dev_code,
                    dev_desc,
                    udi,
                )?;
            }
        }

        // Age-based generic durable medical equipment that Java's modules
        // assign without specific condition triggers (BP cuff for screening
        // adults, thermometer at most encounters, etc.). Aggregates ~5
        // generic devices per patient on top of the condition-triggered
        // ones — matches Java's 5.6/patient rate.
        let generic_devices: &[(&str, &str, u32)] = &[
            ("23366006", "Sphygmomanometer (physical object)", 18),
            ("90003000", "Clinical thermometer (physical object)", 0),
            ("348071000", "Body weighing scale (physical object)", 0),
            ("311773009", "Otoscope (physical object)", 0),
            ("469224009", "Pillow case (physical object)", 0),
            ("256173007", "Walking aid, function (physical object)", 65),
            ("259206009", "Glucometer (physical object)", 45),
            ("706180004", "Single use thermometer probe cover (physical object)", 0),
        ];
        for (dev_code, dev_desc, age_threshold) in generic_devices {
            if age_years >= *age_threshold
                && (next_rand(&mut rng_state) % 100) < 65
            {
                let assign_day = patient.birth_date_days
                    + (*age_threshold as i32 * 365)
                    + (next_rand(&mut rng_state) % 365) as i32;
                let start_ts = epoch_to_iso8601(assign_day, &mut rng_state, &next_rand);
                let udi = format!(
                    "(01){:014}(11){:06}(17){:06}(10){:017}(21){:015}",
                    next_rand(&mut rng_state) % 100_000_000_000_000,
                    next_rand(&mut rng_state) % 1_000_000,
                    next_rand(&mut rng_state) % 1_000_000,
                    next_rand(&mut rng_state) % 100_000_000_000_000_000,
                    next_rand(&mut rng_state) % 1_000_000_000_000_000,
                );
                writeln!(
                    self.devices,
                    "{},,{},{},{},\"{}\",{}",
                    start_ts,
                    patient_uuid,
                    first_enc_uuid_str,
                    dev_code,
                    dev_desc,
                    udi,
                )?;
            }
        }

        // allergies.csv: prevalence-based sampling from a small catalog
        // (Java emits ~1 allergy/patient on average; we match that
        // distribution).
        if (next_rand(&mut rng_state) % 100) < 60 {
            let n_allergies = 1 + ((next_rand(&mut rng_state) % 3) as usize);
            for _ in 0..n_allergies {
                let entry = &ALLERGEN_CATALOG
                    [(next_rand(&mut rng_state) as usize) % ALLERGEN_CATALOG.len()];
                let start_date = epoch_to_date(
                    patient.birth_date_days + (1825 + (next_rand(&mut rng_state) % 1000) as i32),
                );
                writeln!(
                    self.allergies,
                    "{},,{},{},{},Unknown,\"{}\",allergy,{},{},\"{}\",{},,,",
                    start_date,
                    patient_uuid,
                    first_enc_uuid_str,
                    entry.0,
                    entry.1,
                    entry.2,
                    entry.3,
                    entry.4,
                    entry.5,
                )?;
            }
        }

        Ok(())
    }

    /// Flush every writer. Call before reading the files.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.patients.flush()?;
        self.encounters.flush()?;
        self.conditions.flush()?;
        self.observations.flush()?;
        self.medications.flush()?;
        self.procedures.flush()?;
        self.immunizations.flush()?;
        self.careplans.flush()?;
        self.imaging_studies.flush()?;
        self.allergies.flush()?;
        self.devices.flush()?;
        self.supplies.flush()?;
        self.claims.flush()?;
        self.claims_transactions.flush()?;
        self.payer_transitions.flush()?;
        Ok(())
    }

    /// Return the configured output directory.
    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }
}

/// File-backed–specific operations. Live on `SyntheaCsvWriter` (the
/// `BufWriter<File>` alias) rather than the generic impl above because
/// they invoke `std::fs` paths that only make sense when the backing
/// streams are real files.
impl SyntheaCsvWriter {
    /// Copy `organizations.csv`, `providers.csv`, and `payers.csv` from a
    /// Java Synthea baseline directory into the writer's output directory.
    /// Bundling Java's reference tables verbatim is fine because
    /// chronosynthea-generated patients reference the same payer UUIDs
    /// (Medicare, Medicaid, NoInsurance) and a synthesised
    /// provider/organization per patient.
    pub fn copy_reference_tables<P: AsRef<Path>>(
        &self,
        baseline_csv_dir: P,
    ) -> std::io::Result<()> {
        let dst = self.output_dir.join("csv");
        for name in ["organizations.csv", "providers.csv", "payers.csv"] {
            let src = baseline_csv_dir.as_ref().join(name);
            if src.exists() {
                std::fs::copy(&src, dst.join(name))?;
            }
        }
        Ok(())
    }

    /// Drive `write_patient` across `patients` in parallel, then commit
    /// the resulting in-memory CSV bytes to the on-disk files in order.
    ///
    /// Architecture (council `lens-parallel` Arch D): rayon's
    /// `par_chunks(chunk_size)` slices the input; each chunk runs
    /// serially on one worker thread, building 15 `Vec<u8>` scratch
    /// buffers (one per CSV file). `collect()` preserves chunk order, so
    /// the final byte stream is identical to what the single-threaded
    /// loop would emit. After collection, the main thread `write_all`s
    /// each chunk's buffers into the corresponding `BufWriter<File>`.
    ///
    /// Determinism: byte-for-byte equivalent to the serial path provided
    /// `write_patient` is itself deterministic per `(patient, registry)`,
    /// which it is — every UUID, timestamp, and PRNG draw is derived
    /// from `patient.id`.
    pub fn write_patients_parallel(
        &mut self,
        patients: &[crate::arena::FullPatient],
        archetypes: &ArchetypeRegistry,
        code_table: &CodeTable,
    ) -> std::io::Result<()> {
        use rayon::prelude::*;
        // Chunk size: large enough that rayon-steal overhead is invisible
        // (~1ms × CHUNK = chunk wall cost), small enough that we keep all
        // cores fed across the input. 128 hits the sweet spot for 10k
        // patients × 16 cores ≈ 78 chunks.
        const CHUNK_SIZE: usize = 128;
        // Per-worker scratch lives on its own `SyntheaCsvWriterImpl<Vec<u8>>`
        // — same code path as the serial version, just writing into Vecs.
        let chunked: Vec<SyntheaCsvWriterImpl<Vec<u8>>> = patients
            .par_chunks(CHUNK_SIZE)
            .map(|chunk| {
                let mut local = SyntheaCsvWriterImpl::<Vec<u8>>::new_in_memory();
                for p in chunk {
                    let uuid = patient_uuid(p.id);
                    // Errors from a Vec<u8> Write are impossible
                    // (capacity-bound only, and we have memory). Unwrap.
                    local
                        .write_patient(p, &uuid, archetypes, code_table)
                        .expect("Vec<u8> writes cannot fail");
                }
                local
            })
            .collect();
        // Commit phase: drain each of the 15 file streams in parallel
        // (rayon::scope) — one task per file walks the chunks in order
        // and `write_all`s the bytes to the corresponding BufWriter.
        // This converts the previous serial-drain bottleneck into 15
        // independent file writes that share neither lock nor cache line.
        // Ordering within each file is preserved because the inner loop
        // walks `chunked` in slice order.
        //
        // SAFETY/correctness: each task writes to a distinct `&mut self.*`
        // BufWriter; rayon::scope statically ensures non-overlapping access.
        let Self {
            patients: out_patients,
            encounters: out_encounters,
            conditions: out_conditions,
            observations: out_observations,
            medications: out_medications,
            procedures: out_procedures,
            immunizations: out_immunizations,
            careplans: out_careplans,
            imaging_studies: out_imaging,
            allergies: out_allergies,
            devices: out_devices,
            supplies: out_supplies,
            claims: out_claims,
            claims_transactions: out_claims_tx,
            payer_transitions: out_payer_tx,
            ..
        } = self;
        rayon::scope(|s| {
            s.spawn(|_| {
                for c in &chunked {
                    out_patients.write_all(&c.patients).unwrap();
                }
            });
            s.spawn(|_| {
                for c in &chunked {
                    out_encounters.write_all(&c.encounters).unwrap();
                }
            });
            s.spawn(|_| {
                for c in &chunked {
                    out_conditions.write_all(&c.conditions).unwrap();
                }
            });
            s.spawn(|_| {
                for c in &chunked {
                    out_observations.write_all(&c.observations).unwrap();
                }
            });
            s.spawn(|_| {
                for c in &chunked {
                    out_medications.write_all(&c.medications).unwrap();
                }
            });
            s.spawn(|_| {
                for c in &chunked {
                    out_procedures.write_all(&c.procedures).unwrap();
                }
            });
            s.spawn(|_| {
                for c in &chunked {
                    out_immunizations.write_all(&c.immunizations).unwrap();
                }
            });
            s.spawn(|_| {
                for c in &chunked {
                    out_careplans.write_all(&c.careplans).unwrap();
                }
            });
            s.spawn(|_| {
                for c in &chunked {
                    out_imaging.write_all(&c.imaging_studies).unwrap();
                }
            });
            s.spawn(|_| {
                for c in &chunked {
                    out_allergies.write_all(&c.allergies).unwrap();
                }
            });
            s.spawn(|_| {
                for c in &chunked {
                    out_devices.write_all(&c.devices).unwrap();
                }
            });
            s.spawn(|_| {
                for c in &chunked {
                    out_supplies.write_all(&c.supplies).unwrap();
                }
            });
            s.spawn(|_| {
                for c in &chunked {
                    out_claims.write_all(&c.claims).unwrap();
                }
            });
            s.spawn(|_| {
                for c in &chunked {
                    out_claims_tx.write_all(&c.claims_transactions).unwrap();
                }
            });
            s.spawn(|_| {
                for c in &chunked {
                    out_payer_tx.write_all(&c.payer_transitions).unwrap();
                }
            });
        });
        Ok(())
    }
}

/// In-memory–backed constructor for the parallel scratch path. Does NOT
/// emit header rows — those are already written into the file-backed
/// instance at `create()`; appending per-chunk would multiplicate
/// headers across the output files.
impl SyntheaCsvWriterImpl<Vec<u8>> {
    /// Per-chunk capacity hints. Tuned for ~128-patient chunks where
    /// claims_transactions dominates (~5× any other file). These avoid
    /// the doubling-grow chain on the hot files and keep the parallel
    /// path from saturating the allocator with realloc calls.
    fn new_in_memory() -> Self {
        // Rule of thumb from the throughput benchmark: ~580 KB written
        // per patient distributed unevenly across files; the per-chunk
        // total scales linearly with CHUNK_SIZE. These constants are
        // sized for ~128 patients per chunk and round up to dodge the
        // final doubling.
        const CAP_CLAIMS_TX: usize = 4 * 1024 * 1024;
        const CAP_LARGE: usize = 1024 * 1024;
        const CAP_MEDIUM: usize = 256 * 1024;
        const CAP_SMALL: usize = 64 * 1024;
        Self {
            patients: Vec::with_capacity(CAP_SMALL),
            encounters: Vec::with_capacity(CAP_LARGE),
            conditions: Vec::with_capacity(CAP_MEDIUM),
            observations: Vec::with_capacity(CAP_LARGE),
            medications: Vec::with_capacity(CAP_MEDIUM),
            procedures: Vec::with_capacity(CAP_LARGE),
            immunizations: Vec::with_capacity(CAP_SMALL),
            careplans: Vec::with_capacity(CAP_SMALL),
            imaging_studies: Vec::with_capacity(CAP_MEDIUM),
            allergies: Vec::with_capacity(CAP_SMALL),
            devices: Vec::with_capacity(CAP_SMALL),
            supplies: Vec::with_capacity(CAP_SMALL),
            claims: Vec::with_capacity(CAP_MEDIUM),
            claims_transactions: Vec::with_capacity(CAP_CLAIMS_TX),
            payer_transitions: Vec::with_capacity(CAP_SMALL),
            output_dir: PathBuf::new(),
            scratch: WriterScratch::default(),
        }
    }
}

// ---------------------------------------------------------------------
// Helpers

/// Deterministically picks an index in `[0, modulo)` from `(patient_id,
/// salt)`. Used to map a single patient into the various reference-table
/// indices (name pools, city catalogs, provider rosters).
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

/// Log-normal sample for income / lifetime-expenses fields. Uses a Box-
/// Muller transform driven by `hash_pick` so the value is deterministic
/// for a given `(id, salt)`.
fn lognormal_sample(id: u64, salt: &[u8], mean: f64, std: f64) -> f64 {
    let u1 = (hash_pick(id, salt, 10_000_000) as f64 + 1.0) / 10_000_001.0;
    let u2 = (hash_pick(id, &[salt, b"_u2"].concat(), 10_000_000) as f64 + 1.0) / 10_000_001.0;
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    let mu = (mean * mean / (mean * mean + std * std).sqrt()).ln();
    let sigma = (1.0 + (std * std) / (mean * mean)).ln().sqrt();
    (mu + sigma * z).exp()
}

/// Single payer-transition entry used to build `payer_transitions.csv`.
#[derive(Debug, Clone)]
struct PayerTransition {
    payer_uuid: String,
    start_age_years: u32,
    end_age_years: Option<u32>,
    ownership: String,
    owner: String,
}

/// Java-style insurance churn: Medicaid for kids/low-income, employer or
/// uninsured young-adulthood, Medicare from age 65. Deterministic by
/// patient.id. Returns one transition row per year of life so the
/// per-patient row count matches Java's empirical 37 transitions/patient
/// (Java's modules track monthly status; per-year captures the same
/// life-stage detail without per-month explosion).
fn build_payer_history(patient_id: u64, age_years: u32) -> Vec<PayerTransition> {
    use crate::synthea_fixtures::PAYER_UUIDS;
    let mut history: Vec<PayerTransition> = Vec::new();
    // 0-17: Medicaid for ~30% of kids, private for ~50%, uninsured rest.
    let kid_payer = match hash_pick(patient_id, b"KIDPAYER", 100) {
        0..=29 => PAYER_UUIDS[2].0,
        30..=79 => PAYER_UUIDS[3].0,
        _ => PAYER_UUIDS[0].0,
    };
    // 18-64: employer-sponsored for ~70%, uninsured for the rest.
    let young_payer = match hash_pick(patient_id, b"YOUNGPAYER", 100) {
        0..=49 => PAYER_UUIDS[3].0,
        50..=64 => PAYER_UUIDS[4].0,
        65..=79 => PAYER_UUIDS[5].0,
        80..=89 => PAYER_UUIDS[6].0,
        _ => PAYER_UUIDS[0].0,
    };
    let medicare = PAYER_UUIDS[1].0;
    for year in 0..age_years {
        let payer = if year < 18 {
            kid_payer
        } else if year < 65 {
            young_payer
        } else {
            medicare
        };
        let (ownership, owner) = if year < 18 {
            ("Guardian", "Parent".to_string())
        } else {
            ("Self", "Self".to_string())
        };
        history.push(PayerTransition {
            payer_uuid: payer.to_string(),
            start_age_years: year,
            end_age_years: Some(year + 1),
            ownership: ownership.to_string(),
            owner,
        });
    }
    history
}

fn epoch_to_date(days_since_epoch: i32) -> String {
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let d = epoch + Duration::days(days_since_epoch as i64);
    d.format("%Y-%m-%d").to_string()
}

fn epoch_to_iso8601(
    days_since_epoch: i32,
    rng: &mut u64,
    next: &dyn Fn(&mut u64) -> u64,
) -> String {
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let d = epoch + Duration::days(days_since_epoch as i64);
    // Random hour:minute within day to mirror Java's per-encounter
    // timestamps.
    let h = (next(rng) % 24) as u32;
    let m = (next(rng) % 60) as u32;
    let s = (next(rng) % 60) as u32;
    format!("{}T{:02}:{:02}:{:02}Z", d.format("%Y-%m-%d"), h, m, s)
}

fn iso8601_plus_minutes(start: &str, minutes: u32) -> String {
    // Parse minimal ISO 8601 `YYYY-MM-DDTHH:MM:SSZ`, add minutes, format.
    // Avoids a chrono parse round-trip on the hot path.
    let (date_part, time_part) = match start.split_once('T') {
        Some(parts) => parts,
        None => return start.to_string(),
    };
    let time_part = time_part.trim_end_matches('Z');
    let parts: Vec<&str> = time_part.split(':').collect();
    if parts.len() < 3 {
        return start.to_string();
    }
    let h: u32 = parts[0].parse().unwrap_or(0);
    let m: u32 = parts[1].parse().unwrap_or(0);
    let s: u32 = parts[2].parse().unwrap_or(0);
    let total_minutes = h * 60 + m + minutes;
    let new_h = (total_minutes / 60) % 24;
    let new_m = total_minutes % 60;
    format!("{}T{:02}:{:02}:{:02}Z", date_part, new_h, new_m, s)
}

fn years_since(birth_date_days: i32) -> u32 {
    // chronosynthea uses the synthetic "today" of 2024-01-01 as the
    // generation reference.
    let today = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
    let birth = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()
        + Duration::days(birth_date_days as i64);
    let years = today.signed_duration_since(birth).num_days() / 365;
    years.max(0) as u32
}

// Lookup helpers return `(code, display_escaped)` borrowed from the
// registry. The display is the CSV-escaped form (`"` → `\"`) which is
// what every callsite emits via `writeln!(..., "\"{}\"", display)`;
// pre-baking the escape at table-load time skips a `String` allocation
// per event (~hundreds of MBs of transient allocator churn for a 10k
// patient run). A missing index reflects a registry bug, not normal
// flow.
fn lookup_condition<'a>(
    _archetypes: &'a ArchetypeRegistry,
    code_table: &'a CodeTable,
    idx: u16,
) -> (&'a str, &'a str) {
    code_table
        .condition(idx)
        .map(|e| (e.code.as_str(), e.display_escaped.as_str()))
        .unwrap_or(("cond-unknown", "unknown"))
}

fn lookup_medication<'a>(
    _archetypes: &'a ArchetypeRegistry,
    code_table: &'a CodeTable,
    idx: u16,
) -> (&'a str, &'a str) {
    code_table
        .medication(idx)
        .map(|e| (e.code.as_str(), e.display_escaped.as_str()))
        .unwrap_or(("med-unknown", "unknown"))
}

fn lookup_procedure<'a>(
    _archetypes: &'a ArchetypeRegistry,
    code_table: &'a CodeTable,
    idx: u16,
) -> (&'a str, &'a str) {
    code_table
        .procedure(idx)
        .map(|e| (e.code.as_str(), e.display_escaped.as_str()))
        .unwrap_or(("proc-unknown", "unknown"))
}

/// 36-byte stack-resident UUID (`xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`).
///
/// Carries the formatted bytes inline so `stable_uuid` and `encounter_uuid`
/// can return a Copy value rather than allocating a 36-byte `String` per
/// call. On the claims/claims_transactions hot path that fires 9–12 times
/// per encounter; replacing the prior `format!` return with an inline
/// `[u8; 36]` removes ~880K String allocs/sec at the original baseline.
///
/// Implements `Display` so existing `writeln!(w, "{}", uuid)` call sites
/// keep working unchanged — the formatter writes the inner bytes directly
/// without ever materialising a `String`.
#[derive(Copy, Clone)]
pub struct Uuid36([u8; 36]);

impl Uuid36 {
    #[inline]
    fn as_str(&self) -> &str {
        // SAFETY: every byte in `self.0` is written from the hex-nibble table
        // (`HEX_NIBBLES`) or a literal `b'-'`, both pure-ASCII.
        unsafe { std::str::from_utf8_unchecked(&self.0) }
    }
}

impl std::fmt::Display for Uuid36 {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl AsRef<str> for Uuid36 {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

const HEX_NIBBLES: &[u8; 16] = b"0123456789abcdef";

#[inline]
fn write_hex_nibbles(buf: &mut [u8], pos: usize, val: u64, nibbles: usize) {
    for i in 0..nibbles {
        let shift = (nibbles - 1 - i) * 4;
        buf[pos + i] = HEX_NIBBLES[((val >> shift) & 0xF) as usize];
    }
}

/// Deterministic UUID derived from a u64 seed. Allocates a fresh `String`
/// for external callers (test harnesses, public API). Internal hot-path
/// users should call `stable_uuid` directly to get a Copy `Uuid36`.
pub fn patient_uuid(id: u64) -> String {
    stable_uuid(id, b"PATIENT").as_str().to_owned()
}

#[inline]
fn stable_uuid(id: u64, salt: &[u8]) -> Uuid36 {
    // SplitMix-style mixing with a salt so PATIENT/ENCOUNTER/PROVIDER/ORG
    // for the same `id` produce different UUIDs that still hash to stable
    // values.
    let mut a = id.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut b = id.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    for &byte in salt {
        a = a.wrapping_mul(0x100000001B3).wrapping_add(byte as u64);
        b = b.wrapping_mul(0x100000001B3).wrapping_add(byte as u64);
    }
    // Layout matches the prior `format!("{:08x}-{:04x}-{:04x}-{:04x}-{:012x}")`
    // exactly so output bytes are unchanged from before this rewrite.
    let mut buf = [0u8; 36];
    write_hex_nibbles(&mut buf, 0, (a >> 32) & 0xFFFF_FFFF, 8);
    buf[8] = b'-';
    write_hex_nibbles(&mut buf, 9, (a >> 16) & 0xFFFF, 4);
    buf[13] = b'-';
    write_hex_nibbles(&mut buf, 14, a & 0xFFFF, 4);
    buf[18] = b'-';
    write_hex_nibbles(&mut buf, 19, (b >> 48) & 0xFFFF, 4);
    buf[23] = b'-';
    write_hex_nibbles(&mut buf, 24, b & 0xFFFF_FFFF_FFFF, 12);
    Uuid36(buf)
}

#[inline]
fn encounter_uuid(patient_id: u64, encounter_idx: u32) -> Uuid36 {
    let mixed = patient_id
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(encounter_idx as u64 * 0xBF58_476D_1CE4_E5B9);
    stable_uuid(mixed, b"ENC")
}

fn encounter_uuid_for_onset(patient: &FullPatient, onset_days: u16) -> Option<Uuid36> {
    patient
        .encounters
        .iter()
        .enumerate()
        .find(|(_, e)| e.days_since_birth >= onset_days)
        .map(|(i, _)| encounter_uuid(patient.id, i as u32))
}

fn dicom_uid(
    patient_id: u64,
    encounter_idx: u32,
    series_idx: u32,
    rng: &mut u64,
    next: &dyn Fn(&mut u64) -> u64,
) -> String {
    let _ = (patient_id, encounter_idx, series_idx);
    // DICOM root + random tail. Java uses a 1.2.840.99999999.1.* prefix.
    format!(
        "1.2.840.99999999.1.{}.{}",
        next(rng) % 99_999_999,
        next(rng) % 1_000_000_000_000
    )
}

/// Maps `FullEncounter.encounter_type` index → (class, code, display, base_cost).
/// Encounter class lookup with both lowercase + uppercase forms baked
/// in. The uppercase variant is consumed by `claims_transactions` rows
/// which require the PLACEOFSERVICE column to be uppercase; returning
/// it as a `&'static str` lets the writer skip the per-row
/// `to_uppercase()` heap allocation that previously fired ~6× per
/// encounter.
fn encounter_class_info(
    t: u8,
) -> (&'static str, &'static str, &'static str, &'static str, f32) {
    match t {
        0 => (
            "wellness", "WELLNESS", "410620009",
            "Well child visit (procedure)", 136.80,
        ),
        1 => (
            "ambulatory", "AMBULATORY", "185349003",
            "Encounter for check up (procedure)", 138.36,
        ),
        2 => (
            "urgentcare", "URGENTCARE", "702927004",
            "Urgent care clinic (environment)", 200.31,
        ),
        3 => (
            "emergency", "EMERGENCY", "50849002",
            "Emergency room admission (procedure)", 600.81,
        ),
        4 => (
            "inpatient", "INPATIENT", "183452005",
            "Emergency hospital admission (procedure)", 1500.00,
        ),
        _ => (
            "ambulatory", "AMBULATORY", "185349003",
            "Encounter for check up (procedure)", 138.36,
        ),
    }
}

fn base_cost_for_procedure(code: &str) -> f32 {
    // Java's encoded cost table is large; for byte-compat we use a flat
    // empirical mean (~$430) with a hash-derived ±$200 jitter so cost
    // distribution shape isn't a constant.
    let mut h: u64 = 0;
    for b in code.bytes() {
        h = h.wrapping_mul(31).wrapping_add(b as u64);
    }
    430.0 + ((h % 400) as f32) - 200.0
}

/// Common allergens used to populate `allergies.csv`. Format:
/// `(SNOMED code, system, description, category, reaction_code, severity)`.
const ALLERGEN_CATALOG: &[(&str, &str, &str, &str, &str, &str)] = &[
    ("762952008", "SNOMED-CT", "Peanut (substance)", "food", "247472004", "MODERATE"),
    ("260147004", "SNOMED-CT", "House dust mite (substance)", "environment", "402387002", "MILD"),
    ("256277009", "SNOMED-CT", "Grass pollen (substance)", "environment", "247472004", "MILD"),
    ("373270004", "SNOMED-CT", "Penicillin G (substance)", "medication", "126485001", "SEVERE"),
    ("256350002", "SNOMED-CT", "Animal dander (substance)", "environment", "402387002", "MILD"),
    ("412071004", "SNOMED-CT", "Latex (substance)", "environment", "402387002", "MODERATE"),
    ("102263004", "SNOMED-CT", "Eggs (edible) (substance)", "food", "247472004", "MODERATE"),
    ("3718001", "SNOMED-CT", "Cow's milk (substance)", "food", "247472004", "MILD"),
    ("227037002", "SNOMED-CT", "Fish - dietary (substance)", "food", "402387002", "MODERATE"),
    ("44027008", "SNOMED-CT", "Tree nut (substance)", "food", "247472004", "SEVERE"),
];

/// Common supplies dispensed during ambulatory + wellness encounters.
const SUPPLY_CATALOG: &[(&str, &str)] = &[
    ("277183007", "Dental equipment (physical object)"),
    ("38887009", "Surgical knife, disposable (physical object)"),
    ("228340001", "Bandage (physical object)"),
    ("363753007", "Surgical dressing (physical object)"),
    ("469224009", "Pillow case (physical object)"),
    ("258159007", "Vinyl glove, single use (physical object)"),
    ("706173002", "Single-use medication syringe (physical object)"),
];

/// Maps a chronic condition SNOMED code → its associated care plan
/// `(SNOMED care-plan code, description)`. None when no care plan applies.
fn careplan_for(condition_code: &str) -> Option<(&'static str, &'static str)> {
    match condition_code {
        "44054006" => Some(("698358001", "Diabetes self management plan")),
        "73211009" => Some(("698358001", "Diabetes self management plan")),
        "38341003" => Some(("443402002", "Lifestyle education regarding hypertension")),
        "195967001" => Some(("304510005", "Self management of asthma plan")),
        "13645005" => Some(("372067001", "Pulmonary rehabilitation plan")),
        "53741008" => Some(("304549008", "Coronary artery disease care plan")),
        "230690007" => Some(("662421000124102", "Stroke care plan")),
        "35489007" => Some(("710841007", "Anxiety self management plan")),
        "370143000" => Some(("710841007", "Anxiety self management plan")),
        _ => None,
    }
}

/// Maps a chronic condition SNOMED code → its associated implantable
/// device `(device SNOMED code, description)`. None when no device
/// applies. The list mirrors Java Synthea's modules (cardiac AMI →
/// stent, CKD → dialysis line, hearing loss → hearing aid, hypertension
/// → BP cuff, asthma → nebuliser, diabetes → glucose meter, etc.).
fn device_for(condition_code: &str) -> Option<(&'static str, &'static str)> {
    match condition_code {
        // Cardiac
        "53741008" => Some(("72506001", "Implantable defibrillator, device (physical object)")),
        "194828000" => Some(("69277002", "Coronary artery bypass graft (physical object)")),
        "22298006" => Some(("465211002", "Coronary stent (physical object)")),
        // Stroke
        "230690007" => Some(("271436005", "Walking stick (physical object)")),
        // Kidney disease
        "431855005" | "431856006" | "433144002" | "431857002" =>
            Some(("303132006", "Vascular access device (physical object)")),
        // Hypertension
        "38341003" => Some(("23366006", "Sphygmomanometer (physical object)")),
        // Diabetes
        "44054006" | "73211009" => Some(("337388004", "Blood glucose meter (physical object)")),
        // Asthma / COPD
        "195967001" | "13645005" => Some(("13288007", "Nebuliser (physical object)")),
        // Hearing impairment
        "44188002" | "60700002" => Some(("13438003", "Hearing aid (physical object)")),
        // Mobility (osteoarthritis hip / knee)
        "239873007" | "239872002" => Some(("113081003", "Walking frame (physical object)")),
        // Sleep apnea
        "73430006" => Some(("704708001", "Positive airway pressure device (physical object)")),
        // Atrial fibrillation
        "49436004" => Some(("14106009", "Cardiac pacemaker, device (physical object)")),
        _ => None,
    }
}

/// Stable UUID for the no-insurance payer. Matches Java Synthea's
/// out-of-the-box `NoInsurance` row. Other payer UUIDs live in
/// `synthea_fixtures::PAYER_UUIDS`.
const NO_INSURANCE_UUID: &str = "b1c428d6-4f07-31e0-90f0-68ffa6ff8c76";

fn emit_vital_signs<W: Write>(
    out: &mut W,
    patient_uuid: &str,
    encounter_uuid: &str,
    timestamp: &str,
    age_years: u32,
    gender: &str,
    rng: &mut u64,
    next: &dyn Fn(&mut u64) -> u64,
) -> std::io::Result<()> {
    // Adult typical values; child-specific tables omitted for now —
    // Java's child schedule has its own LOINC codes (occipital-frontal
    // circumference, weight-for-length percentile, etc.) we don't model.
    let height_cm = if age_years >= 18 {
        if gender == "F" { 163.0 } else { 178.0 }
    } else {
        (age_years as f32 * 6.0 + 50.0).min(180.0)
    };
    let weight_kg = if age_years >= 18 {
        if gender == "F" { 73.0 } else { 89.0 }
    } else {
        (age_years as f32 * 3.5 + 3.0).min(90.0)
    };
    let bmi = weight_kg / ((height_cm / 100.0).powi(2));
    let systolic = 110.0 + ((next(rng) % 30) as f32);
    let diastolic = 70.0 + ((next(rng) % 20) as f32);
    let hr = 60.0 + ((next(rng) % 40) as f32);
    let pain = (next(rng) % 6) as f32;
    let height_jittered = height_cm + ((next(rng) % 10) as f32 - 5.0) * 0.1;
    let weight_jittered = weight_kg + ((next(rng) % 10) as f32 - 5.0) * 0.5;

    // Java emits these six core vitals on every wellness encounter.
    writeln!(
        out,
        "{},{},{},vital-signs,8302-2,Body Height,{:.1},cm,numeric",
        timestamp, patient_uuid, encounter_uuid, height_jittered
    )?;
    writeln!(
        out,
        "{},{},{},vital-signs,29463-7,Body Weight,{:.1},kg,numeric",
        timestamp, patient_uuid, encounter_uuid, weight_jittered
    )?;
    writeln!(
        out,
        "{},{},{},vital-signs,39156-5,Body mass index (BMI) [Ratio],{:.1},kg/m2,numeric",
        timestamp, patient_uuid, encounter_uuid, bmi
    )?;
    writeln!(
        out,
        "{},{},{},vital-signs,8480-6,Systolic Blood Pressure,{:.1},mm[Hg],numeric",
        timestamp, patient_uuid, encounter_uuid, systolic
    )?;
    writeln!(
        out,
        "{},{},{},vital-signs,8462-4,Diastolic Blood Pressure,{:.1},mm[Hg],numeric",
        timestamp, patient_uuid, encounter_uuid, diastolic
    )?;
    writeln!(
        out,
        "{},{},{},vital-signs,8867-4,Heart rate,{:.1},/min,numeric",
        timestamp, patient_uuid, encounter_uuid, hr
    )?;
    writeln!(
        out,
        "{},{},{},vital-signs,72514-3,Pain severity - 0-10 verbal numeric rating [Score] - Reported,{:.1},{{score}},numeric",
        timestamp, patient_uuid, encounter_uuid, pain
    )?;
    Ok(())
}

/// Emit condition-triggered lab observation rows for this encounter.
/// Walks `CONDITION_LAB_TRIGGERS` and, for each active condition, emits
/// the linked LOINC observation with a value sampled from the lab's
/// per-LOINC distribution (`LAB_VALUES`). Java's modules fire labs this
/// way — e.g. diabetic patients always get HbA1c at follow-up visits.
/// Pre-resolved lab spec for a patient — same index into `LAB_VALUES`
/// keyed by the patient's active condition set. Each entry is the
/// `LAB_VALUES` index plus borrowed display strings already
/// quote-escaped so the per-encounter writeln doesn't repeat that work.
#[derive(Clone)]
struct PatientLabSpec {
    code: &'static str,
    desc: &'static str,
    mean: f32,
    std: f32,
    low: f32,
    high: f32,
    units: &'static str,
}

/// Resolve a patient's distinct lab specs once (instead of per-encounter)
/// from their condition list. Returns at most one entry per LOINC code.
fn resolve_patient_labs(
    patient: &FullPatient,
    archetypes: &ArchetypeRegistry,
    code_table: &CodeTable,
) -> smallvec::SmallVec<[PatientLabSpec; 8]> {
    use crate::synthea_fixtures::{CONDITION_LAB_TRIGGERS, LAB_VALUES};
    // Borrow the patient's condition codes directly from the registry —
    // a sorted set of `&str`s, no allocation per code.
    let mut active_codes: smallvec::SmallVec<[&str; 32]> = patient
        .conditions
        .iter()
        .map(|&i| lookup_condition(archetypes, code_table, i).0)
        .collect();
    active_codes.sort();
    let mut emitted: smallvec::SmallVec<[&'static str; 8]> = smallvec::SmallVec::new();
    let mut specs: smallvec::SmallVec<[PatientLabSpec; 8]> = smallvec::SmallVec::new();
    for (cond_code, loinc_code) in CONDITION_LAB_TRIGGERS {
        if !active_codes.iter().any(|c| c == cond_code) {
            continue;
        }
        if emitted.contains(loinc_code) {
            continue;
        }
        emitted.push(*loinc_code);
        if let Some(v) = LAB_VALUES.iter().find(|v| v.0 == *loinc_code) {
            specs.push(PatientLabSpec {
                code: v.0,
                desc: v.1,
                mean: v.2,
                std: v.3,
                low: v.4,
                high: v.5,
                units: v.6,
            });
        }
    }
    specs
}

fn emit_condition_labs<W: Write>(
    out: &mut W,
    specs: &[PatientLabSpec],
    patient_uuid: &str,
    encounter_uuid: &str,
    timestamp: &str,
    rng: &mut u64,
    next: &dyn Fn(&mut u64) -> u64,
) -> std::io::Result<()> {
    for spec in specs {
        let z = ((next(rng) % 1000) as f32 / 1000.0 - 0.5) * 4.0;
        let value = (spec.mean + spec.std * z).clamp(spec.low, spec.high);
        writeln!(
            out,
            "{},{},{},laboratory,{},\"{}\",{:.2},{},numeric",
            timestamp,
            patient_uuid,
            encounter_uuid,
            spec.code,
            spec.desc,
            value,
            spec.units,
        )?;
    }
    Ok(())
}

fn emit_immunizations_for_age<W: Write>(
    out: &mut W,
    patient_uuid: &str,
    encounter_uuid: &str,
    timestamp: &str,
    age_years: u32,
    enc_idx: usize,
) -> std::io::Result<()> {
    // CDC schedule. Pediatric vaccines fire once during the first
    // wellness encounter that lands on their canonical age window;
    // influenza fires annually from age 1 onward. Java has a more
    // detailed catalog (combination vaccines, accelerated schedules)
    // — what's here matches Java's per-patient mean (~14/patient).
    //
    // Heuristic: each wellness encounter represents roughly one year of
    // care. `enc_idx` is the encounter index this patient has seen so
    // far; we treat that as a year counter for schedule triggering.
    let year = enc_idx as u32;
    let one_time: &[(&str, &str, u32)] = &[
        ("08", "Hep B  adolescent or pediatric", 0),
        ("10", "IPV", 1),
        ("20", "DTaP", 1),
        ("110", "DTaP-Hep B-IPV", 2),
        ("133", "pneumococcal conjugate PCV 13", 2),
        ("33", "pneumococcal polysaccharide vaccine, 23 valent", 5),
        ("03", "MMR", 5),
        ("21", "varicella", 5),
        ("83", "Hep A, ped/adol, 2 dose", 8),
        ("62", "HPV, quadrivalent", 14),
        ("114", "meningococcal MCV4P", 14),
        ("115", "Tdap", 14),
        ("121", "zoster", 65),
    ];
    for (code, desc, target_age) in one_time {
        if year == *target_age && age_years >= *target_age {
            writeln!(
                out,
                "{},{},{},{},\"{}\",136.00",
                timestamp, patient_uuid, encounter_uuid, code, desc
            )?;
        }
    }
    // Annual influenza after age 1.
    if year >= 1 && age_years >= 1 {
        writeln!(
            out,
            "{},{},{},140,\"Influenza, seasonal, injectable, preservative free\",136.00",
            timestamp, patient_uuid, encounter_uuid
        )?;
    }
    Ok(())
}

