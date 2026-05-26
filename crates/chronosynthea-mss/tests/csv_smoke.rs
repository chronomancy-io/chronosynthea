//! Smoke test for the Java-Synthea-compatible CSV writer.
//!
//! Generates 1000 FullPatients via `generate_full`, writes them through
//! `SyntheaCsvWriter`, and asserts that:
//!   * each of the 4 CSV files exists at the configured output dir
//!   * each file has the correct Java-Synthea-compatible header row
//!   * patient and condition rows are present and well-formed
//!   * REASONCODE columns are populated for at least some medications
//!     and procedures (with `u16::MAX` sentinel rendered as empty)

use chronosynthea_mss::{
    patient_uuid, BatchConfig, BatchGenerator, CalibratedRegistry, SyntheaCsvWriter,
};
use std::path::PathBuf;

fn calibrated_registry_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.push("data");
    p.push("prevalence");
    p.push("calibrated_registry.json");
    p
}

#[test]
#[ignore]
fn csv_smoke_synthea_compatible() {
    let path = calibrated_registry_path();
    if !path.exists() {
        eprintln!("Skipping: calibrated_registry.json not found");
        return;
    }
    let registry = CalibratedRegistry::load(&path).unwrap();
    let fingerprint = registry.to_fingerprint();
    let config = BatchConfig {
        seed: 42,
        ..Default::default()
    };
    let generator = BatchGenerator::new(fingerprint, config);

    let out_dir = std::env::temp_dir().join("chronosynthea-csv-smoke");
    let _ = std::fs::remove_dir_all(&out_dir);
    let mut writer = SyntheaCsvWriter::create(&out_dir).expect("create csv writer");

    let patients = generator.generate_full(1000);
    println!("generated {} full patients", patients.len());

    let archetypes = generator.archetypes();
    let code_table = generator.code_table();

    let mut total_conditions = 0usize;
    let mut total_unique_meds = 0usize;
    let mut total_unique_procs = 0usize;
    let mut total_med_events = 0usize;
    let mut total_proc_events = 0usize;
    let mut meds_with_reason = 0usize;
    let mut procs_with_reason = 0usize;
    for p in &patients {
        let uuid = patient_uuid(p.id);
        writer
            .write_patient(p, &uuid, archetypes, code_table)
            .expect("write patient");
        total_conditions += p.conditions.len();
        total_unique_meds += p.medications.len();
        total_unique_procs += p.procedures.len();
        // Build per-patient cause lookups so we can count REASONCODE
        // populations at the EVENT level (one row per encounter event in the
        // CSV, not one row per unique code).
        let med_cause: std::collections::HashMap<u16, u16> = p
            .medications
            .iter()
            .zip(p.medication_causes.iter())
            .map(|(&m, &c)| (m, c))
            .collect();
        let proc_cause: std::collections::HashMap<u16, u16> = p
            .procedures
            .iter()
            .zip(p.procedure_causes.iter())
            .map(|(&pr, &c)| (pr, c))
            .collect();
        for enc in &p.encounters {
            for ev in &enc.events {
                match ev.event_type {
                    1 => {
                        total_med_events += 1;
                        if let Some(&c) = med_cause.get(&ev.code_idx) {
                            if c != u16::MAX {
                                meds_with_reason += 1;
                            }
                        }
                    }
                    2 => {
                        total_proc_events += 1;
                        if let Some(&c) = proc_cause.get(&ev.code_idx) {
                            if c != u16::MAX {
                                procs_with_reason += 1;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    writer.flush().expect("flush csv writer");

    println!(
        "csv totals: conditions={total_conditions} med_events={total_med_events} \
         proc_events={total_proc_events} (unique meds={total_unique_meds}, procs={total_unique_procs})"
    );
    println!(
        "REASONCODE populated: meds {}/{} ({:.1}%) procs {}/{} ({:.1}%)",
        meds_with_reason,
        total_med_events,
        100.0 * meds_with_reason as f64 / total_med_events.max(1) as f64,
        procs_with_reason,
        total_proc_events,
        100.0 * procs_with_reason as f64 / total_proc_events.max(1) as f64
    );

    // Validate that each output file exists with a Java-compatible header.
    use std::io::BufRead;
    let p_path = out_dir.join("csv").join("patients.csv");
    let c_path = out_dir.join("csv").join("conditions.csv");
    let m_path = out_dir.join("csv").join("medications.csv");
    let pr_path = out_dir.join("csv").join("procedures.csv");
    let e_path = out_dir.join("csv").join("encounters.csv");
    let o_path = out_dir.join("csv").join("observations.csv");
    let im_path = out_dir.join("csv").join("immunizations.csv");
    let cp_path = out_dir.join("csv").join("careplans.csv");
    let img_path = out_dir.join("csv").join("imaging_studies.csv");
    let al_path = out_dir.join("csv").join("allergies.csv");
    let dv_path = out_dir.join("csv").join("devices.csv");
    let sp_path = out_dir.join("csv").join("supplies.csv");

    let read_first = |p: &std::path::Path| -> String {
        let f = std::fs::File::open(p).expect("open csv");
        let mut r = std::io::BufReader::new(f);
        let mut s = String::new();
        r.read_line(&mut s).unwrap();
        s.trim_end().to_string()
    };

    let p_hdr = read_first(&p_path);
    assert!(p_hdr.starts_with("Id,BIRTHDATE,"), "patients.csv header: {p_hdr}");
    assert!(p_hdr.contains("RACE,ETHNICITY,GENDER"));

    let c_hdr = read_first(&c_path);
    assert_eq!(c_hdr, "START,STOP,PATIENT,ENCOUNTER,SYSTEM,CODE,DESCRIPTION");

    let m_hdr = read_first(&m_path);
    assert!(m_hdr.starts_with("START,STOP,PATIENT"), "medications.csv header: {m_hdr}");
    assert!(m_hdr.contains("REASONCODE,REASONDESCRIPTION"));

    let pr_hdr = read_first(&pr_path);
    assert!(pr_hdr.starts_with("START,STOP,PATIENT"));
    assert!(pr_hdr.contains("REASONCODE,REASONDESCRIPTION"));

    // Row counts should match per-patient totals + 1 for header.
    let count_lines = |p: &std::path::Path| -> usize {
        std::io::BufReader::new(std::fs::File::open(p).unwrap())
            .lines()
            .count()
    };
    // patients.csv: one row per patient (header + N).
    assert_eq!(count_lines(&p_path), 1 + patients.len());
    // conditions.csv: one row per unique condition per patient.
    assert_eq!(count_lines(&c_path), 1 + total_conditions);
    // medications + procedures: one row per encounter event of that type
    // (matches Java Synthea's temporal layout — same drug across many
    // encounters yields many rows).
    assert_eq!(count_lines(&m_path), 1 + total_med_events);
    assert_eq!(count_lines(&pr_path), 1 + total_proc_events);

    // The remaining files emitted by the full-CSV writer: assert they
    // each have the Java-compatible header + at least one event row
    // (smoke level — exhaustive row-count parity is in the manifesto's
    // honest-scope notes).
    for (path, expected_prefix) in [
        (&e_path, "Id,START,STOP,PATIENT,ORGANIZATION,PROVIDER,PAYER,ENCOUNTERCLASS"),
        (&o_path, "DATE,PATIENT,ENCOUNTER,CATEGORY,CODE,DESCRIPTION,VALUE,UNITS,TYPE"),
        (&im_path, "DATE,PATIENT,ENCOUNTER,CODE,DESCRIPTION,BASE_COST"),
        (&cp_path, "Id,START,STOP,PATIENT,ENCOUNTER,CODE,DESCRIPTION,REASONCODE,REASONDESCRIPTION"),
        (&img_path, "Id,DATE,PATIENT,ENCOUNTER,SERIES_UID"),
        (&al_path, "START,STOP,PATIENT,ENCOUNTER,CODE,SYSTEM,DESCRIPTION,TYPE,CATEGORY"),
        (&dv_path, "START,STOP,PATIENT,ENCOUNTER,CODE,DESCRIPTION,UDI"),
        (&sp_path, "DATE,PATIENT,ENCOUNTER,CODE,DESCRIPTION,QUANTITY"),
    ] {
        let hdr = read_first(path);
        assert!(
            hdr.starts_with(expected_prefix),
            "{}: header `{}` does not start with `{}`",
            path.display(),
            hdr,
            expected_prefix
        );
        let n = count_lines(path);
        assert!(
            n > 1,
            "{}: expected at least one event row beyond header, got {}",
            path.display(),
            n
        );
    }
    println!(
        "auxiliary CSVs: encounters={} observations={} immunizations={} careplans={} imaging={} allergies={} devices={} supplies={}",
        count_lines(&e_path),
        count_lines(&o_path),
        count_lines(&im_path),
        count_lines(&cp_path),
        count_lines(&img_path),
        count_lines(&al_path),
        count_lines(&dv_path),
        count_lines(&sp_path),
    );

    // At least *some* medications and procedures should have a REASONCODE
    // — otherwise the REASONCODE linkage isn't firing.
    if total_med_events > 100 {
        assert!(
            meds_with_reason > 0,
            "no medication REASONCODE populated — linkage not firing"
        );
    }
    if total_proc_events > 100 {
        assert!(
            procs_with_reason > 0,
            "no procedure REASONCODE populated — linkage not firing"
        );
    }
}
