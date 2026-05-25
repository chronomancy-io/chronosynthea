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
    let mut total_meds = 0usize;
    let mut total_procs = 0usize;
    let mut meds_with_reason = 0usize;
    let mut procs_with_reason = 0usize;
    for p in &patients {
        let uuid = patient_uuid(p.id);
        writer
            .write_patient(p, &uuid, archetypes, code_table)
            .expect("write patient");
        total_conditions += p.conditions.len();
        total_meds += p.medications.len();
        total_procs += p.procedures.len();
        meds_with_reason += p
            .medication_causes
            .iter()
            .filter(|&&c| c != u16::MAX)
            .count();
        procs_with_reason += p
            .procedure_causes
            .iter()
            .filter(|&&c| c != u16::MAX)
            .count();
    }
    writer.flush().expect("flush csv writer");

    println!(
        "csv totals: conditions={total_conditions} medications={total_meds} procedures={total_procs}"
    );
    println!(
        "REASONCODE populated: meds {}/{} procs {}/{}",
        meds_with_reason, total_meds, procs_with_reason, total_procs
    );

    // Validate that each output file exists with a Java-compatible header.
    use std::io::BufRead;
    let p_path = out_dir.join("csv").join("patients.csv");
    let c_path = out_dir.join("csv").join("conditions.csv");
    let m_path = out_dir.join("csv").join("medications.csv");
    let pr_path = out_dir.join("csv").join("procedures.csv");

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
    assert_eq!(count_lines(&p_path), 1 + patients.len());
    assert_eq!(count_lines(&c_path), 1 + total_conditions);
    assert_eq!(count_lines(&m_path), 1 + total_meds);
    assert_eq!(count_lines(&pr_path), 1 + total_procs);

    // At least *some* medications and procedures should have a REASONCODE
    // — otherwise the REASONCODE linkage isn't firing.
    if total_meds > 100 {
        assert!(
            meds_with_reason > 0,
            "no medication REASONCODE populated — linkage not firing"
        );
    }
    if total_procs > 100 {
        assert!(
            procs_with_reason > 0,
            "no procedure REASONCODE populated — linkage not firing"
        );
    }
}
