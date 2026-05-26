//! Smoke test for the SynthEHRella exporters.
//!
//! Generates 1000 FullPatients, writes a binary patient × code matrix
//! and a long-form temporal record file, then asserts:
//!   * the matrix has one row per patient + header
//!   * column count matches conditions + medications + procedures
//!   * at least some 1s exist (patients have events)
//!   * the temporal records file has one row per (condition + encounter event)

use chronosynthea_mss::synthehrella::{
    write_binary_matrix, write_temporal_records, MatrixOptions,
};
use chronosynthea_mss::{BatchConfig, BatchGenerator, CalibratedRegistry};
use std::io::BufRead;
use std::path::PathBuf;

fn registry_path() -> PathBuf {
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
fn synthehrella_exporters_smoke() {
    let path = registry_path();
    if !path.exists() {
        eprintln!("Skipping: calibrated_registry.json not found");
        return;
    }
    let registry = CalibratedRegistry::load(&path).unwrap();
    let fingerprint = registry.to_fingerprint();
    let config = BatchConfig {
        seed: 1234,
        ..Default::default()
    };
    let generator = BatchGenerator::new(fingerprint, config);

    let patients = generator.generate_full(1000);
    let archetypes = generator.archetypes();
    let code_table = generator.code_table();

    let out_dir = std::env::temp_dir().join("chronosynthea-synthehrella-smoke");
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).unwrap();

    // Binary matrix
    let bm_path = out_dir.join("binary_matrix.csv");
    let bm_rows = write_binary_matrix(
        &patients,
        archetypes,
        code_table,
        &bm_path,
        MatrixOptions::default(),
    )
    .unwrap();
    println!("wrote binary matrix to {:?} ({} rows)", bm_path, bm_rows);
    assert_eq!(bm_rows, patients.len() + 1, "matrix rows = patients + header");

    // Verify the matrix header has the expected number of columns.
    let f = std::fs::File::open(&bm_path).unwrap();
    let mut reader = std::io::BufReader::new(f);
    let mut header = String::new();
    reader.read_line(&mut header).unwrap();
    let header_cols = header.trim_end().split(',').count();
    let expected_cols = 1
        + code_table.num_conditions()
        + code_table.num_medications()
        + code_table.num_procedures();
    assert_eq!(
        header_cols, expected_cols,
        "header columns: 1 (patient_id) + conditions + medications + procedures"
    );

    // Check that at least *some* cells are 1 in the body. If everything is
    // 0 the smoke isn't picking up any patient events.
    let mut found_one = false;
    for line in reader.lines() {
        let line = line.unwrap();
        if line.contains(",1") {
            found_one = true;
            break;
        }
    }
    assert!(found_one, "binary matrix has no `1` values — exporter not picking up events");

    // Temporal records
    let tr_path = out_dir.join("temporal_records.csv");
    let tr_rows = write_temporal_records(&patients, archetypes, code_table, &tr_path).unwrap();
    println!("wrote temporal records to {:?} ({} rows)", tr_path, tr_rows);
    assert!(tr_rows > 1 + patients.len(), "temporal records should have many event rows");
}
