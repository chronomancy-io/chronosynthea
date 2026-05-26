//! Pure-CSV write benchmark: how fast can we drop 10k patients into
//! the 15-file Java schema? The previous perf benchmarks measured
//! *generation* throughput (in-memory FullPatient construction); this
//! one measures the write hot path separately so we can see where the
//! cost actually lives.

// mimalloc matches the CLI binary's allocator so the benchmark is a
// fair proxy for production throughput rather than a glibc-malloc lower
// bound.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use chronosynthea_mss::{patient_uuid, BatchConfig, BatchGenerator, CalibratedRegistry, SyntheaCsvWriter};
use std::path::PathBuf;
use std::time::Instant;

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
fn csv_write_throughput() {
    let path = registry_path();
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

    let n = 10_000;
    let t0 = Instant::now();
    let patients = generator.generate_full(n);
    let gen_dt = t0.elapsed();
    let total_events: usize = patients
        .iter()
        .map(|p| p.encounters.iter().map(|e| e.events.len()).sum::<usize>())
        .sum();
    eprintln!(
        "[gen] {} patients in {:.3}s = {:.0}/sec; {:.0}M events",
        n,
        gen_dt.as_secs_f64(),
        n as f64 / gen_dt.as_secs_f64(),
        total_events as f64 / 1e6
    );

    let archetypes = generator.archetypes();
    let code_table = generator.code_table();

    // Run #1 — serial path (the historical baseline)
    let out_dir = std::env::temp_dir().join("chronosynthea-csv-throughput");
    let _ = std::fs::remove_dir_all(&out_dir);
    let mut writer = SyntheaCsvWriter::create(&out_dir).unwrap();
    let t1 = Instant::now();
    for p in &patients {
        let uuid = patient_uuid(p.id);
        writer.write_patient(p, &uuid, archetypes, code_table).unwrap();
    }
    writer.flush().unwrap();
    let write_dt = t1.elapsed();
    eprintln!(
        "[csv][serial] wrote {} patients ({:.1}M events) in {:.3}s = {:.0}/sec; {:.1} Mevents/sec",
        n,
        total_events as f64 / 1e6,
        write_dt.as_secs_f64(),
        n as f64 / write_dt.as_secs_f64(),
        total_events as f64 / 1e6 / write_dt.as_secs_f64()
    );
    let mut total_bytes: u64 = 0;
    for entry in std::fs::read_dir(out_dir.join("csv")).unwrap() {
        let entry = entry.unwrap();
        if let Ok(meta) = entry.metadata() {
            total_bytes += meta.len();
        }
    }
    eprintln!(
        "[csv][serial] total bytes written: {:.1} MB ({:.0} MB/sec)",
        total_bytes as f64 / 1e6,
        total_bytes as f64 / 1e6 / write_dt.as_secs_f64()
    );

    // Run #2 — parallel chunked path. Rayon's par_chunks preserves
    // chunk order via collect, so the file bytes match the serial
    // path's bytes exactly given the same patient slice.
    let out_dir_par = std::env::temp_dir().join("chronosynthea-csv-throughput-par");
    let _ = std::fs::remove_dir_all(&out_dir_par);
    let mut writer_par = SyntheaCsvWriter::create(&out_dir_par).unwrap();
    let t2 = Instant::now();
    writer_par
        .write_patients_parallel(&patients, archetypes, code_table)
        .unwrap();
    writer_par.flush().unwrap();
    let par_dt = t2.elapsed();
    eprintln!(
        "[csv][parallel] wrote {} patients ({:.1}M events) in {:.3}s = {:.0}/sec; {:.1} Mevents/sec  ({:.2}× vs serial)",
        n,
        total_events as f64 / 1e6,
        par_dt.as_secs_f64(),
        n as f64 / par_dt.as_secs_f64(),
        total_events as f64 / 1e6 / par_dt.as_secs_f64(),
        write_dt.as_secs_f64() / par_dt.as_secs_f64()
    );
    let mut par_bytes: u64 = 0;
    for entry in std::fs::read_dir(out_dir_par.join("csv")).unwrap() {
        let entry = entry.unwrap();
        if let Ok(meta) = entry.metadata() {
            par_bytes += meta.len();
        }
    }
    eprintln!(
        "[csv][parallel] total bytes written: {:.1} MB ({:.0} MB/sec)",
        par_bytes as f64 / 1e6,
        par_bytes as f64 / 1e6 / par_dt.as_secs_f64()
    );
}
