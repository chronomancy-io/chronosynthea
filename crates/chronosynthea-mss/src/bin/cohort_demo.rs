//! Cohort query demo — proves the Phase 4 cohort API end-to-end.
//!
//! Runs three increasingly selective cohort queries and reports
//! match counts + latency + bytes-on-disk. Each cohort writes to its
//! own `summary.parquet` so downstream consumers can drop in directly.
//!
//! Build/run:
//!     cargo run --release --features parquet --bin cohort_demo

#[cfg(not(feature = "parquet"))]
fn main() {
    eprintln!("cohort_demo requires --features parquet");
    std::process::exit(2);
}

#[cfg(feature = "parquet")]
fn main() {
    use chronosynthea_mss::cohort::FilterExpr;
    use chronosynthea_mss::parquet_writer::SyntheaStatsParquetWriter;
    use chronosynthea_mss::reproducibility::CohortManifest;
    use chronosynthea_mss::{
        BatchConfig, BatchGenerator, CalibratedRegistry,
    };
    use std::path::PathBuf;
    use std::time::Instant;

    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.push("data");
    p.push("prevalence");
    p.push("calibrated_registry.json");
    if !p.exists() {
        eprintln!("registry not found at {}", p.display());
        std::process::exit(1);
    }
    let registry = CalibratedRegistry::load(&p).unwrap();
    let fingerprint = registry.to_fingerprint();
    let seed: u64 = 42;
    let generator = BatchGenerator::new(
        fingerprint,
        BatchConfig {
            seed,
            ..Default::default()
        },
    );
    let fp_hash = *generator.fingerprint_hash();
    let bench_root: PathBuf = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache/chronosynthea-bench");

    let cohorts: &[(&str, FilterExpr, usize, usize)] = &[
        // (label, filter, target_count, max_scan)
        (
            "elderly-male",
            FilterExpr::And {
                children: vec![
                    FilterExpr::AgeRange { lo: 60, hi: 80 },
                    FilterExpr::Sex { value: "M".into() },
                ],
            },
            1000,
            50_000,
        ),
        (
            "stroke",
            FilterExpr::HasCondition {
                code: "230690007".into(),
            },
            1000,
            100_000,
        ),
        (
            "stroke-AND-diabetes",
            FilterExpr::And {
                children: vec![
                    FilterExpr::HasCondition {
                        code: "230690007".into(),
                    },
                    FilterExpr::HasCondition {
                        code: "44054006".into(),
                    },
                ],
            },
            500,
            200_000,
        ),
    ];

    println!();
    println!(
        "{:>22}  {:>8}  {:>8}  {:>7}  {:>10}  {:>10}  {:>10}",
        "cohort",
        "target",
        "matched",
        "select",
        "scanned",
        "wall (s)",
        "out (KB)",
    );
    println!("{}", "-".repeat(86));

    for (label, filter, target, max_scan) in cohorts {
        let dir = bench_root.join(format!("cohort-{}", label));
        let _ = std::fs::remove_dir_all(&dir);
        let mut writer = SyntheaStatsParquetWriter::create(&dir).unwrap();
        let t = Instant::now();
        let result = generator.cohort(filter, *target, *max_scan, |p| {
            writer.write_patient(p).unwrap();
        });
        writer.finish().unwrap();
        let dt = t.elapsed();
        let out_path = dir.join("parquet/summary.parquet");
        let bytes = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);

        // Emit a manifest reflecting the cohort definition so an
        // auditor can re-run the exact same query.
        let mut manifest = CohortManifest::new(
            &fp_hash,
            seed,
            result.matched,
            &format!("parquet-stats-cohort:{}", label),
        );
        manifest.output_bytes = bytes;
        manifest
            .write_json(dir.join("parquet/manifest.json"))
            .unwrap();
        // Drop the filter expression in the same dir for auditability.
        std::fs::write(
            dir.join("parquet/filter.json"),
            serde_json::to_string_pretty(filter).unwrap(),
        )
        .unwrap();

        println!(
            "{:>22}  {:>8}  {:>8}  {:>6.2}%  {:>10}  {:>10.3}  {:>10.1}",
            label,
            target,
            result.matched,
            result.selectivity() * 100.0,
            result.scanned,
            dt.as_secs_f64(),
            bytes as f64 / 1024.0,
        );
    }

    println!();
    println!(
        "(outputs in {} — each cohort dir contains summary.parquet, manifest.json, filter.json)",
        bench_root.display(),
    );
}
