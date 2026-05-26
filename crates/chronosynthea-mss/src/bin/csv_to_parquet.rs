//! Convert existing Synthea CSV output into Parquet — used to measure
//! real on-disk size ratios across all 15 output files without
//! re-implementing the writer logic for each file.
//!
//! Build/run (after producing CSVs with `csv_bench`):
//!     cargo run --release --features parquet --bin csv_to_parquet -- \
//!         ~/.cache/chronosynthea-bench/parallel/csv \
//!         ~/.cache/chronosynthea-bench/parquet

#[cfg(not(feature = "parquet"))]
fn main() {
    eprintln!("csv_to_parquet requires --features parquet");
    std::process::exit(2);
}

#[cfg(feature = "parquet")]
fn main() {
    use arrow::csv::ReaderBuilder;
    use parquet::arrow::ArrowWriter;
    use parquet::basic::{Compression, ZstdLevel};
    use parquet::file::properties::WriterProperties;
    use std::fs::{self, File};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: {} <csv_input_dir> <parquet_output_dir>",
            args.first().map(String::as_str).unwrap_or("csv_to_parquet")
        );
        std::process::exit(2);
    }
    let csv_dir = PathBuf::from(&args[1]);
    let parquet_dir = PathBuf::from(&args[2]);
    fs::create_dir_all(&parquet_dir).expect("mkdir parquet out");

    // Iterate all .csv files in csv_dir, convert each to .parquet,
    // report per-file size + ratio.
    let mut entries: Vec<_> = fs::read_dir(&csv_dir)
        .expect("read csv_dir")
        .filter_map(|r| r.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|x| x == "csv")
                .unwrap_or(false)
        })
        .collect();
    entries.sort_by_key(|e| e.path());

    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        .build();

    println!(
        "{:>22}  {:>10}  {:>10}  {:>6}  {:>8}",
        "file", "csv (MB)", "parq (MB)", "ratio", "convert (ms)"
    );
    println!("{}", "-".repeat(64));

    let mut total_csv = 0u64;
    let mut total_parq = 0u64;
    for entry in entries {
        let csv_path = entry.path();
        let name = csv_path.file_stem().unwrap().to_string_lossy().to_string();
        let parq_path = parquet_dir.join(format!("{name}.parquet"));

        let csv_size = fs::metadata(&csv_path).map(|m| m.len()).unwrap_or(0);
        if csv_size == 0 {
            continue;
        }

        let t = Instant::now();
        match convert_one(&csv_path, &parq_path, props.clone()) {
            Ok(()) => {
                let parq_size = fs::metadata(&parq_path).map(|m| m.len()).unwrap_or(0);
                let dt_ms = t.elapsed().as_secs_f64() * 1000.0;
                let ratio = csv_size as f64 / parq_size.max(1) as f64;
                println!(
                    "{:>22}  {:>10.2}  {:>10.2}  {:>5.2}×  {:>8.0}",
                    name,
                    csv_size as f64 / 1e6,
                    parq_size as f64 / 1e6,
                    ratio,
                    dt_ms,
                );
                total_csv += csv_size;
                total_parq += parq_size;
            }
            Err(e) => {
                eprintln!("[{name}] convert failed: {e}");
            }
        }
    }

    println!("{}", "-".repeat(64));
    println!(
        "{:>22}  {:>10.2}  {:>10.2}  {:>5.2}×",
        "TOTAL",
        total_csv as f64 / 1e6,
        total_parq as f64 / 1e6,
        total_csv as f64 / total_parq.max(1) as f64,
    );

    fn convert_one(
        csv_path: &Path,
        parq_path: &Path,
        props: WriterProperties,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Two-pass: first infer schema with arrow's CSV reader, then
        // re-read with that schema and stream batches into Parquet.
        let format = arrow::csv::reader::Format::default()
            .with_header(true)
            .with_delimiter(b',');
        let mut f = File::open(csv_path)?;
        let (schema, _) = format.infer_schema(&mut f, Some(8192))?;
        use std::io::Seek;
        f.seek(std::io::SeekFrom::Start(0))?;

        let reader = ReaderBuilder::new(std::sync::Arc::new(schema.clone()))
            .with_header(true)
            .with_batch_size(8192)
            .build(f)?;

        let out = File::create(parq_path)?;
        let mut writer =
            ArrowWriter::try_new(out, std::sync::Arc::new(schema), Some(props))?;
        for batch_res in reader {
            let batch = batch_res?;
            writer.write(&batch)?;
        }
        writer.close()?;
        Ok(())
    }
}
