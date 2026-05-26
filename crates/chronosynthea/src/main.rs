//! ChronoSynthea CLI - High-performance synthetic patient generation.

use std::path::PathBuf;
use std::time::Instant;

// mimalloc is the global allocator for the CLI. The CSV emit path
// allocates ~per-event String/Vec churn that bumpalo can't capture;
// mimalloc reclaims that traffic 5–15% faster than glibc malloc on
// typical workloads. See the council perf-pass synthesis for the
// measurement.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::{Parser, Subcommand};

use chronosynthea_cde::{
    compute_metrics, compute_signature, encode_module, encode_module_structural,
};
use chronosynthea_core::load_module;
use chronosynthea_gen::{
    CuratedRegistry, DemographicProfile, Generator, GeneratorConfig, OptimizedRegistry,
    ParallelGenerator,
};
use chronosynthea_io::{create_file_writer, write_patients_parallel, OutputFormat};

#[derive(Parser)]
#[command(name = "chronosynthea")]
#[command(
    about = "High-performance synthetic patient generation with Coleman Dimensional Encoding"
)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate synthetic patients
    Generate {
        /// Number of patients to generate
        #[arg(short, long, default_value = "1000")]
        count: usize,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Output format (jsonl, compact, codes-only, json, msgpack)
        #[arg(short, long, default_value = "jsonl")]
        format: String,

        /// Path to prevalence registry
        #[arg(short, long)]
        registry: Option<PathBuf>,

        /// Random seed for reproducibility
        #[arg(short, long, default_value = "42")]
        seed: u64,

        /// Number of worker threads (0 = auto)
        #[arg(short, long, default_value = "0")]
        workers: usize,

        /// Use parallel generation
        #[arg(long, default_value = "true")]
        parallel: bool,

        /// Use streaming mode (overlaps generation and I/O for better throughput)
        #[arg(long, default_value = "false")]
        streaming: bool,
    },

    /// Validate a Synthea module
    Validate {
        /// Path to module JSON file
        #[arg(short, long)]
        module: PathBuf,

        /// Show detailed output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Encode a module using CDE
    Encode {
        /// Path to module JSON file
        #[arg(short, long)]
        module: PathBuf,

        /// Output file for encoding report
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Use structural encoding only (no semantic features)
        #[arg(long)]
        structural_only: bool,

        /// Show encoding metrics
        #[arg(long)]
        metrics: bool,
    },

    /// Compute module signature
    Signature {
        /// Path to module JSON file
        #[arg(short, long)]
        module: PathBuf,

        /// Use short signature (16 chars)
        #[arg(long)]
        short: bool,
    },

    /// Compare generated output with reference
    Compare {
        /// Path to generated file
        #[arg(short, long)]
        generated: PathBuf,

        /// Path to reference file
        #[arg(short, long)]
        reference: PathBuf,
    },

    /// Run benchmarks
    Bench {
        /// Number of patients for benchmark
        #[arg(short, long, default_value = "10000")]
        count: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            count,
            output,
            format,
            registry,
            seed,
            workers,
            parallel,
            streaming,
        } => {
            run_generate(
                count, output, format, registry, seed, workers, parallel, streaming,
            );
        }
        Commands::Validate { module, verbose } => {
            run_validate(module, verbose);
        }
        Commands::Encode {
            module,
            output,
            structural_only,
            metrics,
        } => {
            run_encode(module, output, structural_only, metrics);
        }
        Commands::Signature { module, short } => {
            run_signature(module, short);
        }
        Commands::Compare {
            generated,
            reference,
        } => {
            run_compare(generated, reference);
        }
        Commands::Bench { count } => {
            run_bench(count);
        }
    }
}

fn run_generate(
    count: usize,
    output: PathBuf,
    format_str: String,
    registry_path: Option<PathBuf>,
    seed: u64,
    workers: usize,
    parallel: bool,
    streaming: bool,
) {
    let format = OutputFormat::from_str(&format_str).unwrap_or_else(|| {
        eprintln!("Invalid format: {}. Using jsonl.", format_str);
        OutputFormat::Jsonl
    });

    // Load or create registry
    let registry = if let Some(path) = registry_path {
        match OptimizedRegistry::load(&path) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Failed to load registry: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // Create default registry
        create_default_registry()
    };

    let config = GeneratorConfig::with_patients(count)
        .with_seed(seed)
        .with_workers(workers);

    println!("Generating {} patients...", count);
    let start = Instant::now();

    if streaming && parallel {
        // Streaming mode: generate and write simultaneously
        println!("Using streaming mode (overlapped generation + I/O)");
        let generator = ParallelGenerator::new(config, registry);
        println!("Using {} workers", generator.num_workers());

        let mut writer = match create_file_writer(&output, format) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("Failed to create output file: {}", e);
                std::process::exit(1);
            }
        };

        // Get the streaming receiver
        let rx = generator.generate_streaming(count);

        // Consume patients and write them
        let mut written = 0;
        for patient in rx {
            if let Err(e) = writer.write(&patient) {
                eprintln!("Failed to write patient: {}", e);
                std::process::exit(1);
            }
            written += 1;
        }

        let total_duration = start.elapsed();
        let _ = writer.finish();
        let file_size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
        println!(
            "Streamed {} patients in {:.2}s ({:.0} patients/sec, {:.1} MB)",
            written,
            total_duration.as_secs_f64(),
            written as f64 / total_duration.as_secs_f64(),
            file_size as f64 / 1_000_000.0
        );
        println!("Total time: {:.2}s", total_duration.as_secs_f64());
        return;
    }

    // Batch mode (original)
    let patients = if parallel {
        let generator = ParallelGenerator::new(config, registry);
        println!("Using {} workers", generator.num_workers());
        generator.generate(count).unwrap()
    } else {
        let generator = Generator::new(config, registry);
        generator.generate(count).unwrap()
    };

    let gen_duration = start.elapsed();
    println!(
        "Generated {} patients in {:.2}s ({:.0} patients/sec)",
        patients.len(),
        gen_duration.as_secs_f64(),
        patients.len() as f64 / gen_duration.as_secs_f64()
    );

    // Write to file
    println!("Writing to {:?} (format: {})...", output, format);
    let write_start = Instant::now();

    match write_patients_parallel(&output, &patients, format) {
        Ok(n) => {
            let write_duration = write_start.elapsed();
            let file_size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
            println!(
                "Wrote {} patients in {:.2}s ({:.1} MB)",
                n,
                write_duration.as_secs_f64(),
                file_size as f64 / 1_000_000.0
            );
        }
        Err(e) => {
            eprintln!("Failed to write output: {}", e);
            std::process::exit(1);
        }
    }

    let total = start.elapsed();
    println!("Total time: {:.2}s", total.as_secs_f64());
}

fn run_validate(module_path: PathBuf, verbose: bool) {
    match load_module(&module_path) {
        Ok(module) => {
            println!("Module: {}", module.name);
            println!("States: {}", module.state_count());
            println!("Edges: {}", module.edge_count());
            println!("Has Initial: {}", module.has_initial_state());
            println!("Has Terminal: {}", module.has_terminal_state());

            if verbose {
                println!("\nStates:");
                for name in module.state_names() {
                    if let Some(state) = module.states.get(name) {
                        println!("  {} ({})", name, state.state_type);
                    }
                }
            }

            if module.has_initial_state() && module.has_terminal_state() {
                println!("\nValidation: PASSED");
            } else {
                println!("\nValidation: FAILED (missing Initial or Terminal state)");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("Failed to load module: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_encode(
    module_path: PathBuf,
    output: Option<PathBuf>,
    structural_only: bool,
    show_metrics: bool,
) {
    let module = match load_module(&module_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load module: {}", e);
            std::process::exit(1);
        }
    };

    let report = if structural_only {
        encode_module_structural(&module).unwrap()
    } else {
        encode_module(&module).unwrap()
    };

    println!("Module: {}", module.name);
    println!("Vectors: {}", report.vectors.len());
    println!("Collisions: {}", report.collisions.len());

    if show_metrics {
        let metrics = compute_metrics(&report, 16);
        println!("\nMetrics:");
        println!("  Compression ratio: {:.1}x", metrics.compression_ratio);
        println!("  Collision count: {}", metrics.collision_count);
        println!("  Near-collision count: {}", metrics.near_collision_count);
        println!("  Mean pairwise L1: {:.4}", metrics.mean_pairwise_l1);
        println!("  Saturation: {:.2}%", metrics.saturation * 100.0);
    }

    if let Some(path) = output {
        let json = serde_json::to_string_pretty(&report).unwrap();
        std::fs::write(&path, json).unwrap();
        println!("\nWrote report to {:?}", path);
    }
}

fn run_signature(module_path: PathBuf, short: bool) {
    let module = match load_module(&module_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load module: {}", e);
            std::process::exit(1);
        }
    };

    let sig = if short {
        chronosynthea_cde::compute_short_signature(&module)
    } else {
        compute_signature(&module)
    };

    println!("{}", sig);
}

fn run_compare(_generated: PathBuf, _reference: PathBuf) {
    println!("Compare functionality not yet implemented");
}

fn run_bench(count: usize) {
    println!("Benchmarking with {} patients...\n", count);

    // Sequential benchmark
    let registry1 = create_default_registry();
    let config = GeneratorConfig::with_patients(count).with_seed(42);
    let generator = Generator::new(config, registry1);

    let start = Instant::now();
    let _ = generator.generate(count).unwrap();
    let seq_duration = start.elapsed();

    println!(
        "Sequential: {:.2}s ({:.0} patients/sec)",
        seq_duration.as_secs_f64(),
        count as f64 / seq_duration.as_secs_f64()
    );

    // Parallel benchmark
    let registry2 = create_default_registry();
    let config = GeneratorConfig::with_patients(count).with_seed(42);
    let generator = ParallelGenerator::new(config, registry2);

    let start = Instant::now();
    let _ = generator.generate(count).unwrap();
    let par_duration = start.elapsed();

    println!(
        "Parallel ({} workers): {:.2}s ({:.0} patients/sec)",
        generator.num_workers(),
        par_duration.as_secs_f64(),
        count as f64 / par_duration.as_secs_f64()
    );

    let speedup = seq_duration.as_secs_f64() / par_duration.as_secs_f64();
    println!("\nSpeedup: {:.1}x", speedup);
}

fn create_default_registry() -> OptimizedRegistry {
    use ahash::AHashMap;

    let mut age_dist = AHashMap::new();
    age_dist.insert("0-17".to_string(), 0.22);
    age_dist.insert("18-44".to_string(), 0.35);
    age_dist.insert("45-64".to_string(), 0.26);
    age_dist.insert("65+".to_string(), 0.17);

    let mut gender_dist = AHashMap::new();
    gender_dist.insert("M".to_string(), 0.49);
    gender_dist.insert("F".to_string(), 0.51);

    let mut race_dist = AHashMap::new();
    race_dist.insert("white".to_string(), 0.60);
    race_dist.insert("black".to_string(), 0.13);
    race_dist.insert("asian".to_string(), 0.06);
    race_dist.insert("hispanic".to_string(), 0.18);
    race_dist.insert("other".to_string(), 0.03);

    let mut ethnicity_dist = AHashMap::new();
    ethnicity_dist.insert("nonhispanic".to_string(), 0.82);
    ethnicity_dist.insert("hispanic".to_string(), 0.18);

    OptimizedRegistry::new(CuratedRegistry {
        version: "1.0".to_string(),
        conditions: vec![],
        medications: vec![],
        observations: vec![],
        procedures: vec![],
        demographics: DemographicProfile {
            age_distribution: age_dist,
            gender_distribution: gender_dist,
            race_distribution: race_dist,
            ethnicity_distribution: ethnicity_dist,
        },
    })
}
