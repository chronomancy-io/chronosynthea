//! SynthEHRella-compatible exporters for community-standard fidelity
//! evaluation.
//!
//! SynthEHRella (arXiv:2411.04281, 2024) is a multi-faceted evaluation
//! harness for synthetic EHR generators. It accepts two primary input
//! formats:
//!
//!   1. **Patient × code binary matrix** — one row per patient, columns
//!      indexed by SNOMED/RxNorm/LOINC code, values `1` if the patient
//!      ever received that code, `0` otherwise. Used for fidelity
//!      statistics (cosine, MMD, dimension-wise distance), discriminator
//!      training, and utility downstream-task evaluation.
//!
//!   2. **Long-form temporal records** — `(patient_id, code, timestamp)`
//!      triples used for trajectory-aware metrics (sequence alignment,
//!      Markov-chain entropy comparisons).
//!
//! This module writes both. The CSV output from `SyntheaCsvWriter` already
//! gives the temporal long-form; this module adds the binary matrix.
//!
//! ## Usage
//!
//! ```no_run
//! use chronosynthea_mss::{BatchConfig, BatchGenerator, CalibratedRegistry};
//! use chronosynthea_mss::synthehrella::{write_binary_matrix, MatrixOptions};
//! # let registry = CalibratedRegistry::load("data/prevalence/calibrated_registry.json").unwrap();
//! # let fingerprint = registry.to_fingerprint();
//! # let generator = BatchGenerator::new(fingerprint, BatchConfig::default());
//! let patients = generator.generate_full(10_000);
//! let archetypes = generator.archetypes();
//! let code_table = generator.code_table();
//! write_binary_matrix(
//!     &patients,
//!     archetypes,
//!     code_table,
//!     "/tmp/chronosynthea-synthehrella-input/binary_matrix.csv",
//!     MatrixOptions::default(),
//! ).unwrap();
//! ```
//!
//! Then point SynthEHRella at the output directory; see the SynthEHRella
//! readme for the exact CLI invocation. We don't bundle SynthEHRella
//! itself — this module is the bridge.

use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::archetype::ArchetypeRegistry;
use crate::arena::FullPatient;
use crate::tables::CodeTable;

/// Options controlling the patient × code binary matrix layout.
#[derive(Debug, Clone)]
pub struct MatrixOptions {
    /// Include conditions in the matrix (one column per condition code).
    pub include_conditions: bool,
    /// Include medications in the matrix.
    pub include_medications: bool,
    /// Include procedures in the matrix.
    pub include_procedures: bool,
    /// When true, prepends a row containing the column codes (header).
    pub write_header: bool,
}

impl Default for MatrixOptions {
    fn default() -> Self {
        Self {
            include_conditions: true,
            include_medications: true,
            include_procedures: true,
            write_header: true,
        }
    }
}

/// Writes a patient × code binary matrix to `output_path`. Returns the
/// number of rows written (one per patient + 1 for the header when
/// `write_header` is set).
///
/// Format: comma-separated, no quoting needed (codes are alphanumeric +
/// hyphens). Cell values are `0` or `1`. Column order: condition codes
/// (sorted by index), then medication codes, then procedure codes.
pub fn write_binary_matrix<P: AsRef<Path>>(
    patients: &[FullPatient],
    _archetypes: &ArchetypeRegistry,
    code_table: &CodeTable,
    output_path: P,
    options: MatrixOptions,
) -> std::io::Result<usize> {
    if let Some(parent) = output_path.as_ref().parent() {
        create_dir_all(parent)?;
    }
    let f = File::create(output_path)?;
    let mut w = BufWriter::new(f);

    // Resolve the column set. We use the code_table's native ordering so
    // the same model run produces a stable column schema; downstream
    // SynthEHRella tooling keys on column names, not positions.
    let n_cond = code_table.num_conditions();
    let n_med = code_table.num_medications();
    let n_proc = code_table.num_procedures();

    if options.write_header {
        write!(w, "patient_id")?;
        if options.include_conditions {
            for i in 0..n_cond {
                if let Some(e) = code_table.condition(i as u16) {
                    write!(w, ",{}", e.code)?;
                }
            }
        }
        if options.include_medications {
            for i in 0..n_med {
                if let Some(e) = code_table.medication(i as u16) {
                    write!(w, ",{}", e.code)?;
                }
            }
        }
        if options.include_procedures {
            for i in 0..n_proc {
                if let Some(e) = code_table.procedure(i as u16) {
                    write!(w, ",{}", e.code)?;
                }
            }
        }
        writeln!(w)?;
    }

    let mut rows = if options.write_header { 1 } else { 0 };
    // Per-patient bitset over each axis. Sized once; reused per patient.
    let cap_cond = n_cond.max(1);
    let cap_med = n_med.max(1);
    let cap_proc = n_proc.max(1);
    let mut cond_seen = vec![0u8; cap_cond];
    let mut med_seen = vec![0u8; cap_med];
    let mut proc_seen = vec![0u8; cap_proc];

    for patient in patients {
        if options.include_conditions {
            cond_seen.iter_mut().for_each(|b| *b = 0);
            for &c in &patient.conditions {
                let i = c as usize;
                if i < cond_seen.len() {
                    cond_seen[i] = 1;
                }
            }
        }
        if options.include_medications {
            med_seen.iter_mut().for_each(|b| *b = 0);
            for &m in &patient.medications {
                let i = m as usize;
                if i < med_seen.len() {
                    med_seen[i] = 1;
                }
            }
        }
        if options.include_procedures {
            proc_seen.iter_mut().for_each(|b| *b = 0);
            for &p in &patient.procedures {
                let i = p as usize;
                if i < proc_seen.len() {
                    proc_seen[i] = 1;
                }
            }
        }

        write!(w, "{}", patient.id)?;
        if options.include_conditions {
            for &b in &cond_seen {
                write!(w, ",{}", b)?;
            }
        }
        if options.include_medications {
            for &b in &med_seen {
                write!(w, ",{}", b)?;
            }
        }
        if options.include_procedures {
            for &b in &proc_seen {
                write!(w, ",{}", b)?;
            }
        }
        writeln!(w)?;
        rows += 1;
    }

    w.flush()?;
    Ok(rows)
}

/// Writes a long-form temporal record set to `output_path`. Format:
/// `patient_id,event_type,code,timestamp_days_since_birth` — same shape
/// SynthEHRella's trajectory metrics consume. `event_type` is one of
/// `condition`, `medication`, `procedure`, `observation`.
pub fn write_temporal_records<P: AsRef<Path>>(
    patients: &[FullPatient],
    _archetypes: &ArchetypeRegistry,
    code_table: &CodeTable,
    output_path: P,
) -> std::io::Result<usize> {
    if let Some(parent) = output_path.as_ref().parent() {
        create_dir_all(parent)?;
    }
    let f = File::create(output_path)?;
    let mut w = BufWriter::new(f);
    writeln!(
        w,
        "patient_id,event_type,code,days_since_birth"
    )?;

    let mut rows = 1usize;
    for patient in patients {
        // Conditions stamped at their per-condition onset day.
        for (i, &c) in patient.conditions.iter().enumerate() {
            let onset = patient
                .condition_onset_days
                .get(i)
                .copied()
                .unwrap_or(0);
            let code = code_table
                .condition(c)
                .map(|e| e.code.as_str())
                .unwrap_or("?");
            writeln!(w, "{},condition,{},{}", patient.id, code, onset)?;
            rows += 1;
        }
        // Medication, procedure, observation events from encounters.
        for enc in &patient.encounters {
            for ev in &enc.events {
                let (kind, code) = match ev.event_type {
                    1 => (
                        "medication",
                        code_table
                            .medication(ev.code_idx)
                            .map(|e| e.code.as_str())
                            .unwrap_or("?"),
                    ),
                    2 => (
                        "procedure",
                        code_table
                            .procedure(ev.code_idx)
                            .map(|e| e.code.as_str())
                            .unwrap_or("?"),
                    ),
                    3 => (
                        "observation",
                        code_table
                            .observation(ev.code_idx)
                            .map(|e| e.code.as_str())
                            .unwrap_or("?"),
                    ),
                    _ => continue,
                };
                writeln!(
                    w,
                    "{},{},{},{}",
                    patient.id, kind, code, enc.days_since_birth
                )?;
                rows += 1;
            }
        }
    }
    w.flush()?;
    Ok(rows)
}
