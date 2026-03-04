//! Configuration types for patient generation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Configuration for the synthetic patient generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorConfig {
    /// Number of patients to generate.
    pub num_patients: usize,

    /// Maximum encounters per patient.
    #[serde(default = "default_max_encounters")]
    pub max_encounters_per_patient: usize,

    /// Random seed for deterministic generation.
    pub seed: u64,

    /// Base date for timeline generation.
    pub start_date: DateTime<Utc>,

    /// Number of years of history to generate.
    #[serde(default = "default_time_span")]
    pub time_span_years: u32,

    /// Path to prevalence registry file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prevalence_registry_path: Option<String>,

    /// Number of worker threads (0 = auto-detect).
    #[serde(default)]
    pub num_workers: usize,
}

fn default_max_encounters() -> usize {
    50
}

fn default_time_span() -> u32 {
    10
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            num_patients: 1000,
            max_encounters_per_patient: 10, // Match Go benchmark config
            seed: 42,
            start_date: Utc::now(),
            time_span_years: 10,
            prevalence_registry_path: None,
            num_workers: 0,
        }
    }
}

impl GeneratorConfig {
    /// Creates a new configuration with the specified number of patients.
    pub fn with_patients(num_patients: usize) -> Self {
        Self {
            num_patients,
            ..Default::default()
        }
    }

    /// Sets the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the number of workers.
    pub fn with_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }

    /// Returns the effective number of workers (auto-detect if 0).
    pub fn effective_workers(&self) -> usize {
        if self.num_workers == 0 {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4)
        } else {
            self.num_workers
        }
    }
}
