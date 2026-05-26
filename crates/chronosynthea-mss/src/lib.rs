//! Minimally Sufficient Statistic (MSS) encoding for ultra-fast patient generation.
//!
//! This crate provides the core infrastructure for achieving 1M patients/second
//! by pre-computing all statistical distributions from Java Synthea output and
//! enabling O(1) amortized patient generation through:
//!
//! - Patient archetypes with pre-computed condition probabilities
//! - SIMD-accelerated batch sampling
//! - Arena-based zero-allocation generation
//! - Compile-time interned string tables

pub mod archetype;
pub mod arena;
pub mod batch;
pub mod cascade;
pub mod causal_dag;
pub mod csv_writer;
pub mod error;
pub mod extractor;
pub mod fingerprint;
pub mod java_compat;
pub mod sampler;
pub mod stats;
pub mod synthea_fixtures;
pub mod synthehrella;
pub mod tables;

#[cfg(feature = "gpu")]
pub mod gpu;
pub mod types;

pub use archetype::{ArchetypeRegistry, CooccurrenceModel, PatientArchetype};
pub use arena::{
    CompactEncounter, CompactEvent, CompactPatient, EventCounts, FullEncounter, FullPatient,
    WorkerArena,
};
pub use batch::{AtomicStatistics, BatchConfig, BatchGenerator, GenerationResult};
pub use error::{MssError, MssResult};
pub use extractor::MssExtractor;
pub use fingerprint::MssFingerprint;
pub use java_compat::{CalibratedRegistry, JavaValidation};
pub use cascade::{CascadeRule, CausalCascadeModel};
pub use causal_dag::{CausalDagModel, FitReport, GIBBS_ITERATIONS};
pub use csv_writer::{patient_uuid, SyntheaCsvWriter};
pub use sampler::{EventBitset, EventSampler, SimdSampler};
pub use stats::StreamingStatistics;
pub use types::{ArchetypeId, ConditionIndex};
