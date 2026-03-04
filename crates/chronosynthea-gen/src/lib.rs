//! High-performance patient generation for ChronoSynthea.
//!
//! This crate provides:
//! - Prevalence-based patient generation
//! - Vose alias method for O(1) sampling
//! - Parallel generation with Rayon
//! - Arena-based allocation for minimal GC pressure

mod alias;
mod buffer;
mod config;
mod error;
mod generator;
mod parallel;
mod prevalence;

pub use alias::*;
pub use buffer::*;
pub use config::*;
pub use error::*;
pub use generator::*;
pub use parallel::*;
pub use prevalence::*;
