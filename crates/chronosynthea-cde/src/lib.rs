//! Coleman Dimensional Encoding (CDE) for ChronoSynthea.
//!
//! CDE encodes Synthea healthcare simulation modules into low-dimensional vectors
//! for efficient representation and analysis.

mod axis;
mod config;
mod encode;
mod error;
mod features;
mod metrics;
mod signature;

pub use axis::*;
pub use config::*;
pub use encode::*;
pub use error::*;
pub use features::*;
pub use metrics::*;
pub use signature::*;
