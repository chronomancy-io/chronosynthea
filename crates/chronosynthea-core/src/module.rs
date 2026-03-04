//! Synthea module types and loading functions.
//!
//! This module provides types for representing Synthea healthcare simulation modules,
//! which are state machines defining patient pathways through the healthcare system.

mod edge;
mod loader;
mod state;
mod types;

pub use edge::*;
pub use loader::*;
pub use state::*;
pub use types::*;
