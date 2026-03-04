//! Core types for ChronoSynthea synthetic patient generation.
//!
//! This crate provides the foundational types for:
//! - Synthea module representation (Module, State, Edge)
//! - Patient records (Patient, Encounter, Event)
//! - Module loading and validation

#![allow(clippy::field_reassign_with_default)]

mod error;
mod module;
mod patient;

pub use error::*;
pub use module::*;
pub use patient::*;
