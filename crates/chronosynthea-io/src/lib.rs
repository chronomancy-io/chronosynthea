//! I/O formats and streaming for ChronoSynthea.
//!
//! This crate provides:
//! - Multiple output formats (JSONL, compact, codes-only)
//! - Streaming output with buffered writers
//! - Gzip compression support

mod error;
mod format;
mod stream;

pub use error::*;
pub use format::*;
pub use stream::*;
