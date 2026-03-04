//! Error types for I/O operations.

use thiserror::Error;

/// Errors that can occur during I/O operations.
#[derive(Error, Debug)]
pub enum IoError {
    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization error (serde_json).
    #[error("JSON serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// JSON serialization error (simd-json).
    #[error("SIMD JSON serialization error: {0}")]
    SimdJsonSerialization(#[from] simd_json::Error),

    /// MessagePack serialization error.
    #[error("MessagePack serialization error: {0}")]
    MessagePackEncode(#[from] rmp_serde::encode::Error),

    /// Invalid format specified.
    #[error("invalid format: {0}")]
    InvalidFormat(String),
}

/// Result type alias for I/O operations.
pub type IoResult<T> = Result<T, IoError>;
