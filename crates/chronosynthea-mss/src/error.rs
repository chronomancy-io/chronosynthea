//! Error types for MSS operations.

use thiserror::Error;

/// MSS-specific errors.
#[derive(Error, Debug)]
pub enum MssError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("MessagePack error: {0}")]
    MsgPack(#[from] rmp_serde::decode::Error),

    #[error("MessagePack encode error: {0}")]
    MsgPackEncode(#[from] rmp_serde::encode::Error),

    #[error("Invalid fingerprint format: {0}")]
    InvalidFormat(String),

    #[error("Missing required data: {0}")]
    MissingData(String),

    #[error("Statistical validation failed: {0}")]
    ValidationFailed(String),
}

/// Result type for MSS operations.
pub type MssResult<T> = Result<T, MssError>;
