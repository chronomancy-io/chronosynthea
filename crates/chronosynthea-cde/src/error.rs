//! Error types for CDE operations.

use thiserror::Error;

/// Errors that can occur during CDE encoding.
#[derive(Error, Debug)]
pub enum CdeError {
    /// Feature extraction failed.
    #[error("feature extraction failed: {0}")]
    FeatureExtractionError(String),

    /// Encoding failed.
    #[error("encoding failed: {0}")]
    EncodingError(String),

    /// Invalid configuration.
    #[error("invalid configuration: {0}")]
    ConfigError(String),

    /// Calibration failed.
    #[error("calibration failed: {0}")]
    CalibrationError(String),
}

/// Result type alias for CDE operations.
pub type CdeResult<T> = Result<T, CdeError>;
