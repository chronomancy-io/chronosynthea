//! Error types for patient generation.

use thiserror::Error;

/// Errors that can occur during patient generation.
#[derive(Error, Debug)]
pub enum GeneratorError {
    /// Failed to load prevalence registry.
    #[error("failed to load prevalence registry: {0}")]
    RegistryLoadError(String),

    /// Invalid configuration.
    #[error("invalid generator configuration: {0}")]
    ConfigError(String),

    /// I/O error during generation.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON serialization error.
    #[error("serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Channel was closed (streaming mode).
    #[error("channel closed unexpectedly")]
    ChannelClosed,

    /// Generic I/O or callback error.
    #[error("I/O error: {0}")]
    Io(String),
}

/// Result type alias for generator operations.
pub type GeneratorResult<T> = Result<T, GeneratorError>;
