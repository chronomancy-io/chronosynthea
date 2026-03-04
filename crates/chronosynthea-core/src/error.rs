//! Error types for ChronoSynthea core operations.

use thiserror::Error;

/// Errors that can occur during module operations.
#[derive(Error, Debug)]
pub enum ModuleError {
    /// Failed to read module file from disk.
    #[error("failed to read module file: {0}")]
    IoError(#[from] std::io::Error),

    /// Failed to parse module JSON.
    #[error("failed to parse module JSON: {0}")]
    ParseError(String),

    /// Module is missing required fields.
    #[error("module validation failed: {0}")]
    ValidationError(String),

    /// Module file does not exist.
    #[error("module file does not exist: {path}")]
    FileNotFound { path: String },
}

/// Errors that can occur during patient generation.
#[derive(Error, Debug)]
pub enum GeneratorError {
    /// Failed to load prevalence registry.
    #[error("failed to load prevalence registry: {0}")]
    RegistryError(String),

    /// Invalid configuration provided.
    #[error("invalid generator configuration: {0}")]
    ConfigError(String),

    /// I/O error during generation.
    #[error("I/O error during generation: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type alias for module operations.
pub type ModuleResult<T> = Result<T, ModuleError>;

/// Result type alias for generator operations.
pub type GeneratorResult<T> = Result<T, GeneratorError>;
