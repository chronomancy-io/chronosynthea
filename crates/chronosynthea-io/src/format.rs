//! Output format definitions.

use serde::{Deserialize, Serialize};

/// Output format for patient data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    /// Full JSON Lines format with all fields.
    Jsonl,
    /// Compact format with short field names (~2x smaller).
    Compact,
    /// Codes-only format with just unique codes per patient (~15x smaller).
    CodesOnly,
    /// JSON array format (all patients in one array).
    Json,
    /// MessagePack binary format (~3-5x faster than JSON).
    #[serde(rename = "msgpack")]
    MessagePack,
}

impl OutputFormat {
    /// Returns the file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            OutputFormat::Jsonl => "jsonl",
            OutputFormat::Compact => "compact.jsonl",
            OutputFormat::CodesOnly => "codes.jsonl",
            OutputFormat::Json => "json",
            OutputFormat::MessagePack => "msgpack",
        }
    }

    /// Parses a format from a string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "jsonl" => Some(OutputFormat::Jsonl),
            "compact" => Some(OutputFormat::Compact),
            "codes-only" | "codesonly" | "codes" => Some(OutputFormat::CodesOnly),
            "json" => Some(OutputFormat::Json),
            "msgpack" | "messagepack" | "mp" => Some(OutputFormat::MessagePack),
            _ => None,
        }
    }

    /// Returns a description of the format.
    pub fn description(&self) -> &'static str {
        match self {
            OutputFormat::Jsonl => "Full JSON Lines with all fields",
            OutputFormat::Compact => "Compact format with short field names",
            OutputFormat::CodesOnly => "Minimal format with unique codes only",
            OutputFormat::Json => "JSON array format",
            OutputFormat::MessagePack => "MessagePack binary format (3-5x faster)",
        }
    }

    /// Returns true if this is a binary format.
    pub fn is_binary(&self) -> bool {
        matches!(self, OutputFormat::MessagePack)
    }
}

impl Default for OutputFormat {
    fn default() -> Self {
        OutputFormat::Jsonl
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Jsonl => write!(f, "jsonl"),
            OutputFormat::Compact => write!(f, "compact"),
            OutputFormat::CodesOnly => write!(f, "codes-only"),
            OutputFormat::Json => write!(f, "json"),
            OutputFormat::MessagePack => write!(f, "msgpack"),
        }
    }
}
