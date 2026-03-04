//! Configuration types for CDE encoding.

use ahash::AHashMap;
use serde::{Deserialize, Serialize};

/// Configuration for the linear axis model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearModelConfig {
    /// Number of decimal places for precision (default: 6).
    #[serde(default = "default_precision")]
    pub precision_decimals: u32,

    /// Axis definitions.
    pub axes: Vec<AxisConfig>,

    /// Weight matrix: axis_name -> feature_name -> weight.
    #[serde(default)]
    pub weights: AHashMap<String, AHashMap<String, f64>>,
}

fn default_precision() -> u32 {
    6
}

impl Default for LinearModelConfig {
    fn default() -> Self {
        Self {
            precision_decimals: 6,
            axes: default_axes(),
            weights: default_weights(),
        }
    }
}

/// Configuration for a single axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisConfig {
    /// Axis name.
    pub name: String,

    /// Human-readable description.
    pub description: String,

    /// Minimum value (typically 0.0).
    #[serde(default)]
    pub min_value: f64,

    /// Maximum value (typically 1.0).
    #[serde(default = "default_max_value")]
    pub max_value: f64,

    /// Optional units.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub units: Option<String>,
}

fn default_max_value() -> f64 {
    1.0
}

/// Returns the default 4-axis configuration.
pub fn default_axes() -> Vec<AxisConfig> {
    vec![
        AxisConfig {
            name: "temporal_proxy".to_string(),
            description: "Temporal progression through workflow".to_string(),
            min_value: 0.0,
            max_value: 1.0,
            units: None,
        },
        AxisConfig {
            name: "clinical_intensity".to_string(),
            description: "Intervention intensity".to_string(),
            min_value: 0.0,
            max_value: 1.0,
            units: None,
        },
        AxisConfig {
            name: "branching_uncertainty".to_string(),
            description: "Decision complexity".to_string(),
            min_value: 0.0,
            max_value: 1.0,
            units: None,
        },
        AxisConfig {
            name: "terminality_risk".to_string(),
            description: "Risk of terminal outcomes".to_string(),
            min_value: 0.0,
            max_value: 1.0,
            units: None,
        },
    ]
}

/// Returns the default weight matrix for the linear model.
pub fn default_weights() -> AHashMap<String, AHashMap<String, f64>> {
    let mut weights = AHashMap::new();

    // Temporal proxy weights
    let mut temporal = AHashMap::new();
    temporal.insert("depth_from_initial".to_string(), 1.0);
    weights.insert("temporal_proxy".to_string(), temporal);

    // Clinical intensity weights
    let mut clinical = AHashMap::new();
    clinical.insert("has_condition".to_string(), 0.3);
    clinical.insert("has_medication".to_string(), 0.3);
    clinical.insert("has_procedure".to_string(), 0.3);
    clinical.insert("has_observation".to_string(), 0.1);
    weights.insert("clinical_intensity".to_string(), clinical);

    // Branching uncertainty weights
    let mut branching = AHashMap::new();
    branching.insert("out_degree".to_string(), 0.5);
    branching.insert("has_conditional_transition".to_string(), 0.25);
    branching.insert("has_distributed_transition".to_string(), 0.25);
    weights.insert("branching_uncertainty".to_string(), branching);

    // Terminality risk weights
    let mut terminality = AHashMap::new();
    terminality.insert("is_terminal".to_string(), 1.0);
    weights.insert("terminality_risk".to_string(), terminality);

    weights
}
