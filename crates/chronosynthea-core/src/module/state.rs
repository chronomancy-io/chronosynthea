//! State types for Synthea modules.

use ahash::AHashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A state in a Synthea module state machine.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct State {
    /// State type (e.g., "Initial", "Terminal", "Encounter", "Condition", etc.)
    #[serde(rename = "type")]
    pub state_type: String,

    /// Optional remarks/documentation
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub remarks: Vec<String>,

    // === Transition fields (only one should be present) ===
    /// Direct transition to another state
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direct_transition: Option<String>,

    /// Complex transition with conditions and nested transitions
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub complex_transition: Vec<ComplexTransition>,

    /// Conditional transition based on conditions
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conditional_transition: Vec<ConditionalTransition>,

    /// Distributed transition with probabilities
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub distributed_transition: Vec<DistributedTransition>,

    // === State-specific fields ===
    /// Attribute name for attribute-based states
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attribute: Option<String>,

    /// Value for attribute states
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,

    /// Value code for coded values
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value_code: Option<Code>,

    /// Distribution for probabilistic states
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distribution: Option<Distribution>,

    /// Medical codes (SNOMED, ICD, etc.)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub codes: Vec<Code>,

    /// Attribute to assign result to
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assign_to_attribute: Option<String>,

    /// Category for observations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,

    /// Unit for measurements
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,

    /// Vital sign type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vital_sign: Option<String>,

    /// Numeric range
    #[serde(skip_serializing_if = "Option::is_none")]
    pub range: Option<ValueRange>,

    /// Encounter class
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encounter_class: Option<String>,

    /// Reason reference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,

    /// Telemedicine possibility flag
    #[serde(skip_serializing_if = "Option::is_none")]
    pub telemedicine_possibility: Option<String>,

    /// Observations in multi-observation states
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub observations: Vec<Observation>,

    /// Number of observations to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub number_of_observations: Option<i32>,

    /// Activities in care plan
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub activities: Vec<Activity>,

    /// Goals in care plan
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub goals: Vec<Goal>,

    /// Submodule reference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub submodule: Option<String>,

    /// Exact time quantity
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exact: Option<TimeQuantity>,

    /// Action type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,

    /// Wellness encounter flag
    #[serde(default)]
    pub wellness: bool,

    /// Minimum value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum: Option<i32>,

    /// Conditions for conditional logic
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conditions: Vec<Condition>,

    /// Operator for conditions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operator: Option<String>,

    /// Quantity value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantity: Option<f64>,

    /// Gender filter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gender: Option<String>,

    /// Race filter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub race: Option<String>,

    /// Default value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<f64>,

    /// Specialty for encounters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub specialty: Option<String>,

    /// Raw JSON data for semantic feature extraction.
    /// Captured during deserialization for accessing unknown fields.
    #[serde(flatten)]
    pub raw: AHashMap<String, Value>,
}

/// Complex transition with conditions and nested transitions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComplexTransition {
    /// Condition for this transition
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<Condition>,

    /// Target state
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transition: Option<String>,

    /// Nested distribution transitions
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub distributions: Vec<DistributionTransition>,
}

/// Conditional transition based on a condition.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConditionalTransition {
    /// Condition to evaluate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<Condition>,

    /// Target state if condition is true
    pub transition: String,
}

/// Distributed transition with probability.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistributedTransition {
    /// Target state
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transition: Option<String>,

    /// Probability of this transition (0.0 - 1.0)
    #[serde(default)]
    pub distribution: f64,

    /// Nested distributions
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub distributions: Vec<DistributionTransition>,
}

/// Distribution transition with probability.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistributionTransition {
    /// Probability of this transition
    pub distribution: f64,

    /// Target state
    pub transition: String,
}

/// Condition for transitions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Condition {
    /// Type of condition
    pub condition_type: String,

    /// Attribute to check
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attribute: Option<String>,

    /// Comparison operator
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operator: Option<String>,

    /// Value to compare against
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,

    /// Nested conditions for compound logic
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conditions: Vec<Condition>,

    /// Minimum value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum: Option<i32>,

    /// Gender filter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gender: Option<String>,

    /// Race filter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub race: Option<String>,

    /// Quantity value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantity: Option<f64>,

    /// Unit for quantity
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,

    /// Codes for condition
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub codes: Vec<Code>,
}

/// Medical code with coding system.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Code {
    /// Coding system (SNOMED-CT, ICD-10, LOINC, RxNorm, etc.)
    pub system: String,

    /// Code value (can be string or number)
    pub code: Value,

    /// Human-readable display text
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<String>,
}

impl Code {
    /// Returns the code as a string, converting numbers if necessary.
    pub fn code_string(&self) -> String {
        match &self.code {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            _ => self.code.to_string(),
        }
    }
}

/// Probability distribution specification.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Distribution {
    /// Distribution type (normal, uniform, etc.)
    pub kind: String,

    /// Distribution parameters
    #[serde(default)]
    pub parameters: AHashMap<String, Value>,

    /// Optional value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,

    /// Default value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<f64>,

    /// Attribute reference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attribute: Option<String>,

    /// Whether to round the result
    #[serde(default)]
    pub round: bool,

    /// Standard deviation for normal distribution
    #[serde(rename = "standardDeviation", skip_serializing_if = "Option::is_none")]
    pub standard_deviation: Option<f64>,

    /// Mean for normal distribution
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean: Option<f64>,
}

/// Numeric range with low and high bounds.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValueRange {
    pub low: f64,
    pub high: f64,
}

/// Time quantity with unit.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimeQuantity {
    pub quantity: f64,
    pub unit: String,
}

/// Observation in a multi-observation state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Observation {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub vital_sign: Option<String>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub codes: Vec<Code>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub range: Option<ValueRange>,
}

/// Activity in a care plan.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Activity {
    pub system: String,
    pub code: String,
    pub display: String,
}

/// Goal in a care plan.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Goal {
    #[serde(default)]
    pub addresses: Vec<String>,
    pub text: String,
}
