//! Feature extraction for CDE encoding.

use ahash::{AHashMap, AHashSet};
use chronosynthea_core::Module;
use serde_json::Value;
use std::collections::VecDeque;

/// A unified feature vector for a state, ready for axis encoding.
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// State name.
    pub state: String,

    /// Normalized features (all in [0, 1] range).
    pub features: AHashMap<String, f64>,
}

/// Raw feature set extracted from a state before normalization.
#[derive(Debug, Clone)]
pub struct FeatureSet {
    /// State name.
    pub state: String,

    /// State type (e.g., "Initial", "Terminal", "Condition").
    pub state_type: String,

    /// Boolean flags.
    pub flags: AHashMap<String, bool>,

    /// Count values (pre-normalization).
    pub counts: AHashMap<String, f64>,

    /// Numeric values (pre-normalization).
    pub values: AHashMap<String, f64>,
}

/// Extracts semantic features from all states in a module.
///
/// Analyzes the JSON structure and metadata to derive clinically-relevant signals.
pub fn extract_state_features(module: &Module) -> Vec<FeatureSet> {
    let mut features = Vec::with_capacity(module.state_count());

    // Get all unique state types for one-hot encoding
    let state_types = get_unique_state_types(module);

    // Compute edge information once
    let edges = module.edges();
    let mut out_degrees: AHashMap<&str, usize> = AHashMap::new();
    let mut in_degrees: AHashMap<&str, usize> = AHashMap::new();

    for edge in &edges {
        *out_degrees.entry(edge.from.as_str()).or_insert(0) += 1;
        *in_degrees.entry(edge.to.as_str()).or_insert(0) += 1;
    }

    // Process each state in deterministic order
    for state_name in module.state_names() {
        if let Some(state) = module.states.get(state_name) {
            let mut feature_set = FeatureSet {
                state: state_name.to_string(),
                state_type: state.state_type.clone(),
                flags: AHashMap::new(),
                counts: AHashMap::new(),
                values: AHashMap::new(),
            };

            // Extract transition type flags
            feature_set.flags.insert(
                "has_direct_transition".to_string(),
                state.direct_transition.is_some(),
            );
            feature_set.flags.insert(
                "has_distributed_transition".to_string(),
                !state.distributed_transition.is_empty(),
            );
            feature_set.flags.insert(
                "has_conditional_transition".to_string(),
                !state.conditional_transition.is_empty(),
            );

            // Extract action presence flags from raw JSON
            extract_action_flags(&mut feature_set, &state.raw);

            // Add state type one-hot encoding (stable ordering)
            for state_type in &state_types {
                let key = format!("state_type_{}", state_type);
                feature_set
                    .flags
                    .insert(key, state.state_type == *state_type);
            }

            // Add structural counts
            let out_count = out_degrees.get(state_name).copied().unwrap_or(0);
            let in_count = in_degrees.get(state_name).copied().unwrap_or(0);
            feature_set
                .counts
                .insert("out_edge_count".to_string(), out_count as f64);
            feature_set
                .counts
                .insert("in_edge_count".to_string(), in_count as f64);

            features.push(feature_set);
        }
    }

    features
}

/// Extracts action presence flags from raw JSON data.
fn extract_action_flags(feature_set: &mut FeatureSet, raw: &AHashMap<String, Value>) {
    // has_condition
    feature_set
        .flags
        .insert("has_condition".to_string(), raw.contains_key("condition"));

    // has_encounter
    let has_encounter = raw.contains_key("encounter_class")
        || (raw.contains_key("codes") && is_encounter_code(raw));
    feature_set
        .flags
        .insert("has_encounter".to_string(), has_encounter);

    // has_medication
    let has_medication = raw.contains_key("administration")
        || (raw.contains_key("codes") && is_medication_code(raw));
    feature_set
        .flags
        .insert("has_medication".to_string(), has_medication);

    // has_observation
    let has_observation =
        raw.contains_key("category") && raw.contains_key("codes") && is_observation_code(raw);
    feature_set
        .flags
        .insert("has_observation".to_string(), has_observation);

    // has_procedure
    let has_procedure = raw.contains_key("codes") && is_procedure_code(raw);
    feature_set
        .flags
        .insert("has_procedure".to_string(), has_procedure);

    // has_immunization
    let has_immunization = raw.contains_key("codes") && is_immunization_code(raw);
    feature_set
        .flags
        .insert("has_immunization".to_string(), has_immunization);
}

/// Gets all unique state types in deterministic sorted order.
fn get_unique_state_types(module: &Module) -> Vec<String> {
    let type_set: AHashSet<_> = module
        .states
        .values()
        .map(|s| s.state_type.clone())
        .collect();
    let mut types: Vec<_> = type_set.into_iter().collect();
    types.sort();
    types
}

/// Checks if codes contain encounter-related codes.
fn is_encounter_code(raw: &AHashMap<String, Value>) -> bool {
    if raw.contains_key("encounter_class") {
        return true;
    }

    if let Some(Value::Array(codes)) = raw.get("codes") {
        for code in codes {
            if let Some(system) = code.get("system").and_then(|s| s.as_str()) {
                if system == "SNOMED-CT" {
                    if let Some(code_str) = code.get("code").and_then(|c| c.as_str()) {
                        // Common encounter code patterns
                        if code_str.len() >= 7
                            && (code_str.starts_with("3083350")
                                || code_str.starts_with("3711530")
                                || code_str.starts_with("183"))
                        {
                            return true;
                        }
                    }
                }
            }
        }
    }

    false
}

/// Checks if codes contain medication-related codes (RxNorm).
fn is_medication_code(raw: &AHashMap<String, Value>) -> bool {
    if let Some(Value::Array(codes)) = raw.get("codes") {
        for code in codes {
            if let Some(system) = code.get("system").and_then(|s| s.as_str()) {
                if system == "RxNorm" {
                    return true;
                }
            }
        }
    }
    false
}

/// Checks if codes contain observation-related codes (LOINC with category).
fn is_observation_code(raw: &AHashMap<String, Value>) -> bool {
    if let Some(category) = raw.get("category").and_then(|c| c.as_str()) {
        if category == "vital-signs" || category == "laboratory" {
            if let Some(Value::Array(codes)) = raw.get("codes") {
                for code in codes {
                    if let Some(system) = code.get("system").and_then(|s| s.as_str()) {
                        if system == "LOINC" {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

/// Checks if codes contain procedure-related codes (SNOMED procedures).
fn is_procedure_code(raw: &AHashMap<String, Value>) -> bool {
    if let Some(Value::Array(codes)) = raw.get("codes") {
        for code in codes {
            if let Some(system) = code.get("system").and_then(|s| s.as_str()) {
                if system == "SNOMED-CT" {
                    if let Some(code_str) = code.get("code").and_then(|c| c.as_str()) {
                        // SNOMED procedure codes are typically 8-10 digits
                        if (8..=10).contains(&code_str.len()) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

/// Checks if codes contain immunization-related codes (CVX).
fn is_immunization_code(raw: &AHashMap<String, Value>) -> bool {
    if let Some(Value::Array(codes)) = raw.get("codes") {
        for code in codes {
            if let Some(system) = code.get("system").and_then(|s| s.as_str()) {
                if system == "CVX" {
                    return true;
                }
            }
        }
    }
    false
}

/// Structural features for a state (pre-computed for efficiency).
#[derive(Debug, Clone, Default)]
struct StructuralFeatures {
    out_degree: f64,
    in_degree: f64,
    depth_from_initial: f64,
    is_terminal: f64,
}

/// Computes structural features for all states in a module.
pub fn compute_structural_features(module: &Module) -> AHashMap<String, FeatureVector> {
    let mut raw_features: AHashMap<String, StructuralFeatures> = AHashMap::new();

    // Compute degrees
    let edges = module.edges();
    let mut out_degrees: AHashMap<&str, usize> = AHashMap::new();
    let mut in_degrees: AHashMap<&str, usize> = AHashMap::new();

    for edge in &edges {
        *out_degrees.entry(edge.from.as_str()).or_insert(0) += 1;
        *in_degrees.entry(edge.to.as_str()).or_insert(0) += 1;
    }

    // Compute depths from Initial using BFS
    let depths = compute_depths_from_initial(module);

    // Collect raw values
    let mut out_values = Vec::new();
    let mut in_values = Vec::new();
    let mut depth_values = Vec::new();

    for state_name in module.state_names() {
        let out_deg = out_degrees.get(state_name).copied().unwrap_or(0) as f64;
        let in_deg = in_degrees.get(state_name).copied().unwrap_or(0) as f64;
        let depth = depths.get(state_name).copied().unwrap_or(0) as f64;
        let is_terminal = if module
            .states
            .get(state_name)
            .map(|s| s.state_type == "Terminal")
            .unwrap_or(false)
        {
            1.0
        } else {
            0.0
        };

        out_values.push(out_deg);
        in_values.push(in_deg);
        depth_values.push(depth);

        raw_features.insert(
            state_name.to_string(),
            StructuralFeatures {
                out_degree: out_deg,
                in_degree: in_deg,
                depth_from_initial: depth,
                is_terminal,
            },
        );
    }

    // Compute min/max for normalization
    let (out_min, out_max) = min_max(&out_values);
    let (in_min, in_max) = min_max(&in_values);
    let (depth_min, depth_max) = min_max(&depth_values);

    // Normalize and create feature vectors
    let mut result = AHashMap::new();

    for (state_name, feat) in raw_features {
        let mut features = AHashMap::new();
        features.insert(
            "out_degree".to_string(),
            normalize(feat.out_degree, out_min, out_max),
        );
        features.insert(
            "in_degree".to_string(),
            normalize(feat.in_degree, in_min, in_max),
        );
        features.insert(
            "depth_from_initial".to_string(),
            normalize(feat.depth_from_initial, depth_min, depth_max),
        );
        features.insert("is_terminal".to_string(), feat.is_terminal);

        result.insert(
            state_name.clone(),
            FeatureVector {
                state: state_name,
                features,
            },
        );
    }

    result
}

/// Computes graph distance from Initial state for each state using BFS.
fn compute_depths_from_initial(module: &Module) -> AHashMap<String, usize> {
    let mut depths = AHashMap::new();

    if !module.has_initial_state() {
        // All depths 0 if no Initial state
        for state_name in module.state_names() {
            depths.insert(state_name.to_string(), 0);
        }
        return depths;
    }

    let mut visited = AHashSet::new();
    let mut queue = VecDeque::new();

    // Build adjacency list
    let edges = module.edges();
    let mut adj: AHashMap<&str, Vec<&str>> = AHashMap::new();
    for edge in &edges {
        adj.entry(edge.from.as_str())
            .or_default()
            .push(edge.to.as_str());
    }

    // BFS from Initial
    queue.push_back(("Initial", 0));
    visited.insert("Initial");
    depths.insert("Initial".to_string(), 0);

    while let Some((current, current_depth)) = queue.pop_front() {
        if let Some(neighbors) = adj.get(current) {
            for &neighbor in neighbors {
                if !visited.contains(neighbor) {
                    visited.insert(neighbor);
                    depths.insert(neighbor.to_string(), current_depth + 1);
                    queue.push_back((neighbor, current_depth + 1));
                }
            }
        }
    }

    // Any unvisited states get depth 0
    for state_name in module.state_names() {
        if !visited.contains(state_name) {
            depths.insert(state_name.to_string(), 0);
        }
    }

    depths
}

/// Returns min and max values from a slice.
#[inline]
fn min_max(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }

    let mut min = values[0];
    let mut max = values[0];

    for &v in &values[1..] {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }

    (min, max)
}

/// Normalizes a value to [0, 1] using min-max scaling.
#[inline]
fn normalize(value: f64, min: f64, max: f64) -> f64 {
    if (max - min).abs() < f64::EPSILON {
        0.5 // If all values are the same, use 0.5
    } else {
        (value - min) / (max - min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chronosynthea_core::load_module_from_str;

    const TEST_MODULE: &str = r#"{
        "name": "Test Module",
        "states": {
            "Initial": {"type": "Initial", "direct_transition": "Middle"},
            "Middle": {"type": "Simple", "direct_transition": "Terminal"},
            "Terminal": {"type": "Terminal"}
        }
    }"#;

    #[test]
    fn test_extract_state_features() {
        let module = load_module_from_str(TEST_MODULE).unwrap();
        let features = extract_state_features(&module);

        assert_eq!(features.len(), 3);

        // Check that all states have features
        let states: Vec<_> = features.iter().map(|f| f.state.as_str()).collect();
        assert!(states.contains(&"Initial"));
        assert!(states.contains(&"Middle"));
        assert!(states.contains(&"Terminal"));
    }

    #[test]
    fn test_compute_structural_features() {
        let module = load_module_from_str(TEST_MODULE).unwrap();
        let features = compute_structural_features(&module);

        assert_eq!(features.len(), 3);

        // Initial should have depth 0 (normalized to 0.0)
        let initial = features.get("Initial").unwrap();
        assert_eq!(initial.features["depth_from_initial"], 0.0);

        // Terminal should be marked as terminal
        let terminal = features.get("Terminal").unwrap();
        assert_eq!(terminal.features["is_terminal"], 1.0);
    }

    #[test]
    fn test_depths_computed_correctly() {
        let module = load_module_from_str(TEST_MODULE).unwrap();
        let depths = compute_depths_from_initial(&module);

        assert_eq!(depths["Initial"], 0);
        assert_eq!(depths["Middle"], 1);
        assert_eq!(depths["Terminal"], 2);
    }
}
