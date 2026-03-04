//! CDE encoding functions.
//!
//! WASP Role: E(r) — Implements CDE Phase 2 (Coordinate Encoding) for Synthea module
//! graph analysis. Encodes module states along structural axes (branching factor,
//! guard complexity, terminal density, transition entropy).
//!
//! NOTE: This is a module-analysis CDE operating on the Synthea state machine graph,
//! distinct from the WASP-level patient-generation dimensions (Seed, Trajectory,
//! Timing, Schema) defined in the top-level README.

use ahash::AHashMap;
use chronosynthea_core::Module;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::axis::{AxisModel, Vector, WeightedLinearAxisModel};
use crate::config::{default_axes, AxisConfig};
use crate::error::CdeResult;
use crate::features::{compute_structural_features, extract_state_features, FeatureVector};

/// Report from CDE encoding containing vectors and collision information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodeReport {
    /// Axis definitions used for encoding.
    pub axes: Vec<AxisConfig>,

    /// Encoded vectors for each state.
    pub vectors: Vec<Vector>,

    /// Collisions (states with identical hashes).
    pub collisions: Vec<Collision>,
}

/// A collision between states that have the same hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collision {
    /// The shared hash.
    pub hash: String,

    /// States that share this hash.
    pub states: Vec<String>,
}

/// Options for encoding.
#[derive(Debug, Clone, Default)]
pub struct EncodeOptions {
    /// Number of decimal places for precision (default: 6).
    pub precision_decimals: u32,

    /// Include structural features (out_degree, in_degree, depth, is_terminal).
    pub include_structural: bool,

    /// Include semantic features (has_condition, has_medication, etc.).
    pub include_semantic: bool,

    /// Maximum number of state types for one-hot encoding.
    pub max_one_hot_types: usize,
}

impl EncodeOptions {
    /// Creates options for structural-only encoding.
    pub fn structural_only() -> Self {
        Self {
            precision_decimals: 6,
            include_structural: true,
            include_semantic: false,
            max_one_hot_types: 20,
        }
    }

    /// Creates options for full semantic encoding.
    pub fn full_semantic() -> Self {
        Self {
            precision_decimals: 6,
            include_structural: true,
            include_semantic: true,
            max_one_hot_types: 20,
        }
    }
}

/// Encodes a module using structural features only (placeholder encoding).
///
/// This creates CDE vectors based on graph structure:
/// - out_degree: normalized number of outgoing edges
/// - in_degree: normalized number of incoming edges
/// - depth_from_initial: normalized graph distance from Initial state
/// - is_terminal: 1.0 if Terminal state, 0.0 otherwise
pub fn encode_module_structural(module: &Module) -> CdeResult<EncodeReport> {
    let axes = default_axes();
    let structural_features = compute_structural_features(module);

    let mut vectors = Vec::with_capacity(module.state_count());
    let mut hash_to_states: AHashMap<String, Vec<String>> = AHashMap::new();

    for state_name in module.state_names() {
        if let Some(fv) = structural_features.get(state_name) {
            // Create dimensions map from structural features
            let dims = fv.features.clone();
            let hash = hash_dimensions(&dims);

            let vector = Vector {
                state: state_name.to_string(),
                dims,
                hash: hash.clone(),
            };

            vectors.push(vector);
            hash_to_states
                .entry(hash)
                .or_default()
                .push(state_name.to_string());
        }
    }

    // Find collisions
    let mut collisions: Vec<Collision> = hash_to_states
        .into_iter()
        .filter(|(_, states)| states.len() > 1)
        .map(|(hash, mut states)| {
            states.sort();
            Collision { hash, states }
        })
        .collect();
    collisions.sort_by(|a, b| a.hash.cmp(&b.hash));

    Ok(EncodeReport {
        axes,
        vectors,
        collisions,
    })
}

/// Encodes a module using an axis model.
///
/// Combines structural and semantic features, then applies the axis model
/// to produce low-dimensional vectors.
pub fn encode_module_with_model(
    module: &Module,
    model: &impl AxisModel,
    options: &EncodeOptions,
) -> CdeResult<EncodeReport> {
    let axes = model.axes().to_vec();

    // Extract features
    let structural_features = compute_structural_features(module);
    let semantic_features = if options.include_semantic {
        Some(extract_state_features(module))
    } else {
        None
    };

    // Create combined feature vectors
    let mut combined_features: AHashMap<String, FeatureVector> = AHashMap::new();

    for state_name in module.state_names() {
        let mut features = AHashMap::new();

        // Add structural features
        if options.include_structural {
            if let Some(sf) = structural_features.get(state_name) {
                for (k, v) in &sf.features {
                    features.insert(k.clone(), *v);
                }
            }
        }

        // Add semantic features
        if let Some(ref semantic) = semantic_features {
            if let Some(fs) = semantic.iter().find(|f| f.state == state_name) {
                // Convert boolean flags to f64
                for (k, v) in &fs.flags {
                    features.insert(k.clone(), if *v { 1.0 } else { 0.0 });
                }
                // Add counts (raw values, will be normalized by axis model)
                for (k, v) in &fs.counts {
                    features.insert(k.clone(), *v);
                }
                // Add values
                for (k, v) in &fs.values {
                    features.insert(k.clone(), *v);
                }
            }
        }

        combined_features.insert(
            state_name.to_string(),
            FeatureVector {
                state: state_name.to_string(),
                features,
            },
        );
    }

    // Encode using the model
    let mut vectors = Vec::with_capacity(module.state_count());
    let mut hash_to_states: AHashMap<String, Vec<String>> = AHashMap::new();

    for state_name in module.state_names() {
        if let Some(fv) = combined_features.get(state_name) {
            let vector = model.encode(fv)?;
            hash_to_states
                .entry(vector.hash.clone())
                .or_default()
                .push(state_name.to_string());
            vectors.push(vector);
        }
    }

    // Find collisions
    let mut collisions: Vec<Collision> = hash_to_states
        .into_iter()
        .filter(|(_, states)| states.len() > 1)
        .map(|(hash, mut states)| {
            states.sort();
            Collision { hash, states }
        })
        .collect();
    collisions.sort_by(|a, b| a.hash.cmp(&b.hash));

    Ok(EncodeReport {
        axes,
        vectors,
        collisions,
    })
}

/// Encodes a module with the default weighted linear model.
pub fn encode_module(module: &Module) -> CdeResult<EncodeReport> {
    let model = WeightedLinearAxisModel::default_model();
    let options = EncodeOptions::full_semantic();
    encode_module_with_model(module, &model, &options)
}

/// Creates a deterministic hash of dimension values.
fn hash_dimensions(dims: &AHashMap<String, f64>) -> String {
    let mut hasher = Sha256::new();

    // Sort keys for determinism
    let mut keys: Vec<_> = dims.keys().collect();
    keys.sort();

    for key in keys {
        hasher.update(key.as_bytes());
        if let Some(&value) = dims.get(key) {
            hasher.update(format!("{:.6}", value).as_bytes());
        }
    }

    // Return first 16 characters of hex for readability
    let result = hasher.finalize();
    format!("{:x}", result)[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chronosynthea_core::load_module_from_str;

    const TEST_MODULE: &str = r#"{
        "name": "Test Module",
        "states": {
            "Initial": {"type": "Initial", "direct_transition": "Condition1"},
            "Condition1": {"type": "Condition", "direct_transition": "Terminal", "codes": [{"system": "SNOMED-CT", "code": "123456789"}]},
            "Terminal": {"type": "Terminal"}
        }
    }"#;

    #[test]
    fn test_encode_module_structural() {
        let module = load_module_from_str(TEST_MODULE).unwrap();
        let report = encode_module_structural(&module).unwrap();

        assert_eq!(report.vectors.len(), 3);

        // Check that all states have vectors
        let states: Vec<_> = report.vectors.iter().map(|v| v.state.as_str()).collect();
        assert!(states.contains(&"Initial"));
        assert!(states.contains(&"Condition1"));
        assert!(states.contains(&"Terminal"));
    }

    #[test]
    fn test_encode_deterministic() {
        let module = load_module_from_str(TEST_MODULE).unwrap();

        let report1 = encode_module_structural(&module).unwrap();
        let report2 = encode_module_structural(&module).unwrap();

        // Same module should produce same hashes
        for (v1, v2) in report1.vectors.iter().zip(report2.vectors.iter()) {
            assert_eq!(v1.hash, v2.hash);
        }
    }

    #[test]
    fn test_encode_with_model() {
        let module = load_module_from_str(TEST_MODULE).unwrap();
        let report = encode_module(&module).unwrap();

        assert_eq!(report.vectors.len(), 3);
        assert!(!report.axes.is_empty());
    }
}
