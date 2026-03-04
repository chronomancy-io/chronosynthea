//! Axis model for CDE encoding.

use ahash::AHashMap;
use serde::{Deserialize, Serialize};

use crate::config::{AxisConfig, LinearModelConfig};
use crate::error::CdeResult;
use crate::features::FeatureVector;

/// A CDE vector representing an encoded state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector {
    /// State name this vector represents.
    pub state: String,

    /// Dimension values keyed by axis name.
    pub dims: AHashMap<String, f64>,

    /// Deterministic hash of the vector.
    pub hash: String,
}

/// Trait for axis encoding models.
pub trait AxisModel: Send + Sync {
    /// Returns the model name.
    fn name(&self) -> &str;

    /// Returns the axis configurations.
    fn axes(&self) -> &[AxisConfig];

    /// Encodes a feature vector into a CDE vector.
    fn encode(&self, fv: &FeatureVector) -> CdeResult<Vector>;

    /// Returns explanation of how features contributed to each axis.
    fn explain(&self, fv: &FeatureVector) -> AHashMap<String, AHashMap<String, f64>>;
}

/// Weighted linear axis model.
///
/// Computes axis values as weighted linear combinations of features,
/// then clamps to [0, 1] and rounds to specified precision.
#[derive(Debug, Clone)]
pub struct WeightedLinearAxisModel {
    config: LinearModelConfig,
    precision_factor: f64,
}

impl WeightedLinearAxisModel {
    /// Creates a new weighted linear axis model from configuration.
    pub fn new(config: LinearModelConfig) -> Self {
        let precision_factor = 10_f64.powi(config.precision_decimals as i32);
        Self {
            config,
            precision_factor,
        }
    }

    /// Creates a model with default configuration.
    pub fn default_model() -> Self {
        Self::new(LinearModelConfig::default())
    }

    /// Rounds a value to the configured precision.
    #[inline]
    fn round_to_precision(&self, value: f64) -> f64 {
        (value * self.precision_factor).round() / self.precision_factor
    }

    /// Clamps a value to [0, 1].
    #[inline]
    fn clamp_01(value: f64) -> f64 {
        value.clamp(0.0, 1.0)
    }

    /// Computes axis value from features using weighted linear combination.
    fn compute_axis_value(&self, axis_name: &str, fv: &FeatureVector) -> f64 {
        let weights = match self.config.weights.get(axis_name) {
            Some(w) => w,
            None => return 0.0,
        };

        let mut sum = 0.0;
        for (feature_name, weight) in weights {
            if let Some(&feature_value) = fv.features.get(feature_name) {
                sum += weight * feature_value;
            }
        }

        self.round_to_precision(Self::clamp_01(sum))
    }

    /// Generates a deterministic hash for the vector.
    fn compute_hash(&self, state: &str, dims: &AHashMap<String, f64>) -> String {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        hasher.update(state.as_bytes());

        // Sort keys for determinism
        let mut keys: Vec<_> = dims.keys().collect();
        keys.sort();

        for key in keys {
            hasher.update(key.as_bytes());
            if let Some(&value) = dims.get(key) {
                hasher.update(format!("{:.6}", value).as_bytes());
            }
        }

        format!("{:x}", hasher.finalize())
    }
}

impl AxisModel for WeightedLinearAxisModel {
    fn name(&self) -> &str {
        "weighted_linear"
    }

    fn axes(&self) -> &[AxisConfig] {
        &self.config.axes
    }

    fn encode(&self, fv: &FeatureVector) -> CdeResult<Vector> {
        let mut dims = AHashMap::new();

        for axis in &self.config.axes {
            let value = self.compute_axis_value(&axis.name, fv);
            dims.insert(axis.name.clone(), value);
        }

        let hash = self.compute_hash(&fv.state, &dims);

        Ok(Vector {
            state: fv.state.clone(),
            dims,
            hash,
        })
    }

    fn explain(&self, fv: &FeatureVector) -> AHashMap<String, AHashMap<String, f64>> {
        let mut explanations = AHashMap::new();

        for axis in &self.config.axes {
            let mut axis_explanation = AHashMap::new();

            if let Some(weights) = self.config.weights.get(&axis.name) {
                for (feature_name, weight) in weights {
                    if let Some(&feature_value) = fv.features.get(feature_name) {
                        let contribution = weight * feature_value;
                        if contribution.abs() > 1e-10 {
                            axis_explanation.insert(feature_name.clone(), contribution);
                        }
                    }
                }
            }

            explanations.insert(axis.name.clone(), axis_explanation);
        }

        explanations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_feature_vector() -> FeatureVector {
        let mut features = AHashMap::new();
        features.insert("depth_from_initial".to_string(), 0.5);
        features.insert("out_degree".to_string(), 0.3);
        features.insert("is_terminal".to_string(), 0.0);
        features.insert("has_condition".to_string(), 1.0);
        features.insert("has_medication".to_string(), 0.0);
        features.insert("has_procedure".to_string(), 0.0);
        features.insert("has_observation".to_string(), 1.0);

        FeatureVector {
            state: "TestState".to_string(),
            features,
        }
    }

    #[test]
    fn test_weighted_linear_model_encode() {
        let model = WeightedLinearAxisModel::default_model();
        let fv = create_test_feature_vector();

        let vector = model.encode(&fv).unwrap();

        assert_eq!(vector.state, "TestState");
        assert!(vector.dims.contains_key("temporal_proxy"));
        assert!(vector.dims.contains_key("clinical_intensity"));
        assert!(vector.dims.contains_key("branching_uncertainty"));
        assert!(vector.dims.contains_key("terminality_risk"));
    }

    #[test]
    fn test_encode_deterministic() {
        let model = WeightedLinearAxisModel::default_model();
        let fv = create_test_feature_vector();

        let v1 = model.encode(&fv).unwrap();
        let v2 = model.encode(&fv).unwrap();

        assert_eq!(v1.hash, v2.hash);
        for (k, v) in &v1.dims {
            assert_eq!(v2.dims.get(k), Some(v));
        }
    }

    #[test]
    fn test_clamp_values() {
        let mut features = AHashMap::new();
        features.insert("depth_from_initial".to_string(), 5.0); // Should clamp to 1.0

        let fv = FeatureVector {
            state: "Test".to_string(),
            features,
        };

        let model = WeightedLinearAxisModel::default_model();
        let vector = model.encode(&fv).unwrap();

        assert!(vector.dims["temporal_proxy"] <= 1.0);
    }
}
