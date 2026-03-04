//! Metrics computation for CDE encoding quality.

use ahash::AHashMap;
use serde::{Deserialize, Serialize};

use crate::axis::Vector;
use crate::encode::EncodeReport;

/// Metrics for evaluating CDE encoding quality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    /// Number of input feature dimensions.
    pub feature_dims: usize,

    /// Number of output axis dimensions.
    pub axis_dims: usize,

    /// Compression ratio (feature_dims / axis_dims).
    pub compression_ratio: f64,

    /// Number of exact hash collisions.
    pub collision_count: usize,

    /// Number of near-collisions (L1 distance <= 0.001).
    pub near_collision_count: usize,

    /// Mean pairwise L1 distance between vectors.
    pub mean_pairwise_l1: f64,

    /// Entropy per axis (measure of value distribution).
    pub mean_axis_entropy: AHashMap<String, f64>,

    /// Saturation: fraction of values at 0.0 or 1.0.
    pub saturation: f64,
}

/// Computes metrics for an encoding report.
pub fn compute_metrics(report: &EncodeReport, feature_dims: usize) -> Metrics {
    let axis_dims = report.axes.len();
    let vectors = &report.vectors;

    // Compression ratio
    let compression_ratio = if axis_dims > 0 {
        feature_dims as f64 / axis_dims as f64
    } else {
        0.0
    };

    // Collision count
    let collision_count: usize = report.collisions.iter().map(|c| c.states.len() - 1).sum();

    // Near-collision detection
    let near_collision_count = count_near_collisions(vectors, 0.001);

    // Mean pairwise L1 distance
    let mean_pairwise_l1 = compute_mean_pairwise_l1(vectors);

    // Axis entropy
    let mean_axis_entropy = compute_axis_entropy(vectors, &report.axes);

    // Saturation
    let saturation = compute_saturation(vectors);

    Metrics {
        feature_dims,
        axis_dims,
        compression_ratio,
        collision_count,
        near_collision_count,
        mean_pairwise_l1,
        mean_axis_entropy,
        saturation,
    }
}

/// Counts pairs of vectors with L1 distance <= threshold.
fn count_near_collisions(vectors: &[Vector], threshold: f64) -> usize {
    let n = vectors.len();
    if n <= 1 {
        return 0;
    }

    let mut count = 0;

    // For large n, use sampling to avoid O(n²) cost
    let sample_size = if n > 100 { 100 } else { n };
    let step = n / sample_size;

    for i in 0..sample_size {
        let vi = &vectors[i * step];
        for j in (i + 1)..sample_size {
            let vj = &vectors[j * step];
            let l1 = compute_l1_distance(&vi.dims, &vj.dims);
            if l1 <= threshold && vi.hash != vj.hash {
                count += 1;
            }
        }
    }

    // Scale up if we sampled
    if n > 100 {
        count = count * n * n / (sample_size * sample_size);
    }

    count
}

/// Computes L1 (Manhattan) distance between two dimension maps.
fn compute_l1_distance(a: &AHashMap<String, f64>, b: &AHashMap<String, f64>) -> f64 {
    let mut sum = 0.0;

    for (key, &va) in a {
        let vb = b.get(key).copied().unwrap_or(0.0);
        sum += (va - vb).abs();
    }

    // Handle keys in b but not in a
    for (key, &vb) in b {
        if !a.contains_key(key) {
            sum += vb.abs();
        }
    }

    sum
}

/// Computes mean pairwise L1 distance.
fn compute_mean_pairwise_l1(vectors: &[Vector]) -> f64 {
    let n = vectors.len();
    if n <= 1 {
        return 0.0;
    }

    // For large n, use deterministic sampling
    let sample_size = if n > 100 { 100 } else { n };
    let step = n / sample_size;

    let mut total = 0.0;
    let mut count = 0;

    for i in 0..sample_size {
        let vi = &vectors[i * step];
        for j in (i + 1)..sample_size {
            let vj = &vectors[j * step];
            total += compute_l1_distance(&vi.dims, &vj.dims);
            count += 1;
        }
    }

    if count > 0 {
        total / count as f64
    } else {
        0.0
    }
}

/// Computes entropy for each axis using 10-bin histogram.
fn compute_axis_entropy(
    vectors: &[Vector],
    axes: &[crate::config::AxisConfig],
) -> AHashMap<String, f64> {
    let mut result = AHashMap::new();
    let num_bins = 10;

    for axis in axes {
        // Collect values for this axis
        let values: Vec<f64> = vectors
            .iter()
            .filter_map(|v| v.dims.get(&axis.name).copied())
            .collect();

        if values.is_empty() {
            result.insert(axis.name.clone(), 0.0);
            continue;
        }

        // Build histogram
        let mut bins = vec![0usize; num_bins];
        for &v in &values {
            let bin = ((v * num_bins as f64) as usize).min(num_bins - 1);
            bins[bin] += 1;
        }

        // Compute Shannon entropy
        let n = values.len() as f64;
        let mut entropy = 0.0;
        for &count in &bins {
            if count > 0 {
                let p = count as f64 / n;
                entropy -= p * p.log2();
            }
        }

        // Normalize to [0, 1] where max entropy = log2(num_bins)
        let max_entropy = (num_bins as f64).log2();
        let normalized_entropy = if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        };

        result.insert(axis.name.clone(), normalized_entropy);
    }

    result
}

/// Computes saturation: fraction of values at exactly 0.0 or 1.0.
fn compute_saturation(vectors: &[Vector]) -> f64 {
    if vectors.is_empty() {
        return 0.0;
    }

    let mut saturated = 0;
    let mut total = 0;

    for v in vectors {
        for &value in v.dims.values() {
            total += 1;
            if value <= f64::EPSILON || (1.0 - value).abs() <= f64::EPSILON {
                saturated += 1;
            }
        }
    }

    if total > 0 {
        saturated as f64 / total as f64
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::encode_module_structural;
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
    fn test_compute_metrics() {
        let module = load_module_from_str(TEST_MODULE).unwrap();
        let report = encode_module_structural(&module).unwrap();
        let metrics = compute_metrics(&report, 4);

        assert_eq!(metrics.axis_dims, 4);
        assert!(metrics.compression_ratio > 0.0);
        assert!(metrics.saturation >= 0.0 && metrics.saturation <= 1.0);
    }

    #[test]
    fn test_l1_distance() {
        let mut a = AHashMap::new();
        a.insert("x".to_string(), 0.0);
        a.insert("y".to_string(), 1.0);

        let mut b = AHashMap::new();
        b.insert("x".to_string(), 1.0);
        b.insert("y".to_string(), 0.0);

        let distance = compute_l1_distance(&a, &b);
        assert!((distance - 2.0).abs() < f64::EPSILON);
    }
}
