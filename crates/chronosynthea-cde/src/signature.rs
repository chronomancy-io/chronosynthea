//! Module signature computation for deterministic identification.

use chronosynthea_core::Module;
use sha2::{Digest, Sha256};

/// Computes a deterministic signature for a module.
///
/// The signature is based on:
/// - Module name
/// - Sorted state names
/// - State types
/// - Edge structure
///
/// This provides a stable identifier for regression testing and caching.
pub fn compute_signature(module: &Module) -> String {
    let mut hasher = Sha256::new();

    // Include module name
    hasher.update(module.name.as_bytes());
    hasher.update(b"|");

    // Include sorted state names and types
    for state_name in module.state_names() {
        hasher.update(state_name.as_bytes());
        hasher.update(b":");
        if let Some(state) = module.states.get(state_name) {
            hasher.update(state.state_type.as_bytes());
        }
        hasher.update(b",");
    }
    hasher.update(b"|");

    // Include edges
    for edge in module.edges() {
        hasher.update(edge.from.as_bytes());
        hasher.update(b"->");
        hasher.update(edge.to.as_bytes());
        hasher.update(b":");
        hasher.update(edge.kind.as_str().as_bytes());
        hasher.update(b",");
    }

    // Return full hex hash
    format!("{:x}", hasher.finalize())
}

/// Computes a short signature (first 16 hex characters).
pub fn compute_short_signature(module: &Module) -> String {
    compute_signature(module)[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chronosynthea_core::load_module_from_str;

    const TEST_MODULE: &str = r#"{
        "name": "Test Module",
        "states": {
            "Initial": {"type": "Initial", "direct_transition": "Terminal"},
            "Terminal": {"type": "Terminal"}
        }
    }"#;

    #[test]
    fn test_signature_deterministic() {
        let module = load_module_from_str(TEST_MODULE).unwrap();

        let sig1 = compute_signature(&module);
        let sig2 = compute_signature(&module);

        assert_eq!(sig1, sig2);
    }

    #[test]
    fn test_signature_length() {
        let module = load_module_from_str(TEST_MODULE).unwrap();

        let full = compute_signature(&module);
        let short = compute_short_signature(&module);

        assert_eq!(full.len(), 64); // SHA256 hex = 64 chars
        assert_eq!(short.len(), 16);
    }

    #[test]
    fn test_different_modules_different_signatures() {
        let module1 = load_module_from_str(TEST_MODULE).unwrap();

        let module2_json = r#"{
            "name": "Different Module",
            "states": {
                "Initial": {"type": "Initial", "direct_transition": "Terminal"},
                "Terminal": {"type": "Terminal"}
            }
        }"#;
        let module2 = load_module_from_str(module2_json).unwrap();

        assert_ne!(compute_signature(&module1), compute_signature(&module2));
    }
}
