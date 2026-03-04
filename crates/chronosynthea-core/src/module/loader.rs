//! Module loading functions.

use std::fs;
use std::path::Path;

use super::types::Module;
use crate::error::{ModuleError, ModuleResult};

/// Loads a Synthea module from a JSON file path.
///
/// # Arguments
/// * `path` - Path to the module JSON file
///
/// # Returns
/// * `Ok(Module)` - Successfully loaded and validated module
/// * `Err(ModuleError)` - Loading or validation failed
///
/// # Example
/// ```ignore
/// use chronosynthea_core::load_module;
/// let module = load_module("modules/allergies.json")?;
/// println!("Loaded module: {}", module.name);
/// ```
pub fn load_module<P: AsRef<Path>>(path: P) -> ModuleResult<Module> {
    let path = path.as_ref();

    // Check if file exists
    if !path.exists() {
        return Err(ModuleError::FileNotFound {
            path: path.display().to_string(),
        });
    }

    // Read the file
    let data = fs::read(path)?;

    // Parse and validate
    load_module_from_bytes(&data)
}

/// Loads a module from JSON bytes.
///
/// Useful for testing or when module data is already in memory.
///
/// # Arguments
/// * `data` - JSON bytes representing the module
///
/// # Returns
/// * `Ok(Module)` - Successfully parsed and validated module
/// * `Err(ModuleError)` - Parsing or validation failed
pub fn load_module_from_bytes(data: &[u8]) -> ModuleResult<Module> {
    // Try simd-json first for performance, fall back to serde_json
    let module: Module = match simd_json::from_slice(&mut data.to_vec()) {
        Ok(m) => m,
        Err(_) => {
            // Fall back to standard serde_json
            serde_json::from_slice(data).map_err(|e| ModuleError::ParseError(e.to_string()))?
        }
    };

    // Validate basic structure
    validate_module(&module)?;

    Ok(module)
}

/// Loads a module from a JSON string.
///
/// # Arguments
/// * `json` - JSON string representing the module
///
/// # Returns
/// * `Ok(Module)` - Successfully parsed and validated module
/// * `Err(ModuleError)` - Parsing or validation failed
pub fn load_module_from_str(json: &str) -> ModuleResult<Module> {
    load_module_from_bytes(json.as_bytes())
}

/// Validates a module's basic structure.
fn validate_module(module: &Module) -> ModuleResult<()> {
    if module.name.is_empty() {
        return Err(ModuleError::ValidationError(
            "module missing required 'name' field".to_string(),
        ));
    }

    if module.states.is_empty() {
        return Err(ModuleError::ValidationError(
            "module has no states".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_MODULE: &str = r#"{
        "name": "Test Module",
        "states": {
            "Initial": {
                "type": "Initial",
                "direct_transition": "Terminal"
            },
            "Terminal": {
                "type": "Terminal"
            }
        }
    }"#;

    #[test]
    fn test_load_module_from_str() {
        let module = load_module_from_str(MINIMAL_MODULE).unwrap();
        assert_eq!(module.name, "Test Module");
        assert_eq!(module.state_count(), 2);
        assert!(module.has_initial_state());
        assert!(module.has_terminal_state());
    }

    #[test]
    fn test_load_module_validates_name() {
        let json = r#"{"name": "", "states": {"Initial": {"type": "Initial"}}}"#;
        let result = load_module_from_str(json);
        assert!(result.is_err());
        assert!(matches!(result, Err(ModuleError::ValidationError(_))));
    }

    #[test]
    fn test_load_module_validates_states() {
        let json = r#"{"name": "Test", "states": {}}"#;
        let result = load_module_from_str(json);
        assert!(result.is_err());
        assert!(matches!(result, Err(ModuleError::ValidationError(_))));
    }

    #[test]
    fn test_module_edges() {
        let module = load_module_from_str(MINIMAL_MODULE).unwrap();
        let edges = module.edges();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].from, "Initial");
        assert_eq!(edges[0].to, "Terminal");
    }
}
