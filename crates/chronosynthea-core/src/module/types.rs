//! Core module types.

use ahash::AHashMap;
use serde::{Deserialize, Serialize};

use super::state::State;

/// A Synthea module representing a healthcare simulation state machine.
///
/// Modules define patient pathways through various states (encounters, conditions,
/// procedures, etc.) with transitions between them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Module {
    /// Module name (e.g., "Allergies", "Diabetes")
    pub name: String,

    /// Optional remarks/documentation
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub remarks: Vec<String>,

    /// States in the module, keyed by state name
    pub states: AHashMap<String, State>,

    /// GMF (Generic Module Framework) version
    #[serde(default)]
    pub gmf_version: i32,
}

impl Module {
    /// Returns true if the module has at least one Terminal state.
    #[inline]
    pub fn has_terminal_state(&self) -> bool {
        self.states.values().any(|s| s.state_type == "Terminal")
    }

    /// Returns true if the module has an "Initial" state.
    #[inline]
    pub fn has_initial_state(&self) -> bool {
        self.states.contains_key("Initial")
    }

    /// Returns the number of states in the module.
    #[inline]
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// Returns state names in deterministic sorted order.
    pub fn state_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.states.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
    }

    /// Returns state names as owned strings in deterministic sorted order.
    pub fn state_names_owned(&self) -> Vec<String> {
        let mut names: Vec<String> = self.states.keys().cloned().collect();
        names.sort_unstable();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_state_names_sorted() {
        let mut states = AHashMap::new();
        states.insert("Zebra".to_string(), State::default());
        states.insert("Apple".to_string(), State::default());
        states.insert("Middle".to_string(), State::default());

        let module = Module {
            name: "Test".to_string(),
            remarks: vec![],
            states,
            gmf_version: 2,
        };

        let names = module.state_names();
        assert_eq!(names, vec!["Apple", "Middle", "Zebra"]);
    }
}
