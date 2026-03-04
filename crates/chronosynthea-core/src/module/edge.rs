//! Edge types for module graph representation.

use super::types::Module;

/// A directed edge in the module state graph.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Edge {
    /// Source state name
    pub from: String,
    /// Target state name
    pub to: String,
    /// Edge type (direct, complex, conditional, distributed, etc.)
    pub kind: EdgeKind,
}

/// The type of transition an edge represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EdgeKind {
    Direct,
    Complex,
    ComplexDistribution,
    Conditional,
    Distributed,
}

impl EdgeKind {
    /// Returns the string representation of the edge kind.
    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeKind::Direct => "direct",
            EdgeKind::Complex => "complex",
            EdgeKind::ComplexDistribution => "complex_distribution",
            EdgeKind::Conditional => "conditional",
            EdgeKind::Distributed => "distributed",
        }
    }
}

impl Module {
    /// Returns all edges in the module as a deterministically ordered slice.
    ///
    /// Edges are sorted by (from, to, kind) for stable graph views,
    /// which is essential for deterministic CDE encoding.
    pub fn edges(&self) -> Vec<Edge> {
        let mut edges = Vec::new();

        // Process each state in sorted order for determinism
        for from_state in self.state_names() {
            if let Some(state) = self.states.get(from_state) {
                // Direct transition
                if let Some(ref to) = state.direct_transition {
                    edges.push(Edge {
                        from: from_state.to_string(),
                        to: to.clone(),
                        kind: EdgeKind::Direct,
                    });
                }

                // Complex transitions
                for ct in &state.complex_transition {
                    if let Some(ref to) = ct.transition {
                        edges.push(Edge {
                            from: from_state.to_string(),
                            to: to.clone(),
                            kind: EdgeKind::Complex,
                        });
                    }
                    // Add distribution transitions
                    for dt in &ct.distributions {
                        edges.push(Edge {
                            from: from_state.to_string(),
                            to: dt.transition.clone(),
                            kind: EdgeKind::ComplexDistribution,
                        });
                    }
                }

                // Conditional transitions
                for ct in &state.conditional_transition {
                    edges.push(Edge {
                        from: from_state.to_string(),
                        to: ct.transition.clone(),
                        kind: EdgeKind::Conditional,
                    });
                }

                // Distributed transitions
                for dt in &state.distributed_transition {
                    if let Some(ref to) = dt.transition {
                        edges.push(Edge {
                            from: from_state.to_string(),
                            to: to.clone(),
                            kind: EdgeKind::Distributed,
                        });
                    }
                }
            }
        }

        // Sort edges deterministically (by from, then to, then kind)
        edges.sort();

        edges
    }

    /// Returns the total number of edges in the module graph.
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edges().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::state::State;
    use ahash::AHashMap;

    #[test]
    fn test_edges_sorted_deterministically() {
        let mut states = AHashMap::new();

        let mut initial = State::default();
        initial.state_type = "Initial".to_string();
        initial.direct_transition = Some("Middle".to_string());
        states.insert("Initial".to_string(), initial);

        let mut middle = State::default();
        middle.state_type = "Simple".to_string();
        middle.direct_transition = Some("Terminal".to_string());
        states.insert("Middle".to_string(), middle);

        let mut terminal = State::default();
        terminal.state_type = "Terminal".to_string();
        states.insert("Terminal".to_string(), terminal);

        let module = Module {
            name: "Test".to_string(),
            remarks: vec![],
            states,
            gmf_version: 2,
        };

        let edges = module.edges();
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].from, "Initial");
        assert_eq!(edges[0].to, "Middle");
        assert_eq!(edges[1].from, "Middle");
        assert_eq!(edges[1].to, "Terminal");
    }
}
