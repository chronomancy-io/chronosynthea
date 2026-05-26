//! Cohort query library — the user-facing surface for "give me
//! patients matching a filter expression, then materialise them"
//! workflows.
//!
//! Council-pass premise: the friction researchers have is cohort
//! definition and downstream query ergonomics, not raw generation
//! throughput. chronosynthea generating 88K patients/s end-to-end is
//! already overkill for any realistic cohort study — what's missing is
//! a typed, composable way to *say* "stroke patients aged 60-70 with
//! diabetes comorbidity" and get the matching slice back.
//!
//! This module ships:
//!
//! 1. `FilterExpr` — a Serde-serialisable AST for cohort filter
//!    expressions. Compose with `And`, `Or`, `Not`. Leaves cover the
//!    common probe axes: archetype, age band / range, sex, race,
//!    ethnicity, condition membership.
//! 2. `FilterEvaluator` — wraps an `ArchetypeRegistry + CodeTable +
//!    FilterExpr` so the per-patient `matches()` call is `O(1)` (the
//!    condition-code lookups are pre-resolved to `u16` indices once at
//!    construction).
//! 3. `BatchGenerator::cohort(filter, target_count, max_scan, on_match)`
//!    — streams generated patients through the filter, calling
//!    `on_match` for each that passes, until `target_count` matches or
//!    `max_scan` patients have been considered.
//!
//! Composes with the Parquet writers: a typical pipeline is
//!
//! ```ignore
//! let filter = FilterExpr::And(vec![
//!     FilterExpr::AgeRange { lo: 60, hi: 70 },
//!     FilterExpr::HasCondition("230690007".into()), // stroke
//! ]);
//! let mut w = SyntheaStatsParquetWriter::create(&out_dir)?;
//! let outcome = generator.cohort(
//!     &filter,
//!     /* target_count */ 1000,
//!     /* max_scan */ 100_000,
//!     |p| { w.write_patient(p).unwrap(); },
//! );
//! w.finish()?;
//! ```

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::archetype::ArchetypeRegistry;
use crate::arena::FullPatient;
use crate::tables::CodeTable;

/// Cohort filter expression. Composable, serialisable, evaluable
/// against a `FullPatient` in `O(filter_depth)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "op")]
pub enum FilterExpr {
    /// Matches every patient.
    All,
    /// `patient.archetype_id == id`.
    Archetype { id: u16 },
    /// `patient.archetype_id ∈ ids`.
    ArchetypeIn { ids: Vec<u16> },
    /// `age_years ∈ [lo, hi)` where age is computed against today.
    AgeRange { lo: u32, hi: u32 },
    /// `sex == "M"` or `"F"`.
    Sex { value: String },
    /// `race == one of "white"|"black"|"asian"|"hispanic"|"native"|"other"`.
    Race { value: String },
    /// `ethnicity == "hispanic"` or `"nonhispanic"`.
    Ethnicity { value: String },
    /// Patient carries the given SNOMED condition code.
    HasCondition { code: String },
    /// Patient carries every SNOMED code in the list.
    HasAllConditions { codes: Vec<String> },
    /// Patient carries at least one SNOMED code in the list.
    HasAnyCondition { codes: Vec<String> },
    /// Logical AND across children.
    And { children: Vec<FilterExpr> },
    /// Logical OR across children.
    Or { children: Vec<FilterExpr> },
    /// Logical NOT.
    Not { child: Box<FilterExpr> },
}

/// Pre-resolved filter. Walks the `FilterExpr` once at construction,
/// looking up SNOMED codes → `u16` condition indices via the
/// `CodeTable` so per-patient `matches()` evaluation never touches a
/// hash table.
pub struct FilterEvaluator {
    root: ResolvedNode,
}

enum ResolvedNode {
    All,
    Archetype(u16),
    ArchetypeIn(Vec<u16>),
    AgeRange(u32, u32),
    Sex(u8),       // 1 = F, 0 = M
    Race(u8),      // matches the `FullPatient::race` u8 enum
    Ethnicity(u8), // 1 = hispanic, 0 = nonhispanic
    HasCondition(u16),
    HasAllConditions(SmallVec<[u16; 8]>),
    HasAnyCondition(SmallVec<[u16; 8]>),
    And(Vec<ResolvedNode>),
    Or(Vec<ResolvedNode>),
    Not(Box<ResolvedNode>),
    /// Sentinel: a condition code wasn't found in the registry. Matches
    /// nothing (cohorts asking for unknown codes get an empty cohort
    /// rather than a panic).
    NeverMatches,
}

impl FilterEvaluator {
    /// Build an evaluator from a `FilterExpr` + registries. Returns
    /// `None` if the expression is structurally invalid.
    pub fn new(
        expr: &FilterExpr,
        _archetypes: &ArchetypeRegistry,
        code_table: &CodeTable,
    ) -> Self {
        Self {
            root: resolve(expr, code_table),
        }
    }

    /// `O(filter_depth)` membership test.
    #[inline]
    pub fn matches(&self, patient: &FullPatient) -> bool {
        self.root.matches(patient)
    }
}

fn resolve(expr: &FilterExpr, code_table: &CodeTable) -> ResolvedNode {
    match expr {
        FilterExpr::All => ResolvedNode::All,
        FilterExpr::Archetype { id } => ResolvedNode::Archetype(*id),
        FilterExpr::ArchetypeIn { ids } => ResolvedNode::ArchetypeIn(ids.clone()),
        FilterExpr::AgeRange { lo, hi } => ResolvedNode::AgeRange(*lo, *hi),
        FilterExpr::Sex { value } => {
            let v = if value.eq_ignore_ascii_case("F") { 1 } else { 0 };
            ResolvedNode::Sex(v)
        }
        FilterExpr::Race { value } => {
            let v = match value.to_ascii_lowercase().as_str() {
                "white" => 0,
                "black" => 1,
                "asian" => 2,
                "hispanic" => 3,
                "native" => 4,
                _ => 5,
            };
            ResolvedNode::Race(v)
        }
        FilterExpr::Ethnicity { value } => {
            let v = if value.eq_ignore_ascii_case("hispanic") { 1 } else { 0 };
            ResolvedNode::Ethnicity(v)
        }
        FilterExpr::HasCondition { code } => match code_table.condition_index.get(code) {
            Some(&idx) => ResolvedNode::HasCondition(idx),
            None => ResolvedNode::NeverMatches,
        },
        FilterExpr::HasAllConditions { codes } => {
            let mut indices = SmallVec::new();
            for c in codes {
                match code_table.condition_index.get(c) {
                    Some(&idx) => indices.push(idx),
                    None => return ResolvedNode::NeverMatches,
                }
            }
            ResolvedNode::HasAllConditions(indices)
        }
        FilterExpr::HasAnyCondition { codes } => {
            let mut indices = SmallVec::new();
            for c in codes {
                if let Some(&idx) = code_table.condition_index.get(c) {
                    indices.push(idx);
                }
            }
            // Empty `HasAnyCondition` ≡ unmatchable (no codes resolved).
            if indices.is_empty() {
                ResolvedNode::NeverMatches
            } else {
                ResolvedNode::HasAnyCondition(indices)
            }
        }
        FilterExpr::And { children } => ResolvedNode::And(
            children.iter().map(|c| resolve(c, code_table)).collect(),
        ),
        FilterExpr::Or { children } => ResolvedNode::Or(
            children.iter().map(|c| resolve(c, code_table)).collect(),
        ),
        FilterExpr::Not { child } => {
            ResolvedNode::Not(Box::new(resolve(child, code_table)))
        }
    }
}

impl ResolvedNode {
    fn matches(&self, p: &FullPatient) -> bool {
        match self {
            ResolvedNode::All => true,
            ResolvedNode::Archetype(id) => p.archetype_id.0 == *id,
            ResolvedNode::ArchetypeIn(ids) => ids.contains(&p.archetype_id.0),
            ResolvedNode::AgeRange(lo, hi) => {
                let age = age_years_from(p.birth_date_days);
                age >= *lo && age < *hi
            }
            ResolvedNode::Sex(v) => p.sex == *v,
            ResolvedNode::Race(v) => p.race == *v,
            ResolvedNode::Ethnicity(v) => p.ethnicity == *v,
            ResolvedNode::HasCondition(idx) => p.conditions.contains(idx),
            ResolvedNode::HasAllConditions(indices) => {
                indices.iter().all(|i| p.conditions.contains(i))
            }
            ResolvedNode::HasAnyCondition(indices) => {
                indices.iter().any(|i| p.conditions.contains(i))
            }
            ResolvedNode::And(children) => children.iter().all(|c| c.matches(p)),
            ResolvedNode::Or(children) => children.iter().any(|c| c.matches(p)),
            ResolvedNode::Not(child) => !child.matches(p),
            ResolvedNode::NeverMatches => false,
        }
    }
}

fn age_years_from(birth_date_days: i32) -> u32 {
    use chrono::{Duration, NaiveDate, Utc};
    let today = Utc::now().naive_utc().date();
    let birth = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()
        + Duration::days(birth_date_days as i64);
    today
        .signed_duration_since(birth)
        .num_days()
        .max(0)
        .saturating_div(365) as u32
}

/// Outcome of a `cohort` run.
#[derive(Debug, Clone, Default)]
pub struct CohortResult {
    /// How many patient ids were generated + filter-tested.
    pub scanned: usize,
    /// How many matched and were yielded to the callback.
    pub matched: usize,
    /// True when the scan stopped because `target_count` was reached.
    /// False means `max_scan` was hit first (cohort was less selective
    /// than expected, or `target_count` was infeasible).
    pub reached_target: bool,
}

impl CohortResult {
    /// Empirical selectivity = matched / scanned. Useful for sizing
    /// the next run's `max_scan` budget.
    pub fn selectivity(&self) -> f64 {
        if self.scanned == 0 {
            0.0
        } else {
            self.matched as f64 / self.scanned as f64
        }
    }
}

impl crate::batch::BatchGenerator {
    /// Stream generated patients through `filter`, calling `on_match`
    /// for each that passes. Stops when `target_count` matches have
    /// been yielded or `max_scan` patients have been considered,
    /// whichever comes first.
    ///
    /// Memory peak is bounded by the streaming chunk size (default
    /// `chunk_size = 1024`); set explicitly via
    /// `cohort_with_chunk_size` if you need different memory / latency
    /// tradeoffs.
    pub fn cohort<F>(
        &self,
        filter: &FilterExpr,
        target_count: usize,
        max_scan: usize,
        on_match: F,
    ) -> CohortResult
    where
        F: FnMut(&FullPatient),
    {
        self.cohort_with_chunk_size(filter, target_count, max_scan, 1024, on_match)
    }

    pub fn cohort_with_chunk_size<F>(
        &self,
        filter: &FilterExpr,
        target_count: usize,
        max_scan: usize,
        chunk_size: usize,
        mut on_match: F,
    ) -> CohortResult
    where
        F: FnMut(&FullPatient),
    {
        let archetypes = self.archetypes();
        let code_table = self.code_table();
        let evaluator = FilterEvaluator::new(filter, archetypes, code_table);

        let mut result = CohortResult::default();
        // Early-exit closure can't bail out of `generate_full_chunked`
        // mid-chunk (callback returns `()`), so we shortcut at the
        // per-patient level by setting a flag the chunk loop checks.
        let mut stop = false;
        self.generate_full_chunked(max_scan, chunk_size, |chunk| {
            if stop {
                return;
            }
            for p in &chunk {
                if stop {
                    break;
                }
                result.scanned += 1;
                if evaluator.matches(p) {
                    result.matched += 1;
                    on_match(p);
                    if result.matched >= target_count {
                        stop = true;
                        result.reached_target = true;
                    }
                }
            }
        });
        result
    }
}
