//! d5 = `causal-DAG` — single-site Gibbs sampler over the binary condition
//! vector with Ising-model parameters extracted from the empirical pairwise
//! data.
//!
//! ## Parameters
//!
//! For each archetype, each condition `i` has a bias `h_i`:
//!
//! ```text
//! h_i = logit(p_i) = log(p_i / (1 - p_i))
//! ```
//!
//! For each pair `(i, j)` with empirical conditional `P(j | i)` and marginal
//! `P(j)`, the interaction parameter is the log-lift:
//!
//! ```text
//! J_{ij} = log( P(j | i) / P(j) )
//! ```
//!
//! Positive J ⇒ positive correlation, negative J ⇒ negative correlation,
//! zero J ⇒ independence.
//!
//! ## Algorithm
//!
//! For each patient:
//!   1. Initialise `x_1..x_n` independently from Bernoulli(p_i).
//!   2. For K Gibbs iterations (default 3):
//!        For each active condition i in the archetype:
//!          log_odds = h_i + Σ_{j: x_j = 1} J_{ji}
//!          x_i ~ Bernoulli(sigmoid(log_odds))
//!   3. Emit `{i : x_i = 1}`.
//!
//! Handles negative correlations correctly (multi-trigger subtraction
//! doesn't stack catastrophically — each condition resamples conditional on
//! the current full state, not via independent per-pair adjustments). Also
//! captures three-way+ joint structure through iterative dependence on the
//! rest of the vector.

use ahash::AHashMap;
use rand::Rng;
use smallvec::SmallVec;

use crate::archetype::PatientArchetype;
use crate::fingerprint::MssFingerprint;

/// Per-condition Ising-model parameters consumed by the Gibbs sampler.
#[derive(Debug, Clone, Default)]
pub struct CausalDagModel {
    /// Sparse interaction table: trigger_idx → list of (dependent_idx, J_{trigger→dep}).
    /// Both positive and negative log-lifts are stored.
    pub interactions: AHashMap<u16, Vec<(u16, f32)>>,

    /// REVERSE interaction table: dependent_idx → list of (trigger_idx, J_{trigger→dep}).
    /// Lets the inner Gibbs loop iterate just the ~5 triggers of condition `i`
    /// instead of scanning all ~30 active conditions for hits to `i`. This is
    /// the difference between O(N²) (1,270× slowdown) and O(N · k) (~30× of
    /// marginal-only).
    pub reverse_interactions: AHashMap<u16, Vec<(u16, f32)>>,

    /// Per-condition global logit (fallback bias for un-archetyped conditions).
    pub fallback_logits: Vec<f32>,
}

/// Number of Gibbs iterations per patient.
///
/// **Experimental**: this Gibbs sampler is wired and dispatching correctly
/// but the empirical joint it produces does not yet match the input
/// pairwise conditionals. The architectural reason: the J_ij parameters
/// derived directly from observed log-lifts do not generally specify an
/// Ising-Boltzmann distribution whose Gibbs-sampled marginals reproduce
/// the source empirical marginals. Fitting J via pseudo-likelihood or
/// Boltzmann learning is documented future work (see manifesto's "Open
/// future work"). For production joint-correlation modelling today, the
/// `JointMode::PairwiseEmpirical` (additive-boost + two-knob recalibration)
/// path is the validated mode.
pub const GIBBS_ITERATIONS: u8 = 3;

#[inline(always)]
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

#[inline(always)]
fn logit(p: f32) -> f32 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

impl CausalDagModel {
    /// Build the model from a fingerprint. Translates each
    /// `cooccurrence[(trigger, dependent)] = P(dep | trigger)` into the
    /// Ising J parameter via log-lift.
    pub fn from_fingerprint(fp: &MssFingerprint) -> Self {
        let mut model = Self::default();
        model.fallback_logits = fp
            .conditions
            .iter()
            .map(|c| logit(c.prevalence as f32))
            .collect();

        let code_to_idx: AHashMap<&str, u16> = fp
            .conditions
            .iter()
            .enumerate()
            .map(|(i, c)| (c.code.as_str(), i as u16))
            .collect();
        let marginal: Vec<f32> = fp
            .conditions
            .iter()
            .map(|c| c.prevalence as f32)
            .collect();

        for ((trigger_code, dep_code), &cond_prob) in &fp.cooccurrence {
            if let (Some(&t), Some(&d)) = (
                code_to_idx.get(trigger_code.as_str()),
                code_to_idx.get(dep_code.as_str()),
            ) {
                if let Some(&p_d) = marginal.get(d as usize) {
                    if p_d > 1e-6 {
                        let j = ((cond_prob as f32) / p_d).ln();
                        model.interactions.entry(t).or_default().push((d, j));
                        model.reverse_interactions.entry(d).or_default().push((t, j));
                    }
                }
            }
        }

        model
    }

    /// Returns whether the model has any interactions.
    pub fn is_empty(&self) -> bool {
        self.interactions.is_empty()
    }

    /// Total number of (trigger, dependent) interaction entries.
    pub fn len(&self) -> usize {
        self.interactions.values().map(|v| v.len()).sum()
    }

    /// Sample a patient's conditions via single-site Gibbs over the
    /// archetype's active conditions. Emits the indices of `x_i = 1` into
    /// `output`.
    ///
    /// Cost per patient: K × a × k where a = active conditions (~30),
    /// k = mean reverse-interactions per condition (~3–5). At K=3 that's
    /// ~450 ops + ~90 sigmoid evals per patient.
    #[inline(always)]
    pub fn sample<R: Rng>(
        &self,
        archetype: &PatientArchetype,
        output: &mut SmallVec<[u16; 8]>,
        rng: &mut R,
    ) {
        output.clear();

        // State as parallel arrays for cache-friendliness:
        //   conds[i]: condition idx (u16)
        //   bits[i]:  current sample bit (u8)
        // Plus an `EventBitset` (512-bit) for O(1) "is cond_j active?" lookup
        // during the Gibbs inner sum over reverse interactions.
        let n = archetype.conditions.len();
        let mut conds: SmallVec<[u16; 32]> = SmallVec::with_capacity(n);
        let mut bits: SmallVec<[u8; 32]> = SmallVec::with_capacity(n);
        let mut active = crate::sampler::EventBitset::default();

        for &(idx, prob) in &archetype.conditions {
            let bit = if rng.gen::<f32>() < prob { 1u8 } else { 0u8 };
            conds.push(idx);
            bits.push(bit);
            if bit == 1 {
                active.test_and_set(idx);
            }
        }

        for _ in 0..GIBBS_ITERATIONS {
            // Rebuild the active bitset from current bit state. `EventBitset`
            // has no clear-bit operation, so this is the cheap way to keep
            // `active.test()` truthful when a slot flips 1→0 within a sweep.
            active.clear();
            for i in 0..n {
                if bits[i] == 1 {
                    active.test_and_set(conds[i]);
                }
            }

            for slot_i in 0..n {
                let cond_i = conds[slot_i];
                // Bias = population-level logit (matches the level the J_ij
                // log-lifts are extracted at). Per-archetype calibration is
                // applied separately via the recalibration loop if needed.
                let log_p = self
                    .fallback_logits
                    .get(cond_i as usize)
                    .copied()
                    .unwrap_or(-3.0);
                let mut log_odds = log_p;

                // Iterate REVERSE interactions for cond_i: list of triggers j
                // and the J coefficient for j → i. O(k) where k ≈ 3–5 (sparse).
                if let Some(triggers) = self.reverse_interactions.get(&cond_i) {
                    for &(trigger_j, j_val) in triggers {
                        if active.test(trigger_j) {
                            log_odds += j_val;
                        }
                    }
                }

                let p = sigmoid(log_odds).clamp(0.0, 1.0);
                let prev_bit = bits[slot_i];
                let new_bit = if rng.gen::<f32>() < p { 1u8 } else { 0u8 };
                if new_bit != prev_bit {
                    bits[slot_i] = new_bit;
                    if new_bit == 1 {
                        active.test_and_set(cond_i);
                    } else {
                        active.clear_bit(cond_i);
                    }
                }
            }
        }

        for i in 0..n {
            if bits[i] == 1 {
                output.push(conds[i]);
            }
        }
    }
}
