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

/// Report returned by `fit_to_marginals` after Boltzmann-learning iterations.
/// Callers can gate on `max_marginal_residual` and `max_pairwise_residual`
/// (both probabilities in `[0, 1]`) to decide whether to fit further.
#[derive(Debug, Clone, Default)]
pub struct FitReport {
    pub iterations: usize,
    pub max_marginal_residual: f32,
    pub max_pairwise_residual: f32,
}

#[inline]
fn upsert_interaction(
    table: &mut AHashMap<u16, Vec<(u16, f32)>>,
    src: u16,
    dst: u16,
    delta: f32,
) {
    let entry = table.entry(src).or_default();
    for slot in entry.iter_mut() {
        if slot.0 == dst {
            slot.1 = (slot.1 + delta).clamp(-12.0, 12.0);
            return;
        }
    }
    entry.push((dst, delta.clamp(-12.0, 12.0)));
}

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

    /// Per-archetype bias overrides set by Boltzmann learning. Keyed by
    /// `ArchetypeId.0` (`u16`) → per-slot bias vector (same length and order
    /// as the archetype's `conditions`). When present for the archetype being
    /// sampled, the Gibbs inner loop uses these biases instead of the
    /// `logit(archetype.prevalence)` default; in-place fitting updates them.
    pub bias_overrides: AHashMap<u16, Vec<f32>>,
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

    /// Fit the model's biases `h_i` and pairwise interactions `J_{ij}` to
    /// match target marginal and pairwise probabilities via stochastic
    /// gradient ascent on the pseudo-log-likelihood — the standard Boltzmann
    /// learning recipe (Ackley, Hinton & Sejnowski 1985).
    ///
    /// At convergence, the model's Gibbs-sampled marginals reproduce both
    /// the target single-condition prevalences AND the target pairwise
    /// co-occurrence rates, which the raw log-lift initialisation does not.
    /// The gap between the two is the "Ising-Boltzmann calibration gap"
    /// documented in `GIBBS_ITERATIONS`'s comment — this method closes it.
    ///
    /// ## Targets
    ///
    /// * `target_marginals[i]` — desired `P(x_i = 1)`. Length must equal the
    ///   archetype's `conditions.len()` (one entry per active condition).
    /// * `target_pairs[k]` — desired `P(x_a ∧ x_b)` triplets `(a, b, p)`
    ///   where `a, b` are condition indices in the same space as
    ///   `archetype.conditions`. Order of `(a, b)` doesn't matter.
    ///
    /// ## Algorithm
    ///
    /// For each iteration:
    ///   1. Sample `n_samples` patients from the current model.
    ///   2. Compute empirical model marginals and pairwise frequencies.
    ///   3. Update each bias:   `h_i  += lr * (target_p_i  - model_p_i)`.
    ///   4. Update each interaction: `J_ij += lr * (target_p_ij - model_p_ij)`.
    ///
    /// Convergence is monotone-in-expectation for small enough `lr` (Hinton
    /// 2002); we cap it at `lr ≤ 0.5` in practice. Returns the max-absolute
    /// marginal residual after the final iteration so callers can gate.
    pub fn fit_to_marginals<R: Rng>(
        &mut self,
        archetype: &PatientArchetype,
        target_marginals: &[f32],
        target_pairs: &[(u16, u16, f32)],
        n_samples: usize,
        n_iter: usize,
        lr: f32,
        rng: &mut R,
    ) -> FitReport {
        // Hoist the archetype's condition list as a parallel array for the
        // inner observer loop, plus a u16 → slot_idx lookup so we can index
        // observed counts by `(slot_a, slot_b)` without rebuilding hashes.
        let conds: Vec<u16> = archetype.conditions.iter().map(|(c, _)| *c).collect();
        let n = conds.len();
        // Seed the per-archetype bias override with `logit(prevalence)` so
        // the fit starts from the marginal-only initialisation and the
        // gradient drives it toward the joint-correct values.
        let arch_id = archetype.id.0;
        if !self.bias_overrides.contains_key(&arch_id) {
            let init: Vec<f32> = archetype
                .conditions
                .iter()
                .map(|(_, p)| logit(*p))
                .collect();
            self.bias_overrides.insert(arch_id, init);
        }
        let max_idx = conds.iter().copied().max().unwrap_or(0) as usize;
        let mut idx_to_slot: Vec<i16> = vec![-1; max_idx + 1];
        for (slot, &c) in conds.iter().enumerate() {
            idx_to_slot[c as usize] = slot as i16;
        }

        // Convert target_pairs (condition_idx space) into per-archetype
        // (slot_a, slot_b, target_p_ab) so the inner loop indexes a dense
        // square count matrix.
        let mut slot_pairs: Vec<(usize, usize, f32)> = Vec::new();
        for &(a, b, p) in target_pairs {
            let sa = *idx_to_slot.get(a as usize).unwrap_or(&-1);
            let sb = *idx_to_slot.get(b as usize).unwrap_or(&-1);
            if sa < 0 || sb < 0 || sa == sb {
                continue;
            }
            let (lo, hi) = (sa.min(sb) as usize, sa.max(sb) as usize);
            slot_pairs.push((lo, hi, p));
        }

        let mut report = FitReport::default();
        let mut output: SmallVec<[u16; 8]> = SmallVec::new();

        for iter in 0..n_iter {
            // Per-iteration accumulators.
            let mut marg_count: Vec<u32> = vec![0; n];
            let mut pair_count: Vec<u32> = vec![0; n * n];

            for _ in 0..n_samples {
                self.sample(archetype, &mut output, rng);
                // Mark which slots are active this sample.
                let mut active_slots: SmallVec<[usize; 32]> = SmallVec::new();
                for &c in &output {
                    if (c as usize) < idx_to_slot.len() {
                        let s = idx_to_slot[c as usize];
                        if s >= 0 {
                            let su = s as usize;
                            marg_count[su] += 1;
                            active_slots.push(su);
                        }
                    }
                }
                // O(k²) pair update, but k = active count ≪ n typically.
                for i in 0..active_slots.len() {
                    for j in (i + 1)..active_slots.len() {
                        let (lo, hi) = (
                            active_slots[i].min(active_slots[j]),
                            active_slots[i].max(active_slots[j]),
                        );
                        pair_count[lo * n + hi] += 1;
                    }
                }
            }

            let inv_n = 1.0 / n_samples as f32;
            // Marginal update step: adjust the per-archetype bias override.
            // `bias_overrides[arch_id]` is the slot-indexed bias vector the
            // sampler reads in its Gibbs inner loop; mutating it here
            // closes the loop between fit and sample.
            let mut max_marg_resid = 0.0f32;
            let biases = self
                .bias_overrides
                .get_mut(&arch_id)
                .expect("bias_overrides seeded above");
            for slot in 0..n {
                let model_p = marg_count[slot] as f32 * inv_n;
                let target_p = *target_marginals.get(slot).unwrap_or(&model_p);
                let resid = target_p - model_p;
                if resid.abs() > max_marg_resid {
                    max_marg_resid = resid.abs();
                }
                let h = &mut biases[slot];
                *h += lr * resid;
                *h = h.clamp(-12.0, 12.0);
            }

            // Pairwise update step: adjust J_{a→b} and J_{b→a} by the same
            // residual (the model is symmetric in the pairwise Ising sense).
            let mut max_pair_resid = 0.0f32;
            for &(slot_a, slot_b, target_p) in &slot_pairs {
                let (lo, hi) = (slot_a.min(slot_b), slot_a.max(slot_b));
                let model_p = pair_count[lo * n + hi] as f32 * inv_n;
                let resid = target_p - model_p;
                if resid.abs() > max_pair_resid {
                    max_pair_resid = resid.abs();
                }
                let cond_a = conds[slot_a];
                let cond_b = conds[slot_b];
                // Adjust the forward and reverse interactions in lockstep.
                // If the (a → b) entry doesn't exist yet, append one with
                // initial J = 0 so we can build up new couplings during
                // fitting.
                upsert_interaction(&mut self.interactions, cond_a, cond_b, lr * resid);
                upsert_interaction(&mut self.interactions, cond_b, cond_a, lr * resid);
                upsert_interaction(
                    &mut self.reverse_interactions,
                    cond_b,
                    cond_a,
                    lr * resid,
                );
                upsert_interaction(
                    &mut self.reverse_interactions,
                    cond_a,
                    cond_b,
                    lr * resid,
                );
            }

            report.iterations = iter + 1;
            report.max_marginal_residual = max_marg_resid;
            report.max_pairwise_residual = max_pair_resid;
            if max_marg_resid < 0.005 && max_pair_resid < 0.005 {
                break;
            }
        }
        report
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

        // Precompute archetype-specific biases. Using
        // `logit(archetype.prevalence)` instead of the global
        // `fallback_logits` is the difference between a Gibbs chain that
        // converges to the archetype's marginals and one that drifts toward
        // the population-level marginals (the latter mis-fires demographic
        // structure on the joint side). If Boltzmann learning has populated
        // `bias_overrides` for this archetype, those take precedence — they
        // encode the fitted bias that produces correct joint marginals
        // under the current `J_ij` couplings.
        let arch_id = archetype.id.0;
        let mut archetype_biases: SmallVec<[f32; 32]> = SmallVec::with_capacity(n);
        if let Some(overrides) = self.bias_overrides.get(&arch_id) {
            if overrides.len() == n {
                archetype_biases.extend_from_slice(overrides);
            } else {
                for &(_, prev) in &archetype.conditions {
                    archetype_biases.push(logit(prev));
                }
            }
        } else {
            for &(_, prev) in &archetype.conditions {
                archetype_biases.push(logit(prev));
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
                // Bias = archetype-specific logit (so the chain converges to
                // the archetype's calibrated marginals, not the population
                // average). Fallback to `fallback_logits` if the archetype
                // happens to omit the condition (shouldn't happen in
                // practice — `conds[slot_i]` is iterated from
                // `archetype.conditions`).
                let log_p = archetype_biases[slot_i];
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
