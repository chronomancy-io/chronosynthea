//! SIMD-accelerated batch sampling for conditions and events.
//!
//! WASP Role: {F_q} — Local filter implementing `rand < threshold[i]` per condition.
//! CDE Phase: Phase 4 (Local Filtering) — converts index probes to exact patient records.
//!
//! This module provides vectorized sampling that processes multiple
//! patients or conditions simultaneously using SIMD instructions.
//!
//! WASP Guarantees:
//! - Exactness: SIMD comparison produces exact condition assignments (no false positives)
//! - Bounded Work: O(c/8) per patient where c = conditions (f32x8 vectorization)

use rand::Rng;
use smallvec::SmallVec;
use wide::{f32x8, CmpLt};

use crate::archetype::ArchetypeRegistry;

/// Fixed-size bitset for O(1) deduplication of event indices.
/// Supports up to 512 items (8 u64 words × 64 bits).
#[derive(Clone)]
pub struct EventBitset {
    words: [u64; 8],
}

impl EventBitset {
    /// Creates a new empty bitset.
    #[inline]
    pub const fn new() -> Self {
        Self { words: [0; 8] }
    }

    /// Clears all bits.
    #[inline]
    pub fn clear(&mut self) {
        self.words = [0; 8];
    }

    /// Sets a bit and returns true if it was already set.
    #[inline]
    pub fn test_and_set(&mut self, idx: u16) -> bool {
        let word_idx = (idx >> 6) as usize; // idx / 64
        let bit_idx = idx & 63; // idx % 64

        if word_idx >= 8 {
            return false; // Out of range, treat as not set
        }

        let mask = 1u64 << bit_idx;
        let was_set = (self.words[word_idx] & mask) != 0;
        self.words[word_idx] |= mask;
        was_set
    }

    /// Tests if a bit is set.
    #[inline]
    pub fn test(&self, idx: u16) -> bool {
        let word_idx = (idx >> 6) as usize;
        let bit_idx = idx & 63;

        if word_idx >= 8 {
            return false;
        }

        (self.words[word_idx] & (1u64 << bit_idx)) != 0
    }

    /// Returns the number of set bits.
    pub fn count(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }
}

impl Default for EventBitset {
    fn default() -> Self {
        Self::new()
    }
}

/// SIMD sampler for batch condition sampling.
pub struct SimdSampler {
    /// Pre-allocated random buffer (8 floats for SIMD).
    rand_buffer: [f32; 8],

    /// Number of conditions.
    num_conditions: usize,

    /// Threshold stride (padded to multiple of 8).
    threshold_stride: usize,
}

impl SimdSampler {
    /// Creates a new SIMD sampler.
    pub fn new(num_conditions: usize) -> Self {
        Self {
            rand_buffer: [0.0; 8],
            num_conditions,
            threshold_stride: (num_conditions + 7) & !7,
        }
    }

    /// Creates from an archetype registry.
    pub fn from_registry(registry: &ArchetypeRegistry) -> Self {
        Self::new(registry.num_conditions())
    }

    /// Samples conditions for a single patient using SIMD comparisons.
    ///
    /// This is faster than scalar sampling when there are many conditions (>16).
    #[inline]
    pub fn sample_conditions<R: Rng>(
        &mut self,
        thresholds: &[f32],
        rng: &mut R,
        output: &mut SmallVec<[u16; 8]>,
    ) {
        output.clear();

        // Process 8 conditions at a time
        let chunks = thresholds.len() / 8;

        for chunk in 0..chunks {
            // Generate 8 random numbers
            for i in 0..8 {
                self.rand_buffer[i] = rng.gen();
            }

            let base = chunk * 8;
            let thresh = f32x8::from(&thresholds[base..base + 8]);
            let rands = f32x8::new(self.rand_buffer);

            // Compare: rand < threshold
            let mask = rands.cmp_lt(thresh);

            // Extract mask bits
            let mask_bits = mask.move_mask();

            // Process set bits
            if mask_bits != 0 {
                for bit in 0..8 {
                    if (mask_bits & (1 << bit)) != 0 {
                        // Also check threshold > 0 to skip inactive conditions
                        if thresholds[base + bit] > 0.0 {
                            output.push((base + bit) as u16);
                        }
                    }
                }
            }
        }

        // Handle remaining conditions (scalar)
        let remainder_start = chunks * 8;
        for i in remainder_start..thresholds.len() {
            if thresholds[i] > 0.0 && rng.gen::<f32>() < thresholds[i] {
                output.push(i as u16);
            }
        }
    }

    /// Samples conditions for a batch of patients.
    ///
    /// This is more efficient than calling sample_conditions repeatedly
    /// because it generates random numbers in larger batches.
    pub fn sample_conditions_batch<R: Rng>(
        &mut self,
        registry: &ArchetypeRegistry,
        archetype_ids: &[u16],
        rng: &mut R,
        outputs: &mut [SmallVec<[u16; 8]>],
    ) {
        debug_assert_eq!(archetype_ids.len(), outputs.len());

        for (i, &arch_id) in archetype_ids.iter().enumerate() {
            let thresholds = registry.condition_thresholds(arch_id);
            self.sample_conditions(thresholds, rng, &mut outputs[i]);
        }
    }

    /// Samples observations using SIMD (for per-encounter sampling).
    #[inline]
    pub fn sample_observations<R: Rng>(
        &mut self,
        frequencies: &[f32],
        rng: &mut R,
        output: &mut SmallVec<[u16; 16]>,
    ) {
        output.clear();

        let chunks = frequencies.len() / 8;

        for chunk in 0..chunks {
            for i in 0..8 {
                self.rand_buffer[i] = rng.gen();
            }

            let base = chunk * 8;
            let freq = f32x8::from(&frequencies[base..base + 8]);
            let rands = f32x8::new(self.rand_buffer);

            let mask = rands.cmp_lt(freq);
            let mask_bits = mask.move_mask();

            if mask_bits != 0 {
                for bit in 0..8 {
                    if (mask_bits & (1 << bit)) != 0 && frequencies[base + bit] > 0.0 {
                        output.push((base + bit) as u16);
                    }
                }
            }
        }

        // Handle remainder
        let remainder_start = chunks * 8;
        for i in remainder_start..frequencies.len() {
            if frequencies[i] > 0.0 && rng.gen::<f32>() < frequencies[i] {
                output.push(i as u16);
            }
        }
    }
}

/// Batch sampler that pre-generates random numbers for multiple patients.
pub struct BatchRandomSource {
    /// Pre-generated random floats.
    buffer: Vec<f32>,
    /// Current position in buffer.
    position: usize,
    /// Buffer size.
    capacity: usize,
}

impl BatchRandomSource {
    /// Creates a new batch random source.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0.0; capacity],
            position: 0,
            capacity,
        }
    }

    /// Refills the buffer with random numbers.
    pub fn refill<R: Rng>(&mut self, rng: &mut R) {
        for x in &mut self.buffer {
            *x = rng.gen();
        }
        self.position = 0;
    }

    /// Gets the next random float.
    #[inline]
    pub fn next(&mut self) -> f32 {
        if self.position >= self.capacity {
            // Wrap around (caller should refill before this happens)
            self.position = 0;
        }
        let val = self.buffer[self.position];
        self.position += 1;
        val
    }

    /// Gets a slice of the next n random floats.
    #[inline]
    pub fn next_slice(&mut self, n: usize) -> &[f32] {
        let start = self.position;
        self.position += n;
        if self.position > self.capacity {
            self.position = self.capacity;
        }
        &self.buffer[start..self.position]
    }

    /// Returns remaining random numbers.
    pub fn remaining(&self) -> usize {
        self.capacity - self.position
    }
}

/// Vectorized threshold comparison for batch processing.
#[inline]
pub fn compare_threshold_batch(thresholds: &[f32], randoms: &[f32], output: &mut Vec<u16>) {
    output.clear();
    debug_assert_eq!(thresholds.len(), randoms.len());

    let len = thresholds.len();
    let chunks = len / 8;

    for chunk in 0..chunks {
        let base = chunk * 8;
        let thresh = f32x8::from(&thresholds[base..base + 8]);
        let rands = f32x8::from(&randoms[base..base + 8]);

        let mask = rands.cmp_lt(thresh);
        let mask_bits = mask.move_mask();

        if mask_bits != 0 {
            for bit in 0..8 {
                if (mask_bits & (1 << bit)) != 0 && thresholds[base + bit] > 0.0 {
                    output.push((base + bit) as u16);
                }
            }
        }
    }

    // Remainder
    for i in (chunks * 8)..len {
        if thresholds[i] > 0.0 && randoms[i] < thresholds[i] {
            output.push(i as u16);
        }
    }
}

/// Scalar fallback for threshold comparison.
#[inline]
pub fn compare_threshold_scalar<R: Rng>(
    thresholds: &[f32],
    rng: &mut R,
    output: &mut SmallVec<[u16; 8]>,
) {
    output.clear();
    for (i, &threshold) in thresholds.iter().enumerate() {
        if threshold > 0.0 && rng.gen::<f32>() < threshold {
            output.push(i as u16);
        }
    }
}

/// Event sampler for medications, observations, and procedures.
///
/// This sampler generates events based on:
/// - Condition-linked medications and procedures
/// - Per-encounter observations (vitals, labs)
pub struct EventSampler {
    /// Medication indices sampled for current patient.
    medication_buffer: SmallVec<[u16; 16]>,
    /// Observation indices sampled for current encounter.
    observation_buffer: SmallVec<[u16; 16]>,
    /// Procedure indices sampled for current patient.
    procedure_buffer: SmallVec<[u16; 16]>,
    /// Scratch buffer for SIMD sampling.
    rand_buffer: [f32; 8],
    /// Bitset for O(1) observation deduplication.
    observation_seen: EventBitset,
    /// Bitset for O(1) procedure deduplication.
    procedure_seen: EventBitset,
}

impl EventSampler {
    /// Creates a new event sampler.
    pub fn new() -> Self {
        Self {
            medication_buffer: SmallVec::new(),
            observation_buffer: SmallVec::new(),
            procedure_buffer: SmallVec::new(),
            rand_buffer: [0.0; 8],
            observation_seen: EventBitset::new(),
            procedure_seen: EventBitset::new(),
        }
    }

    /// Resets the sampler for a new patient.
    /// Call this before sampling across multiple encounters for the same patient.
    #[inline]
    pub fn reset_patient(&mut self) {
        self.observation_buffer.clear();
        self.procedure_buffer.clear();
        self.observation_seen.clear();
        self.procedure_seen.clear();
    }

    /// Samples observations for an encounter and accumulates into patient-level buffer.
    /// Uses O(1) bitset for deduplication instead of O(n) contains().
    #[inline]
    pub fn sample_observations_accumulate<R: Rng>(&mut self, frequencies: &[f32], rng: &mut R) {
        let chunks = frequencies.len() / 8;

        for chunk in 0..chunks {
            for i in 0..8 {
                self.rand_buffer[i] = rng.gen();
            }

            let base = chunk * 8;
            let freq = f32x8::from(&frequencies[base..base + 8]);
            let rands = f32x8::new(self.rand_buffer);

            let mask = rands.cmp_lt(freq);
            let mask_bits = mask.move_mask();

            if mask_bits != 0 {
                for bit in 0..8 {
                    if (mask_bits & (1 << bit)) != 0 && frequencies[base + bit] > 0.0 {
                        let idx = (base + bit) as u16;
                        // O(1) deduplication with bitset
                        if !self.observation_seen.test_and_set(idx) {
                            self.observation_buffer.push(idx);
                        }
                    }
                }
            }
        }

        // Handle remainder
        let remainder_start = chunks * 8;
        for i in remainder_start..frequencies.len() {
            if frequencies[i] > 0.0 && rng.gen::<f32>() < frequencies[i] {
                let idx = i as u16;
                if !self.observation_seen.test_and_set(idx) {
                    self.observation_buffer.push(idx);
                }
            }
        }
    }

    /// Samples procedures for an encounter and accumulates into patient-level buffer.
    /// Uses O(1) bitset for deduplication instead of O(n) contains().
    #[inline]
    pub fn sample_procedures_accumulate<R: Rng>(&mut self, frequencies: &[f32], rng: &mut R) {
        let chunks = frequencies.len() / 8;

        for chunk in 0..chunks {
            for i in 0..8 {
                self.rand_buffer[i] = rng.gen();
            }

            let base = chunk * 8;
            let freq = f32x8::from(&frequencies[base..base + 8]);
            let rands = f32x8::new(self.rand_buffer);

            let mask = rands.cmp_lt(freq);
            let mask_bits = mask.move_mask();

            if mask_bits != 0 {
                for bit in 0..8 {
                    if (mask_bits & (1 << bit)) != 0 && frequencies[base + bit] > 0.0 {
                        let idx = (base + bit) as u16;
                        // O(1) deduplication with bitset
                        if !self.procedure_seen.test_and_set(idx) {
                            self.procedure_buffer.push(idx);
                        }
                    }
                }
            }
        }

        // Handle remainder
        let remainder_start = chunks * 8;
        for i in remainder_start..frequencies.len() {
            if frequencies[i] > 0.0 && rng.gen::<f32>() < frequencies[i] {
                let idx = i as u16;
                if !self.procedure_seen.test_and_set(idx) {
                    self.procedure_buffer.push(idx);
                }
            }
        }
    }

    /// Returns the accumulated observations buffer.
    pub fn accumulated_observations(&self) -> &[u16] {
        &self.observation_buffer
    }

    /// Returns the accumulated procedures buffer.
    pub fn accumulated_procedures(&self) -> &[u16] {
        &self.procedure_buffer
    }

    /// Samples medications directly from archetype thresholds using SIMD.
    /// This is faster than per-condition sampling because thresholds are pre-computed.
    #[inline]
    pub fn sample_medications_simd<R: Rng>(&mut self, thresholds: &[f32], rng: &mut R) -> &[u16] {
        self.medication_buffer.clear();

        let chunks = thresholds.len() / 8;

        for chunk in 0..chunks {
            for i in 0..8 {
                self.rand_buffer[i] = rng.gen();
            }

            let base = chunk * 8;
            let thresh = f32x8::from(&thresholds[base..base + 8]);
            let rands = f32x8::new(self.rand_buffer);

            let mask = rands.cmp_lt(thresh);
            let mask_bits = mask.move_mask();

            if mask_bits != 0 {
                for bit in 0..8 {
                    if (mask_bits & (1 << bit)) != 0 && thresholds[base + bit] > 0.0 {
                        self.medication_buffer.push((base + bit) as u16);
                    }
                }
            }
        }

        // Handle remainder
        let remainder_start = chunks * 8;
        for i in remainder_start..thresholds.len() {
            if thresholds[i] > 0.0 && rng.gen::<f32>() < thresholds[i] {
                self.medication_buffer.push(i as u16);
            }
        }

        &self.medication_buffer
    }

    /// Samples observations and procedures for multiple encounters in a batch.
    /// Uses probability scaling: P(sampled at least once in N encounters) ≈ min(p * N, 1.0)
    #[inline]
    pub fn sample_events_batch<R: Rng>(
        &mut self,
        obs_frequencies: &[f32],
        proc_frequencies: &[f32],
        num_encounters: u32,
        rng: &mut R,
    ) {
        self.observation_buffer.clear();
        self.procedure_buffer.clear();

        let n = num_encounters as f32;
        let n_vec = f32x8::splat(n);
        let one_vec = f32x8::splat(1.0);

        // Sample observations with SIMD scaling
        let obs_chunks = obs_frequencies.len() / 8;
        for chunk in 0..obs_chunks {
            for i in 0..8 {
                self.rand_buffer[i] = rng.gen();
            }

            let base = chunk * 8;
            let freq = f32x8::from(&obs_frequencies[base..base + 8]);
            let scaled = (freq * n_vec).min(one_vec);
            let rands = f32x8::new(self.rand_buffer);

            let mask = rands.cmp_lt(scaled);
            let mask_bits = mask.move_mask();

            if mask_bits != 0 {
                for bit in 0..8 {
                    if (mask_bits & (1 << bit)) != 0 {
                        self.observation_buffer.push((base + bit) as u16);
                    }
                }
            }
        }

        // Handle observation remainder
        let obs_remainder = obs_chunks * 8;
        for i in obs_remainder..obs_frequencies.len() {
            let scaled_p = (obs_frequencies[i] * n).min(1.0);
            if rng.gen::<f32>() < scaled_p {
                self.observation_buffer.push(i as u16);
            }
        }

        // Sample procedures with SIMD scaling
        let proc_chunks = proc_frequencies.len() / 8;
        for chunk in 0..proc_chunks {
            for i in 0..8 {
                self.rand_buffer[i] = rng.gen();
            }

            let base = chunk * 8;
            let freq = f32x8::from(&proc_frequencies[base..base + 8]);
            let scaled = (freq * n_vec).min(one_vec);
            let rands = f32x8::new(self.rand_buffer);

            let mask = rands.cmp_lt(scaled);
            let mask_bits = mask.move_mask();

            if mask_bits != 0 {
                for bit in 0..8 {
                    if (mask_bits & (1 << bit)) != 0 {
                        self.procedure_buffer.push((base + bit) as u16);
                    }
                }
            }
        }

        // Handle procedure remainder
        let proc_remainder = proc_chunks * 8;
        for i in proc_remainder..proc_frequencies.len() {
            let scaled_p = (proc_frequencies[i] * n).min(1.0);
            if rng.gen::<f32>() < scaled_p {
                self.procedure_buffer.push(i as u16);
            }
        }
    }

    /// Samples medications for a patient based on their conditions.
    ///
    /// For each condition the patient has, we sample from the medications
    /// linked to that condition using their indication frequencies.
    #[inline]
    pub fn sample_medications_for_conditions<R: Rng>(
        &mut self,
        conditions: &[u16],
        registry: &crate::archetype::ArchetypeRegistry,
        rng: &mut R,
    ) -> &[u16] {
        self.medication_buffer.clear();

        for &cond_idx in conditions {
            let meds = registry.medications_for_condition(cond_idx);
            for &(med_idx, frequency) in meds {
                // Sample based on frequency (probability per patient with this condition)
                if rng.gen::<f32>() < frequency {
                    // Avoid duplicates
                    if !self.medication_buffer.contains(&med_idx) {
                        self.medication_buffer.push(med_idx);
                    }
                }
            }
        }

        &self.medication_buffer
    }

    /// Samples procedures for a patient based on their conditions.
    /// Falls back to frequency-based sampling if no indications are defined.
    #[inline]
    pub fn sample_procedures_for_conditions<R: Rng>(
        &mut self,
        conditions: &[u16],
        registry: &crate::archetype::ArchetypeRegistry,
        rng: &mut R,
    ) -> &[u16] {
        self.procedure_buffer.clear();

        // First, try indication-based sampling
        for &cond_idx in conditions {
            let procs = registry.procedures_for_condition(cond_idx);
            for &(proc_idx, frequency) in procs {
                if rng.gen::<f32>() < frequency {
                    if !self.procedure_buffer.contains(&proc_idx) {
                        self.procedure_buffer.push(proc_idx);
                    }
                }
            }
        }

        &self.procedure_buffer
    }

    /// Samples procedures for an encounter using per-encounter frequencies.
    /// This is the fallback when procedures don't have indication codes.
    #[inline]
    pub fn sample_procedures_for_encounter<R: Rng>(
        &mut self,
        frequencies: &[f32],
        rng: &mut R,
    ) -> &[u16] {
        self.procedure_buffer.clear();

        let chunks = frequencies.len() / 8;

        for chunk in 0..chunks {
            for i in 0..8 {
                self.rand_buffer[i] = rng.gen();
            }

            let base = chunk * 8;
            let freq = f32x8::from(&frequencies[base..base + 8]);
            let rands = f32x8::new(self.rand_buffer);

            let mask = rands.cmp_lt(freq);
            let mask_bits = mask.move_mask();

            if mask_bits != 0 {
                for bit in 0..8 {
                    if (mask_bits & (1 << bit)) != 0 && frequencies[base + bit] > 0.0 {
                        self.procedure_buffer.push((base + bit) as u16);
                    }
                }
            }
        }

        // Handle remainder
        let remainder_start = chunks * 8;
        for i in remainder_start..frequencies.len() {
            if frequencies[i] > 0.0 && rng.gen::<f32>() < frequencies[i] {
                self.procedure_buffer.push(i as u16);
            }
        }

        &self.procedure_buffer
    }

    /// Samples observations for a single encounter using SIMD.
    ///
    /// Observations are sampled based on per-encounter frequency,
    /// not linked to specific conditions.
    #[inline]
    pub fn sample_observations_for_encounter<R: Rng>(
        &mut self,
        frequencies: &[f32],
        rng: &mut R,
    ) -> &[u16] {
        self.observation_buffer.clear();

        let chunks = frequencies.len() / 8;

        for chunk in 0..chunks {
            for i in 0..8 {
                self.rand_buffer[i] = rng.gen();
            }

            let base = chunk * 8;
            let freq = f32x8::from(&frequencies[base..base + 8]);
            let rands = f32x8::new(self.rand_buffer);

            let mask = rands.cmp_lt(freq);
            let mask_bits = mask.move_mask();

            if mask_bits != 0 {
                for bit in 0..8 {
                    if (mask_bits & (1 << bit)) != 0 && frequencies[base + bit] > 0.0 {
                        self.observation_buffer.push((base + bit) as u16);
                    }
                }
            }
        }

        // Handle remainder
        let remainder_start = chunks * 8;
        for i in remainder_start..frequencies.len() {
            if frequencies[i] > 0.0 && rng.gen::<f32>() < frequencies[i] {
                self.observation_buffer.push(i as u16);
            }
        }

        &self.observation_buffer
    }

    /// Clears all internal buffers.
    pub fn clear(&mut self) {
        self.medication_buffer.clear();
        self.observation_buffer.clear();
        self.procedure_buffer.clear();
    }

    /// Returns the sampled medications.
    pub fn medications(&self) -> &[u16] {
        &self.medication_buffer
    }

    /// Returns the sampled observations.
    pub fn observations(&self) -> &[u16] {
        &self.observation_buffer
    }

    /// Returns the sampled procedures.
    pub fn procedures(&self) -> &[u16] {
        &self.procedure_buffer
    }
}

impl Default for EventSampler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_simd_sampler() {
        let mut sampler = SimdSampler::new(32);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut output = SmallVec::new();

        // Thresholds: all zeros except a few
        let mut thresholds = vec![0.0f32; 32];
        thresholds[0] = 1.0; // Always
        thresholds[5] = 1.0; // Always
        thresholds[16] = 1.0; // Always
        thresholds[31] = 0.5; // Sometimes

        sampler.sample_conditions(&thresholds, &mut rng, &mut output);

        // Should always include 0, 5, 16
        assert!(output.contains(&0));
        assert!(output.contains(&5));
        assert!(output.contains(&16));
    }

    #[test]
    fn test_batch_random_source() {
        let mut source = BatchRandomSource::new(100);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

        source.refill(&mut rng);

        // Should get different values
        let v1 = source.next();
        let v2 = source.next();
        assert_ne!(v1, v2);

        // Remaining should decrease
        assert_eq!(source.remaining(), 98);
    }

    #[test]
    fn test_compare_threshold_batch() {
        let thresholds = vec![1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0];
        let randoms = vec![0.5, 0.5, 0.5, 0.3, 0.5, 0.5, 0.5, 0.5];
        let mut output = Vec::new();

        compare_threshold_batch(&thresholds, &randoms, &mut output);

        // Should include 0, 2, 3 (where random < threshold and threshold > 0)
        assert!(output.contains(&0));
        assert!(output.contains(&2));
        assert!(output.contains(&3));
        assert!(!output.contains(&1));
    }
}
