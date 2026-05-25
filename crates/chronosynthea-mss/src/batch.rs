//! Batch generator for ultra-high-throughput patient generation.
//!
//! This module implements the parallel architecture with:
//! - Lock-free statistics collection using atomic counters
//! - Per-worker arenas for zero-allocation generation
//! - Batch processing for SIMD efficiency

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use chrono::NaiveDate;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use smallvec::SmallVec;

use crate::archetype::{ArchetypeRegistry, CooccurrenceModel, PatientArchetype};
use crate::arena::{CompactEvent, CompactPatient, FullEncounter, FullPatient};
use crate::causal_dag::CausalDagModel;
use crate::fingerprint::MssFingerprint;
use crate::sampler::{EventSampler, SimdSampler};
use crate::stats::StreamingStatistics;
use crate::tables::CodeTable;

/// Joint-structure sampling mode. d5 axis value in code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointMode {
    /// Independent per-condition Bernoulli draws against calibrated marginals.
    /// Default when no cooccurrence data is loaded.
    MarginalOnly,
    /// Additive boost from empirical conditional probabilities with two-knob
    /// recalibration. Activated when `cooccurrence_pairs` is populated.
    PairwiseEmpirical,
    /// Single-site Gibbs sampler over the full condition vector — handles
    /// both positive and negative correlations and three-way+ joint structure
    /// via iteration. Activated when `CHRONOSYNTHEA_JOINT_MODE = "causal-dag"`
    /// AND a cooccurrence file is loaded.
    CausalDag,
}

/// Configuration for batch generation.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Base seed for deterministic generation.
    pub seed: u64,
    /// Reference date for timestamps.
    pub reference_date: NaiveDate,
    /// Time span in years for encounter distribution.
    pub time_span_years: u32,
    /// Maximum encounters per patient.
    pub max_encounters: u32,
    /// Number of worker threads (0 = auto).
    pub num_workers: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            reference_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            time_span_years: 10,
            max_encounters: 30,
            num_workers: 0,
        }
    }
}

impl BatchConfig {
    /// Returns the effective number of workers.
    pub fn effective_workers(&self) -> usize {
        if self.num_workers == 0 {
            rayon::current_num_threads()
        } else {
            self.num_workers
        }
    }

    /// Returns reference date as days since epoch.
    pub fn reference_days(&self) -> i32 {
        (self.reference_date - NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()).num_days() as i32
    }
}

/// 64-byte-aligned wrapper around an `AtomicU64` so adjacent counters live on
/// separate cache lines. Eliminates the false-sharing penalty that hits the
/// `record_full` hot path when many Rayon threads update neighbouring counter
/// indices — without padding, 8 `AtomicU64`s share a 64-byte line and
/// concurrent updates trigger cache-line migrations.
#[repr(align(64))]
#[derive(Default)]
pub struct PaddedAtomic(pub AtomicU64);

/// Lock-free atomic statistics for parallel collection. Every counter is
/// cache-padded to avoid false sharing — the 3 global running totals
/// (patients/encounters/events) are hammered by every thread on every
/// patient, and the per-condition counters get hit by every patient's
/// condition list; without padding 8 counters fit on one 64-byte cache
/// line, causing massive cross-core invalidation.
pub struct AtomicStatistics {
    /// Total patients generated.
    pub patients: PaddedAtomic,
    /// Total encounters generated.
    pub encounters: PaddedAtomic,
    /// Total events generated.
    pub events: PaddedAtomic,
    /// Condition occurrence counts (cache-padded).
    pub condition_counts: Vec<PaddedAtomic>,
    /// Medication occurrence counts (cache-padded).
    pub medication_counts: Vec<PaddedAtomic>,
    /// Observation occurrence counts (cache-padded).
    pub observation_counts: Vec<PaddedAtomic>,
    /// Procedure occurrence counts (cache-padded).
    pub procedure_counts: Vec<PaddedAtomic>,
}

impl AtomicStatistics {
    fn padded_vec(n: usize) -> Vec<PaddedAtomic> {
        (0..n).map(|_| PaddedAtomic(AtomicU64::new(0))).collect()
    }

    /// Creates new atomic statistics.
    pub fn new(num_conditions: usize) -> Self {
        Self {
            patients: PaddedAtomic(AtomicU64::new(0)),
            encounters: PaddedAtomic(AtomicU64::new(0)),
            events: PaddedAtomic(AtomicU64::new(0)),
            condition_counts: Self::padded_vec(num_conditions),
            medication_counts: Vec::new(),
            observation_counts: Vec::new(),
            procedure_counts: Vec::new(),
        }
    }

    /// Creates new atomic statistics with full event tracking.
    pub fn new_full(
        num_conditions: usize,
        num_medications: usize,
        num_observations: usize,
        num_procedures: usize,
    ) -> Self {
        Self {
            patients: PaddedAtomic(AtomicU64::new(0)),
            encounters: PaddedAtomic(AtomicU64::new(0)),
            events: PaddedAtomic(AtomicU64::new(0)),
            condition_counts: Self::padded_vec(num_conditions),
            medication_counts: Self::padded_vec(num_medications),
            observation_counts: Self::padded_vec(num_observations),
            procedure_counts: Self::padded_vec(num_procedures),
        }
    }

    /// Records a patient generation.
    #[inline]
    pub fn record(&self, conditions: &[u16], encounters: u64, events: u64) {
        self.patients.0.fetch_add(1, Ordering::Relaxed);
        self.encounters.0.fetch_add(encounters, Ordering::Relaxed);
        self.events.0.fetch_add(events, Ordering::Relaxed);

        for &cond_idx in conditions {
            if (cond_idx as usize) < self.condition_counts.len() {
                self.condition_counts[cond_idx as usize]
                    .0
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Records a full patient with all event types.
    #[inline(always)]
    pub fn record_full(
        &self,
        conditions: &[u16],
        medications: &[u16],
        observations: &[u16],
        procedures: &[u16],
        encounters: u64,
        events: u64,
    ) {
        // Batch the scalar updates
        self.patients.0.fetch_add(1, Ordering::Relaxed);
        self.encounters.0.fetch_add(encounters, Ordering::Relaxed);
        self.events.0.fetch_add(events, Ordering::Relaxed);

        // SAFETY: All indices come from valid sampling within known bounds
        unsafe {
            for &cond_idx in conditions {
                self.condition_counts
                    .get_unchecked(cond_idx as usize)
                    .0
                    .fetch_add(1, Ordering::Relaxed);
            }

            for &med_idx in medications {
                self.medication_counts
                    .get_unchecked(med_idx as usize)
                    .0
                    .fetch_add(1, Ordering::Relaxed);
            }

            for &obs_idx in observations {
                self.observation_counts
                    .get_unchecked(obs_idx as usize)
                    .0
                    .fetch_add(1, Ordering::Relaxed);
            }

            for &proc_idx in procedures {
                self.procedure_counts
                    .get_unchecked(proc_idx as usize)
                    .0
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Converts to streaming statistics.
    pub fn to_streaming(&self) -> StreamingStatistics {
        let patients = self.patients.0.load(Ordering::Relaxed);
        let mut stats = StreamingStatistics::new(self.condition_counts.len());

        stats.total_patients = patients;
        stats.total_encounters = self.encounters.0.load(Ordering::Relaxed);
        stats.total_events = self.events.0.load(Ordering::Relaxed);

        for (i, count) in self.condition_counts.iter().enumerate() {
            stats.condition_counts[i] = count.0.load(Ordering::Relaxed);
        }

        // Extend medication counts if needed
        if !self.medication_counts.is_empty() {
            stats.medication_counts = self
                .medication_counts
                .iter()
                .map(|c| c.0.load(Ordering::Relaxed))
                .collect();
        }

        // Extend observation counts if needed
        if !self.observation_counts.is_empty() {
            stats.observation_counts = self
                .observation_counts
                .iter()
                .map(|c| c.0.load(Ordering::Relaxed))
                .collect();
        }

        // Extend procedure counts if needed
        if !self.procedure_counts.is_empty() {
            stats.procedure_counts = self
                .procedure_counts
                .iter()
                .map(|c| c.0.load(Ordering::Relaxed))
                .collect();
        }

        stats
    }
}

/// High-performance batch generator.
pub struct BatchGenerator {
    /// Generation configuration.
    config: BatchConfig,

    /// Archetype registry.
    archetypes: Arc<ArchetypeRegistry>,

    /// Code table.
    code_table: Arc<CodeTable>,

    /// Fingerprint for event frequencies. Held for lifetime + future event-path
    /// extensions; the current hot paths consume `archetypes` + `code_table` directly.
    #[allow(dead_code)]
    fingerprint: Arc<MssFingerprint>,

    /// Co-occurrence model for condition dependencies.
    cooccurrence: Arc<CooccurrenceModel>,

    /// Optional Gibbs Ising-model sampler for d5 = `causal-DAG`. Constructed
    /// from the fingerprint's cooccurrence map but only consulted when
    /// `joint_mode == CausalDag`. `Arc` so it can be shared across Rayon
    /// workers without per-thread cloning of the interaction table.
    causal_dag: Arc<CausalDagModel>,

    /// Active d5 joint-structure mode. Set once at construction from the
    /// fingerprint contents and `CHRONOSYNTHEA_JOINT_MODE` env var.
    joint_mode: JointMode,
}

impl BatchGenerator {
    /// Creates a new batch generator from an MSS fingerprint.
    pub fn new(fingerprint: MssFingerprint, config: BatchConfig) -> Self {
        let archetypes = Arc::new(ArchetypeRegistry::from_fingerprint(&fingerprint));
        let code_table = Arc::new(CodeTable::from_fingerprint(&fingerprint));
        let cooccurrence = Arc::new(CooccurrenceModel::from_fingerprint(&fingerprint));
        let causal_dag = Arc::new(CausalDagModel::from_fingerprint(&fingerprint));

        // d5 mode selection. Defaults to MarginalOnly; promotes to
        // PairwiseEmpirical when cooccurrence is loaded; promotes further
        // to CausalDag when the env var asks for it (and there's data to
        // build a meaningful Ising model from).
        let joint_mode = if std::env::var("CHRONOSYNTHEA_JOINT_MODE").as_deref()
            == Ok("causal-dag")
            && !causal_dag.is_empty()
        {
            JointMode::CausalDag
        } else if !cooccurrence.is_empty() {
            JointMode::PairwiseEmpirical
        } else {
            JointMode::MarginalOnly
        };

        let fingerprint = Arc::new(fingerprint);

        Self {
            config,
            archetypes,
            code_table,
            fingerprint,
            cooccurrence,
            causal_dag,
            joint_mode,
        }
    }

    /// Returns the active d5 joint-structure mode.
    pub fn joint_mode(&self) -> JointMode {
        self.joint_mode
    }

    /// Generates patients and returns aggregate statistics only (fastest path).
    ///
    /// Per-thread non-atomic accumulators via Rayon `fold`+`reduce` — no
    /// atomic ops on the hot path. The merge at end is O(num_threads × num_conditions).
    pub fn generate_stats_only(&self, count: usize) -> StreamingStatistics {
        let num_conditions = self.archetypes.num_conditions();
        let base_seed = self.config.seed;
        let cooccurrence = Arc::clone(&self.cooccurrence);
        let causal_dag = Arc::clone(&self.causal_dag);
        let joint_mode = self.joint_mode;
        let archetypes = Arc::clone(&self.archetypes);

        struct LocalStats {
            rng: Xoshiro256PlusPlus,
            sampler: SimdSampler,
            condition_buffer: SmallVec<[u16; 8]>,
            patients: u64,
            encounters: u64,
            events: u64,
            cond_counts: Vec<u64>,
        }

        let merged = (0..count)
            .into_par_iter()
            .fold(
                || {
                    let thread_id = rayon::current_thread_index().unwrap_or(0);
                    let seed = base_seed.wrapping_add(thread_id as u64 * 1_000_000);
                    LocalStats {
                        rng: Xoshiro256PlusPlus::seed_from_u64(seed),
                        sampler: SimdSampler::from_registry(&archetypes),
                        condition_buffer: SmallVec::new(),
                        patients: 0,
                        encounters: 0,
                        events: 0,
                        cond_counts: vec![0u64; num_conditions],
                    }
                },
                |mut s, patient_id| {
                    // Per-patient reseed preserves deterministic-per-patient
                    // output that the original generate_stats_only contract
                    // promised.
                    let patient_seed = base_seed.wrapping_add(patient_id as u64);
                    s.rng = Xoshiro256PlusPlus::seed_from_u64(patient_seed);

                    let archetype = archetypes.sample(&mut s.rng);

                    match joint_mode {


                        JointMode::CausalDag => {


                            causal_dag.sample(archetype, &mut s.condition_buffer, &mut s.rng);


                        }


                        JointMode::PairwiseEmpirical => {


                            archetype.sample_conditions_with_cooccurrence(


                                &mut s.rng,


                                &mut s.condition_buffer,


                                &cooccurrence,


                            );


                        }


                        JointMode::MarginalOnly => {


                            let (thr, idx) = archetypes.active_view(archetype.id);


                            s.sampler.sample_active(thr, idx, &mut s.rng, &mut s.condition_buffer);


                        }


                    }

                    let encounter_count = self.estimate_encounters(archetype, &mut s.rng);
                    let event_count =
                        self.estimate_events(archetype, encounter_count, &mut s.rng);

                    s.patients += 1;
                    s.encounters += encounter_count;
                    s.events += event_count;
                    for &c in s.condition_buffer.iter() {
                        if (c as usize) < s.cond_counts.len() {
                            unsafe { *s.cond_counts.get_unchecked_mut(c as usize) += 1 };
                        }
                    }
                    s
                },
            )
            .reduce(
                || LocalStats {
                    rng: Xoshiro256PlusPlus::seed_from_u64(0),
                    sampler: SimdSampler::from_registry(&archetypes),
                    condition_buffer: SmallVec::new(),
                    patients: 0,
                    encounters: 0,
                    events: 0,
                    cond_counts: vec![0u64; num_conditions],
                },
                |mut a, b| {
                    a.patients += b.patients;
                    a.encounters += b.encounters;
                    a.events += b.events;
                    for (x, y) in a.cond_counts.iter_mut().zip(b.cond_counts.iter()) {
                        *x += y;
                    }
                    a
                },
            );

        let mut stats = StreamingStatistics::new(num_conditions);
        stats.total_patients = merged.patients;
        stats.total_encounters = merged.encounters;
        stats.total_events = merged.events;
        stats.condition_counts = merged.cond_counts;
        stats
    }

    /// Generates compact patients (in-memory, minimal allocations).
    pub fn generate_compact(&self, count: usize) -> Vec<CompactPatient> {
        let base_seed = self.config.seed;
        let reference_days = self.config.reference_days();
        let cooccurrence = Arc::clone(&self.cooccurrence);
        // causal_dag / joint_mode dispatch happens inside the `generate_compact_patient`
        // helper method through `self.causal_dag` / `self.joint_mode` — no local capture
        // needed here.

        (0..count)
            .into_par_iter()
            .map_init(
                || {
                    let thread_id = rayon::current_thread_index().unwrap_or(0);
                    let seed = base_seed.wrapping_add(thread_id as u64 * 1_000_000);
                    (
                        Xoshiro256PlusPlus::seed_from_u64(seed),
                        SimdSampler::from_registry(&self.archetypes),
                        SmallVec::<[u16; 8]>::new(),
                    )
                },
                |(rng, sampler, condition_buffer), patient_id| {
                    // Deterministic per-patient seed
                    let patient_seed = base_seed.wrapping_add(patient_id as u64);
                    *rng = Xoshiro256PlusPlus::seed_from_u64(patient_seed);

                    self.generate_compact_patient(
                        patient_id as u64,
                        rng,
                        sampler,
                        condition_buffer,
                        reference_days,
                        &cooccurrence,
                    )
                },
            )
            .collect()
    }

    /// Generates a single compact patient.
    #[inline]
    fn generate_compact_patient(
        &self,
        id: u64,
        rng: &mut Xoshiro256PlusPlus,
        sampler: &mut SimdSampler,
        condition_buffer: &mut SmallVec<[u16; 8]>,
        _reference_days: i32,
        cooccurrence: &CooccurrenceModel,
    ) -> CompactPatient {
        use crate::tables::{ethnicity_to_idx, race_to_idx};

        // Sample archetype
        let archetype = self.archetypes.sample(rng);

        // Sample conditions with co-occurrence modeling — dense SIMD active-view
        // on the fast path; the scalar branch fires only when a populated
        // co-occurrence model is plugged in.
        match self.joint_mode {
            JointMode::CausalDag => {
                self.causal_dag.sample(archetype, condition_buffer, rng);
            }
            JointMode::PairwiseEmpirical => {
                archetype.sample_conditions_with_cooccurrence(rng, condition_buffer, cooccurrence);
            }
            JointMode::MarginalOnly => {
                let (thr, idx) = self.archetypes.active_view(archetype.id);
                sampler.sample_active(thr, idx, rng, condition_buffer);
            }
        }

        // Sample age within archetype range
        let age = archetype.sample_age(rng);
        let max_age_days = (age as u32).saturating_mul(365) + (age as u32) / 4; // approx age × 365.25

        // Calculate birth date
        let birth_year = 2024 - age as i32;
        let birth_date = NaiveDate::from_ymd_opt(birth_year, 1, 1).unwrap();
        let birth_date_days =
            (birth_date - NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()).num_days() as i32;

        // d5 = `temporal-ordered`: sample per-condition onset days from the
        // empirical Java Synthea distribution loaded into the registry. Sort
        // the (condition, onset) pairs ascending by onset so the patient's
        // trajectory is in temporal order — equivalent to Java Synthea's
        // state-machine-generated condition timeline.
        let mut condition_onset_days: SmallVec<[u16; 8]> = SmallVec::new();
        for &c in condition_buffer.iter() {
            let days = self.archetypes.sample_onset_days(c, max_age_days, rng);
            condition_onset_days.push(days);
        }
        // Sort by onset; we have two parallel SmallVecs, so build a temp index.
        let mut order: SmallVec<[u8; 8]> = (0..condition_buffer.len() as u8).collect();
        order.sort_by_key(|&i| condition_onset_days[i as usize]);
        let sorted_conds: SmallVec<[u16; 8]> = order
            .iter()
            .map(|&i| condition_buffer[i as usize])
            .collect();
        let sorted_onsets: SmallVec<[u16; 8]> = order
            .iter()
            .map(|&i| condition_onset_days[i as usize])
            .collect();

        // Extract demographics from archetype
        let sex =
            if archetype.demographics.gender == "female" || archetype.demographics.gender == "F" {
                1
            } else {
                0
            };
        let race = race_to_idx(&archetype.demographics.race);
        let ethnicity = ethnicity_to_idx(&archetype.demographics.ethnicity);

        // Estimate encounters
        let encounter_count = self.estimate_encounters(archetype, rng).min(255) as u8;

        CompactPatient {
            id,
            birth_date_days,
            sex,
            race,
            ethnicity,
            encounter_count,
            condition_count: sorted_conds.len().min(255) as u8,
            archetype_id: archetype.id,
            conditions: sorted_conds,
            condition_onset_days: sorted_onsets,
        }
    }

    /// Estimates the number of encounters for a patient.
    #[inline(always)]
    fn estimate_encounters(&self, archetype: &PatientArchetype, rng: &mut impl rand::Rng) -> u64 {
        // Use Poisson-ish distribution around mean
        let mean = archetype.mean_encounters;
        let variance = mean.sqrt();
        let count = (mean + rng.gen::<f32>() * variance * 2.0 - variance).max(1.0);
        count.min(self.config.max_encounters as f32) as u64
    }

    /// Estimates the number of events for a patient.
    #[inline(always)]
    fn estimate_events(
        &self,
        archetype: &PatientArchetype,
        encounters: u64,
        rng: &mut impl rand::Rng,
    ) -> u64 {
        let mean_per_enc = archetype.mean_events_per_encounter;
        let variance = mean_per_enc.sqrt();
        let events_per = (mean_per_enc + rng.gen::<f32>() * variance * 2.0 - variance).max(1.0);
        (encounters as f32 * events_per) as u64
    }

    /// Returns the archetype registry.
    /// Mutable access to the shared archetype registry Arc — used by the
    /// recalibration loop to apply per-condition prevalence scales in place.
    /// Returns `Some` only when this generator holds the sole reference.
    pub fn archetypes_arc_mut(&mut self) -> &mut std::sync::Arc<ArchetypeRegistry> {
        &mut self.archetypes
    }

    /// Mutable access to the shared co-occurrence model Arc — used by the
    /// recalibration loop to adjust per-dependent boost scales in place.
    pub fn cooccurrence_arc_mut(&mut self) -> &mut std::sync::Arc<CooccurrenceModel> {
        &mut self.cooccurrence
    }

    pub fn archetypes(&self) -> &ArchetypeRegistry {
        &self.archetypes
    }

    /// Returns the code table.
    pub fn code_table(&self) -> &CodeTable {
        &self.code_table
    }

    /// Returns the configuration.
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }

    /// Returns the number of conditions.
    pub fn num_conditions(&self) -> usize {
        self.archetypes.num_conditions()
    }

    /// Generates full statistics including medications, observations, and procedures.
    ///
    /// This is slower than generate_stats_only but tracks all event types.
    /// Uses Rayon's `fold` + `reduce` pattern with per-thread non-atomic
    /// `LocalStats` accumulators — no atomic ops on the hot path, merge once
    /// per thread at the end. Eliminates ~170M atomic ops/sec that the
    /// previous `record_full` path was issuing at 5.6M patients/sec.
    pub fn generate_full_stats_only(&self, count: usize) -> StreamingStatistics {
        let num_conditions = self.archetypes.num_conditions();
        let num_medications = self.archetypes.num_medications();
        let num_observations = self.archetypes.num_observations();
        let num_procedures = self.archetypes.num_procedures();

        let base_seed = self.config.seed;
        let cooccurrence = Arc::clone(&self.cooccurrence);
        let causal_dag = Arc::clone(&self.causal_dag);
        let joint_mode = self.joint_mode;
        let archetypes = Arc::clone(&self.archetypes);

        let obs_freqs = archetypes.observation_frequencies();
        let proc_freqs = archetypes.procedure_frequencies();
        let max_encounters_f = self.config.max_encounters as f32;

        struct LocalStats {
            cond_rng: Xoshiro256PlusPlus,
            med_rng: Xoshiro256PlusPlus,
            evt_rng: Xoshiro256PlusPlus,
            condition_buffer: SmallVec<[u16; 8]>,
            event_sampler: EventSampler,
            sampler: SimdSampler,
            patients: u64,
            encounters: u64,
            events: u64,
            cond_counts: Vec<u64>,
            med_counts: Vec<u64>,
            obs_counts: Vec<u64>,
            proc_counts: Vec<u64>,
        }

        let merged = (0..count)
            .into_par_iter()
            .fold(
                || {
                    let thread_id = rayon::current_thread_index().unwrap_or(0);
                    let base = base_seed.wrapping_add(thread_id as u64);
                    let cond_seed = base.wrapping_mul(0x9E3779B97F4A7C15);
                    let med_seed = base.wrapping_mul(0xBF58476D1CE4E5B9);
                    let evt_seed = base.wrapping_mul(0x94D049BB133111EB);
                    LocalStats {
                        cond_rng: Xoshiro256PlusPlus::seed_from_u64(cond_seed),
                        med_rng: Xoshiro256PlusPlus::seed_from_u64(med_seed),
                        evt_rng: Xoshiro256PlusPlus::seed_from_u64(evt_seed),
                        condition_buffer: SmallVec::new(),
                        event_sampler: EventSampler::new(),
                        sampler: SimdSampler::from_registry(&archetypes),
                        patients: 0,
                        encounters: 0,
                        events: 0,
                        cond_counts: vec![0u64; num_conditions],
                        med_counts: vec![0u64; num_medications],
                        obs_counts: vec![0u64; num_observations],
                        proc_counts: vec![0u64; num_procedures],
                    }
                },
                |mut s, _patient_id| {
                    let archetype = archetypes.sample(&mut s.cond_rng);

                    match joint_mode {
                        JointMode::CausalDag => {
                            causal_dag.sample(
                                archetype,
                                &mut s.condition_buffer,
                                &mut s.cond_rng,
                            );
                        }
                        JointMode::PairwiseEmpirical => {
                            archetype.sample_conditions_with_cooccurrence(
                                &mut s.cond_rng,
                                &mut s.condition_buffer,
                                &cooccurrence,
                            );
                        }
                        JointMode::MarginalOnly => {
                            let (thr, idx) = archetypes.active_view(archetype.id);
                            s.sampler.sample_active(
                                thr,
                                idx,
                                &mut s.cond_rng,
                                &mut s.condition_buffer,
                            );
                        }
                    }

                    let med_thresholds = archetypes.medication_thresholds(archetype.id);
                    s.event_sampler
                        .sample_medications_simd(med_thresholds, &mut s.med_rng);

                    let mean = archetype.mean_encounters;
                    let encounter_count = (mean
                        + (s.evt_rng.gen::<f32>() - 0.5) * mean.sqrt() * 2.0)
                        .max(1.0)
                        .min(max_encounters_f)
                        as u64;

                    s.event_sampler.sample_events_batch(
                        obs_freqs,
                        proc_freqs,
                        encounter_count as u32,
                        &mut s.evt_rng,
                    );

                    let events_per = archetype.mean_events_per_encounter;
                    let event_count = (encounter_count as f32 * events_per) as u64;

                    // Non-atomic local accumulation. Vec writes only.
                    s.patients += 1;
                    s.encounters += encounter_count;
                    s.events += event_count;
                    for &c in s.condition_buffer.iter() {
                        unsafe { *s.cond_counts.get_unchecked_mut(c as usize) += 1 };
                    }
                    for &m in s.event_sampler.medications() {
                        unsafe { *s.med_counts.get_unchecked_mut(m as usize) += 1 };
                    }
                    for &o in s.event_sampler.accumulated_observations() {
                        unsafe { *s.obs_counts.get_unchecked_mut(o as usize) += 1 };
                    }
                    for &p in s.event_sampler.accumulated_procedures() {
                        unsafe { *s.proc_counts.get_unchecked_mut(p as usize) += 1 };
                    }
                    s
                },
            )
            .reduce(
                || LocalStats {
                    cond_rng: Xoshiro256PlusPlus::seed_from_u64(0),
                    med_rng: Xoshiro256PlusPlus::seed_from_u64(0),
                    evt_rng: Xoshiro256PlusPlus::seed_from_u64(0),
                    condition_buffer: SmallVec::new(),
                    event_sampler: EventSampler::new(),
                    sampler: SimdSampler::from_registry(&archetypes),
                    patients: 0,
                    encounters: 0,
                    events: 0,
                    cond_counts: vec![0u64; num_conditions],
                    med_counts: vec![0u64; num_medications],
                    obs_counts: vec![0u64; num_observations],
                    proc_counts: vec![0u64; num_procedures],
                },
                |mut a, b| {
                    a.patients += b.patients;
                    a.encounters += b.encounters;
                    a.events += b.events;
                    for (x, y) in a.cond_counts.iter_mut().zip(b.cond_counts.iter()) {
                        *x += y;
                    }
                    for (x, y) in a.med_counts.iter_mut().zip(b.med_counts.iter()) {
                        *x += y;
                    }
                    for (x, y) in a.obs_counts.iter_mut().zip(b.obs_counts.iter()) {
                        *x += y;
                    }
                    for (x, y) in a.proc_counts.iter_mut().zip(b.proc_counts.iter()) {
                        *x += y;
                    }
                    a
                },
            );

        let mut stats = StreamingStatistics::new(num_conditions);
        stats.total_patients = merged.patients;
        stats.total_encounters = merged.encounters;
        stats.total_events = merged.events;
        stats.condition_counts = merged.cond_counts;
        stats.medication_counts = merged.med_counts;
        stats.observation_counts = merged.obs_counts;
        stats.procedure_counts = merged.proc_counts;
        stats
    }

    /// Generates full patients with complete encounter and event data.
    ///
    /// This is slower than generate_stats_only but produces complete patient
    /// records with medications, observations, and procedures.
    pub fn generate_full(&self, count: usize) -> Vec<FullPatient> {
        let base_seed = self.config.seed;
        let cooccurrence = Arc::clone(&self.cooccurrence);
        // Dispatch via `self.joint_mode` / `self.causal_dag` inside the
        // `generate_full_patient` helper — no local capture needed.
        let archetypes = Arc::clone(&self.archetypes);

        (0..count)
            .into_par_iter()
            .map_init(
                || {
                    let thread_id = rayon::current_thread_index().unwrap_or(0);
                    let seed = base_seed.wrapping_add(thread_id as u64 * 1_000_000);
                    (
                        Xoshiro256PlusPlus::seed_from_u64(seed),
                        SmallVec::<[u16; 8]>::new(),
                        EventSampler::new(),
                        SimdSampler::from_registry(&archetypes),
                    )
                },
                |(rng, condition_buffer, event_sampler, sampler), patient_id| {
                    // Deterministic per-patient seed
                    let patient_seed = base_seed.wrapping_add(patient_id as u64);
                    *rng = Xoshiro256PlusPlus::seed_from_u64(patient_seed);

                    self.generate_full_patient(
                        patient_id as u64,
                        rng,
                        sampler,
                        condition_buffer,
                        event_sampler,
                        &cooccurrence,
                        &archetypes,
                    )
                },
            )
            .collect()
    }

    /// Generates a single full patient with all events.
    fn generate_full_patient(
        &self,
        id: u64,
        rng: &mut Xoshiro256PlusPlus,
        sampler: &mut SimdSampler,
        condition_buffer: &mut SmallVec<[u16; 8]>,
        event_sampler: &mut EventSampler,
        cooccurrence: &CooccurrenceModel,
        archetypes: &ArchetypeRegistry,
    ) -> FullPatient {
        use crate::tables::{encounter_type_to_idx, ethnicity_to_idx, race_to_idx};
        use rand::Rng;

        // Sample archetype
        let archetype = archetypes.sample(rng);

        // Sample conditions with d5 dispatch — dense SIMD active-view on the marginal path.
        match self.joint_mode {
            JointMode::CausalDag => {
                self.causal_dag.sample(archetype, condition_buffer, rng);
            }
            JointMode::PairwiseEmpirical => {
                archetype.sample_conditions_with_cooccurrence(rng, condition_buffer, cooccurrence);
            }
            JointMode::MarginalOnly => {
                let (thr, idx) = archetypes.active_view(archetype.id);
                sampler.sample_active(thr, idx, rng, condition_buffer);
            }
        }

        // Sample age within archetype range
        let age = archetype.sample_age(rng);
        let max_age_days = (age as u32).saturating_mul(365) + (age as u32) / 4;
        let birth_year = 2024 - age as i32;
        let birth_date = NaiveDate::from_ymd_opt(birth_year, 1, 1).unwrap();
        let birth_date_days =
            (birth_date - NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()).num_days() as i32;

        // Extract demographics
        let sex =
            if archetype.demographics.gender == "female" || archetype.demographics.gender == "F" {
                1
            } else {
                0
            };
        let race = race_to_idx(&archetype.demographics.race);
        let ethnicity = ethnicity_to_idx(&archetype.demographics.ethnicity);

        // d5 = `temporal-ordered`: sample per-condition onset days and sort
        // (condition, onset) ascending so the patient's trajectory walks in
        // temporal order. Mirrors the CompactPatient path; the encounter-level
        // CSV writer uses these to stamp the conditions.csv `START` column
        // with each condition's actual onset date instead of the patient's
        // birth date.
        let mut raw_onsets: SmallVec<[u16; 8]> = SmallVec::new();
        for &c in condition_buffer.iter() {
            raw_onsets.push(archetypes.sample_onset_days(c, max_age_days, rng));
        }
        let mut order: SmallVec<[u8; 8]> =
            (0..condition_buffer.len() as u8).collect();
        order.sort_by_key(|&i| raw_onsets[i as usize]);
        let sorted_conds: SmallVec<[u16; 8]> = order
            .iter()
            .map(|&i| condition_buffer[i as usize])
            .collect();
        let sorted_onsets: SmallVec<[u16; 8]> =
            order.iter().map(|&i| raw_onsets[i as usize]).collect();
        // Keep `condition_buffer` aligned with the sorted patient view so
        // downstream samplers (medications, procedures, REASONCODE) see the
        // same per-patient condition set.
        condition_buffer.clear();
        condition_buffer.extend_from_slice(&sorted_conds);

        // Create patient
        let mut patient = FullPatient::new(id, birth_date_days, sex, race, ethnicity, archetype.id);
        patient.conditions = sorted_conds;
        patient.condition_onset_days = sorted_onsets;

        // Sample medications based on conditions
        let meds =
            event_sampler.sample_medications_for_conditions(condition_buffer, archetypes, rng);
        patient.medications = SmallVec::from_slice(meds);

        // REASONCODE linkage: for each sampled medication, pick the
        // condition that caused it (matches Java Synthea's
        // medications.csv:REASONCODE column).
        patient.medication_causes = patient
            .medications
            .iter()
            .map(|&m| archetypes.sample_medication_cause(m, condition_buffer, rng))
            .collect();

        // Generate encounters and sample procedures per encounter
        let encounter_count = self.estimate_encounters(archetype, rng).min(30) as u32;
        let proc_freqs = archetypes.procedure_frequencies();

        // Encounter type distribution
        let encounter_types = [
            "wellness",
            "ambulatory",
            "ambulatory",
            "ambulatory",
            "urgentcare",
        ];

        // Reset sampler for this patient's encounters (uses O(1) bitset clearing)
        event_sampler.reset_patient();

        for enc_idx in 0..encounter_count {
            // Spread encounters across patient lifetime
            let days_since_birth = if age > 0 {
                rng.gen_range(0..(age * 365)) as u16
            } else {
                0
            };

            // Sample encounter type
            let enc_type_str = encounter_types[rng.gen_range(0..encounter_types.len())];
            let enc_type = encounter_type_to_idx(enc_type_str);

            let mut encounter = FullEncounter::new(enc_type, days_since_birth);

            // Add condition diagnosis events (chronic conditions appear at
            // ~30% of encounters; acute conditions at the first encounter).
            for &cond_idx in condition_buffer.iter() {
                if enc_idx == 0 || rng.gen::<f32>() < 0.3 {
                    encounter.add_event(CompactEvent {
                        event_type: 0, // diagnosis
                        system_idx: 0, // SNOMED-CT
                        code_idx: cond_idx,
                        display_idx: cond_idx,
                        timestamp_offset: 0,
                    });
                }
            }

            // Medication events: each patient-level medication fires at
            // ~50% of encounters (chronic admin); REASONCODE is preserved
            // via the per-patient `medication_causes` lookup in csv_writer.
            for &med_idx in patient.medications.iter() {
                if rng.gen::<f32>() < 0.5 {
                    encounter.add_event(CompactEvent {
                        event_type: 1, // medication
                        system_idx: 1, // RxNorm
                        code_idx: med_idx,
                        display_idx: med_idx,
                        timestamp_offset: 0,
                    });
                }
            }

            // Observation events.
            let obs_freqs = archetypes.observation_frequencies();
            let obs = event_sampler.sample_observations_for_encounter(obs_freqs, rng);
            for &obs_idx in obs {
                encounter.add_event(CompactEvent {
                    event_type: 3, // observation
                    system_idx: 2, // LOINC
                    code_idx: obs_idx,
                    display_idx: obs_idx,
                    timestamp_offset: 0,
                });
            }

            // Procedure events: per-encounter sampling at the natural
            // empirical rate (each procedure's `frequency` field is already
            // a per-encounter probability extracted from Java's output).
            // Also adds condition-triggered procedures so chronic conditions
            // (dialysis for CKD, mammograms for breast screening) fire at
            // each encounter at their per-condition rate. Both paths
            // accumulate into the patient-level `procedure_seen` bitset for
            // the REASONCODE lookup and dedup against the patient.procedures
            // membership set.
            let enc_procs = event_sampler.sample_procedures_for_encounter(proc_freqs, rng);
            for &proc_idx in enc_procs {
                encounter.add_event(CompactEvent {
                    event_type: 2, // procedure
                    system_idx: 0, // SNOMED-CT
                    code_idx: proc_idx,
                    display_idx: proc_idx,
                    timestamp_offset: 0,
                });
            }
            // Condition-triggered per-encounter procedure firing.
            for &cond_idx in condition_buffer.iter() {
                let procs = archetypes.procedures_for_condition(cond_idx);
                for &(p, freq) in procs {
                    if freq > 0.0 && rng.gen::<f32>() < freq {
                        encounter.add_event(CompactEvent {
                            event_type: 2, // procedure
                            system_idx: 0,
                            code_idx: p,
                            display_idx: p,
                            timestamp_offset: 0,
                        });
                    }
                }
            }

            patient.encounters.push(encounter);
        }

        // Build the patient-level unique procedure set from encounter events
        // (rather than carrying a parallel patient buffer that has to be
        // synced with what the encounter loop actually emitted). Bitset dedup
        // keeps this O(n_events) with no quadratic contains() check.
        let mut proc_seen = [0u64; (u16::MAX as usize / 64) + 1];
        for enc in &patient.encounters {
            for ev in &enc.events {
                if ev.event_type == 2 {
                    let idx = ev.code_idx as usize;
                    let word = idx / 64;
                    let bit = 1u64 << (idx % 64);
                    if proc_seen[word] & bit == 0 {
                        proc_seen[word] |= bit;
                        patient.procedures.push(ev.code_idx);
                    }
                }
            }
        }

        // REASONCODE linkage for procedures (matches Java Synthea's
        // procedures.csv:REASONCODE column).
        patient.procedure_causes = patient
            .procedures
            .iter()
            .map(|&p| archetypes.sample_procedure_cause(p, condition_buffer, rng))
            .collect();

        patient
    }
}

/// Result of batch generation.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Number of patients generated.
    pub patient_count: u64,
    /// Total encounters generated.
    pub encounter_count: u64,
    /// Total events generated.
    pub event_count: u64,
    /// Generation time in milliseconds.
    pub generation_time_ms: u64,
    /// Patients per second.
    pub patients_per_second: f64,
    /// Streaming statistics.
    pub statistics: StreamingStatistics,
}

impl GenerationResult {
    /// Creates a new generation result.
    pub fn new(stats: StreamingStatistics, duration_ms: u64) -> Self {
        let patient_count = stats.total_patients;
        let patients_per_second = if duration_ms > 0 {
            patient_count as f64 * 1000.0 / duration_ms as f64
        } else {
            0.0
        };

        Self {
            patient_count,
            encounter_count: stats.total_encounters,
            event_count: stats.total_events,
            generation_time_ms: duration_ms,
            patients_per_second,
            statistics: stats,
        }
    }

    /// Returns whether the target of 1M patients/second was achieved.
    pub fn achieved_target(&self) -> bool {
        self.patients_per_second >= 1_000_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fingerprint::{
        ConditionStats, DemographicBucket, EncounterStats, JointDemographics,
    };
    use ahash::AHashMap;

    fn create_test_fingerprint() -> MssFingerprint {
        let mut buckets = AHashMap::new();
        buckets.insert(
            DemographicBucket::new("18-44", "male", "white", "nonhispanic"),
            0.3,
        );
        buckets.insert(
            DemographicBucket::new("18-44", "female", "white", "nonhispanic"),
            0.3,
        );
        buckets.insert(
            DemographicBucket::new("45-64", "male", "white", "nonhispanic"),
            0.2,
        );
        buckets.insert(
            DemographicBucket::new("65+", "female", "white", "nonhispanic"),
            0.2,
        );

        MssFingerprint {
            version: "1.0".to_string(),
            source: "test".to_string(),
            total_patients: 1000,
            total_encounters: 10000,
            joint_demographics: JointDemographics {
                buckets,
                total_patients: 1000,
            },
            conditions: vec![
                ConditionStats {
                    code: "38341003".to_string(),
                    display: "Hypertension".to_string(),
                    prevalence: 0.3,
                    by_age_bucket: AHashMap::new(),
                    by_gender: AHashMap::new(),
                    by_race: AHashMap::new(),
                    chronic: true,
                    mean_onset_age: 50.0,
                },
                ConditionStats {
                    code: "44054006".to_string(),
                    display: "Diabetes".to_string(),
                    prevalence: 0.1,
                    by_age_bucket: AHashMap::new(),
                    by_gender: AHashMap::new(),
                    by_race: AHashMap::new(),
                    chronic: true,
                    mean_onset_age: 55.0,
                },
            ],
            medications: vec![],
            observations: vec![],
            procedures: vec![],
            cooccurrence: AHashMap::new(),
            cooccurrence_dependent_scale: AHashMap::new(),
            onset_stats: Vec::new(),
            encounter_stats: EncounterStats {
                mean_by_age: AHashMap::new(),
                type_distribution: AHashMap::new(),
                mean_events_per_encounter: 5.0,
            },
        }
    }

    #[test]
    fn test_batch_generator_stats_only() {
        let fp = create_test_fingerprint();
        let config = BatchConfig::default();
        let generator = BatchGenerator::new(fp, config);

        let stats = generator.generate_stats_only(1000);

        assert_eq!(stats.total_patients, 1000);
        assert!(stats.total_encounters > 0);
    }

    #[test]
    fn test_batch_generator_compact() {
        let fp = create_test_fingerprint();
        let config = BatchConfig::default();
        let generator = BatchGenerator::new(fp, config);

        let patients = generator.generate_compact(100);

        assert_eq!(patients.len(), 100);

        // Check IDs are unique
        let mut ids: Vec<_> = patients.iter().map(|p| p.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 100);
    }

    #[test]
    fn test_atomic_statistics() {
        let stats = AtomicStatistics::new(10);

        stats.record(&[0, 1, 2], 5, 20);
        stats.record(&[1, 2], 3, 15);

        let streaming = stats.to_streaming();

        assert_eq!(streaming.total_patients, 2);
        assert_eq!(streaming.total_encounters, 8);
        assert_eq!(streaming.total_events, 35);
        assert_eq!(streaming.condition_counts[0], 1);
        assert_eq!(streaming.condition_counts[1], 2);
        assert_eq!(streaming.condition_counts[2], 2);
    }
}
