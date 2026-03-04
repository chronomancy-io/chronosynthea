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
use crate::fingerprint::MssFingerprint;
use crate::sampler::{EventSampler, SimdSampler};
use crate::stats::StreamingStatistics;
use crate::tables::CodeTable;

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

/// Lock-free atomic statistics for parallel collection.
pub struct AtomicStatistics {
    /// Total patients generated.
    pub patients: AtomicU64,
    /// Total encounters generated.
    pub encounters: AtomicU64,
    /// Total events generated.
    pub events: AtomicU64,
    /// Condition occurrence counts.
    pub condition_counts: Vec<AtomicU64>,
    /// Medication occurrence counts.
    pub medication_counts: Vec<AtomicU64>,
    /// Observation occurrence counts.
    pub observation_counts: Vec<AtomicU64>,
    /// Procedure occurrence counts.
    pub procedure_counts: Vec<AtomicU64>,
}

impl AtomicStatistics {
    /// Creates new atomic statistics.
    pub fn new(num_conditions: usize) -> Self {
        Self {
            patients: AtomicU64::new(0),
            encounters: AtomicU64::new(0),
            events: AtomicU64::new(0),
            condition_counts: (0..num_conditions).map(|_| AtomicU64::new(0)).collect(),
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
            patients: AtomicU64::new(0),
            encounters: AtomicU64::new(0),
            events: AtomicU64::new(0),
            condition_counts: (0..num_conditions).map(|_| AtomicU64::new(0)).collect(),
            medication_counts: (0..num_medications).map(|_| AtomicU64::new(0)).collect(),
            observation_counts: (0..num_observations).map(|_| AtomicU64::new(0)).collect(),
            procedure_counts: (0..num_procedures).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// Records a patient generation.
    #[inline]
    pub fn record(&self, conditions: &[u16], encounters: u64, events: u64) {
        self.patients.fetch_add(1, Ordering::Relaxed);
        self.encounters.fetch_add(encounters, Ordering::Relaxed);
        self.events.fetch_add(events, Ordering::Relaxed);

        for &cond_idx in conditions {
            if (cond_idx as usize) < self.condition_counts.len() {
                self.condition_counts[cond_idx as usize].fetch_add(1, Ordering::Relaxed);
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
        self.patients.fetch_add(1, Ordering::Relaxed);
        self.encounters.fetch_add(encounters, Ordering::Relaxed);
        self.events.fetch_add(events, Ordering::Relaxed);

        // SAFETY: All indices come from valid sampling within known bounds
        unsafe {
            for &cond_idx in conditions {
                self.condition_counts
                    .get_unchecked(cond_idx as usize)
                    .fetch_add(1, Ordering::Relaxed);
            }

            for &med_idx in medications {
                self.medication_counts
                    .get_unchecked(med_idx as usize)
                    .fetch_add(1, Ordering::Relaxed);
            }

            for &obs_idx in observations {
                self.observation_counts
                    .get_unchecked(obs_idx as usize)
                    .fetch_add(1, Ordering::Relaxed);
            }

            for &proc_idx in procedures {
                self.procedure_counts
                    .get_unchecked(proc_idx as usize)
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Converts to streaming statistics.
    pub fn to_streaming(&self) -> StreamingStatistics {
        let patients = self.patients.load(Ordering::Relaxed);
        let mut stats = StreamingStatistics::new(self.condition_counts.len());

        stats.total_patients = patients;
        stats.total_encounters = self.encounters.load(Ordering::Relaxed);
        stats.total_events = self.events.load(Ordering::Relaxed);

        for (i, count) in self.condition_counts.iter().enumerate() {
            stats.condition_counts[i] = count.load(Ordering::Relaxed);
        }

        // Extend medication counts if needed
        if !self.medication_counts.is_empty() {
            stats.medication_counts = self
                .medication_counts
                .iter()
                .map(|c| c.load(Ordering::Relaxed))
                .collect();
        }

        // Extend observation counts if needed
        if !self.observation_counts.is_empty() {
            stats.observation_counts = self
                .observation_counts
                .iter()
                .map(|c| c.load(Ordering::Relaxed))
                .collect();
        }

        // Extend procedure counts if needed
        if !self.procedure_counts.is_empty() {
            stats.procedure_counts = self
                .procedure_counts
                .iter()
                .map(|c| c.load(Ordering::Relaxed))
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

    /// Fingerprint for event frequencies.
    fingerprint: Arc<MssFingerprint>,

    /// Co-occurrence model for condition dependencies.
    cooccurrence: Arc<CooccurrenceModel>,
}

impl BatchGenerator {
    /// Creates a new batch generator from an MSS fingerprint.
    pub fn new(fingerprint: MssFingerprint, config: BatchConfig) -> Self {
        let archetypes = Arc::new(ArchetypeRegistry::from_fingerprint(&fingerprint));
        let code_table = Arc::new(CodeTable::from_fingerprint(&fingerprint));
        let cooccurrence = Arc::new(CooccurrenceModel::from_fingerprint(&fingerprint));
        let fingerprint = Arc::new(fingerprint);

        Self {
            config,
            archetypes,
            code_table,
            fingerprint,
            cooccurrence,
        }
    }

    /// Generates patients and returns aggregate statistics only (fastest path).
    ///
    /// This is the fastest generation mode - patients are generated and their
    /// statistics are accumulated, but individual patients are not stored.
    pub fn generate_stats_only(&self, count: usize) -> StreamingStatistics {
        let num_conditions = self.archetypes.num_conditions();
        let stats = Arc::new(AtomicStatistics::new(num_conditions));
        let base_seed = self.config.seed;
        let cooccurrence = Arc::clone(&self.cooccurrence);

        // Parallel generation using rayon
        (0..count).into_par_iter().for_each_init(
            || {
                let thread_id = rayon::current_thread_index().unwrap_or(0);
                let seed = base_seed.wrapping_add(thread_id as u64 * 1_000_000);
                (
                    Xoshiro256PlusPlus::seed_from_u64(seed),
                    SimdSampler::from_registry(&self.archetypes),
                    SmallVec::<[u16; 8]>::new(),
                )
            },
            |(rng, _sampler, condition_buffer), patient_id| {
                // Deterministic per-patient seed
                let patient_seed = base_seed.wrapping_add(patient_id as u64);
                *rng = Xoshiro256PlusPlus::seed_from_u64(patient_seed);

                // Sample archetype
                let archetype = self.archetypes.sample(rng);

                // Sample conditions with co-occurrence modeling
                if cooccurrence.is_empty() {
                    // Fast path: no co-occurrence, use flat sampling
                    self.archetypes
                        .sample_conditions_flat(archetype.id, rng, condition_buffer);
                } else {
                    // Apply co-occurrence model
                    archetype.sample_conditions_with_cooccurrence(
                        rng,
                        condition_buffer,
                        &cooccurrence,
                    );
                }

                // Estimate encounters and events
                let encounter_count = self.estimate_encounters(archetype, rng);
                let event_count = self.estimate_events(archetype, encounter_count, rng);

                // Record statistics atomically
                stats.record(condition_buffer, encounter_count, event_count);
            },
        );

        stats.to_streaming()
    }

    /// Generates compact patients (in-memory, minimal allocations).
    pub fn generate_compact(&self, count: usize) -> Vec<CompactPatient> {
        let base_seed = self.config.seed;
        let reference_days = self.config.reference_days();
        let cooccurrence = Arc::clone(&self.cooccurrence);

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
                |(rng, _sampler, condition_buffer), patient_id| {
                    // Deterministic per-patient seed
                    let patient_seed = base_seed.wrapping_add(patient_id as u64);
                    *rng = Xoshiro256PlusPlus::seed_from_u64(patient_seed);

                    self.generate_compact_patient(
                        patient_id as u64,
                        rng,
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
        condition_buffer: &mut SmallVec<[u16; 8]>,
        reference_days: i32,
        cooccurrence: &CooccurrenceModel,
    ) -> CompactPatient {
        use crate::tables::{ethnicity_to_idx, race_to_idx};
        use rand::Rng;

        // Sample archetype
        let archetype = self.archetypes.sample(rng);

        // Sample conditions with co-occurrence modeling
        if cooccurrence.is_empty() {
            self.archetypes
                .sample_conditions_flat(archetype.id, rng, condition_buffer);
        } else {
            archetype.sample_conditions_with_cooccurrence(rng, condition_buffer, cooccurrence);
        }

        // Sample age within archetype range
        let age = archetype.sample_age(rng);

        // Calculate birth date
        let birth_year = 2024 - age as i32;
        let birth_date = NaiveDate::from_ymd_opt(birth_year, 1, 1).unwrap();
        let birth_date_days =
            (birth_date - NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()).num_days() as i32;

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
            condition_count: condition_buffer.len().min(255) as u8,
            archetype_id: archetype.id,
            conditions: condition_buffer.clone(),
        }
    }

    /// Estimates the number of encounters for a patient.
    #[inline]
    fn estimate_encounters(&self, archetype: &PatientArchetype, rng: &mut impl rand::Rng) -> u64 {
        // Use Poisson-ish distribution around mean
        let mean = archetype.mean_encounters;
        let variance = mean.sqrt();
        let count = (mean + rng.gen::<f32>() * variance * 2.0 - variance).max(1.0);
        count.min(self.config.max_encounters as f32) as u64
    }

    /// Estimates the number of events for a patient.
    #[inline]
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
    pub fn generate_full_stats_only(&self, count: usize) -> StreamingStatistics {
        let num_conditions = self.archetypes.num_conditions();
        let num_medications = self.archetypes.num_medications();
        let num_observations = self.archetypes.num_observations();
        let num_procedures = self.archetypes.num_procedures();

        let stats = Arc::new(AtomicStatistics::new_full(
            num_conditions,
            num_medications,
            num_observations,
            num_procedures,
        ));
        let base_seed = self.config.seed;
        let cooccurrence = Arc::clone(&self.cooccurrence);
        let archetypes = Arc::clone(&self.archetypes);

        // Cache frequency arrays outside the loop
        let obs_freqs = archetypes.observation_frequencies();
        let proc_freqs = archetypes.procedure_frequencies();
        let use_cooccurrence = !cooccurrence.is_empty();

        (0..count).into_par_iter().for_each_init(
            || {
                let thread_id = rayon::current_thread_index().unwrap_or(0);
                // Use thread-local RNG that advances naturally (no per-patient reseeding)
                let seed =
                    base_seed.wrapping_add((thread_id as u64).wrapping_mul(0x9E3779B97F4A7C15));
                (
                    Xoshiro256PlusPlus::seed_from_u64(seed),
                    SmallVec::<[u16; 8]>::new(),
                    EventSampler::new(),
                )
            },
            |(rng, condition_buffer, event_sampler), _patient_id| {
                // Sample archetype
                let archetype = archetypes.sample(rng);

                // Sample conditions
                if use_cooccurrence {
                    archetype.sample_conditions_with_cooccurrence(
                        rng,
                        condition_buffer,
                        &cooccurrence,
                    );
                } else {
                    archetypes.sample_conditions_flat(archetype.id, rng, condition_buffer);
                }

                // Sample medications using pre-computed SIMD thresholds
                let med_thresholds = archetypes.medication_thresholds(archetype.id);
                event_sampler.sample_medications_simd(med_thresholds, rng);

                // Estimate encounter count (simplified inline)
                let mean = archetype.mean_encounters;
                let encounter_count = (mean + (rng.gen::<f32>() - 0.5) * mean.sqrt() * 2.0)
                    .max(1.0)
                    .min(self.config.max_encounters as f32)
                    as u64;

                // Sample observations and procedures in batch
                event_sampler.sample_events_batch(
                    obs_freqs,
                    proc_freqs,
                    encounter_count as u32,
                    rng,
                );

                // Estimate total events (simplified inline)
                let events_per = archetype.mean_events_per_encounter;
                let event_count = (encounter_count as f32 * events_per) as u64;

                // Record statistics directly from sampler buffers
                stats.record_full(
                    condition_buffer,
                    event_sampler.medications(),
                    event_sampler.accumulated_observations(),
                    event_sampler.accumulated_procedures(),
                    encounter_count,
                    event_count,
                );
            },
        );

        stats.to_streaming()
    }

    /// Generates full patients with complete encounter and event data.
    ///
    /// This is slower than generate_stats_only but produces complete patient
    /// records with medications, observations, and procedures.
    pub fn generate_full(&self, count: usize) -> Vec<FullPatient> {
        let base_seed = self.config.seed;
        let cooccurrence = Arc::clone(&self.cooccurrence);
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
                    )
                },
                |(rng, condition_buffer, event_sampler), patient_id| {
                    // Deterministic per-patient seed
                    let patient_seed = base_seed.wrapping_add(patient_id as u64);
                    *rng = Xoshiro256PlusPlus::seed_from_u64(patient_seed);

                    self.generate_full_patient(
                        patient_id as u64,
                        rng,
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
        condition_buffer: &mut SmallVec<[u16; 8]>,
        event_sampler: &mut EventSampler,
        cooccurrence: &CooccurrenceModel,
        archetypes: &ArchetypeRegistry,
    ) -> FullPatient {
        use crate::tables::{encounter_type_to_idx, ethnicity_to_idx, race_to_idx};
        use rand::Rng;

        // Sample archetype
        let archetype = archetypes.sample(rng);

        // Sample conditions with co-occurrence modeling
        if cooccurrence.is_empty() {
            archetypes.sample_conditions_flat(archetype.id, rng, condition_buffer);
        } else {
            archetype.sample_conditions_with_cooccurrence(rng, condition_buffer, cooccurrence);
        }

        // Sample age within archetype range
        let age = archetype.sample_age(rng);
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

        // Create patient
        let mut patient = FullPatient::new(id, birth_date_days, sex, race, ethnicity, archetype.id);
        patient.conditions = condition_buffer.clone();

        // Sample medications based on conditions
        let meds =
            event_sampler.sample_medications_for_conditions(condition_buffer, archetypes, rng);
        patient.medications = SmallVec::from_slice(meds);

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
            // Sample procedures for this encounter with O(1) bitset deduplication
            event_sampler.sample_procedures_accumulate(proc_freqs, rng);

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

            // Add condition diagnosis events (for chronic conditions, add to some encounters)
            for &cond_idx in condition_buffer.iter() {
                // Add condition to ~30% of encounters (for chronic) or just first encounter (acute)
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

            // Add medication events
            for &med_idx in patient.medications.iter() {
                // Medications appear in ~50% of encounters (chronic meds)
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

            // Add observation events (sample for this encounter)
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

            // Add procedure events from sampler's accumulated procedures
            for &proc_idx in event_sampler.accumulated_procedures() {
                // Procedures appear in ~20% of encounters
                if rng.gen::<f32>() < 0.2 {
                    encounter.add_event(CompactEvent {
                        event_type: 2, // procedure
                        system_idx: 0, // SNOMED-CT
                        code_idx: proc_idx,
                        display_idx: proc_idx,
                        timestamp_offset: 0,
                    });
                }
            }

            patient.encounters.push(encounter);
        }

        // Copy accumulated procedures to patient
        patient.procedures = SmallVec::from_slice(event_sampler.accumulated_procedures());

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
