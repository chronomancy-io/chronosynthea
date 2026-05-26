//! Arena-based zero-allocation generation.
//!
//! This module provides bump allocators for each worker thread,
//! enabling O(1) batch resets and minimizing allocation overhead.

use std::cell::UnsafeCell;

use bumpalo::Bump;
use smallvec::SmallVec;

/// Per-worker arena for zero-allocation patient generation.
///
/// Each worker thread gets its own arena that is reset between batches,
/// avoiding global allocator contention and enabling O(1) batch resets.
pub struct WorkerArena {
    /// Bump allocator for patient data.
    bump: Bump,

    /// Pre-allocated scratch buffer for serialization.
    scratch: Vec<u8>,

    /// Pre-allocated condition buffer (avoids per-patient allocation).
    condition_buffer: Vec<u16>,

    /// Pre-allocated encounter buffer.
    encounter_buffer: Vec<CompactEncounter>,

    /// Pre-allocated event buffer.
    event_buffer: Vec<CompactEvent>,

    /// Statistics accumulator (no allocation during generation).
    stats: ArenaStats,
}

/// Compact encounter for arena allocation (fixed size).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct CompactEncounter {
    /// Timestamp as days since epoch.
    pub timestamp_days: u16,
    /// Encounter type index.
    pub encounter_type: u8,
    /// Number of events.
    pub event_count: u8,
    /// Start index in event buffer.
    pub event_start: u16,
    /// Padding for alignment.
    _padding: u16,
}

/// Ultra-compact event (8 bytes total).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(8))]
pub struct CompactEvent {
    /// Event type (diagnosis=0, medication=1, procedure=2, observation=3, immunization=4).
    pub event_type: u8,
    /// System index (SNOMED=0, RxNorm=1, LOINC=2, CPT=3).
    pub system_idx: u8,
    /// Code index into static code table.
    pub code_idx: u16,
    /// Display index into static display table.
    pub display_idx: u16,
    /// Timestamp offset from encounter (in hours, max ~2.7 years).
    pub timestamp_offset: u16,
}

// Ensure CompactEvent is exactly 8 bytes
const _: () = assert!(std::mem::size_of::<CompactEvent>() == 8);

/// Compact patient for arena allocation.
#[derive(Debug, Clone)]
pub struct CompactPatient {
    /// Patient ID (monotonic counter).
    pub id: u64,
    /// Birth date as days since epoch (1970-01-01).
    pub birth_date_days: i32,
    /// Sex (0=male, 1=female).
    pub sex: u8,
    /// Race index.
    pub race: u8,
    /// Ethnicity index.
    pub ethnicity: u8,
    /// Number of encounters.
    pub encounter_count: u8,
    /// Number of conditions.
    pub condition_count: u8,
    /// Archetype ID.
    pub archetype_id: crate::types::ArchetypeId,
    /// Condition indices (stored inline for cache efficiency).
    pub conditions: SmallVec<[u16; 8]>,
    /// Per-condition onset age in days since birth — parallel to `conditions`.
    /// Empty when the d5 axis value is not `temporal-ordered` (no onset_stats
    /// loaded). When populated, `conditions[i]` onset at
    /// `birth_date_days + condition_onset_days[i]`, and the pair is sorted
    /// ascending by onset so consumers can walk the patient's trajectory in
    /// temporal order without re-sorting.
    pub condition_onset_days: SmallVec<[u16; 8]>,
}

/// Full patient with complete encounter and event data.
#[derive(Debug, Clone)]
pub struct FullPatient {
    /// Patient ID (monotonic counter).
    pub id: u64,
    /// Birth date as days since epoch (1970-01-01).
    pub birth_date_days: i32,
    /// Sex (0=male, 1=female).
    pub sex: u8,
    /// Race index.
    pub race: u8,
    /// Ethnicity index.
    pub ethnicity: u8,
    /// Archetype ID.
    pub archetype_id: crate::types::ArchetypeId,
    /// Condition indices.
    pub conditions: SmallVec<[u16; 8]>,
    /// Per-condition onset, days since the patient's birth. Parallel to
    /// `conditions`, sorted ascending so the patient's trajectory walks in
    /// temporal order. Mirrors `CompactPatient::condition_onset_days`.
    pub condition_onset_days: SmallVec<[u16; 8]>,
    /// Medication indices.
    pub medications: SmallVec<[u16; 8]>,
    /// REASONCODE per medication — the condition index that caused each
    /// prescription. Parallel to `medications`. `u16::MAX` means "no
    /// active condition is an indication" (prophylactic / routine
    /// prescription, like Java Synthea's empty REASONCODE rows).
    pub medication_causes: SmallVec<[u16; 8]>,
    /// Procedure indices.
    pub procedures: SmallVec<[u16; 8]>,
    /// REASONCODE per procedure — same shape as `medication_causes` but
    /// for the procedure list. Parallel to `procedures`.
    pub procedure_causes: SmallVec<[u16; 8]>,
    /// Encounters with their events. Inline capacity 16 (was 8): the
    /// empirical encounter-per-patient mean is ~12, so the prior cap-8
    /// SmallVec spilled to heap for the majority of patients; bumping
    /// to 16 keeps the encounter array inline for ~80% of cases at the
    /// cost of ~264 bytes/patient stack/arena footprint — well worth it
    /// to eliminate the pointer-chase in the encounter loop.
    pub encounters: SmallVec<[FullEncounter; 16]>,
}

/// Full encounter with event details.
///
/// Events are partitioned by type at sample-time into typed
/// sub-collections rather than living in one mixed vec. The CSV-write
/// path used to do three filtered linear scans per encounter
/// (`events.iter().filter(event_type == X)` for medications, procedures,
/// observations) — at ~22 events × 3 scans × 12 encounters × 10k
/// patients that was ~8M filter comparisons per run. With the typed
/// split below those become direct iterations.
#[derive(Debug, Clone)]
pub struct FullEncounter {
    /// Encounter type index.
    pub encounter_type: u8,
    /// Timestamp as days since patient birth.
    pub days_since_birth: u16,
    /// Medication events (was event_type == 1).
    pub medications: SmallVec<[CompactEvent; 8]>,
    /// Procedure events (was event_type == 2). Inline-cap 16 covers
    /// most real encounters; the few outliers spill to heap.
    pub procedures: SmallVec<[CompactEvent; 16]>,
    /// Observation events (was event_type == 3). Inline-cap 16 covers
    /// vitals + condition-triggered labs typically.
    pub observations: SmallVec<[CompactEvent; 16]>,
}

impl FullPatient {
    /// Creates a new full patient with the given demographics.
    pub fn new(
        id: u64,
        birth_date_days: i32,
        sex: u8,
        race: u8,
        ethnicity: u8,
        archetype_id: crate::types::ArchetypeId,
    ) -> Self {
        Self {
            id,
            birth_date_days,
            sex,
            race,
            ethnicity,
            archetype_id,
            conditions: SmallVec::new(),
            condition_onset_days: SmallVec::new(),
            medications: SmallVec::new(),
            medication_causes: SmallVec::new(),
            procedures: SmallVec::new(),
            procedure_causes: SmallVec::new(),
            encounters: SmallVec::new(),
        }
    }

    /// Returns the total number of events across all encounters.
    pub fn total_events(&self) -> usize {
        self.encounters.iter().map(|e| e.total_events()).sum()
    }

    /// Returns the number of encounters.
    pub fn num_encounters(&self) -> usize {
        self.encounters.len()
    }

    /// Counts events by type. Diagnoses are computed from the patient's
    /// condition list directly (the per-encounter diagnosis events were
    /// dropped when event_type became a coordinate axis — see
    /// `FullEncounter::add_event`). Immunizations are emitted from the
    /// CDC schedule rather than from per-encounter records.
    pub fn event_counts(&self) -> EventCounts {
        let mut counts = EventCounts::default();
        counts.diagnoses = self.conditions.len() as u32;
        for encounter in &self.encounters {
            counts.medications += encounter.medications.len() as u32;
            counts.procedures += encounter.procedures.len() as u32;
            counts.observations += encounter.observations.len() as u32;
        }
        counts
    }
}

impl FullEncounter {
    /// Creates a new encounter.
    pub fn new(encounter_type: u8, days_since_birth: u16) -> Self {
        Self {
            encounter_type,
            days_since_birth,
            medications: SmallVec::new(),
            procedures: SmallVec::new(),
            observations: SmallVec::new(),
        }
    }

    /// Total events across all three typed sub-vecs (used by stats /
    /// throughput counters).
    #[inline]
    pub fn total_events(&self) -> usize {
        self.medications.len() + self.procedures.len() + self.observations.len()
    }

    /// Routes a `CompactEvent` to its typed sub-vec based on its
    /// `event_type` byte (1=medication, 2=procedure, 3=observation).
    /// Diagnosis (type 0) and immunization (type 4) events are dropped
    /// because nothing in the CSV writer or stats consumes them — the
    /// writer reads `patient.conditions` directly for diagnoses and
    /// emits immunizations from the CDC schedule, not from
    /// per-encounter event records.
    #[inline]
    pub fn add_event(&mut self, event: CompactEvent) {
        match event.event_type {
            1 => self.medications.push(event),
            2 => self.procedures.push(event),
            3 => self.observations.push(event),
            // 0 (diagnosis) and 4 (immunization) intentionally dropped;
            // see doc comment above.
            _ => {}
        }
    }
}

/// Event counts by type.
#[derive(Debug, Clone, Default)]
pub struct EventCounts {
    pub diagnoses: u32,
    pub medications: u32,
    pub procedures: u32,
    pub observations: u32,
    pub immunizations: u32,
}

impl EventCounts {
    /// Returns total events.
    pub fn total(&self) -> u32 {
        self.diagnoses + self.medications + self.procedures + self.observations + self.immunizations
    }
}

/// Per-arena statistics (no allocation).
#[derive(Debug, Clone, Default)]
pub struct ArenaStats {
    /// Patients generated in this arena.
    pub patients_generated: u64,
    /// Total encounters generated.
    pub encounters_generated: u64,
    /// Total events generated.
    pub events_generated: u64,
    /// Condition occurrence counts (indexed by condition ID).
    pub condition_counts: Vec<u64>,
    /// Peak memory usage in bytes.
    pub peak_memory: usize,
}

impl WorkerArena {
    /// Creates a new worker arena with pre-allocated buffers.
    pub fn new(max_conditions: usize) -> Self {
        Self {
            bump: Bump::with_capacity(1024 * 1024), // 1MB initial
            scratch: Vec::with_capacity(64 * 1024), // 64KB scratch
            condition_buffer: Vec::with_capacity(32),
            encounter_buffer: Vec::with_capacity(32),
            event_buffer: Vec::with_capacity(256),
            stats: ArenaStats {
                condition_counts: vec![0; max_conditions],
                ..Default::default()
            },
        }
    }

    /// Creates with specified capacity.
    pub fn with_capacity(bump_capacity: usize, max_conditions: usize) -> Self {
        Self {
            bump: Bump::with_capacity(bump_capacity),
            scratch: Vec::with_capacity(64 * 1024),
            condition_buffer: Vec::with_capacity(32),
            encounter_buffer: Vec::with_capacity(32),
            event_buffer: Vec::with_capacity(256),
            stats: ArenaStats {
                condition_counts: vec![0; max_conditions],
                ..Default::default()
            },
        }
    }

    /// Resets the arena for the next batch.
    ///
    /// This is O(1) - just resets the bump pointer without freeing memory.
    #[inline]
    pub fn reset(&mut self) {
        self.bump.reset();
        self.condition_buffer.clear();
        self.encounter_buffer.clear();
        self.event_buffer.clear();
        // Note: scratch and stats are intentionally NOT reset
    }

    /// Resets statistics (call between batches if needed).
    pub fn reset_stats(&mut self) {
        self.stats.patients_generated = 0;
        self.stats.encounters_generated = 0;
        self.stats.events_generated = 0;
        self.stats.condition_counts.fill(0);
    }

    /// Returns a mutable reference to the condition buffer.
    #[inline]
    pub fn condition_buffer(&mut self) -> &mut Vec<u16> {
        &mut self.condition_buffer
    }

    /// Returns a mutable reference to the encounter buffer.
    #[inline]
    pub fn encounter_buffer(&mut self) -> &mut Vec<CompactEncounter> {
        &mut self.encounter_buffer
    }

    /// Returns a mutable reference to the event buffer.
    #[inline]
    pub fn event_buffer(&mut self) -> &mut Vec<CompactEvent> {
        &mut self.event_buffer
    }

    /// Returns a mutable reference to the scratch buffer.
    #[inline]
    pub fn scratch_buffer(&mut self) -> &mut Vec<u8> {
        &mut self.scratch
    }

    /// Allocates a slice in the bump arena.
    #[inline]
    pub fn alloc_slice<T: Copy>(&self, slice: &[T]) -> &[T] {
        self.bump.alloc_slice_copy(slice)
    }

    /// Allocates a value in the bump arena.
    #[inline]
    pub fn alloc<T>(&self, value: T) -> &mut T {
        self.bump.alloc(value)
    }

    /// Records a patient generation.
    #[inline]
    pub fn record_patient(
        &mut self,
        condition_ids: &[u16],
        encounter_count: usize,
        event_count: usize,
    ) {
        self.stats.patients_generated += 1;
        self.stats.encounters_generated += encounter_count as u64;
        self.stats.events_generated += event_count as u64;

        for &cond_id in condition_ids {
            if (cond_id as usize) < self.stats.condition_counts.len() {
                self.stats.condition_counts[cond_id as usize] += 1;
            }
        }
    }

    /// Returns the current statistics.
    pub fn stats(&self) -> &ArenaStats {
        &self.stats
    }

    /// Returns mutable statistics for merging.
    pub fn stats_mut(&mut self) -> &mut ArenaStats {
        &mut self.stats
    }

    /// Returns the current memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.bump.allocated_bytes()
    }

    /// Updates peak memory tracking.
    pub fn update_peak_memory(&mut self) {
        let current = self.memory_usage();
        if current > self.stats.peak_memory {
            self.stats.peak_memory = current;
        }
    }
}

/// Pool of worker arenas for parallel generation.
pub struct ArenaPool {
    arenas: Vec<UnsafeCell<WorkerArena>>,
    max_conditions: usize,
}

// Safety: Each arena is accessed by exactly one thread at a time
unsafe impl Sync for ArenaPool {}

impl ArenaPool {
    /// Creates a pool with one arena per worker.
    pub fn new(num_workers: usize, max_conditions: usize) -> Self {
        let arenas = (0..num_workers)
            .map(|_| UnsafeCell::new(WorkerArena::new(max_conditions)))
            .collect();

        Self {
            arenas,
            max_conditions,
        }
    }

    /// Gets the arena for a specific worker.
    ///
    /// # Safety
    /// Caller must ensure no other thread accesses this arena simultaneously.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get_arena(&self, worker_id: usize) -> &mut WorkerArena {
        debug_assert!(worker_id < self.arenas.len());
        &mut *self.arenas[worker_id].get()
    }

    /// Returns the number of arenas.
    pub fn len(&self) -> usize {
        self.arenas.len()
    }

    /// Returns whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.arenas.is_empty()
    }

    /// Merges statistics from all arenas.
    pub fn merge_stats(&self) -> ArenaStats {
        let mut merged = ArenaStats {
            condition_counts: vec![0; self.max_conditions],
            ..Default::default()
        };

        for arena_cell in &self.arenas {
            // Safety: We're the only one accessing during merge
            let arena = unsafe { &*arena_cell.get() };
            let stats = arena.stats();

            merged.patients_generated += stats.patients_generated;
            merged.encounters_generated += stats.encounters_generated;
            merged.events_generated += stats.events_generated;
            merged.peak_memory = merged.peak_memory.max(stats.peak_memory);

            for (i, &count) in stats.condition_counts.iter().enumerate() {
                if i < merged.condition_counts.len() {
                    merged.condition_counts[i] += count;
                }
            }
        }

        merged
    }

    /// Resets all arenas.
    pub fn reset_all(&self) {
        for arena_cell in &self.arenas {
            // Safety: We're the only one accessing during reset
            let arena = unsafe { &mut *arena_cell.get() };
            arena.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compact_event_size() {
        assert_eq!(std::mem::size_of::<CompactEvent>(), 8);
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = WorkerArena::new(100);

        // Allocate some data
        arena.condition_buffer().push(1);
        arena.condition_buffer().push(2);
        arena.encounter_buffer().push(CompactEncounter::default());

        assert_eq!(arena.condition_buffer().len(), 2);
        assert_eq!(arena.encounter_buffer().len(), 1);

        // Reset
        arena.reset();

        assert!(arena.condition_buffer().is_empty());
        assert!(arena.encounter_buffer().is_empty());
    }

    #[test]
    fn test_arena_stats() {
        let mut arena = WorkerArena::new(10);

        arena.record_patient(&[0, 1, 2], 3, 10);
        arena.record_patient(&[1, 2], 2, 8);

        assert_eq!(arena.stats().patients_generated, 2);
        assert_eq!(arena.stats().encounters_generated, 5);
        assert_eq!(arena.stats().events_generated, 18);
        assert_eq!(arena.stats().condition_counts[0], 1);
        assert_eq!(arena.stats().condition_counts[1], 2);
        assert_eq!(arena.stats().condition_counts[2], 2);
    }

    #[test]
    fn test_arena_pool() {
        let pool = ArenaPool::new(4, 100);
        assert_eq!(pool.len(), 4);

        unsafe {
            let arena = pool.get_arena(0);
            arena.record_patient(&[0], 1, 5);
        }

        let stats = pool.merge_stats();
        assert_eq!(stats.patients_generated, 1);
    }
}
