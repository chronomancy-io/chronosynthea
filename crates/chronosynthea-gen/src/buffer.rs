//! Buffer pooling for reduced allocations during patient generation.
//!
//! Uses pre-allocated buffers per worker to minimize GC pressure.

use chronosynthea_core::{Encounter, Event};

use crate::prevalence::SampledCondition;

/// Per-worker buffer for reusable allocations.
///
/// Each worker maintains its own buffer to avoid contention.
/// Buffers are reset (length set to 0) between patients but capacity is retained.
#[derive(Debug, Default)]
pub struct WorkerBuffer {
    /// Reusable condition buffer.
    pub conditions: Vec<SampledCondition>,

    /// Reusable encounter buffer.
    pub encounters: Vec<Encounter>,

    /// Reusable event buffer.
    pub events: Vec<Event>,

    /// Reusable string buffer for codes.
    pub codes: Vec<String>,
}

impl WorkerBuffer {
    /// Creates a new worker buffer with pre-allocated capacity.
    pub fn new() -> Self {
        Self {
            conditions: Vec::with_capacity(16),
            encounters: Vec::with_capacity(64),
            events: Vec::with_capacity(128),
            codes: Vec::with_capacity(32),
        }
    }

    /// Creates a worker buffer with custom capacities.
    pub fn with_capacity(
        conditions: usize,
        encounters: usize,
        events: usize,
        codes: usize,
    ) -> Self {
        Self {
            conditions: Vec::with_capacity(conditions),
            encounters: Vec::with_capacity(encounters),
            events: Vec::with_capacity(events),
            codes: Vec::with_capacity(codes),
        }
    }

    /// Resets all buffers, keeping allocated capacity.
    #[inline]
    pub fn reset(&mut self) {
        self.conditions.clear();
        self.encounters.clear();
        self.events.clear();
        self.codes.clear();
    }

    /// Returns the total capacity across all buffers.
    pub fn total_capacity(&self) -> usize {
        std::mem::size_of::<SampledCondition>() * self.conditions.capacity()
            + std::mem::size_of::<Encounter>() * self.encounters.capacity()
            + std::mem::size_of::<Event>() * self.events.capacity()
            + std::mem::size_of::<String>() * self.codes.capacity()
    }
}

/// Object pool for encounter slices.
pub struct EncounterPool {
    pool: Vec<Vec<Encounter>>,
    capacity: usize,
}

impl EncounterPool {
    /// Creates a new encounter pool.
    pub fn new(capacity: usize) -> Self {
        Self {
            pool: Vec::with_capacity(16),
            capacity,
        }
    }

    /// Gets an encounter vector from the pool or creates a new one.
    pub fn get(&mut self) -> Vec<Encounter> {
        self.pool
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(self.capacity))
    }

    /// Returns an encounter vector to the pool.
    pub fn put(&mut self, mut encounters: Vec<Encounter>) {
        encounters.clear();
        if self.pool.len() < 16 {
            self.pool.push(encounters);
        }
    }
}

/// Object pool for event slices.
pub struct EventPool {
    pool: Vec<Vec<Event>>,
    capacity: usize,
}

impl EventPool {
    /// Creates a new event pool.
    pub fn new(capacity: usize) -> Self {
        Self {
            pool: Vec::with_capacity(16),
            capacity,
        }
    }

    /// Gets an event vector from the pool or creates a new one.
    pub fn get(&mut self) -> Vec<Event> {
        self.pool
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(self.capacity))
    }

    /// Returns an event vector to the pool.
    pub fn put(&mut self, mut events: Vec<Event>) {
        events.clear();
        if self.pool.len() < 16 {
            self.pool.push(events);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_buffer_reset() {
        let mut buffer = WorkerBuffer::new();

        // Add some items
        buffer.codes.push("test".to_string());
        assert_eq!(buffer.codes.len(), 1);

        // Reset
        buffer.reset();
        assert_eq!(buffer.codes.len(), 0);
        assert!(buffer.codes.capacity() >= 32); // Capacity retained
    }

    #[test]
    fn test_encounter_pool() {
        let mut pool = EncounterPool::new(64);

        // Get a vector
        let enc = pool.get();
        assert!(enc.capacity() >= 64);

        // Return it
        pool.put(enc);
        assert_eq!(pool.pool.len(), 1);

        // Get it back
        let enc2 = pool.get();
        assert!(enc2.capacity() >= 64);
        assert_eq!(pool.pool.len(), 0);
    }
}
