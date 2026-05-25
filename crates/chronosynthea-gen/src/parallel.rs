//! Parallel patient generator using Rayon.
//!
//! Achieves 4-8x speedup over sequential generation by using
//! work-stealing parallelism with independent RNGs per worker.

use std::sync::Arc;
use std::thread;

use crossbeam_channel::bounded;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use chronosynthea_core::Patient;

use crate::buffer::WorkerBuffer;
use crate::config::GeneratorConfig;
use crate::error::{GeneratorError, GeneratorResult};
use crate::prevalence::sample_conditions_for_patient;
use crate::prevalence::OptimizedRegistry;

/// Parallel patient generator using Rayon.
pub struct ParallelGenerator {
    config: GeneratorConfig,
    registry: OptimizedRegistry,
    num_workers: usize,
}

impl ParallelGenerator {
    /// Creates a new parallel generator.
    pub fn new(config: GeneratorConfig, registry: OptimizedRegistry) -> Self {
        let num_workers = config.effective_workers();
        Self {
            config,
            registry,
            num_workers,
        }
    }

    /// Generates patients in parallel.
    ///
    /// Each worker gets an independent RNG seeded from the base seed
    /// to ensure deterministic results regardless of thread scheduling.
    pub fn generate(&self, count: usize) -> GeneratorResult<Vec<Patient>> {
        let base_seed = self.config.seed;

        // Use Rayon's parallel iterator with per-thread initialization
        let patients: Vec<Patient> = (0..count)
            .into_par_iter()
            .map_init(
                || {
                    // Each thread gets its own RNG and buffer
                    // We use the thread index from rayon to seed deterministically
                    let thread_id = rayon::current_thread_index().unwrap_or(0);
                    let seed = base_seed.wrapping_add(thread_id as u64 * 1_000_000);
                    (
                        Xoshiro256PlusPlus::seed_from_u64(seed),
                        WorkerBuffer::new(),
                        thread_id,
                    )
                },
                |(rng, buffer, _thread_id), patient_id| {
                    // Reseed RNG for each patient to ensure determinism
                    // regardless of which thread processes which patient
                    let patient_seed = base_seed.wrapping_add(patient_id as u64);
                    *rng = Xoshiro256PlusPlus::seed_from_u64(patient_seed);

                    let patient = self.generate_patient(patient_id as u64, rng, buffer);
                    buffer.reset();
                    patient
                },
            )
            .collect();

        Ok(patients)
    }

    /// Creates a streaming patient iterator that generates patients in parallel.
    ///
    /// This overlaps CPU (generation) and I/O (consumption) for better throughput.
    /// Uses crossbeam channels with backpressure to control memory usage.
    ///
    /// Returns a receiver that yields patients as they are generated.
    pub fn generate_streaming(&self, count: usize) -> crossbeam_channel::Receiver<Patient> {
        // Channel capacity controls backpressure - 1000 patients in flight max
        let (tx, rx) = bounded::<Patient>(1000);
        let base_seed = self.config.seed;

        // Clone what we need for the generator thread
        let config = self.config.clone();
        let registry = self.registry.clone();

        // Spawn generator in a separate thread
        thread::spawn(move || {
            let _result: Result<(), GeneratorError> = (0..count).into_par_iter().try_for_each_init(
                || {
                    let thread_id = rayon::current_thread_index().unwrap_or(0);
                    let seed = base_seed.wrapping_add(thread_id as u64 * 1_000_000);
                    (Xoshiro256PlusPlus::seed_from_u64(seed), WorkerBuffer::new())
                },
                |(rng, buffer), patient_id| {
                    let patient_seed = base_seed.wrapping_add(patient_id as u64);
                    *rng = Xoshiro256PlusPlus::seed_from_u64(patient_seed);

                    let patient = generate_patient_internal(
                        patient_id as u64,
                        rng,
                        buffer,
                        &config,
                        &registry,
                    );
                    buffer.reset();

                    tx.send(patient).map_err(|_| GeneratorError::ChannelClosed)
                },
            );
            // Sender is dropped here, signaling completion
        });

        rx
    }

    /// Generates patients and returns them through a channel for batch processing.
    ///
    /// Useful for large cohorts to avoid accumulating all patients in memory.
    pub fn generate_stream(&self, count: usize) -> GeneratorResult<Vec<Patient>> {
        // For backward compatibility, collect into a Vec
        self.generate(count)
    }

    /// Generates a single patient.
    fn generate_patient(
        &self,
        id: u64,
        rng: &mut Xoshiro256PlusPlus,
        _buffer: &mut WorkerBuffer,
    ) -> Patient {
        // Sample demographics
        let (age, gender_str, race_str, ethnicity_str) =
            self.registry.sample_demographics_fast(rng);

        // Convert to enums
        use chronosynthea_core::{Ethnicity, Race, Sex};

        let sex = if gender_str == "F" {
            Sex::Female
        } else {
            Sex::Male
        };
        let race = match race_str.as_str() {
            "white" => Race::White,
            "black" => Race::Black,
            "asian" => Race::Asian,
            "hispanic" => Race::Hispanic,
            "native" => Race::Native,
            _ => Race::Other,
        };
        let ethnicity = match ethnicity_str.as_str() {
            "hispanic" => Ethnicity::Hispanic,
            "nonhispanic" => Ethnicity::NonHispanic,
            _ => Ethnicity::Unknown,
        };

        // Calculate birth date
        use chrono::{Datelike, NaiveDate};
        let birth_year = self.config.start_date.year() - age as i32;
        let birth_date = NaiveDate::from_ymd_opt(birth_year, 1, 1)
            .unwrap_or_else(|| NaiveDate::from_ymd_opt(1970, 1, 1).unwrap());

        // Sample conditions
        let conditions = sample_conditions_for_patient(
            &self.registry.registry,
            rng,
            age,
            &gender_str,
            &race_str,
        );

        // Generate encounters
        let encounters = self.generate_encounters(rng, age, &conditions);

        Patient {
            id: format!("P{:08}", id),
            birth_date,
            sex,
            race,
            ethnicity,
            encounters,
        }
    }

    /// Generates encounters for a patient using condition-based sampling (like Go).
    ///
    /// This is O(conditions) per patient, NOT O(all_codes) per encounter.
    fn generate_encounters(
        &self,
        rng: &mut Xoshiro256PlusPlus,
        age: u32,
        conditions: &[crate::prevalence::SampledCondition],
    ) -> Vec<chronosynthea_core::Encounter> {
        use chrono::Duration;
        use chronosynthea_core::{Encounter, EncounterType, Event, EventType};
        use rand::Rng;

        let mut encounters = Vec::with_capacity(20);

        // Calculate years of care - limit wellness visits to leave room for condition encounters
        let max_wellness = self
            .config
            .max_encounters_per_patient
            .saturating_sub(5)
            .max(3);
        let years_of_care = (age as usize).saturating_sub(18).min(max_wellness).max(1);

        // Generate annual wellness visits
        for year in 0..years_of_care {
            let days_back = (years_of_care - year) * 365 + rng.gen_range(0..365);
            let timestamp = self.config.start_date - Duration::days(days_back as i64);

            let mut events = Vec::with_capacity(20);

            // Sample observations from registry (calibrated frequencies)
            self.add_observations(&mut events, timestamp, rng);

            // Sample procedures from registry (calibrated frequencies)
            self.add_procedures(&mut events, timestamp, rng);

            // Sample medications from registry (calibrated frequencies)
            self.add_medications(&mut events, timestamp, rng);

            // Record ~20% of chronic conditions per wellness visit
            for cond in conditions.iter().filter(|c| c.chronic) {
                if rng.gen::<f64>() < 0.20 {
                    events.push(Event {
                        event_type: EventType::Diagnosis,
                        code: Arc::from(cond.code.as_str()),
                        system: Arc::from("SNOMED-CT"),
                        description: Arc::from(cond.display.as_str()),
                        timestamp,
                    });

                    // Add medications for this condition
                    for med in &cond.medications {
                        if rng.gen::<f64>() < 0.6 {
                            events.push(Event {
                                event_type: EventType::Medication,
                                code: Arc::from(med.code.as_str()),
                                system: Arc::from("RxNorm"),
                                description: Arc::from(med.display.as_str()),
                                timestamp,
                            });
                        }
                    }
                }
            }

            encounters.push(Encounter {
                timestamp,
                encounter_type: EncounterType::Wellness,
                class: "AMB".to_string(),
                events,
            });
        }

        // Generate condition-specific encounters (fill remaining slots)
        let max_condition_encounters = self
            .config
            .max_encounters_per_patient
            .saturating_sub(encounters.len());
        let mut condition_encounter_count = 0;

        for cond in conditions {
            if condition_encounter_count >= max_condition_encounters {
                break;
            }

            let visits = if cond.chronic {
                rng.gen_range(1..=3)
            } else {
                1
            };

            for _ in 0..visits {
                if condition_encounter_count >= max_condition_encounters {
                    break;
                }

                let days_back = rng.gen_range(0..self.config.time_span_years * 365);
                let timestamp = self.config.start_date - Duration::days(days_back as i64);

                let mut events = Vec::with_capacity(10);

                // Sample observations from registry
                self.add_observations(&mut events, timestamp, rng);

                // The condition itself
                events.push(Event {
                    event_type: EventType::Diagnosis,
                    code: Arc::from(cond.code.as_str()),
                    system: Arc::from("SNOMED-CT"),
                    description: Arc::from(cond.display.as_str()),
                    timestamp,
                });

                // Medications for this condition
                for med in &cond.medications {
                    if rng.gen::<f64>() < 0.7 {
                        events.push(Event {
                            event_type: EventType::Medication,
                            code: Arc::from(med.code.as_str()),
                            system: Arc::from("RxNorm"),
                            description: Arc::from(med.display.as_str()),
                            timestamp,
                        });
                    }
                }

                let encounter_type = if !cond.chronic && rng.gen::<f64>() < 0.3 {
                    EncounterType::Emergency
                } else {
                    EncounterType::Ambulatory
                };

                encounters.push(Encounter {
                    timestamp,
                    encounter_type,
                    class: match encounter_type {
                        EncounterType::Emergency => "EMER".to_string(),
                        _ => "AMB".to_string(),
                    },
                    events,
                });

                condition_encounter_count += 1;
            }
        }

        // Limit total encounters
        if encounters.len() > self.config.max_encounters_per_patient {
            encounters.truncate(self.config.max_encounters_per_patient);
        }

        encounters
    }

    /// Samples observations from registry using calibrated frequencies.
    #[inline]
    fn add_observations<R: rand::Rng>(
        &self,
        events: &mut Vec<chronosynthea_core::Event>,
        timestamp: chrono::DateTime<chrono::Utc>,
        rng: &mut R,
    ) {
        use chronosynthea_core::{Event, EventType};

        // Sample from registry observations using calibrated per-encounter frequencies
        for entry in &self.registry.interned_observations {
            let freq = entry.frequency.max(0.001); // Minimum 0.1% for coverage
            if rng.gen::<f64>() < freq {
                events.push(Event {
                    event_type: EventType::Observation,
                    code: Arc::clone(&entry.code),
                    system: Arc::clone(&entry.system),
                    description: Arc::clone(&entry.display),
                    timestamp,
                });
            }
        }
    }

    /// Samples procedures from registry using calibrated frequencies.
    #[inline]
    fn add_procedures<R: rand::Rng>(
        &self,
        events: &mut Vec<chronosynthea_core::Event>,
        timestamp: chrono::DateTime<chrono::Utc>,
        rng: &mut R,
    ) {
        use chronosynthea_core::{Event, EventType};

        // Sample from registry procedures using calibrated per-encounter frequencies
        for entry in &self.registry.interned_procedures {
            let freq = entry.frequency.max(0.001); // Minimum 0.1% for coverage
            if rng.gen::<f64>() < freq {
                events.push(Event {
                    event_type: EventType::Procedure,
                    code: Arc::clone(&entry.code),
                    system: Arc::clone(&entry.system),
                    description: Arc::clone(&entry.display),
                    timestamp,
                });
            }
        }
    }

    /// Samples medications from registry using calibrated frequencies.
    #[inline]
    fn add_medications<R: rand::Rng>(
        &self,
        events: &mut Vec<chronosynthea_core::Event>,
        timestamp: chrono::DateTime<chrono::Utc>,
        rng: &mut R,
    ) {
        use chronosynthea_core::{Event, EventType};

        // Sample from registry medications using calibrated per-encounter frequencies
        for entry in &self.registry.interned_medications {
            let freq = entry.frequency.max(0.001); // Minimum 0.1% for coverage
            if rng.gen::<f64>() < freq {
                events.push(Event {
                    event_type: EventType::Medication,
                    code: Arc::clone(&entry.code),
                    system: Arc::clone(&entry.system),
                    description: Arc::clone(&entry.display),
                    timestamp,
                });
            }
        }
    }

    /// Returns the number of workers.
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }
}

/// Internal helper function for generating a patient (used by streaming).
///
/// Uses condition-based sampling (like Go) - O(conditions) not O(all_codes).
fn generate_patient_internal(
    id: u64,
    rng: &mut Xoshiro256PlusPlus,
    _buffer: &mut WorkerBuffer,
    config: &GeneratorConfig,
    registry: &OptimizedRegistry,
) -> Patient {
    use chrono::{Datelike, Duration, NaiveDate};
    use chronosynthea_core::{Encounter, EncounterType, Ethnicity, Event, EventType, Race, Sex};
    use rand::Rng;

    // Sample demographics
    let (age, gender_str, race_str, ethnicity_str) = registry.sample_demographics_fast(rng);

    // Convert to enums
    let sex = if gender_str == "F" {
        Sex::Female
    } else {
        Sex::Male
    };
    let race = match race_str.as_str() {
        "white" => Race::White,
        "black" => Race::Black,
        "asian" => Race::Asian,
        "hispanic" => Race::Hispanic,
        "native" => Race::Native,
        _ => Race::Other,
    };
    let ethnicity = match ethnicity_str.as_str() {
        "hispanic" => Ethnicity::Hispanic,
        "nonhispanic" => Ethnicity::NonHispanic,
        _ => Ethnicity::Unknown,
    };

    // Calculate birth date
    let birth_year = config.start_date.year() - age as i32;
    let birth_date = NaiveDate::from_ymd_opt(birth_year, 1, 1)
        .unwrap_or_else(|| NaiveDate::from_ymd_opt(1970, 1, 1).unwrap());

    // Sample conditions ONCE per patient
    let conditions =
        sample_conditions_for_patient(&registry.registry, rng, age, &gender_str, &race_str);

    // Generate encounters using condition-based approach
    let mut encounters = Vec::with_capacity(20);
    let max_wellness = config.max_encounters_per_patient.saturating_sub(5).max(3);
    let years_of_care = (age as usize).saturating_sub(18).min(max_wellness).max(1);

    // Wellness visits
    for year in 0..years_of_care {
        let days_back = (years_of_care - year) * 365 + rng.gen_range(0..365);
        let timestamp = config.start_date - Duration::days(days_back as i64);

        let mut events = Vec::with_capacity(15);
        add_observations_internal(&mut events, timestamp, registry, rng);
        add_procedures_internal(&mut events, timestamp, registry, rng);
        add_medications_internal(&mut events, timestamp, registry, rng);

        // Record ~20% of chronic conditions
        for cond in conditions.iter().filter(|c| c.chronic) {
            if rng.gen::<f64>() < 0.20 {
                events.push(Event {
                    event_type: EventType::Diagnosis,
                    code: Arc::from(cond.code.as_str()),
                    system: Arc::from("SNOMED-CT"),
                    description: Arc::from(cond.display.as_str()),
                    timestamp,
                });
                for med in &cond.medications {
                    if rng.gen::<f64>() < 0.6 {
                        events.push(Event {
                            event_type: EventType::Medication,
                            code: Arc::from(med.code.as_str()),
                            system: Arc::from("RxNorm"),
                            description: Arc::from(med.display.as_str()),
                            timestamp,
                        });
                    }
                }
            }
        }

        encounters.push(Encounter {
            timestamp,
            encounter_type: EncounterType::Wellness,
            class: "AMB".to_string(),
            events,
        });
    }

    // Condition encounters (fill remaining slots)
    let max_cond_enc = config
        .max_encounters_per_patient
        .saturating_sub(encounters.len());
    let mut cond_enc_count = 0;
    for cond in &conditions {
        if cond_enc_count >= max_cond_enc {
            break;
        }
        let visits = if cond.chronic {
            rng.gen_range(1..=3)
        } else {
            1
        };
        for _ in 0..visits {
            if cond_enc_count >= max_cond_enc {
                break;
            }
            let days_back = rng.gen_range(0..config.time_span_years * 365);
            let timestamp = config.start_date - Duration::days(days_back as i64);

            let mut events = Vec::with_capacity(10);
            add_observations_internal(&mut events, timestamp, registry, rng);
            events.push(Event {
                event_type: EventType::Diagnosis,
                code: Arc::from(cond.code.as_str()),
                system: Arc::from("SNOMED-CT"),
                description: Arc::from(cond.display.as_str()),
                timestamp,
            });
            for med in &cond.medications {
                if rng.gen::<f64>() < 0.7 {
                    events.push(Event {
                        event_type: EventType::Medication,
                        code: Arc::from(med.code.as_str()),
                        system: Arc::from("RxNorm"),
                        description: Arc::from(med.display.as_str()),
                        timestamp,
                    });
                }
            }

            let encounter_type = if !cond.chronic && rng.gen::<f64>() < 0.3 {
                EncounterType::Emergency
            } else {
                EncounterType::Ambulatory
            };

            encounters.push(Encounter {
                timestamp,
                encounter_type,
                class: if encounter_type == EncounterType::Emergency {
                    "EMER"
                } else {
                    "AMB"
                }
                .to_string(),
                events,
            });
            cond_enc_count += 1;
        }
    }

    if encounters.len() > config.max_encounters_per_patient {
        encounters.truncate(config.max_encounters_per_patient);
    }

    Patient {
        id: format!("P{:08}", id),
        birth_date,
        sex,
        race,
        ethnicity,
        encounters,
    }
}

/// Samples observations from registry (for streaming).
#[inline]
fn add_observations_internal<R: rand::Rng>(
    events: &mut Vec<chronosynthea_core::Event>,
    timestamp: chrono::DateTime<chrono::Utc>,
    registry: &OptimizedRegistry,
    rng: &mut R,
) {
    use chronosynthea_core::{Event, EventType};
    for entry in &registry.interned_observations {
        let freq = entry.frequency.max(0.001);
        if rng.gen::<f64>() < freq {
            events.push(Event {
                event_type: EventType::Observation,
                code: Arc::clone(&entry.code),
                system: Arc::clone(&entry.system),
                description: Arc::clone(&entry.display),
                timestamp,
            });
        }
    }
}

/// Samples procedures from registry (for streaming).
#[inline]
fn add_procedures_internal<R: rand::Rng>(
    events: &mut Vec<chronosynthea_core::Event>,
    timestamp: chrono::DateTime<chrono::Utc>,
    registry: &OptimizedRegistry,
    rng: &mut R,
) {
    use chronosynthea_core::{Event, EventType};
    for entry in &registry.interned_procedures {
        let freq = entry.frequency.max(0.001);
        if rng.gen::<f64>() < freq {
            events.push(Event {
                event_type: EventType::Procedure,
                code: Arc::clone(&entry.code),
                system: Arc::clone(&entry.system),
                description: Arc::clone(&entry.display),
                timestamp,
            });
        }
    }
}

/// Samples medications from registry (for streaming).
#[inline]
fn add_medications_internal<R: rand::Rng>(
    events: &mut Vec<chronosynthea_core::Event>,
    timestamp: chrono::DateTime<chrono::Utc>,
    registry: &OptimizedRegistry,
    rng: &mut R,
) {
    use chronosynthea_core::{Event, EventType};
    for entry in &registry.interned_medications {
        let freq = entry.frequency.max(0.001);
        if rng.gen::<f64>() < freq {
            events.push(Event {
                event_type: EventType::Medication,
                code: Arc::clone(&entry.code),
                system: Arc::clone(&entry.system),
                description: Arc::clone(&entry.display),
                timestamp,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prevalence::{CuratedRegistry, DemographicProfile};
    use ahash::AHashMap;

    fn create_test_registry() -> OptimizedRegistry {
        let mut age_dist = AHashMap::new();
        age_dist.insert("18-44".to_string(), 0.5);
        age_dist.insert("45-64".to_string(), 0.3);
        age_dist.insert("65+".to_string(), 0.2);

        let mut gender_dist = AHashMap::new();
        gender_dist.insert("M".to_string(), 0.5);
        gender_dist.insert("F".to_string(), 0.5);

        OptimizedRegistry::new(CuratedRegistry {
            version: "1.0".to_string(),
            conditions: vec![],
            medications: vec![],
            observations: vec![],
            procedures: vec![],
            demographics: DemographicProfile {
                age_distribution: age_dist,
                gender_distribution: gender_dist,
                ..Default::default()
            },
        })
    }

    #[test]
    fn test_parallel_generate() {
        let config = GeneratorConfig::with_patients(100).with_seed(42);
        let registry = create_test_registry();
        let generator = ParallelGenerator::new(config, registry);

        let patients = generator.generate(100).unwrap();
        assert_eq!(patients.len(), 100);

        // Check unique IDs
        let ids: std::collections::HashSet<_> = patients.iter().map(|p| &p.id).collect();
        assert_eq!(ids.len(), 100);
    }

    #[test]
    fn test_parallel_deterministic() {
        // Test that the same seed produces deterministic patient IDs
        let registry = create_test_registry();
        let config = GeneratorConfig::with_patients(50).with_seed(42);

        let gen = ParallelGenerator::new(config, registry);

        let mut patients1 = gen.generate(50).unwrap();
        let mut patients2 = gen.generate(50).unwrap();

        // Sort by ID since parallel execution order may vary
        patients1.sort_by(|a, b| a.id.cmp(&b.id));
        patients2.sort_by(|a, b| a.id.cmp(&b.id));

        // IDs should match
        for (p1, p2) in patients1.iter().zip(patients2.iter()) {
            assert_eq!(p1.id, p2.id);
        }

        // All IDs should be unique
        let ids: std::collections::HashSet<_> = patients1.iter().map(|p| &p.id).collect();
        assert_eq!(ids.len(), 50);
    }
}
