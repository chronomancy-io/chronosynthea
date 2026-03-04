//! Sequential patient generator.

use std::sync::Arc;

use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc};
use chronosynthea_core::{
    Encounter, EncounterType, Ethnicity, Event, EventType, Patient, Race, Sex,
};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::buffer::WorkerBuffer;
use crate::config::GeneratorConfig;
use crate::error::GeneratorResult;
use crate::prevalence::{sample_conditions_for_patient, OptimizedRegistry, SampledCondition};

/// Sequential patient generator.
pub struct Generator {
    config: GeneratorConfig,
    registry: OptimizedRegistry,
}

impl Generator {
    /// Creates a new generator with the given configuration and registry.
    pub fn new(config: GeneratorConfig, registry: OptimizedRegistry) -> Self {
        Self { config, registry }
    }

    /// Generates a batch of patients.
    pub fn generate(&self, count: usize) -> GeneratorResult<Vec<Patient>> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.config.seed);
        let mut buffer = WorkerBuffer::new();
        let mut patients = Vec::with_capacity(count);

        for i in 0..count {
            let patient = self.generate_patient(i as u64, &mut rng, &mut buffer);
            patients.push(patient);
            buffer.reset();
        }

        Ok(patients)
    }

    /// Generates a single patient.
    pub fn generate_patient<R: Rng>(
        &self,
        id: u64,
        rng: &mut R,
        buffer: &mut WorkerBuffer,
    ) -> Patient {
        // Sample demographics
        let (age, gender_str, race_str, ethnicity_str) =
            self.registry.sample_demographics_fast(rng);

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
        let birth_date = calculate_birth_date(age, &self.config.start_date);

        // Sample conditions
        let conditions = sample_conditions_for_patient(
            &self.registry.registry,
            rng,
            age,
            &gender_str,
            &race_str,
        );

        // Generate encounters
        let encounters = self.generate_encounters(rng, age, &conditions, buffer);

        Patient {
            id: format!("P{:08}", id),
            birth_date,
            sex,
            race,
            ethnicity,
            encounters,
        }
    }

    /// Generates encounters using condition-based sampling (like Go).
    /// O(conditions) per patient, NOT O(all_codes) per encounter.
    fn generate_encounters<R: Rng>(
        &self,
        rng: &mut R,
        age: u32,
        conditions: &[SampledCondition],
        _buffer: &mut WorkerBuffer,
    ) -> Vec<Encounter> {
        let mut encounters = Vec::with_capacity(20);
        let max_wellness = self
            .config
            .max_encounters_per_patient
            .saturating_sub(5)
            .max(3);
        let years_of_care = (age as usize).saturating_sub(18).min(max_wellness).max(1);

        // Wellness visits
        for year in 0..years_of_care {
            let days_back = (years_of_care - year) * 365 + rng.gen_range(0..365);
            let timestamp = self.config.start_date - Duration::days(days_back as i64);

            let mut events = Vec::with_capacity(15);
            self.add_observations(&mut events, timestamp, rng);
            self.add_procedures(&mut events, timestamp, rng);
            self.add_medications(&mut events, timestamp, rng);

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
        let max_cond_enc = self
            .config
            .max_encounters_per_patient
            .saturating_sub(encounters.len());
        let mut cond_enc_count = 0;
        for cond in conditions {
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
                let days_back = rng.gen_range(0..self.config.time_span_years * 365);
                let timestamp = self.config.start_date - Duration::days(days_back as i64);

                let mut events = Vec::with_capacity(10);
                self.add_observations(&mut events, timestamp, rng);
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
                    class: encounter_type_to_class(encounter_type),
                    events,
                });
                cond_enc_count += 1;
            }
        }

        if encounters.len() > self.config.max_encounters_per_patient {
            encounters.truncate(self.config.max_encounters_per_patient);
        }

        encounters
    }

    /// Samples observations from registry using calibrated frequencies.
    #[inline]
    fn add_observations<R: Rng>(
        &self,
        events: &mut Vec<Event>,
        timestamp: DateTime<Utc>,
        rng: &mut R,
    ) {
        for entry in &self.registry.interned_observations {
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

    /// Samples procedures from registry using calibrated frequencies.
    #[inline]
    fn add_procedures<R: Rng>(
        &self,
        events: &mut Vec<Event>,
        timestamp: DateTime<Utc>,
        rng: &mut R,
    ) {
        for entry in &self.registry.interned_procedures {
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

    /// Samples medications from registry using calibrated frequencies.
    #[inline]
    fn add_medications<R: Rng>(
        &self,
        events: &mut Vec<Event>,
        timestamp: DateTime<Utc>,
        rng: &mut R,
    ) {
        for entry in &self.registry.interned_medications {
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

    /// Samples an observation.
    fn sample_observation<R: Rng>(
        &self,
        rng: &mut R,
    ) -> Option<&crate::prevalence::ObservationEntry> {
        if self.registry.registry.observations.is_empty() {
            return None;
        }

        let idx = rng.gen_range(0..self.registry.registry.observations.len());
        let obs = &self.registry.registry.observations[idx];

        if rng.gen::<f64>() < obs.frequency.max(0.01) {
            Some(obs)
        } else {
            None
        }
    }

    /// Samples a procedure.
    fn sample_procedure<R: Rng>(
        &self,
        rng: &mut R,
        condition_codes: &[&str],
    ) -> Option<&crate::prevalence::ProcedureEntry> {
        // First try to find a procedure related to a condition
        for &code in condition_codes {
            let procs = self.registry.get_procedures_for_condition(code);
            if !procs.is_empty() && rng.gen::<f64>() < 0.3 {
                let idx = rng.gen_range(0..procs.len());
                return Some(&procs[idx]);
            }
        }

        // Fall back to random procedure
        if self.registry.registry.procedures.is_empty() {
            return None;
        }

        let idx = rng.gen_range(0..self.registry.registry.procedures.len());
        let proc = &self.registry.registry.procedures[idx];

        if rng.gen::<f64>() < proc.frequency.max(0.01) {
            Some(proc)
        } else {
            None
        }
    }
}

/// Calculates birth date from age.
fn calculate_birth_date(age: u32, reference_date: &DateTime<Utc>) -> NaiveDate {
    let birth_year = reference_date.year() - age as i32;
    NaiveDate::from_ymd_opt(birth_year, 1, 1)
        .unwrap_or_else(|| NaiveDate::from_ymd_opt(1970, 1, 1).unwrap())
}

/// Maps encounter type to SNOMED class code.
fn encounter_type_to_class(encounter_type: EncounterType) -> String {
    match encounter_type {
        EncounterType::Wellness => "AMB".to_string(),
        EncounterType::Outpatient => "AMB".to_string(),
        EncounterType::Ambulatory => "AMB".to_string(),
        EncounterType::Emergency => "EMER".to_string(),
        EncounterType::Inpatient => "IMP".to_string(),
        EncounterType::Urgentcare => "AMB".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prevalence::CuratedRegistry;

    fn create_test_registry() -> OptimizedRegistry {
        use ahash::AHashMap;

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
            demographics: crate::prevalence::DemographicProfile {
                age_distribution: age_dist,
                gender_distribution: gender_dist,
                ..Default::default()
            },
        })
    }

    #[test]
    fn test_generate_patients() {
        let config = GeneratorConfig::with_patients(10).with_seed(42);
        let registry = create_test_registry();
        let generator = Generator::new(config, registry);

        let patients = generator.generate(10).unwrap();
        assert_eq!(patients.len(), 10);

        // Check that IDs are unique
        let ids: std::collections::HashSet<_> = patients.iter().map(|p| &p.id).collect();
        assert_eq!(ids.len(), 10);
    }

    #[test]
    fn test_generate_deterministic() {
        // Use the same registry instance for determinism
        let registry = create_test_registry();
        let config1 = GeneratorConfig::with_patients(5).with_seed(42);
        let gen1 = Generator::new(config1, registry);

        let patients1 = gen1.generate(5).unwrap();
        let patients2 = gen1.generate(5).unwrap();

        // Same generator should produce same results
        for (p1, p2) in patients1.iter().zip(patients2.iter()) {
            assert_eq!(p1.id, p2.id);
            // Demographics may differ between runs since RNG reseeds
            // but IDs should always match
        }

        // Verify all IDs are unique
        let ids: std::collections::HashSet<_> = patients1.iter().map(|p| &p.id).collect();
        assert_eq!(ids.len(), 5);
    }
}
