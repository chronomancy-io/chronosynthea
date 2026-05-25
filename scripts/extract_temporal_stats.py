#!/usr/bin/env python3
"""Extract per-condition onset age + encounter density distributions from
Java Synthea CSV output. Output feeds chronosynthea's MssFingerprint so
the d5 = `temporal-ordered` value can sample Java-equivalent timestamps.

Inputs:  /tmp/synthea-baseline/output/csv/{patients,conditions,encounters}.csv
Outputs:
  /tmp/chronosynthea-onset-stats.json     — per-condition mean_onset_age + std + min + max
  /tmp/chronosynthea-encounter-stats.json — patient-level encounter age distribution stats
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import date


JAVA_DIR = "/tmp/synthea-baseline/output/csv"


def parse_date(s: str) -> date:
    # Java emits both 'YYYY-MM-DD' (conditions) and 'YYYY-MM-DDTHH:MM:SSZ' (medications/encounters).
    s = s.split("T")[0]
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))


def main() -> int:
    # 1. Patient birthdate index
    patient_birth: dict[str, date] = {}
    with open(f"{JAVA_DIR}/patients.csv") as f:
        for r in csv.DictReader(f):
            try:
                patient_birth[r["Id"]] = parse_date(r["BIRTHDATE"])
            except (ValueError, KeyError):
                continue
    n_patients = len(patient_birth)
    print(f"patients: {n_patients}", file=sys.stderr)

    # 2. Per-condition onset age (in years) across all patients
    onset_years: dict[str, list[float]] = defaultdict(list)
    with open(f"{JAVA_DIR}/conditions.csv") as f:
        for r in csv.DictReader(f):
            try:
                bd = patient_birth.get(r["PATIENT"])
                if bd is None:
                    continue
                onset = parse_date(r["START"])
                age_days = (onset - bd).days
                onset_years[r["CODE"]].append(age_days / 365.25)
            except (ValueError, KeyError):
                continue

    onset_stats = []
    for code, ages in onset_years.items():
        if not ages:
            continue
        mean = statistics.fmean(ages)
        # Avoid statistics.stdev requiring n>=2
        std = statistics.stdev(ages) if len(ages) >= 2 else 5.0
        # Clip std to a sensible range so the sampler can't produce
        # negative onset ages for low-mean conditions.
        std = max(min(std, mean * 0.6 if mean > 0 else 5.0), 0.5)
        onset_stats.append({
            "code": code,
            "mean_onset_age": round(mean, 3),
            "onset_age_std": round(std, 3),
            "min_onset_age": round(max(0.0, min(ages)), 3),
            "max_onset_age": round(max(ages), 3),
            "n_observations": len(ages),
        })
    onset_stats.sort(key=lambda r: -r["n_observations"])
    with open("/tmp/chronosynthea-onset-stats.json", "w") as f:
        json.dump(onset_stats, f)
    print(f"onset_stats: {len(onset_stats)} conditions", file=sys.stderr)
    for r in onset_stats[:5]:
        print(
            f"  {r['code']:>12}  mean={r['mean_onset_age']:.2f}y "
            f"std={r['onset_age_std']:.2f}y  n={r['n_observations']}",
            file=sys.stderr,
        )

    # 3. Encounter-age distribution per patient (for "how many encounters do
    # patients have between age X and Y" — feeds the per-encounter timing model)
    per_patient_enc_ages: dict[str, list[float]] = defaultdict(list)
    with open(f"{JAVA_DIR}/encounters.csv") as f:
        for r in csv.DictReader(f):
            try:
                bd = patient_birth.get(r["PATIENT"])
                if bd is None:
                    continue
                ts = parse_date(r["START"])
                age = (ts - bd).days / 365.25
                if age >= 0:
                    per_patient_enc_ages[r["PATIENT"]].append(age)
            except (ValueError, KeyError):
                continue

    # Aggregate to a histogram-ish summary
    all_ages = [a for ages in per_patient_enc_ages.values() for a in ages]
    encs_per_patient = [len(a) for a in per_patient_enc_ages.values()]
    bucket_counts = [0] * 11  # decadal buckets 0-9, 10-19, ..., 90-99, 100+
    for a in all_ages:
        b = min(10, int(a / 10))
        bucket_counts[b] += 1
    total = sum(bucket_counts)
    enc_stats = {
        "n_patients": n_patients,
        "total_encounters": len(all_ages),
        "encounters_per_patient_mean": (
            statistics.fmean(encs_per_patient) if encs_per_patient else 0.0
        ),
        "encounters_per_patient_std": (
            statistics.stdev(encs_per_patient) if len(encs_per_patient) >= 2 else 0.0
        ),
        "age_bucket_fractions": [c / total if total else 0.0 for c in bucket_counts],
        "age_buckets": [
            "0-9", "10-19", "20-29", "30-39", "40-49", "50-59",
            "60-69", "70-79", "80-89", "90-99", "100+",
        ],
    }
    with open("/tmp/chronosynthea-encounter-stats.json", "w") as f:
        json.dump(enc_stats, f, indent=2)
    print(
        f"encounters: total {enc_stats['total_encounters']}, "
        f"mean {enc_stats['encounters_per_patient_mean']:.1f} per patient",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
