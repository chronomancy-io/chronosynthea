#!/usr/bin/env python3
"""Extract empirical condition-pair onset lags from Java Synthea's
conditions.csv.

For each ordered pair (trigger, downstream) where a meaningful number of
patients exhibit both conditions, computes the mean and standard
deviation of `downstream.onset_days - trigger.onset_days` across those
patients. Patients in whom the downstream condition precedes the trigger
contribute negative lags; only pairs where the mean lag is positive
(downstream truly follows trigger) are emitted.

Output: JSON array of
    {"trigger": "<SNOMED>", "downstream": "<SNOMED>", "mean_days": <int>,
     "std_days": <int>, "probability": <float>}
where `probability` is `P(downstream | trigger, both present)` — the
fraction of trigger-patients who also have the downstream condition
during their lifetime.

Used by chronosynthea's `CausalCascadeModel` to enforce trajectory
ordering on patient-level samples: when both conditions appear in a
sampled patient, the downstream condition's onset is regenerated as a
Gaussian draw centred at `trigger.onset + mean_days`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def days_since_epoch(date_str: str) -> int:
    return (datetime.strptime(date_str, "%Y-%m-%d") - datetime(1970, 1, 1)).days


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--conditions",
        type=Path,
        default=Path("/tmp/synthea-baseline/output/csv/conditions.csv"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/chronosynthea-verify/data/prevalence/cascade_lags.json"),
    )
    ap.add_argument(
        "--min-cooccurrence",
        type=int,
        default=20,
        help="minimum number of patients with both conditions to emit a pair",
    )
    ap.add_argument(
        "--min-mean-lag-days",
        type=int,
        default=180,
        help="minimum positive mean lag to emit a pair (filters synchronous co-occurrence)",
    )
    args = ap.parse_args()

    if not args.conditions.exists():
        print(f"missing {args.conditions}", file=sys.stderr)
        return 1

    # patient_id → {condition_code: earliest_onset_days}
    per_patient: dict[str, dict[str, int]] = defaultdict(dict)
    n_rows = 0
    with open(args.conditions) as f:
        for r in csv.DictReader(f):
            n_rows += 1
            pid = r["PATIENT"]
            code = r["CODE"]
            start = r["START"]
            if not start:
                continue
            try:
                onset = days_since_epoch(start)
            except ValueError:
                continue
            cur = per_patient[pid].get(code)
            if cur is None or onset < cur:
                per_patient[pid][code] = onset

    print(
        f"loaded {n_rows} rows across {len(per_patient)} patients",
        file=sys.stderr,
    )

    # Per (trigger, downstream) pair: list of lag values from patients
    # that have both.
    pair_lags: dict[tuple[str, str], list[int]] = defaultdict(list)
    # Per trigger: count of patients with the trigger (denominator for
    # downstream probabilities).
    trigger_count: dict[str, int] = defaultdict(int)
    for codes in per_patient.values():
        for trigger, t_onset in codes.items():
            trigger_count[trigger] += 1
            for downstream, d_onset in codes.items():
                if trigger == downstream:
                    continue
                pair_lags[(trigger, downstream)].append(d_onset - t_onset)

    print(
        f"observed {len(pair_lags)} ordered condition pairs",
        file=sys.stderr,
    )

    out_pairs = []
    for (trigger, downstream), lags in pair_lags.items():
        if len(lags) < args.min_cooccurrence:
            continue
        mean = sum(lags) / len(lags)
        if mean < args.min_mean_lag_days:
            continue
        var = sum((l - mean) ** 2 for l in lags) / len(lags)
        std = var ** 0.5
        prob = len(lags) / trigger_count[trigger]
        out_pairs.append(
            {
                "trigger": trigger,
                "downstream": downstream,
                "mean_days": int(mean),
                "std_days": int(std),
                "probability": round(prob, 4),
                "n": len(lags),
            }
        )

    # Sort by mean_days for human readability
    out_pairs.sort(key=lambda r: (r["trigger"], r["mean_days"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out_pairs, f, indent=2)

    print(
        f"wrote {len(out_pairs)} cascade pairs to {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
