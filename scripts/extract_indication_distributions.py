#!/usr/bin/env python3
"""Extract per-(procedure|medication) conditional REASONCODE distributions
from Java Synthea CSV output, and patch chronosynthea's calibrated registry
with `indication_distribution: [[code, weight], ...]` entries.

Where `extract_procedure_indications.py` kept only the most-common REASONCODE
per procedure, this script keeps the full empirical distribution: every
`(code, P(reason|procedure))` weight from Java's `procedures.csv` and
`medications.csv`. Downstream consumers can then sample REASONCODE
weighted by Java's actual conditional, giving multi-cause coverage instead
of single-cause coverage (lifts procedure REASONCODE population from
~13% to ~46% — matching Java's overall rate).

Inputs (defaults):
  /tmp/synthea-baseline/output/csv/procedures.csv
  /tmp/synthea-baseline/output/csv/medications.csv

Output: patches /tmp/chronosynthea-verify/data/prevalence/calibrated_registry.json
in place.

Tuning: `--top-k N` keeps only the top-N reasons per procedure (default 8,
which captures >95% of Java's REASONCODE mass for the long tail without
exploding registry size). `--min-weight W` drops weights below W (default
0.005 — half a percent).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def extract_distribution(
    csv_path: Path,
    code_col: str,
    reason_col: str,
    top_k: int,
    min_weight: float,
) -> dict[str, list[tuple[str, float]]]:
    per_code: dict[str, Counter] = defaultdict(Counter)
    n_total = 0
    n_with_reason = 0
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            n_total += 1
            code = r[code_col]
            reason = r.get(reason_col, "")
            if reason:
                n_with_reason += 1
                per_code[code][reason] += 1

    print(
        f"{csv_path.name}: rows={n_total} with_reason={n_with_reason} "
        f"distinct_codes_with_reason={len(per_code)}",
        file=sys.stderr,
    )

    out: dict[str, list[tuple[str, float]]] = {}
    for code, counter in per_code.items():
        total = sum(counter.values())
        if total == 0:
            continue
        # Top-K reasons by frequency
        top = counter.most_common(top_k)
        dist = []
        kept_mass = 0
        for reason_code, count in top:
            w = count / total
            if w < min_weight:
                continue
            dist.append((reason_code, w))
            kept_mass += w
        if not dist:
            continue
        # Renormalise so weights in dist sum to 1 (we sample conditional on
        # at least one being active).
        if kept_mass > 0:
            dist = [(c, w / kept_mass) for c, w in dist]
        out[code] = dist
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--procedures",
        type=Path,
        default=Path("/tmp/synthea-baseline/output/csv/procedures.csv"),
    )
    ap.add_argument(
        "--medications",
        type=Path,
        default=Path("/tmp/synthea-baseline/output/csv/medications.csv"),
    )
    ap.add_argument(
        "--registry",
        type=Path,
        default=Path(
            "/tmp/chronosynthea-verify/data/prevalence/calibrated_registry.json"
        ),
    )
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--min-weight", type=float, default=0.005)
    args = ap.parse_args()

    proc_dist: dict[str, list[tuple[str, float]]] = {}
    med_dist: dict[str, list[tuple[str, float]]] = {}
    if args.procedures.exists():
        proc_dist = extract_distribution(
            args.procedures, "CODE", "REASONCODE", args.top_k, args.min_weight
        )
    else:
        print(f"missing: {args.procedures}", file=sys.stderr)
    if args.medications.exists():
        med_dist = extract_distribution(
            args.medications, "CODE", "REASONCODE", args.top_k, args.min_weight
        )
    else:
        print(f"missing: {args.medications}", file=sys.stderr)

    with open(args.registry) as f:
        reg = json.load(f)

    procedures = reg.get("procedures", [])
    meds = reg.get("medications", [])

    proc_patched = 0
    proc_avg_k = 0
    for proc in procedures:
        code = proc.get("code")
        if code in proc_dist:
            dist = proc_dist[code]
            proc["indication_distribution"] = [[c, float(w)] for c, w in dist]
            proc_patched += 1
            proc_avg_k += len(dist)

    med_patched = 0
    med_avg_k = 0
    for med in meds:
        code = med.get("code")
        if code in med_dist:
            dist = med_dist[code]
            med["indication_distribution"] = [[c, float(w)] for c, w in dist]
            med_patched += 1
            med_avg_k += len(dist)

    print(
        f"procedures patched: {proc_patched}/{len(procedures)}"
        + (
            f" (avg {proc_avg_k / proc_patched:.1f} reasons/proc)"
            if proc_patched
            else ""
        ),
        file=sys.stderr,
    )
    print(
        f"medications patched: {med_patched}/{len(meds)}"
        + (
            f" (avg {med_avg_k / med_patched:.1f} reasons/med)"
            if med_patched
            else ""
        ),
        file=sys.stderr,
    )

    with open(args.registry, "w") as f:
        json.dump(reg, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
