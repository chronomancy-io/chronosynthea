#!/usr/bin/env python3
"""Drive SynthEHRella's fidelity evaluation against chronosynthea output.

SynthEHRella (arXiv:2411.04281) is a multi-faceted evaluator for synthetic
EHR generators. It compares synthetic data to a real reference and reports
fidelity (dimension-wise correlation, marginal alignment, MMD/KL),
utility (downstream prediction transfer), and privacy
(re-identification risk, membership-inference accuracy).

This script:
  1. Generates `n` chronosynthea patients in both `marginal` and
     `pairwise-empirical` modes.
  2. Writes the binary patient × code matrix and long-form temporal
     records via `chronosynthea_mss::synthehrella::write_binary_matrix`
     and `write_temporal_records`.
  3. Invokes SynthEHRella's `evaluate.py` against the chronosynthea output
     using the user-supplied Java Synthea reference matrix as the "real"
     baseline.

We don't bundle SynthEHRella itself — clone it from
`https://github.com/chufangao/SynthEHRella` and pass its repo root via
`--synthehrella-root`.

Example:
    python scripts/synthehrella_evaluate.py \\
        --n 10000 \\
        --reference /tmp/java-synthea-binary-matrix.csv \\
        --synthehrella-root /path/to/SynthEHRella \\
        --out workspace/synthehrella-eval/
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000, help="number of synthetic patients")
    ap.add_argument(
        "--reference",
        type=Path,
        help="path to the reference (Java Synthea) binary matrix CSV",
        required=False,
    )
    ap.add_argument(
        "--synthehrella-root",
        type=Path,
        help="path to a local SynthEHRella checkout (https://github.com/chufangao/SynthEHRella)",
        required=False,
    )
    ap.add_argument("--out", type=Path, default=Path("workspace/synthehrella-eval/"))
    ap.add_argument(
        "--mode",
        choices=("marginal", "pairwise-empirical", "causal-dag"),
        default="pairwise-empirical",
        help="chronosynthea joint mode for the generated synthetic patients",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Step 1: run a Rust-side exporter binary that writes the binary matrix
    # + temporal records for `n` patients in the requested mode.
    print(f"[1/3] generating {args.n} synthetic patients in {args.mode} mode")
    env = os.environ.copy()
    env["CHRONOSYNTHEA_JOINT_MODE"] = args.mode
    env["CHRONOSYNTHEA_SYNTHEHRELLA_OUT"] = str(args.out)
    env["CHRONOSYNTHEA_SYNTHEHRELLA_N"] = str(args.n)
    res = subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            "chronosynthea-mss",
            "--release",
            "--test",
            "synthehrella_export",
            "--",
            "--ignored",
            "--nocapture",
        ],
        env=env,
        cwd=Path(__file__).resolve().parent.parent,
    )
    if res.returncode != 0:
        print("export run failed; not invoking SynthEHRella", file=sys.stderr)
        return res.returncode

    # Step 2: optionally drive SynthEHRella against the output. Without
    # `--synthehrella-root` we just write the inputs and exit.
    if not args.synthehrella_root or not args.reference:
        print(
            "[2/3] synthehrella-root or reference not provided — skipping SynthEHRella invocation"
        )
        print(f"  binary matrix:   {args.out / 'binary_matrix.csv'}")
        print(f"  temporal records: {args.out / 'temporal_records.csv'}")
        return 0

    print(f"[2/3] invoking SynthEHRella at {args.synthehrella_root}")
    se_bin = args.synthehrella_root / "evaluate.py"
    if not se_bin.exists():
        print(f"SynthEHRella evaluate.py not found at {se_bin}", file=sys.stderr)
        return 1
    se_args = [
        sys.executable,
        str(se_bin),
        "--real",
        str(args.reference),
        "--synth",
        str(args.out / "binary_matrix.csv"),
        "--metrics",
        "all",
        "--output",
        str(args.out / "synthehrella-report"),
    ]
    print("  ", " ".join(se_args))
    res = subprocess.run(se_args)
    if res.returncode != 0:
        return res.returncode

    print(f"[3/3] SynthEHRella report written to {args.out / 'synthehrella-report'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
