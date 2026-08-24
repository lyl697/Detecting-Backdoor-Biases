#!/usr/bin/env python3
"""Recompute model-level paper metrics from lightweight released CSV records."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from classify.evaluation.final_decision import combine_rows
from classify.evaluation.metrics import binary_metrics
from classify.evaluation.model_aggregation import aggregate_rows


RESULTS = ROOT / "artifacts" / "paper_results"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty result file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-predictions", type=Path,
                        default=RESULTS / "prompt_predictions.csv")
    parser.add_argument("--stage2", type=Path,
                        default=RESULTS / "stage2_verification_scores.csv")
    parser.add_argument("--stage1-threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not args.prompt_predictions.is_file() or not args.stage2.is_file():
        raise SystemExit(
            "Generate prompt predictions and Stage-II scores first, or pass their paths with "
            "--prompt-predictions and --stage2"
        )
    stage1 = aggregate_rows(read_csv(args.prompt_predictions), args.stage1_threshold)
    write_csv(RESULTS / "stage1_model_scores.csv", stage1)
    final = combine_rows(stage1, read_csv(args.stage2))
    write_csv(RESULTS / "final_model_predictions.csv", final)
    metrics = binary_metrics(
        [int(row["true_label"]) for row in final],
        [int(row["final_prediction"]) for row in final],
    )
    (RESULTS / "final_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
