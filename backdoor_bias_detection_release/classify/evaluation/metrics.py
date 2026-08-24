#!/usr/bin/env python3
"""Compute final binary model-level metrics from released predictions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def binary_metrics(labels: list[int], predictions: list[int]) -> dict:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have equal length")
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    total = len(labels)
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Accuracy": (tp + tn) / total if total else 0.0,
        "TPR": tp / (tp + fn) if tp + fn else 0.0,
        "FPR": fp / (fp + tn) if fp + tn else 0.0,
        "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = binary_metrics(
        [int(row["true_label"]) for row in rows],
        [int(row["final_prediction"]) for row in rows],
    )
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
