#!/usr/bin/env python3
"""Aggregate prompt probabilities into the paper's Stage-I model score.

The scientific definition is ``mean(backdoor_probability)``. Prompt-level
hard predictions are deliberately ignored by this module.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


MODEL_KEYS = ("model_id", "model_tag", "name")


def _model_id(row: dict) -> str:
    for key in MODEL_KEYS:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    raise ValueError("Each prediction row needs model_id or model_tag")


def aggregate_rows(
    rows: list[dict],
    threshold: float,
) -> list[dict]:
    """Return one Stage-I row per model using a strict ``score > threshold``."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("architecture", "")).strip(), _model_id(row))].append(row)

    output = []
    for architecture_key, model_id in sorted(grouped):
        model_rows = grouped[architecture_key, model_id]
        probabilities = []
        prompt_indices = set()
        for row in model_rows:
            try:
                probability = float(row["backdoor_probability"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{model_id}: invalid backdoor_probability in row {row!r}"
                ) from exc
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{model_id}: probability outside [0, 1]: {probability}")
            probabilities.append(probability)
            prompt_index = str(row.get("prompt_index", "")).strip()
            if not prompt_index:
                raise ValueError(f"{model_id}: missing prompt_index")
            if prompt_index in prompt_indices:
                raise ValueError(f"{model_id}: duplicate prompt_index {prompt_index}")
            prompt_indices.add(prompt_index)

        score = sum(probabilities) / len(probabilities)
        architectures = {str(r.get("architecture", "")).strip() for r in model_rows} - {""}
        labels = {str(r.get("true_label", "")).strip() for r in model_rows} - {""}
        if len(architectures) > 1 or len(labels) > 1:
            raise ValueError(f"{model_id}: inconsistent architecture or true_label")
        output.append({
            "model_id": model_id,
            "architecture": next(iter(architectures), ""),
            "true_label": next(iter(labels), ""),
            "num_prompts": len(prompt_indices),
            "stage1_score": score,
            "stage1_threshold": float(threshold),
            "stage1_prediction": int(score > threshold),
        })
    return output


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["model_id", "architecture", "true_label", "num_prompts",
              "stage1_score", "stage1_threshold", "stage1_prediction"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Stage-I model threshold (default: 0.5)",
    )
    args = parser.parse_args()
    write_csv(args.output, aggregate_rows(read_csv(args.input), args.threshold))


if __name__ == "__main__":
    main()
