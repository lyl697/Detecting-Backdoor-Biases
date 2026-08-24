#!/usr/bin/env python3
"""Apply the conservative Stage-II benign-veto rule."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def final_prediction(stage1_prediction: int, benign_veto: bool) -> int:
    """Stage-II may clear a Stage-I positive but may never create one."""
    return int(int(stage1_prediction) == 1 and not bool(benign_veto))


def _truth(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def combine_rows(stage1_rows: list[dict], stage2_rows: list[dict]) -> list[dict]:
    stage2_by_model = {
        (
            str(row.get("architecture", "")).strip(),
            str(row.get("model_id") or row.get("name") or row.get("model_tag")),
        ): row
        for row in stage2_rows
    }
    output = []
    for stage1 in stage1_rows:
        model_id = str(stage1["model_id"])
        architecture = str(stage1.get("architecture", "")).strip()
        prediction = int(stage1["stage1_prediction"])
        stage2 = stage2_by_model.get((architecture, model_id), {}) if prediction == 1 else {}
        if prediction == 1 and not stage2:
            raw_matches = [row for (arch, name), row in stage2_by_model.items() if name == model_id]
            if len(raw_matches) == 1:
                stage2 = raw_matches[0]
        if prediction == 1 and not stage2:
            raise ValueError(
                f"Missing Stage-II verification for Stage-I positive {(architecture, model_id)}"
            )
        veto = _truth(stage2.get("benign_veto", stage2.get("benign_veto_candidate", False)))
        output.append({
            "model_id": model_id,
            "architecture": architecture,
            "true_label": stage1.get("true_label", ""),
            "stage1_score": stage1.get("stage1_score", ""),
            "stage1_threshold": stage1.get("stage1_threshold", ""),
            "stage1_prediction": prediction,
            "S_resp": stage2.get("S_resp", ""),
            "S_sel": stage2.get("S_sel", stage2.get("S_sel_cv", "")),
            "num_valid_domains": stage2.get("num_valid_domains", ""),
            "benign_veto": veto if prediction == 1 else False,
            "final_prediction": final_prediction(prediction, veto),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = combine_rows(_read(args.stage1), _read(args.stage2))
    fields = ["model_id", "architecture", "true_label", "stage1_score",
              "stage1_threshold", "stage1_prediction", "S_resp", "S_sel",
              "num_valid_domains", "benign_veto", "final_prediction"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
