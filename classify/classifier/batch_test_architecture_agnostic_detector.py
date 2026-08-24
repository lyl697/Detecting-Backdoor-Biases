"""Batch prediction for the architecture-agnostic detector.

The script discovers test feature directories under one or more model-family
roots, invokes ``architecture_agnostic_detector.py --mode predict`` once
per model, and writes both per-model predictions and a global summary CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


FAMILY_DIRS = {
    "sd2": "SD2",
    "sd35": "SD35",
    "flux": "flux",
    "sd14": "SD14",
}


def _parse_families(text: str) -> list[str]:
    families = list(dict.fromkeys(item.strip().lower() for item in text.split(",") if item.strip()))
    unknown = sorted(set(families) - set(FAMILY_DIRS))
    if unknown:
        raise ValueError(f"Unknown model families: {unknown}; supported={sorted(FAMILY_DIRS)}")
    if not families:
        raise ValueError("--model_families must contain at least one family")
    return families


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _discover_models(hidden_root: Path, suffix: str) -> list[Path]:
    if not hidden_root.is_dir():
        return []
    return sorted(
        path
        for path in hidden_root.iterdir()
        if path.is_dir()
        and path.name.endswith(suffix)
        and (path / "results.csv").is_file()
    )


def _family_variants(family_dir: str) -> list[str]:
    """Prefer target-only feature directories but retain legacy compatibility."""
    return [f"{family_dir}no_reference", family_dir]


def _first_existing(candidates: list[Path]) -> Path:
    return next((path for path in candidates if path.is_file()), candidates[0])


def _feature_candidates(
    classify_root: Path,
    roots: list[str],
    family_dir: str,
    model_name: str,
    filename: str,
) -> list[Path]:
    return [
        classify_root / root / variant / model_name / filename
        for root in roots
        for variant in _family_variants(family_dir)
    ]


def _required_feature_paths(classify_root: Path, family_dir: str, model_name: str) -> dict[str, Path]:
    hidden = _feature_candidates(
        classify_root, ["lfd_hidden_feature"], family_dir, model_name, "results.csv"
    )
    trace_angle = _feature_candidates(
        classify_root, ["trace_adjacent_latents", "trace_adjacent_latents_feature"],
        family_dir, model_name, "adjacent_z_cos.csv",
    )
    trace_norm = _feature_candidates(
        classify_root, ["trace_adjacent_latents", "trace_adjacent_latents_feature"],
        family_dir, model_name, "adjacent_z_norm.csv",
    )
    trace_delta = _feature_candidates(
        classify_root, ["trace_adjacent_latents", "trace_adjacent_latents_feature"],
        family_dir, model_name, "adjacent_z_delta_norm.csv",
    )
    image = _feature_candidates(
        classify_root, ["image_similarity_features", "image_similarity_feature"],
        family_dir, model_name, "results.csv",
    )
    navi = _feature_candidates(
        classify_root, ["navi_hidden_feature"], family_dir, model_name, "results.csv"
    )
    return {
        "hidden_csv": _first_existing(hidden),
        "trace_angle_csv": _first_existing(trace_angle),
        "trace_norm_csv": _first_existing(trace_norm),
        "trace_delta_csv": _first_existing(trace_delta),
        "image_similarity_csv": _first_existing(image),
        "navi_hidden_csv": _first_existing(navi),
    }


def _count_predictions(path: Path) -> dict:
    rows = _read_rows(path)
    probabilities = []
    backdoor = 0
    for row in rows:
        try:
            backdoor += int(int(row.get("predicted_label", "0")) == 1)
        except (TypeError, ValueError):
            pass
        try:
            probabilities.append(float(row["backdoor_probability"]))
        except (KeyError, TypeError, ValueError):
            pass
    total = len(rows)
    return {
        "total_count": total,
        "backdoor_count": backdoor,
        "clean_count": total - backdoor,
        "backdoor_ratio": backdoor / total if total else 0.0,
        "mean_backdoor_probability": (
            sum(probabilities) / len(probabilities) if probabilities else ""
        ),
    }


def _write_summary(path: Path, rows: list[dict]) -> None:
    fields = [
        "model_family",
        "model_name",
        "status",
        "total_count",
        "backdoor_count",
        "clean_count",
        "backdoor_ratio",
        "mean_backdoor_probability",
        "prediction_csv",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def run(args) -> None:
    detector_script = args.detector_script.resolve()
    model_in = args.model_in.resolve()
    if not detector_script.is_file():
        raise FileNotFoundError(f"Detector script does not exist: {detector_script}")
    if not model_in.is_file():
        raise FileNotFoundError(f"Detector checkpoint does not exist: {model_in}")

    jobs = []
    for family in _parse_families(args.model_families):
        family_dir = FAMILY_DIRS[family]
        hidden_roots = [
            args.classify_root / "lfd_hidden_feature" / variant
            for variant in _family_variants(family_dir)
        ]
        discovered = {}
        for hidden_root in hidden_roots:
            for path in _discover_models(hidden_root, args.test_suffix):
                discovered.setdefault(path.name, path)
        models = [discovered[name] for name in sorted(discovered)]
        if not models:
            print(
                f"[warning] no *{args.test_suffix} models found in "
                + ", ".join(str(path) for path in hidden_roots),
                flush=True,
            )
        jobs.extend((family, family_dir, path.name) for path in models)
    if not jobs:
        raise ValueError("No test feature directories were discovered")

    summary_path = args.output_dir / "summary.csv"
    summary_rows = []
    for position, (family, family_dir, model_name) in enumerate(jobs, start=1):
        prediction_csv = args.output_dir / "predictions" / family / f"{model_name}.csv"
        prediction_csv.parent.mkdir(parents=True, exist_ok=True)
        print(f"[batch {position}/{len(jobs)}] {family}/{model_name}", flush=True)
        row = {
            "model_family": family,
            "model_name": model_name,
            "prediction_csv": str(prediction_csv),
        }
        try:
            feature_paths = _required_feature_paths(args.classify_root, family_dir, model_name)
            missing = [f"{name}={path}" for name, path in feature_paths.items() if not path.is_file()]
            if missing:
                raise FileNotFoundError("Missing feature files: " + "; ".join(missing))

            if args.skip_existing and prediction_csv.is_file():
                row.update(status="skipped_existing", **_count_predictions(prediction_csv))
            else:
                manifest_path = (
                    args.output_dir / "manifests" / family / f"{model_name}.json"
                )
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                with manifest_path.open("w", encoding="utf-8") as stream:
                    json.dump(
                        [{name: str(path.resolve()) for name, path in feature_paths.items()}],
                        stream,
                        ensure_ascii=False,
                        indent=2,
                    )
                command = [
                    sys.executable,
                    str(detector_script),
                    "--mode", "predict",
                    "--model_in", str(model_in),
                    "--dataset_manifest", str(manifest_path.resolve()),
                    "--predict_csv", str(prediction_csv),
                    "--device", args.device,
                    "--batch_size", str(args.batch_size),
                    "--threshold", str(args.threshold),
                ]
                result = subprocess.run(command, check=False)
                if result.returncode != 0:
                    raise RuntimeError(f"detector exited with code {result.returncode}")
                row.update(status="ok", **_count_predictions(prediction_csv))
            print(
                f"[result] {family}/{model_name}: "
                f"backdoor={row['backdoor_count']}/{row['total_count']}",
                flush=True,
            )
        except Exception as exc:
            row.update(status="failed", error=str(exc))
            print(f"[failed] {family}/{model_name}: {exc}", flush=True)
            summary_rows.append(row)
            _write_summary(summary_path, summary_rows)
            if not args.continue_on_error:
                raise
            continue
        summary_rows.append(row)
        _write_summary(summary_path, summary_rows)

    print(f"Done. Summary: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch-test the architecture-agnostic detector")
    parser.add_argument("--model_families", default="sd2,sd35,flux,sd14")
    parser.add_argument("--classify_root", type=Path, default=Path("classify"))
    parser.add_argument("--test_suffix", default="_test_5biases4clean")
    parser.add_argument(
        "--detector_script",
        type=Path,
        default=Path("classify/classifier/architecture_agnostic_detector.py"),
    )
    parser.add_argument(
        "--model_in",
        type=Path,
        default=Path("artifacts/detector_checkpoints/architecture_agnostic_4.pt"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("artifacts/paper_results/architecture_agnostic"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
