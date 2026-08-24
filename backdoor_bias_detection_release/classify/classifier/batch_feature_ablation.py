#!/usr/bin/env python3
"""
Batch prediction for feature-ablation block-channel detectors.

For each test model directory, this script invokes `feature_ablation.py` once
in `predict_all` mode.
That single invocation evaluates a selected set of ablation checkpoints:

    full
    self_only
    reference_only
    wo_lfd
    wo_ltf
    wo_cross
    wo_image
    wo_adf

Output layout
-------------
<output_dir>/
    predictions/
        <test_model_name>/
            full.csv
            self_only.csv
            reference_only.csv
            wo_lfd.csv
            wo_ltf.csv
            wo_cross.csv
            wo_image.csv
            wo_adf.csv
            ablation_prediction_summary.csv
    summary.csv

The global summary.csv contains one row per (test model, ablation variant).

Important
---------
The hard-label counts/ratios in this launcher's summary are diagnostics only.
Paper model scores must be recomputed from each prompt CSV's
``backdoor_probability`` with ``evaluation/model_aggregation.py``.

This batch script targets the canonical feature layout used by
`feature_ablation.py`. It expects precomputed:
    - LFD hidden features
    - LTF trace features
    - reference-latent discrepancy features
    - image similarity features
    - activation-difference features

It intentionally does not change feature extraction or classifier logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple


TEST_SUFFIX = "_test_5biases4clean"

ALL_VARIANTS = (
    "full",
    "self_only",
    "reference_only",
    "wo_lfd",
    "wo_ltf",
    "wo_cross",
    "wo_image",
    "wo_adf",
)

VARIANT_GROUPS: Dict[str, Set[str]] = {
    "full": {"lfd", "ltf", "cross", "image", "adf"},
    "self_only": {"lfd", "ltf"},
    "reference_only": {"cross", "image", "adf"},
    "wo_lfd": {"ltf", "cross", "image", "adf"},
    "wo_ltf": {"lfd", "cross", "image", "adf"},
    "wo_cross": {"lfd", "ltf", "image", "adf"},
    "wo_image": {"lfd", "ltf", "cross", "adf"},
    "wo_adf": {"lfd", "ltf", "cross", "image"},
}

VARIANT_DESCRIPTIONS = {
    "full": "LFD + LTF + Cross + Image + ADF",
    "self_only": "LFD + LTF",
    "reference_only": "Cross + Image + ADF",
    "wo_lfd": "LTF + Cross + Image + ADF",
    "wo_ltf": "LFD + Cross + Image + ADF",
    "wo_cross": "LFD + LTF + Image + ADF",
    "wo_image": "LFD + LTF + Cross + ADF",
    "wo_adf": "LFD + LTF + Cross + Image",
}


def _family_dir(model_family: str) -> str:
    mapping = {
        "sd2": "SD2",
        "sd35": "SD35",
        "flux": "flux",
        "sd14": "SD14",
        "SD2_cmp": "SD2_cmp",
    }
    return mapping[model_family]


def _parse_variants(text: str) -> List[str]:
    if not str(text).strip():
        return list(ALL_VARIANTS)

    variants = [x.strip() for x in str(text).split(",") if x.strip()]
    variants = list(dict.fromkeys(variants))

    unknown = [x for x in variants if x not in VARIANT_GROUPS]
    if unknown:
        raise ValueError(
            "Unknown ablation variants: "
            + ", ".join(unknown)
            + ". Available: "
            + ", ".join(ALL_VARIANTS)
        )
    return variants


def _required_groups(variants: Sequence[str]) -> Set[str]:
    groups: Set[str] = set()
    for variant in variants:
        groups.update(VARIANT_GROUPS[variant])
    return groups


def _discover_models(hidden_root: Path, suffix: str) -> List[Path]:
    if not hidden_root.is_dir():
        raise FileNotFoundError(
            f"Hidden feature root does not exist: {hidden_root}"
        )

    return sorted(
        path
        for path in hidden_root.iterdir()
        if path.is_dir()
        and path.name.endswith(suffix)
        and (path / "results.csv").is_file()
    )


def _read_rows(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _hidden_sample_keys(path: Path) -> Set[Tuple[int, str]]:
    keys: Set[Tuple[int, str]] = set()

    for row in _read_rows(path):
        prompt_text = str(row.get("prompt_index", "")).strip()
        if not prompt_text:
            continue

        for column, value in row.items():
            if column.endswith("_orig_feature_paths") and str(value or "").strip():
                keys.add(
                    (
                        int(prompt_text),
                        column.removesuffix("_orig_feature_paths"),
                    )
                )

    return keys


def _tabular_sample_keys(path: Path) -> Set[Tuple[int, str]]:
    keys: Set[Tuple[int, str]] = set()

    for row in _read_rows(path):
        prompt_text = str(row.get("prompt_index", "")).strip()
        model_tag = str(row.get("model_tag", "")).strip()

        if prompt_text and model_tag:
            keys.add((int(prompt_text), model_tag))

    return keys


def _trace_sample_keys(path: Path) -> Set[Tuple[int, str]]:
    keys: Set[Tuple[int, str]] = set()

    for row in _read_rows(path):
        prompt_text = str(row.get("prompt_index", "")).strip()
        if not prompt_text:
            continue

        for column in (
            "clean_orig",
            "backdoor_orig",
            "test_orig",
        ):
            if column in row and str(row.get(column, "")).strip():
                keys.add(
                    (
                        int(prompt_text),
                        column.removesuffix("_orig"),
                    )
                )

    return keys


def _find_incomplete_features(
    expected: Set[Tuple[int, str]],
    feature_paths: Dict[str, Tuple[Path, str]],
) -> Dict[str, List[Tuple[int, str]]]:
    incomplete: Dict[str, List[Tuple[int, str]]] = {}

    for feature_name, (path, layout) in feature_paths.items():
        if not path.is_file():
            incomplete[feature_name] = sorted(expected)
            continue

        if layout == "trace":
            actual = _trace_sample_keys(path)
        elif layout == "tabular":
            actual = _tabular_sample_keys(path)
        else:
            raise ValueError(f"Unknown feature layout: {layout}")

        missing = sorted(expected - actual)
        if missing:
            incomplete[feature_name] = missing

    return incomplete


def _encode_incomplete(
    incomplete: Dict[str, List[Tuple[int, str]]],
    feature_paths: Dict[str, Tuple[Path, str]],
) -> str:
    return json.dumps(
        {
            name: {
                "file_missing": not feature_paths[name][0].is_file(),
                "path": str(feature_paths[name][0]),
                "missing_count": len(keys),
                "missing_keys": [
                    f"{prompt_index}:{model_tag}"
                    for prompt_index, model_tag in keys
                ],
            }
            for name, keys in incomplete.items()
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _count_predictions(path: Path) -> dict:
    total = 0
    backdoor = 0
    probabilities: List[float] = []

    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1

            try:
                predicted_label = int(row.get("predicted_label", "0"))
            except (TypeError, ValueError):
                predicted_label = 0

            backdoor += int(predicted_label == 1)

            try:
                probabilities.append(float(row["backdoor_probability"]))
            except (KeyError, TypeError, ValueError):
                pass

    return {
        "total_count": total,
        "backdoor_count": backdoor,
        "clean_count": total - backdoor,
        "backdoor_ratio": backdoor / total if total else 0.0,
        "mean_backdoor_probability": (
            sum(probabilities) / len(probabilities)
            if probabilities
            else float("nan")
        ),
    }


def _write_summary(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "model_name",
        "variant",
        "feature_configuration",
        "status",
        "backdoor_count",
        "total_count",
        "clean_count",
        "backdoor_ratio",
        "mean_backdoor_probability",
        "expected_sample_count",
        "incomplete_features",
        "prediction_csv",
        "error",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                field: row.get(field, "")
                for field in fields
            }
            for row in rows
        )


def _validate_checkpoints(
    ablation_model_dir: Path,
    variants: Sequence[str],
) -> None:
    missing = []

    for variant in variants:
        ckpt = ablation_model_dir / variant / "detector.pt"
        if not ckpt.is_file():
            missing.append(str(ckpt))

    if missing:
        raise FileNotFoundError(
            "Missing ablation checkpoints:\n  "
            + "\n  ".join(missing)
        )


def _coverage_paths_for_variants(
    required_groups: Set[str],
    angle_csv: Path,
    norm_csv: Path,
    delta_csv: Path,
    modelcross_csv: Path,
    image_csv: Path,
    navi_csv: Path,
) -> Dict[str, Tuple[Path, str]]:
    """Only require feature files needed by at least one requested variant."""

    paths: Dict[str, Tuple[Path, str]] = {}

    if "ltf" in required_groups:
        paths.update(
            {
                "latent_cosine": (angle_csv, "trace"),
                "latent_norm": (norm_csv, "trace"),
                "latent_update_norm": (delta_csv, "trace"),
            }
        )

    if "cross" in required_groups:
        paths["reference_latent"] = (
            modelcross_csv,
            "tabular",
        )

    if "image" in required_groups:
        paths["image_similarity"] = (
            image_csv,
            "tabular",
        )

    if "adf" in required_groups:
        paths["activation_difference"] = (
            navi_csv,
            "tabular",
        )

    return paths


def _append_existing_rows(
    summary_rows: List[dict],
    model_name: str,
    expected_count: int,
    model_prediction_dir: Path,
    variants: Sequence[str],
    status: str,
) -> None:
    for variant in variants:
        prediction_csv = model_prediction_dir / f"{variant}.csv"
        counts = _count_predictions(prediction_csv)

        summary_rows.append(
            {
                "model_name": model_name,
                "variant": variant,
                "feature_configuration": VARIANT_DESCRIPTIONS[variant],
                "status": status,
                **counts,
                "expected_sample_count": expected_count,
                "incomplete_features": "",
                "prediction_csv": str(prediction_csv),
                "error": "",
            }
        )


def run(args) -> None:
    variants = _parse_variants(args.variants)
    required_groups = _required_groups(variants)

    family_dir = _family_dir(args.model_family)
    classify_root = args.classify_root

    hidden_root = (
        classify_root
        / "lfd"
        / family_dir
    )

    trace_root = (
        classify_root
        / "latent_trajectory"
        / family_dir
    )

    image_root = (
        classify_root
        / "image_similarity"
        / family_dir
    )

    navi_root = (
        classify_root
        / "activation_difference"
        / family_dir
    )

    modelcross_root = (
        classify_root
        / "reference_latent"
        / family_dir
    )

    models = _discover_models(
        hidden_root,
        args.test_suffix,
    )

    if not models:
        raise ValueError(
            f"No *{args.test_suffix} feature directories found in {hidden_root}"
        )

    detector_script = args.detector_script.resolve()
    if not detector_script.is_file():
        raise FileNotFoundError(
            f"Ablation detector script does not exist: {detector_script}"
        )

    _validate_checkpoints(
        args.ablation_model_dir,
        variants,
    )

    predictions_root = (
        args.output_dir
        / "predictions"
    )
    predictions_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows: List[dict] = []
    summary_path = args.output_dir / "summary.csv"

    print("=" * 90)
    print("Feature-ablation batch test")
    print("=" * 90)
    print(f"family             : {args.model_family} -> {family_dir}")
    print(f"num test models    : {len(models)}")
    print(f"variants           : {','.join(variants)}")
    print(f"required groups    : {','.join(sorted(required_groups))}")
    print(f"checkpoint root    : {args.ablation_model_dir}")
    print(f"detector script    : {detector_script}")
    print(f"output             : {args.output_dir}")

    for position, hidden_dir in enumerate(models, start=1):
        model_name = hidden_dir.name
        model_prediction_dir = predictions_root / model_name
        model_prediction_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"\n[batch {position}/{len(models)}] {model_name}",
            flush=True,
        )

        try:
            hidden_csv = hidden_dir / "results.csv"
            if not hidden_csv.is_file():
                raise FileNotFoundError(
                    f"Missing hidden CSV for {model_name}: {hidden_csv}"
                )

            model_trace_dir = trace_root / model_name
            angle_csv = model_trace_dir / "adjacent_z_cos.csv"
            norm_csv = model_trace_dir / "adjacent_z_norm.csv"
            delta_csv = model_trace_dir / "adjacent_z_delta_norm.csv"

            modelcross_csv = (
                modelcross_root
                / model_name
                / "results.csv"
            )

            image_csv = (
                image_root
                / model_name
                / "results.csv"
            )

            navi_csv = (
                navi_root
                / model_name
                / "results.csv"
            )

            expected_keys = _hidden_sample_keys(
                hidden_csv
            )

            if not expected_keys:
                raise ValueError(
                    f"No test samples found in hidden CSV: {hidden_csv}"
                )

            coverage_paths = _coverage_paths_for_variants(
                required_groups=required_groups,
                angle_csv=angle_csv,
                norm_csv=norm_csv,
                delta_csv=delta_csv,
                modelcross_csv=modelcross_csv,
                image_csv=image_csv,
                navi_csv=navi_csv,
            )

            incomplete = _find_incomplete_features(
                expected_keys,
                coverage_paths,
            )

            if incomplete:
                incomplete_text = _encode_incomplete(
                    incomplete,
                    coverage_paths,
                )

                for variant in variants:
                    summary_rows.append(
                        {
                            "model_name": model_name,
                            "variant": variant,
                            "feature_configuration": VARIANT_DESCRIPTIONS[variant],
                            "status": "incomplete_features",
                            "expected_sample_count": len(expected_keys),
                            "incomplete_features": incomplete_text,
                            "prediction_csv": "",
                            "error": "",
                        }
                    )

                print(
                    f"[incomplete] {model_name}: {incomplete_text}",
                    flush=True,
                )

                _write_summary(
                    summary_path,
                    summary_rows,
                )
                continue

            # -------------------------------------------------------------
            # skip_existing works per ablation variant.
            # If only some outputs already exist, only missing variants are
            # sent to predict_all.
            # -------------------------------------------------------------
            existing_variants: List[str] = []
            run_variants: List[str] = []

            for variant in variants:
                prediction_csv = (
                    model_prediction_dir
                    / f"{variant}.csv"
                )

                if args.skip_existing and prediction_csv.is_file():
                    existing_variants.append(variant)
                else:
                    run_variants.append(variant)

            if existing_variants:
                _append_existing_rows(
                    summary_rows=summary_rows,
                    model_name=model_name,
                    expected_count=len(expected_keys),
                    model_prediction_dir=model_prediction_dir,
                    variants=existing_variants,
                    status="skipped_existing",
                )

                for variant in existing_variants:
                    counts = _count_predictions(
                        model_prediction_dir / f"{variant}.csv"
                    )
                    print(
                        f"[existing/{variant}] "
                        f"backdoor={counts['backdoor_count']}/{counts['total_count']}",
                        flush=True,
                    )

            if run_variants:
                # ---------------------------------------------------------
                # ONE subprocess call per test model.
                # The ablation detector internally loops over all requested
                # checkpoints using --mode predict_all.
                # ---------------------------------------------------------
                command = [
                    sys.executable,
                    str(detector_script),
                    "--mode",
                    "predict_all",
                    "--ablation_output_dir",
                    str(args.ablation_model_dir),
                    "--predict_output_dir",
                    str(model_prediction_dir),
                    "--variants",
                    ",".join(run_variants),
                    "--lfd_csv",
                    str(hidden_csv),
                    "--latent_cosine_csv",
                    str(angle_csv),
                    "--latent_norm_csv",
                    str(norm_csv),
                    "--latent_update_norm_csv",
                    str(delta_csv),
                    "--reference_latent_csv",
                    str(modelcross_csv),
                    "--image_similarity_csv",
                    str(image_csv),
                    "--activation_difference_csv",
                    str(navi_csv),
                    "--extra_feature_csv_out",
                    "",
                    "--device",
                    args.device,
                    "--batch_size",
                    str(args.batch_size),
                    "--threshold",
                    str(args.threshold),
                    "--model_id",
                    model_name,
                    "--architecture",
                    "sd2" if args.model_family == "SD2_cmp" else args.model_family,
                    "--prompt_txt",
                    str(args.prompt_txt),
                ]

                print(
                    "[run variants] " + ",".join(run_variants),
                    flush=True,
                )

                result = subprocess.run(
                    command,
                    check=False,
                )

                if result.returncode != 0:
                    raise RuntimeError(
                        "ablation detector exited with code "
                        f"{result.returncode}"
                    )

                # ---------------------------------------------------------
                # Collect each variant's prediction file into one global
                # summary table.
                # ---------------------------------------------------------
                for variant in run_variants:
                    prediction_csv = (
                        model_prediction_dir
                        / f"{variant}.csv"
                    )

                    if not prediction_csv.is_file():
                        raise FileNotFoundError(
                            "Ablation prediction was not created: "
                            f"{prediction_csv}"
                        )

                    counts = _count_predictions(
                        prediction_csv
                    )

                    summary_rows.append(
                        {
                            "model_name": model_name,
                            "variant": variant,
                            "feature_configuration": VARIANT_DESCRIPTIONS[variant],
                            "status": "ok",
                            **counts,
                            "expected_sample_count": len(expected_keys),
                            "incomplete_features": "",
                            "prediction_csv": str(prediction_csv),
                            "error": "",
                        }
                    )

                    print(
                        f"[result/{variant}] {model_name}: "
                        f"backdoor={counts['backdoor_count']}/"
                        f"{counts['total_count']}, "
                        f"mean_prob={counts['mean_backdoor_probability']:.6f}",
                        flush=True,
                    )

        except Exception as exc:
            # If one test model fails, record one failure row for each
            # requested variant that has not already been summarized.
            summarized = {
                row["variant"]
                for row in summary_rows
                if row.get("model_name") == model_name
            }

            for variant in variants:
                if variant in summarized:
                    continue

                summary_rows.append(
                    {
                        "model_name": model_name,
                        "variant": variant,
                        "feature_configuration": VARIANT_DESCRIPTIONS[variant],
                        "status": "failed",
                        "prediction_csv": str(
                            model_prediction_dir / f"{variant}.csv"
                        ),
                        "error": str(exc),
                    }
                )

            print(
                f"[failed] {model_name}: {exc}",
                flush=True,
            )

            _write_summary(
                summary_path,
                summary_rows,
            )

            if not args.continue_on_error:
                raise

        _write_summary(
            summary_path,
            summary_rows,
        )

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print(f"Summary: {summary_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch-test all feature-ablation block-channel classifiers"
        )
    )

    parser.add_argument(
        "--model_family",
        choices=[
            "sd2",
            "sd35",
            "flux",
            "sd14",
            "SD2_cmp",
        ],
        default="flux",
    )
    parser.add_argument("--prompt_txt", type=Path, default=Path("data/prompts/test.txt"))

    parser.add_argument(
        "--classify_root",
        type=Path,
        default=Path("artifacts/features/test"),
    )

    parser.add_argument(
        "--test_suffix",
        default=TEST_SUFFIX,
    )

    # Directory created by the ablation training script:
    #   <dir>/full/detector.pt
    #   <dir>/self_only/detector.pt
    #   ...
    parser.add_argument(
        "--ablation_model_dir",
        type=Path,
        default=Path(
            "artifacts/checkpoints/feature_ablation_flux"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "artifacts/predictions/feature_ablation_flux"
        ),
    )

    parser.add_argument(
        "--detector_script",
        type=Path,
        default=Path(
            "classify/classifier/"
            "feature_ablation.py"
        ),
        help=(
            "Feature-ablation detector/training entry script created for "
            "the current block-channel classifier."
        ),
    )

    parser.add_argument(
        "--variants",
        type=str,
        #default=",".join(ALL_VARIANTS),
        default="full",
        help=(
            "Comma-separated variants. Available: "
            + ",".join(ALL_VARIANTS)
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--continue_on_error",
        action="store_true",
    )

    parser.add_argument(
        "--skip_existing",
        default=True,
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
