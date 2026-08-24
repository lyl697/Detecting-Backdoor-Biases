"""Run reproducible random summaries for one ablation prediction variant.

Each source is reduced to 100 rows with the same rule as
summarize_batch_prediction_subsets.py. In every round, five positions are
randomly sampled without replacement from each consecutive group of ten. The
first one, first two, and all five sampled positions form nested 10-, 20-, and
50-row subsets respectively.

Expected batch-ablation layout::

    predictions/
        <test_model>/
            full.csv
            wo_cross.csv
            ablation_prediction_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path
from typing import List, Sequence, Tuple

from summarize_batch_prediction_subsets import (
    mean_probability,
    parse_label,
    read_rows,
    select_base_rows,
    write_csv,
)


SUBSET_WIDTHS: Tuple[Tuple[int, int], ...] = ((10, 1), (20, 2), (50, 5))
RELEASE_ROOT = Path(__file__).resolve().parents[2]
VARIANT_ALIASES = {
    "wo_crosso_cross": "wo_cross",
}


def normalize_variant(text: str) -> str:
    variant = VARIANT_ALIASES.get(str(text).strip(), str(text).strip())
    if variant == "mixed":
        variant = "auto"
    if variant not in {"wo_cross", "full", "auto"}:
        raise ValueError("--variant must be 'wo_cross', 'full', or 'auto'")
    return variant


def discover_prediction_files(predictions_dir: Path, variant: str) -> List[Path]:
    """Find only ``<model>/<variant>.csv`` files from ablation batch output."""
    if variant == "auto":
        selected = []
        ambiguous = []
        for model_dir in sorted(path for path in predictions_dir.iterdir() if path.is_dir()):
            candidates = [
                model_dir / name
                for name in ("wo_cross.csv", "full.csv")
                if (model_dir / name).is_file()
            ]
            if len(candidates) == 1:
                selected.append(candidates[0])
            elif len(candidates) > 1:
                ambiguous.append(model_dir.name)
        if ambiguous:
            raise ValueError(
                "Auto variant selection is ambiguous because both wo_cross.csv "
                "and full.csv exist for: " + ", ".join(ambiguous)
            )
        return selected

    nested = sorted(
        path
        for path in predictions_dir.glob(f"*/{variant}.csv")
        if path.is_file()
    )
    if nested:
        return nested

    # Retain compatibility with the old flat layout when a directory contains
    # prediction CSVs directly. In that layout every CSV is treated as the
    # selected variant, except known aggregate reports.
    return sorted(
        path
        for path in predictions_dir.glob("*.csv")
        if path.is_file()
        and path.name not in {"summary.csv", "ablation_prediction_summary.csv"}
    )


def model_name_for(path: Path, predictions_dir: Path) -> str:
    return path.parent.name if path.parent != predictions_dir else path.stem


def variant_for_path(path: Path, requested_variant: str) -> str:
    return path.stem if requested_variant == "auto" else requested_variant


def stable_file_seed(
    base_seed: int,
    path: Path,
    predictions_dir: Path,
    round_number: int,
) -> int:
    identity = path.resolve().relative_to(predictions_dir.resolve()).as_posix().encode("utf-8")
    digest = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")
    return (base_seed + digest + round_number * 1_000_003) % (2**63)


def random_group_positions(rng: random.Random) -> List[List[int]]:
    selections = []
    for group_start in range(0, 100, 10):
        offsets = rng.sample(range(10), 5)
        selections.append([group_start + offset for offset in offsets])
    return selections


def positions_for_width(group_positions: Sequence[Sequence[int]], width: int) -> List[int]:
    return [position for group in group_positions for position in group[:width]]


def summarize_file(
    path: Path,
    predictions_dir: Path,
    variant: str,
    rounds: int,
    base_seed: int,
) -> Tuple[List[dict], List[dict]]:
    source_rows = read_rows(path)
    base_rows, source_numbers, source_layout = select_base_rows(source_rows)
    model_name = model_name_for(path, predictions_dir)
    summary_rows = []
    detail_rows = []

    for round_number in range(1, rounds + 1):
        round_seed = stable_file_seed(base_seed, path, predictions_dir, round_number)
        group_positions = random_group_positions(random.Random(round_seed))

        for subset_size, width in SUBSET_WIDTHS:
            positions = positions_for_width(group_positions, width)
            sampled = [base_rows[position] for position in positions]
            sampled_source_numbers = [source_numbers[position] for position in positions]
            labels = [
                parse_label(row, path, source_row)
                for row, source_row in zip(sampled, sampled_source_numbers)
            ]
            backdoor_count = sum(label == 1 for label in labels)
            clean_count = sum(label == 0 for label in labels)
            probability = mean_probability(sampled)

            summary_rows.append(
                {
                    "model_name": model_name,
                    "variant": variant,
                    "prediction_csv": str(path),
                    "source_row_count": len(source_rows),
                    "source_layout": source_layout,
                    "round": round_number,
                    "round_seed": round_seed,
                    "subset_size": subset_size,
                    "clean_count": clean_count,
                    "backdoor_count": backdoor_count,
                    "backdoor_ratio": backdoor_count / subset_size,
                    "mean_backdoor_probability": "" if probability is None else probability,
                    "status": "ok",
                    "error": "",
                }
            )

            for subset_index, (position, source_number, row, label) in enumerate(
                zip(positions, sampled_source_numbers, sampled, labels), start=1
            ):
                detail_rows.append(
                    {
                        "model_name": model_name,
                        "variant": variant,
                        "round": round_number,
                        "round_seed": round_seed,
                        "subset_size": subset_size,
                        "subset_index": subset_index,
                        "base_100_position": position + 1,
                        "source_row_number": source_number,
                        "group_number": position // 10 + 1,
                        "position_in_group": position % 10 + 1,
                        "prompt_index": row.get("prompt_index", ""),
                        "predicted_label": label,
                        "predicted_tag": "backdoor" if label == 1 else "clean",
                        "backdoor_probability": row.get("backdoor_probability", ""),
                        "prediction_csv": str(path),
                    }
                )

    return summary_rows, detail_rows


def safe_row_count(path: Path) -> int | str:
    try:
        return len(read_rows(path))
    except Exception:
        return ""


def run(args) -> None:
    variant = normalize_variant(args.variant)
    files = discover_prediction_files(args.predictions_dir, variant)
    if not files:
        expected = "wo_cross.csv or full.csv" if variant == "auto" else f"{variant}.csv"
        raise ValueError(f"No {expected} prediction files found under {args.predictions_dir}")

    summary_rows: List[dict] = []
    detail_rows: List[dict] = []
    for path in files:
        file_variant = variant_for_path(path, variant)
        try:
            file_summary, file_details = summarize_file(
                path, args.predictions_dir, file_variant, args.rounds, args.seed
            )
            summary_rows.extend(file_summary)
            detail_rows.extend(file_details)
            for round_number in range(1, args.rounds + 1):
                rows = [row for row in file_summary if row["round"] == round_number]
                counts = ", ".join(
                    f"{row['backdoor_count']}/{row['subset_size']} backdoor" for row in rows
                )
                print(f"[{model_name_for(path, args.predictions_dir)}/{file_variant}] round={round_number}: {counts}")
        except Exception as exc:
            summary_rows.append(
                {
                    "model_name": model_name_for(path, args.predictions_dir),
                    "variant": file_variant,
                    "prediction_csv": str(path),
                    "source_row_count": safe_row_count(path),
                    "source_layout": "",
                    "round": "",
                    "round_seed": "",
                    "subset_size": "",
                    "clean_count": "",
                    "backdoor_count": "",
                    "backdoor_ratio": "",
                    "mean_backdoor_probability": "",
                    "status": "skipped",
                    "error": str(exc),
                }
            )
            print(f"[skip] {path.name}: {exc}")
            if not args.continue_on_error:
                raise

    summary_fields = [
        "model_name",
        "variant",
        "prediction_csv",
        "source_row_count",
        "source_layout",
        "round",
        "round_seed",
        "subset_size",
        "clean_count",
        "backdoor_count",
        "backdoor_ratio",
        "mean_backdoor_probability",
        "status",
        "error",
    ]
    detail_fields = [
        "model_name",
        "variant",
        "round",
        "round_seed",
        "subset_size",
        "subset_index",
        "base_100_position",
        "source_row_number",
        "group_number",
        "position_in_group",
        "prompt_index",
        "predicted_label",
        "predicted_tag",
        "backdoor_probability",
        "prediction_csv",
    ]
    write_csv(args.summary_csv, summary_rows, summary_fields)
    write_csv(args.details_csv, detail_rows, detail_fields)
    print(f"Wrote summary to {args.summary_csv}")
    print(f"Wrote selected-row details to {args.details_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run random 10/20/50-row prediction summaries")
    parser.add_argument(
        "--predictions_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--variant",
        default="auto",
        help=(
            "Ablation prediction to summarize: wo_cross, full, or auto. "
            "Auto selects the sole available one per model directory."
        ),
    )
    parser.add_argument("--summary_csv", type=Path, default=None)
    parser.add_argument("--details_csv", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--continue_on_error", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be >= 1")
    variant = normalize_variant(args.variant)
    output_root = RELEASE_ROOT / "artifacts" / "paper_results"
    args.summary_csv = args.summary_csv or output_root / f"random_{variant}_prediction_subset_summary.csv"
    args.details_csv = args.details_csv or output_root / f"random_{variant}_prediction_subset_details.csv"
    return args


if __name__ == "__main__":
    run(parse_args())
