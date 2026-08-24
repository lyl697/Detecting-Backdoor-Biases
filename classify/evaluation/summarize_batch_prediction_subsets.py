"""Summarize fixed 10/20/50-row subsets of batch prediction CSV files.

For a 100-row prediction file, rows are used directly. For a 1000-row file,
the configured one-based source ranges are first concatenated into 100 rows.
The resulting rows are split into ten consecutive groups of ten and sampled
with three schemes: front, back, and middle.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


THOUSAND_ROW_RANGES: Tuple[Tuple[int, int], ...] = (
    (1, 10),
    (81, 90),
    (161, 170),
    (241, 250),
    (321, 330),
    (401, 410),
    (481, 490),
    (561, 570),
    (841, 845),
    (861, 865),
    (881, 885),
    (901, 905),
)

SCHEMES: Dict[str, Dict[int, Tuple[int, ...]]] = {
    "front": {
        10: (0,),
        20: (0, 1),
        50: (0, 1, 2, 3, 4),
    },
    "back": {
        10: (9,),
        20: (8, 9),
        50: (5, 6, 7, 8, 9),
    },
    "middle": {
        10: (4,),
        20: (4, 5),
        50: (3, 4, 5, 6, 7),
    },
}


def read_rows(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_label(row: dict, path: Path, row_number: int) -> int:
    raw = str(row.get("predicted_label", "")).strip()
    if raw in {"0", "1"}:
        return int(raw)

    tag = str(row.get("predicted_tag", "")).strip().lower()
    if tag == "clean":
        return 0
    if tag == "backdoor":
        return 1
    raise ValueError(
        f"Invalid predicted label at {path}, data row {row_number}: "
        f"predicted_label={raw!r}, predicted_tag={tag!r}"
    )


def select_base_rows(rows: Sequence[dict]) -> Tuple[List[dict], List[int], str]:
    if len(rows) == 100:
        return list(rows), list(range(1, 101)), "direct_100"
    if len(rows) != 1000:
        raise ValueError(f"expected 100 or 1000 data rows, found {len(rows)}")

    selected_rows: List[dict] = []
    source_numbers: List[int] = []
    for start, end in THOUSAND_ROW_RANGES:
        selected_rows.extend(rows[start - 1 : end])
        source_numbers.extend(range(start, end + 1))
    if len(selected_rows) != 100:
        raise RuntimeError(
            f"Internal 1000-row range configuration selected {len(selected_rows)} rows"
        )
    return selected_rows, source_numbers, "selected_100_from_1000"


def subset_positions(offsets: Sequence[int]) -> List[int]:
    return [group_start + offset for group_start in range(0, 100, 10) for offset in offsets]


def mean_probability(rows: Sequence[dict]) -> float | None:
    values = []
    for row in rows:
        try:
            values.append(float(row["backdoor_probability"]))
        except (KeyError, TypeError, ValueError):
            continue
    return sum(values) / len(values) if values else None


def summarize_file(path: Path) -> Tuple[List[dict], List[dict]]:
    source_rows = read_rows(path)
    base_rows, source_numbers, source_layout = select_base_rows(source_rows)
    summary_rows = []
    detail_rows = []

    for scheme_name, sizes in SCHEMES.items():
        for requested_size, offsets in sizes.items():
            positions = subset_positions(offsets)
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
                    "model_name": path.stem,
                    "prediction_csv": str(path),
                    "source_row_count": len(source_rows),
                    "source_layout": source_layout,
                    "scheme": scheme_name,
                    "subset_size": requested_size,
                    "clean_count": clean_count,
                    "backdoor_count": backdoor_count,
                    "backdoor_ratio": backdoor_count / requested_size,
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
                        "model_name": path.stem,
                        "scheme": scheme_name,
                        "subset_size": requested_size,
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


def write_csv(path: Path, rows: List[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> None:
    prediction_dir = args.predictions_dir
    files = sorted(
        path
        for path in prediction_dir.glob("*.csv")
        if path.resolve() not in {args.summary_csv.resolve(), args.details_csv.resolve()}
    )
    if not files:
        raise ValueError(f"No prediction CSV files found in {prediction_dir}")

    summary_rows: List[dict] = []
    detail_rows: List[dict] = []
    for path in files:
        try:
            file_summary, file_details = summarize_file(path)
            summary_rows.extend(file_summary)
            detail_rows.extend(file_details)
            counts = ", ".join(
                f"{row['scheme']}-{row['subset_size']}: "
                f"{row['backdoor_count']}/{row['subset_size']} backdoor"
                for row in file_summary
            )
            print(f"[{path.stem}] {counts}")
        except Exception as exc:
            summary_rows.append(
                {
                    "model_name": path.stem,
                    "prediction_csv": str(path),
                    "source_row_count": len(read_rows(path)),
                    "source_layout": "",
                    "scheme": "",
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
        "prediction_csv",
        "source_row_count",
        "source_layout",
        "scheme",
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
        "scheme",
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
    parser = argparse.ArgumentParser(description="Summarize fixed subsets of batch prediction CSV files")
    parser.add_argument(
        "--predictions_dir",
        type=Path,
        default=Path("classify/output/old_detector_800_SD35/predictions"),
    )
    parser.add_argument("--summary_csv", type=Path, default=None)
    parser.add_argument("--details_csv", type=Path, default=None)
    parser.add_argument("--continue_on_error", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    output_root = args.predictions_dir.parent
    args.summary_csv = args.summary_csv or output_root / "prediction_subset_summary.csv"
    args.details_csv = args.details_csv or output_root / "prediction_subset_details.csv"
    return args


if __name__ == "__main__":
    run(parse_args())
