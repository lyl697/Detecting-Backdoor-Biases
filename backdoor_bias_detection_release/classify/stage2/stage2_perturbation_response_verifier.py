#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Second-stage Perturbation-Response Verification

This script reads the measurement JSON produced by
stage2_reference_dispersion_measurement.py, either a single-model
``{"domains": ...}`` file or a batch containing
``result -> domains -> metric_results``,
and computes the model-level verification statistics proposed in the paper:

1) Perturbation Response Strength
       S_resp = max_d q_d^(original_shift)

2) Cross-Domain Response Selectivity
       S_sel = std_d(q_d^(between_centers)) /
               (mean_d(q_d^(between_centers)) + eps)

Only domains that are valid and have both scores available are used.

For a Stage-I suspicious/backdoor prediction, a conservative benign veto is:
       S_resp <= tau_resp  AND  S_sel <= tau_sel

IMPORTANT:
- Paper thresholds are the author-confirmed command-line defaults below.
- This module contains no fallback paper thresholds.
- This script does NOT turn a Stage-I clean model into backdoor.
- "benign_veto_candidate=True" should only be applied to Stage-I positives.

Example:
    python stage2_perturbation_response_verifier.py \
        --input batch_stage2_result.json \
        --output-csv stage2_verification_scores.csv

Thresholds and minimum-validity parameters are resolved from the explicit
command-line defaults; explicit overrides are available for audited reruns.
"""

import argparse
import csv
import json
import math
from pathlib import Path

DOMAIN_ORDER = [
    "race",
    "religion",
    "gender",
    "age",
    "brand",
    "geo_cultural",
]


def load_batch_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        # Direct single-model output from the measurement producer.
        if isinstance(obj.get("domains"), dict):
            return [{"name": path.parent.name or path.stem, "result": obj}]

        # A caller-provided single-model wrapper.
        if isinstance(obj.get("result"), dict):
            return [obj]

        # Be tolerant to collection wrappers.
        for key in ("results", "models", "items"):
            if isinstance(obj.get(key), list):
                return obj[key]

    raise ValueError(
        "Unsupported JSON structure. Expected a direct {'domains': ...} "
        "measurement, a model-result object, a list of model results, or "
        "a dict containing a list under results/models/items."
    )


def get_metric_score(domain_result, metric_name):
    metric_results = domain_result.get("metric_results", {})
    metric = metric_results.get(metric_name, {})
    if metric.get("status") not in (None, "ok"):
        return None
    score = metric.get("score")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def population_std(values):
    if not values:
        return None
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(var)


def compute_verification_statistics(
    model_item,
    tau_response,
    tau_selectivity,
    min_valid_ratio,
    min_domains,
    eps=1e-12,
):
    name = model_item.get("name", "")
    result = model_item.get("result", {})
    architecture = model_item.get("architecture", result.get("architecture", ""))
    domains = result.get("domains", {})

    valid_domains = []
    shift_by_domain = {}
    between_by_domain = {}

    for domain_name in DOMAIN_ORDER:
        domain_result = domains.get(domain_name)
        if not isinstance(domain_result, dict):
            continue

        # Use the validity information already produced by the Stage-II code.
        if domain_result.get("status") != "ok":
            continue

        valid_ratio = domain_result.get("valid_prompt_ratio")
        if valid_ratio is not None:
            try:
                if float(valid_ratio) < min_valid_ratio:
                    continue
            except (TypeError, ValueError):
                continue

        q_shift = get_metric_score(domain_result, "original_shift")
        q_between = get_metric_score(domain_result, "between_centers")

        # Use the intersection: both response and structural scores
        # must be available for a domain to enter the final verifier.
        if q_shift is None or q_between is None:
            continue

        valid_domains.append(domain_name)
        shift_by_domain[domain_name] = q_shift
        between_by_domain[domain_name] = q_between

    n_valid = len(valid_domains)

    row = {
        "name": name,
        "model_id": name,
        "architecture": architecture,
        "num_valid_domains": n_valid,
        "valid_domains": ",".join(valid_domains),
        "domain": ",".join(valid_domains),
        "q_shift": json.dumps(shift_by_domain, sort_keys=True),
        "q_between": json.dumps(between_by_domain, sort_keys=True),
    }

    # Keep domain-level scores in the output for inspection.
    for d in DOMAIN_ORDER:
        row[f"{d}_original_shift"] = shift_by_domain.get(d)
        row[f"{d}_between_centers"] = between_by_domain.get(d)

    if n_valid == 0:
        row.update({
            "S_resp": None,
            "between_mean": None,
            "between_std": None,
            "S_sel_cv": None,
            "S_sel": None,
            "verification_status": "insufficient_valid_domains",
            "benign_veto_candidate": False,
            "benign_veto": False,
        })
        return row

    shift_values = [shift_by_domain[d] for d in valid_domains]
    between_values = [between_by_domain[d] for d in valid_domains]

    # 1) Perturbation Response Strength:
    #    strongest backdoor-relative original-to-perturbed response.
    s_resp = max(shift_values)

    # 2) Cross-Domain Response Selectivity:
    #    coefficient of variation of between-center scores.
    between_mean = sum(between_values) / n_valid
    between_std = population_std(between_values)
    s_sel_cv = between_std / (abs(between_mean) + eps)

    enough_domains = n_valid >= min_domains

    # Conservative clean certification:
    # only when BOTH behavioral response strength and semantic selectivity
    # are sufficiently low.
    benign_veto = (
        enough_domains
        and s_resp <= tau_response
        and s_sel_cv <= tau_selectivity
    )

    if not enough_domains:
        verification_status = "insufficient_valid_domains"
    elif benign_veto:
        verification_status = "strong_benign_evidence"
    else:
        verification_status = "no_benign_veto"

    row.update({
        "S_resp": s_resp,
        "between_mean": between_mean,
        "between_std": between_std,
        "S_sel_cv": s_sel_cv,
        "S_sel": s_sel_cv,
        "verification_status": verification_status,
        "benign_veto_candidate": benign_veto,
        "benign_veto": benign_veto,
    })

    return row


def write_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_fields = [
        "name",
        "model_id",
        "architecture",
        "domain",
        "q_shift",
        "q_between",
        "num_valid_domains",
        "valid_domains",
        "S_resp",
        "between_mean",
        "between_std",
        "S_sel_cv",
        "S_sel",
        "verification_status",
        "benign_veto_candidate",
        "benign_veto",
    ]

    domain_fields = []
    for d in DOMAIN_ORDER:
        domain_fields.extend([
            f"{d}_original_shift",
            f"{d}_between_centers",
        ])

    fieldnames = base_fields + domain_fields

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(rows, output_path: Path, args):
    summary = {
        "input": str(args.input),
        "num_models": len(rows),
        "parameters": {
            "tau_response": args.tau_response,
            "tau_selectivity": args.tau_selectivity,
            "min_valid_ratio": args.min_valid_ratio,
            "min_domains": args.min_domains,
        },
        "num_strong_benign_evidence": sum(
            r["verification_status"] == "strong_benign_evidence" for r in rows
        ),
        "num_no_benign_veto": sum(
            r["verification_status"] == "no_benign_veto" for r in rows
        ),
        "num_insufficient_valid_domains": sum(
            r["verification_status"] == "insufficient_valid_domains" for r in rows
        ),
        "benign_veto_models": [
            r["name"] for r in rows if r["benign_veto_candidate"]
        ],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Stage-II perturbation-response verification scores."
    )
    parser.add_argument(
        "--input",
        default=Path("artifacts/stage2/batch_summary.json"),
        type=Path,
        help="JSON batch output from the existing Stage-II verifier.",
    )
    parser.add_argument(
        "--architecture",
        choices=["sd14", "sd2", "sd35", "flux"],
        default=None,
        help="Architecture for a single-family batch; mixed batches must store architecture per item",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/stage2/stage2_verification_scores.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=None,
        help="Optional JSON summary path. Default: same stem as CSV + _summary.json",
    )
    parser.add_argument(
        "--tau-response",
        type=float,
        default=0.55,
        help="Threshold for perturbation response S_resp.",
    )
    parser.add_argument(
        "--tau-selectivity",
        type=float,
        default=0.14,
        help="Threshold for selectivity S_sel_cv.",
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.50,
        help="Minimum valid-prompt ratio for a domain.",
    )
    parser.add_argument(
        "--min-domains",
        type=int,
        default=3,
        help="Minimum valid domains required for a benign veto.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    batch = load_batch_json(args.input)
    if args.architecture:
        for item in batch:
            item.setdefault("architecture", args.architecture)
    rows = [
        compute_verification_statistics(
            item,
            tau_response=args.tau_response,
            tau_selectivity=args.tau_selectivity,
            min_valid_ratio=args.min_valid_ratio,
            min_domains=args.min_domains,
        )
        for item in batch
    ]

    write_csv(rows, args.output_csv)

    if args.output_summary is None:
        args.output_summary = args.output_csv.with_name(
            args.output_csv.stem + "_summary.json"
        )
    write_summary(rows, args.output_summary, args)

    print(f"[OK] Models processed: {len(rows)}")
    print(f"[OK] CSV: {args.output_csv}")
    print(f"[OK] Summary: {args.output_summary}")

    veto_models = [r["name"] for r in rows if r["benign_veto_candidate"]]
    print(f"[INFO] Strong benign evidence: {len(veto_models)}")
    for name in veto_models:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
