"""Unified single-model and batch feature-generation entry point.

This launcher does not alter numerical feature calculations. A manifest job
selects a feature group, while --model_family selects the preserved pipeline
strategy. One selected job is a single-model run; multiple selected jobs are a
batch run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


CLASSIFY_ROOT = Path(__file__).resolve().parent
BACKENDS = {
    # Intrinsic: LFD + adjacent latent trajectory.
    "intrinsic_train_odd_even": "features/intrinsic/odd_even_joint_lfd_trace_train_features.py",
    "intrinsic_test": "features/intrinsic/test_features.py",
    # Reference assisted: activation + latent + decoded-image discrepancy.
    "reference_train_odd_even": "features/reference/odd_even_joint_reference_train_features.py",
    "reference_test": "features/reference/test_features.py",
}

RUNTIME_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_runtime_value(value):
    if not isinstance(value, str):
        return value
    value = value.replace("${REPO_ROOT}", str(CLASSIFY_ROOT.parent))
    missing = sorted({name for name in RUNTIME_VARIABLE.findall(value) if name not in os.environ})
    if missing:
        raise ValueError("Missing runtime environment variables: " + ", ".join(missing))
    return os.path.expandvars(value)


def cli_arguments(values: dict) -> list[str]:
    result: list[str] = []
    boolean_optional = set(values.pop("__boolean_optional__", []))
    for flag, value in values.items():
        if isinstance(value, bool):
            if value:
                result.append(flag)
            elif flag in boolean_optional:
                result.append("--no-" + flag.removeprefix("--"))
        elif value is not None:
            result.extend((flag, str(value)))
    return result


def read_jobs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError(f"{path} must contain a non-empty 'jobs' list")
    names = [str(job.get("name", "")) for job in jobs]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("Every job needs a unique non-empty name")
    return jobs


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "backend", "status", "exit_code", "started_utc", "finished_utc", "log", "command"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job", action="append", help="Run only this named job; repeat to select several")
    parser.add_argument("--list", action="store_true", help="List jobs and exit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=Path("artifacts/feature_generation_runs"))
    args = parser.parse_args()

    jobs = read_jobs(args.manifest)
    if args.list:
        for job in jobs:
            print(f"{job['name']}\t{job.get('backend', '')}\t{job.get('description', '')}")
        return
    selected = set(args.job or [job["name"] for job in jobs])
    unknown = selected.difference(job["name"] for job in jobs)
    if unknown:
        raise ValueError(f"Unknown jobs: {sorted(unknown)}")

    run_dir = args.log_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows: list[dict] = []
    for job in jobs:
        if job["name"] not in selected:
            continue
        backend_name = job.get("backend")
        if backend_name not in BACKENDS:
            raise ValueError(f"{job['name']}: unsupported backend {backend_name!r}")
        script = CLASSIFY_ROOT / BACKENDS[backend_name]
        values = {
            flag: expand_runtime_value(value)
            for flag, value in job.get("arguments", {}).items()
        }
        command = [sys.executable, str(script), *cli_arguments(values)]
        print(shlex.join(command))
        if args.dry_run:
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / f"{job['name']}.log"
        started = datetime.now(timezone.utc).isoformat()
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=CLASSIFY_ROOT.parent, stdout=log, stderr=subprocess.STDOUT)
        row = {
            "name": job["name"], "backend": backend_name,
            "status": "ok" if result.returncode == 0 else "failed",
            "exit_code": result.returncode, "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "log": str(log_path), "command": shlex.join(command),
        }
        rows.append(row)
        write_summary(run_dir / "summary.csv", rows)
        if result.returncode and not args.continue_on_error:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
