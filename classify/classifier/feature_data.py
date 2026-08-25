"""Load, align, and normalize the five detector feature groups.

Feature CSVs and tensors are joined by prompt/model identity, aligned by
denoising step, and normalized using statistics fitted on training records.
"""

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


Label = int | None
RowKey = Tuple[int, str]
FeatureDict = Dict[str, float]
STEP_RE = re.compile(r"_step_(\d+)")
DEFAULT_STEPS = (0, 10, 20, 30, 40, 49)
LAST_HIDDEN_BLOCKS = 4

LABELS: Dict[str, Label] = {
    "clean": 0,
    "backdoor": 1,
    "test": None,
}


@dataclass
class SampleRecord:
    prompt_index: int
    model_tag: str
    label: Label
    hidden: torch.Tensor
    trace: FeatureDict


def _safe_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _read_csv(path: Path) -> List[dict]:
    if not path or not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _split_paths(path_text: str) -> List[Path]:
    return [Path(p) for p in str(path_text or "").split(";") if p.strip()]


def _haar_dwt_1d(vector: torch.Tensor) -> torch.Tensor:
    """Single-level Haar DWT for one block vector."""

    vector = vector.reshape(-1).float()
    if vector.numel() % 2 == 1:
        vector = torch.nn.functional.pad(vector, (0, 1))
    even = vector[0::2]
    odd = vector[1::2]
    scale = math.sqrt(2.0)
    approx = (even + odd) / scale
    detail = (even - odd) / scale
    return torch.cat([approx, detail], dim=0)


def _dwt_probability_vector(vector: torch.Tensor) -> torch.Tensor:
    coeffs = _haar_dwt_1d(vector)
    energy = coeffs.square()
    total = energy.sum()
    if float(total.item()) <= 1e-12:
        return torch.zeros_like(energy)
    return energy / total


def _dwt_probability_blocks(tensor: torch.Tensor) -> torch.Tensor:
    """Apply DWT and energy normalization independently to every block vector."""

    if tensor.dim() == 1:
        return _dwt_probability_vector(tensor).unsqueeze(0)
    if tensor.dim() == 2:
        return torch.stack([_dwt_probability_vector(row) for row in tensor], dim=0)

    leading_shape = tensor.shape[:-1]
    flat = tensor.reshape(-1, tensor.shape[-1])
    transformed = torch.stack([_dwt_probability_vector(row) for row in flat], dim=0)
    return transformed.reshape(*leading_shape, transformed.shape[-1])


""" def _load_hidden_tensor(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data)
    data = data.detach().float().cpu()
    if data.dim() == 1:
        return data.unsqueeze(0)
    if data.dim() == 2:
        return data[-LAST_HIDDEN_BLOCKS:]
    if data.dim() == 3 and data.shape[0] == 1:
        return data.squeeze(0)[-LAST_HIDDEN_BLOCKS:]
    if data.dim() >= 3:
        data = data[:, -LAST_HIDDEN_BLOCKS:, ...]
        return data.flatten(start_dim=1)
    return data.reshape(1, -1) """


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")



def _parse_metric_payload(payload_text: str) -> dict:
    if not payload_text:
        return {}
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _flatten_trace_payload(payload_text: str, prefix: str) -> FeatureDict:
    payload = _parse_metric_payload(payload_text)
    features: FeatureDict = {}

    if prefix == "trace_norm":
        norm_values: Dict[int, List[float]] = {}
        for step_text, step_payload in payload.items():
            if not isinstance(step_payload, dict):
                continue
            step = int(float(step_text))
            from_step = int(float(step_payload.get("from_step", step)))
            for target_step, metric_name in (
                (from_step, "mean_prev_norm"),
                (step, "mean_curr_norm"),
                (step, "mean_norm"),
            ):
                if metric_name not in step_payload:
                    continue
                value = _safe_float(step_payload.get(metric_name))
                if math.isfinite(value):
                    norm_values.setdefault(target_step, []).append(value)

        for step, values in norm_values.items():
            features[f"{prefix}_step_{step:03d}_mean_norm"] = float(np.mean(values))
        return features

    for step_text, step_payload in payload.items():
        if not isinstance(step_payload, dict):
            continue
        step = int(float(step_text))
        for metric_name, value in step_payload.items():
            if metric_name in {"from_step", "to_step", "num_pairs", "step"}:
                continue
            clean_name = "mean_cosine" if metric_name == "mean_angle_deg" else metric_name
            features[f"{prefix}_step_{step:03d}_{clean_name}"] = _safe_float(value)
    return features


def load_trace_features(
    angle_csv: Path | None,
    norm_csv: Path | None,
    delta_csv: Path | None,
) -> Dict[RowKey, FeatureDict]:
    specs = [
        ("trace_cosine", angle_csv),
        ("trace_norm", norm_csv),
        ("trace_delta", delta_csv),
    ]
    features_by_key: Dict[RowKey, FeatureDict] = {}

    for prefix, csv_path in specs:
        if not csv_path:
            continue
        for row in _read_csv(csv_path):
            prompt_index = int(row["prompt_index"])
            for model_column in ("clean_orig", "backdoor_orig", "test_orig"):
                if model_column not in row:
                    continue
                model_tag = model_column.removesuffix("_orig")
                key = (prompt_index, model_tag)
                features_by_key.setdefault(key, {}).update(
                    _flatten_trace_payload(row[model_column], prefix)
                )

    return features_by_key


def _split_csv_paths(csv_paths) -> List[Path]:
    if not csv_paths:
        return []
    paths = []
    for chunk in str(csv_paths).replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            paths.append(Path(chunk))
    return paths


def load_numeric_feature_csv(
    csv_paths,
    allowed_prefixes: Tuple[str, ...],
) -> Dict[RowKey, FeatureDict]:
    """Load numeric feature columns from one or more CSVs keyed by (prompt_index, model_tag)."""

    features_by_key: Dict[RowKey, FeatureDict] = {}
    paths = _split_csv_paths(csv_paths)
    if not paths:
        return features_by_key

    metadata = {
        "prompt_index",
        "original_prompt",
        "model_tag",
        "label",
        "target_paths",
        "clean_ref_paths",
        "backdoor_ref_paths",
        "ref_paths",
    }
    for csv_path in paths:
        for row in _read_csv(csv_path):
            if not row.get("prompt_index") or not row.get("model_tag"):
                continue
            key = (int(row["prompt_index"]), row["model_tag"])
            for column, value in row.items():
                if column in metadata:
                    continue
                if not any(column.startswith(prefix) for prefix in allowed_prefixes):
                    continue
                features_by_key.setdefault(key, {})[column] = _safe_float(value)
    return features_by_key


def _load_raw_feature_tensor(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data)
    data = data.detach().float().cpu()
    if data.dim() == 0:
        return data.reshape(1, 1)
    if data.dim() == 1:
        return data.unsqueeze(0)
    if data.dim() > 2:
        return data.flatten(start_dim=1)
    return data


def _add_navi_vector_features(
    features: FeatureDict,
    path_text: str,
    prefix: str,
    steps: List[int],
) -> None:
    if not path_text:
        return
    tensor = _load_raw_feature_tensor(Path(path_text))

    for step_idx, step_vector in enumerate(tensor):
        step = steps[step_idx] if step_idx < len(steps) else step_idx
        for dim_idx, value in enumerate(step_vector.reshape(-1)):
            features[f"navi_{prefix}_step_{step:03d}_dim_{dim_idx:04d}"] = float(value.item())


def load_activation_difference_features(csv_paths, steps: List[int]) -> Dict[RowKey, FeatureDict]:
    """Load clean/backdoor reference activation-difference vectors by step."""

    features_by_key: Dict[RowKey, FeatureDict] = {}
    for csv_path in _split_csv_paths(csv_paths):
        for row in _read_csv(csv_path):
            if not row.get("prompt_index") or not row.get("model_tag"):
                continue
            key = (int(row["prompt_index"]), row["model_tag"])
            features = features_by_key.setdefault(key, {})
            _add_navi_vector_features(
                features,
                row.get("clean_ref_difference_vector_path", ""),
                "clean_ref",
                steps,
            )
            _add_navi_vector_features(
                features,
                row.get("backdoor_ref_difference_vector_path", ""),
                "backdoor_ref",
                steps,
            )
    return features_by_key



def parse_steps(step_text: str | None) -> List[int]:
    if not step_text:
        return list(DEFAULT_STEPS)
    steps = [int(item.strip()) for item in str(step_text).split(",") if item.strip()]
    return steps or list(DEFAULT_STEPS)


def optional_path(path_text: str | None) -> Path | None:
    if path_text is None or not str(path_text).strip():
        return None
    return Path(path_text)


def _feature_step_and_name(name: str) -> Tuple[int | None, str]:
    match = STEP_RE.search(name)
    if not match:
        return None, f"global_{name}"
    step = int(match.group(1))
    base_name = f"{name[:match.start()]}{name[match.end():]}"
    return step, base_name


def _build_step_feature_array(
    records: List[SampleRecord],
    steps: List[int],
    feature_names: List[str],
    return_mask: bool = False,
):
    if not feature_names:
        x = np.zeros((len(records), len(steps), 0), dtype=np.float32)
        mask = np.zeros_like(x, dtype=bool)
        return (x, mask) if return_mask else x

    step_to_idx = {step: idx for idx, step in enumerate(steps)}
    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    x = np.zeros((len(records), len(steps), len(feature_names)), dtype=np.float32)
    mask = np.zeros_like(x, dtype=bool)

    for record_idx, record in enumerate(records):
        for raw_name, raw_value in record.trace.items():
            step, feature_name = _feature_step_and_name(raw_name)
            feature_idx = feature_to_idx.get(feature_name)
            if feature_idx is None:
                continue

            value = _safe_float(raw_value)
            if not math.isfinite(value):
                continue
            if step is None:
                x[record_idx, :, feature_idx] = value
                mask[record_idx, :, feature_idx] = True
                continue

            step_idx = step_to_idx.get(step)
            if step_idx is not None:
                x[record_idx, step_idx, feature_idx] = value
                mask[record_idx, step_idx, feature_idx] = True

    return (x, mask) if return_mask else x


def _fit_masked_minmax(x: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    feature_dim = x.shape[-1]
    flat_x = x.reshape(-1, feature_dim)
    flat_mask = mask.reshape(-1, feature_dim)
    mins = np.zeros(feature_dim, dtype=np.float32)
    ranges = np.ones(feature_dim, dtype=np.float32)

    for idx in range(feature_dim):
        values = flat_x[flat_mask[:, idx], idx]
        if values.size == 0:
            continue
        min_value = float(values.min())
        max_value = float(values.max())
        mins[idx] = min_value
        value_range = max_value - min_value
        ranges[idx] = value_range if value_range > 1e-6 else 1.0

    return mins, ranges


def _apply_masked_minmax(
    x: np.ndarray,
    mask: np.ndarray,
    mins: np.ndarray,
    ranges: np.ndarray,
) -> np.ndarray:
    normalized = np.zeros_like(x, dtype=np.float32)
    values = (x - mins.reshape(1, 1, -1)) / ranges.reshape(1, 1, -1)
    normalized[mask] = values[mask]
    return normalized.astype(np.float32)


def fit_trace_preprocessor(records: List[SampleRecord], steps: List[int]):
    feature_names = sorted(
        {_feature_step_and_name(key)[1] for record in records for key in record.trace}
    )
    x, mask = _build_step_feature_array(records, steps, feature_names, return_mask=True)
    if not feature_names:
        return feature_names, np.zeros(0), np.ones(0), x

    mins, ranges = _fit_masked_minmax(x, mask)
    normalized = _apply_masked_minmax(x, mask, mins, ranges)
    return feature_names, mins, ranges, normalized


def transform_trace(
    records: List[SampleRecord],
    steps: List[int],
    feature_names: List[str],
    means,
    stds,
) -> np.ndarray:
    x, mask = _build_step_feature_array(records, steps, feature_names, return_mask=True)
    if not feature_names:
        return x
    mins = np.asarray(means, dtype=np.float32)
    ranges = np.asarray(stds, dtype=np.float32)
    return _apply_masked_minmax(x, mask, mins, ranges)



def split_train_val(
    records: List[SampleRecord],
    val_ratio: float,
    random_state: int,
):
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    indices = np.arange(len(records))
    labels = np.array([int(record.label) for record in records])
    groups = np.array([record.prompt_index for record in records])

    try:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
        train_idx, val_idx = next(splitter.split(indices, labels, groups=groups))
        if len(set(labels[train_idx].tolist())) < 2 or len(set(labels[val_idx].tolist())) < 2:
            raise ValueError
    except ValueError:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=random_state,
            stratify=labels,
        )
    return train_idx, val_idx


