"""Architecture-agnostic step-aligned detector with flat chunk Transformers.

LFD and Navi processing
-----------------------
For each denoising step, all non-step axes are flattened while the step axis is
preserved.  No linear interpolation and no PCA are used.  An arbitrary-length
flat feature vector is split into fixed-width chunks, each chunk is projected to
a Transformer token, variable token counts are handled with padding masks, and
masked attention pooling reduces the sequence to one fixed-size token per step.

Pipeline per LFD/Navi group::

    [step, ...] -> flatten non-step axes -> [step, D]
    -> chunk into [step, ceil(D/chunk_dim), chunk_dim]
    -> token projection + sinusoidal chunk position encoding
    -> masked Transformer -> masked attention pooling
    -> [step, token_dim]

LFD and Navi use separate encoders but share the same chunk/token dimensions.
Their resulting step tokens are concatenated with the original normalized trace
and original image-similarity scalars.  The resulting ``[step, d_total]`` array
is processed by a 1D CNN -> Transformer -> MLP detector.  Trace and image
features are not projected to ``token_dim``.
"""

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


TRAIN_LABELS = {"clean": 0, "backdoor": 1}
TRACE_FAMILIES = ("trace_cosine", "trace_norm", "trace_delta")
IMAGE_RE = re.compile(r"step_similarity_(\d+)_to_(\d+)_(.+)")



@dataclass
class RawSample:
    prompt_index: int
    original_prompt: str
    model_tag: str
    label: int | None
    lfd: np.ndarray                 # [step, arbitrary flattened dimension]
    navi: np.ndarray                # [step, arbitrary flattened dimension]
    trace: Dict[int, Dict[str, float]]
    image: Dict[int, Dict[str, float]]



# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def _read_csv(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _safe_float(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _resolve_path(text: str, source_csv: Path) -> Path:
    path = Path(str(text).strip())
    if path.exists() or path.is_absolute():
        return path
    candidate = source_csv.parent / path
    return candidate if candidate.exists() else path


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _tensor_from_file(path: Path) -> torch.Tensor:
    value = _torch_load(path)
    if isinstance(value, dict):
        if "values" not in value:
            raise ValueError(f"Missing 'values' in activation summary: {path}")
        value = value["values"]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.detach().float().cpu()


def _mean_preserve_shape(path_text: str, source_csv: Path) -> np.ndarray:
    """Average replicas while preserving all original non-step axes.

    Replicas are normally shape-identical.  If they are not, each tensor is
    padded only for the purpose of replica averaging and a per-element count is
    used, so padded zeros never bias the mean.
    """
    tensors: List[torch.Tensor] = []
    for text in str(path_text or "").split(";"):
        if not text.strip():
            continue
        tensor = _tensor_from_file(_resolve_path(text, source_csv))
        if tensor.ndim == 0:
            tensor = tensor.reshape(1, 1)
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        tensors.append(tensor)

    if not tensors:
        return np.empty((0,), dtype=np.float32)

    max_ndim = max(t.ndim for t in tensors)
    normalized = []
    for tensor in tensors:
        while tensor.ndim < max_ndim:
            tensor = tensor.unsqueeze(-1)
        normalized.append(tensor)

    shape = tuple(max(t.shape[d] for t in normalized) for d in range(max_ndim))
    total = torch.zeros(shape, dtype=torch.float32)
    count = torch.zeros(shape, dtype=torch.float32)

    for tensor in normalized:
        slices = tuple(slice(0, size) for size in tensor.shape)
        finite = torch.isfinite(tensor)
        clean = torch.where(finite, tensor, torch.zeros_like(tensor))
        total[slices] += clean
        count[slices] += finite.float()

    mean = total / count.clamp_min(1.0)
    mean[count == 0] = 0.0
    return mean.numpy().astype(np.float32, copy=False)


def _flatten_preserve_step(array: np.ndarray) -> np.ndarray:
    """Flatten every non-step axis and keep the denoising-step axis intact."""
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array[None, :]
    else:
        array = array.reshape(array.shape[0], -1)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _parse_json(text: str) -> dict:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_trace_file(path: Path, family: str, output):
    for row in _read_csv(path):
        prompt_idx = int(row["prompt_index"])
        for column, text in row.items():
            if not column.endswith("_orig") or not text:
                continue
            tag = column.removesuffix("_orig")
            key = (prompt_idx, tag)
            sample = output.setdefault(key, {})
            for step_text, metrics in _parse_json(text).items():
                if not isinstance(metrics, dict):
                    continue
                step = int(float(step_text))
                step_features = sample.setdefault(step, {})
                for metric, value in metrics.items():
                    if metric in {"from_step", "to_step", "num_pairs", "step"}:
                        continue
                    step_features[f"{family}_{metric}"] = _safe_float(value)


def load_trace(paths) -> Dict[Tuple[int, str], Dict[int, Dict[str, float]]]:
    output = {}
    _load_trace_file(Path(paths["trace_angle_csv"]), "trace_cosine", output)
    _load_trace_file(Path(paths["trace_norm_csv"]), "trace_norm", output)
    _load_trace_file(Path(paths["trace_delta_csv"]), "trace_delta", output)
    return output


def load_image(path: Path):
    output = {}
    for row in _read_csv(path):
        prompt_idx = int(row["prompt_index"])
        explicit_tag = str(row.get("model_tag", "")).strip()
        tags = [explicit_tag] if explicit_tag else ["clean", "backdoor"]
        for tag in tags:
            sample = {}
            prefix = "" if explicit_tag else f"{tag}_"
            for column, value in row.items():
                canonical = column[len(prefix):] if prefix and column.startswith(prefix) else column
                match = IMAGE_RE.fullmatch(canonical)
                if match:
                    destination = int(match.group(2))
                    sample.setdefault(destination, {})[f"image_{match.group(3)}"] = _safe_float(value)
            if sample:
                output[(prompt_idx, tag)] = sample
    return output


def load_navi(path: Path):
    output = {}
    for row in _read_csv(path):
        prompt_idx = int(row["prompt_index"])
        explicit_tag = str(row.get("model_tag", "")).strip()
        if explicit_tag and row.get("activation_summary_path"):
            raw = _mean_preserve_shape(row["activation_summary_path"], path)
            output[(prompt_idx, explicit_tag)] = _flatten_preserve_step(raw)
        else:
            for tag in ("clean", "backdoor"):
                column = f"{tag}_activation_summary_path"
                if row.get(column):
                    raw = _mean_preserve_shape(row[column], path)
                    output[(prompt_idx, tag)] = _flatten_preserve_step(raw)
    return output


def load_lfd(path: Path):
    output = {}
    for row in _read_csv(path):
        prompt_idx = int(row["prompt_index"])
        explicit_tag = str(row.get("model_tag", "")).strip()
        if explicit_tag and row.get("feature_paths"):
            raw = _mean_preserve_shape(row["feature_paths"], path)
            output[(prompt_idx, explicit_tag)] = _flatten_preserve_step(raw)
        for tag in ("clean", "backdoor", "test"):
            for column in (f"{tag}_orig_feature_paths", f"{tag}_feature_paths"):
                if row.get(column):
                    raw = _mean_preserve_shape(row[column], path)
                    output[(prompt_idx, tag)] = _flatten_preserve_step(raw)
                    break
    return output


def load_prompt_texts(path: Path) -> Dict[int, str]:
    """Load prompt text from the dataset spine for leakage-safe grouping."""
    output = {}
    for row in _read_csv(path):
        prompt_idx = int(row["prompt_index"])
        output[prompt_idx] = str(row.get("original_prompt", "")).strip()
    return output


def _lookup_prompt_feature(features, prompt_idx: int, preferred_tag: str):
    """Use an exact model tag when possible, otherwise the sole prompt entry.

    This fallback is intended for prediction, where LFD may use ``test`` while
    target-only image/Navi CSVs may retain a user-supplied clean/backdoor tag.
    Labels and tag names must not affect feature alignment at prediction time.
    """
    exact = features.get((prompt_idx, preferred_tag))
    if exact is not None:
        return exact
    candidates = [value for (idx, _tag), value in features.items() if idx == prompt_idx]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _dataset_samples(paths, require_label: bool) -> List[RawSample]:
    hidden_path = Path(paths["hidden_csv"])
    lfd = load_lfd(hidden_path)
    prompt_texts = load_prompt_texts(hidden_path)
    navi = load_navi(Path(paths["navi_hidden_csv"]))
    trace = load_trace(paths)
    image = load_image(Path(paths["image_similarity_csv"]))
    records = []
    if require_label:
        keys = sorted(set(lfd) & set(navi) & set(trace) & set(image))
        for key in keys:
            label = TRAIN_LABELS.get(key[1])
            if label is None or not lfd[key].size or not navi[key].size:
                continue
            records.append(RawSample(
                key[0], prompt_texts.get(key[0], ""), key[1], label,
                lfd[key], navi[key], trace[key], image[key],
            ))
    else:
        # Prediction deliberately ignores labels and tolerates differing source tags.
        for (prompt_idx, tag), lfd_value in sorted(lfd.items()):
            navi_value = _lookup_prompt_feature(navi, prompt_idx, tag)
            trace_value = _lookup_prompt_feature(trace, prompt_idx, tag)
            image_value = _lookup_prompt_feature(image, prompt_idx, tag)
            if navi_value is None or trace_value is None or image_value is None:
                continue
            if not lfd_value.size or not navi_value.size:
                continue
            records.append(RawSample(
                prompt_idx, prompt_texts.get(prompt_idx, ""), "test", None,
                lfd_value, navi_value, trace_value, image_value,
            ))
    print(
        f"aligned={len(records)}, raw keys: lfd={len(lfd)}, trace={len(trace)}, "
        f"image={len(image)}, navi={len(navi)}"
    )
    if records:
        lfd_shapes = sorted({tuple(r.lfd.shape) for r in records})
        navi_shapes = sorted({tuple(r.navi.shape) for r in records})
        print(f"LFD flattened shapes ({len(lfd_shapes)}): {lfd_shapes[:12]}{' ...' if len(lfd_shapes) > 12 else ''}")
        print(f"Navi flattened shapes ({len(navi_shapes)}): {navi_shapes[:12]}{' ...' if len(navi_shapes) > 12 else ''}")
    return records


def load_samples(args, require_label=True):
    names = (
        "hidden_csv", "trace_angle_csv", "trace_norm_csv", "trace_delta_csv",
        "image_similarity_csv", "navi_hidden_csv",
    )
    if args.dataset_manifest:
        with args.dataset_manifest.open("r", encoding="utf-8") as stream:
            datasets = json.load(stream)
    else:
        datasets = [{name: getattr(args, name) for name in names}]
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Dataset manifest must be a non-empty JSON list")

    combined = []
    prompt_group_ids = {}
    for dataset_idx, paths in enumerate(datasets):
        missing = [name for name in names if not paths.get(name)]
        if missing:
            raise ValueError(f"Dataset {dataset_idx} missing paths: {missing}")
        records = _dataset_samples(paths, require_label)
        for record in records:
            # Same prompt text across SD2/SD3.5 receives the same group id.
            # Empty prompt text falls back to a dataset-local identity.
            group_key = (
                ("text", record.original_prompt)
                if record.original_prompt
                else ("dataset_index", dataset_idx, record.prompt_index)
            )
            if group_key not in prompt_group_ids:
                prompt_group_ids[group_key] = len(prompt_group_ids)
            record.prompt_index = prompt_group_ids[group_key]
        combined.extend(records)
    if not combined:
        raise ValueError("No samples contain all required feature groups")
    return combined


def split_by_prompt(samples, val_ratio, seed):
    prompts = sorted({sample.prompt_index for sample in samples})
    if len(prompts) < 2:
        raise ValueError("At least two prompts are required")
    random.Random(seed).shuffle(prompts)
    count = min(len(prompts) - 1, max(1, round(len(prompts) * val_ratio)))
    val_prompts = set(prompts[:count])
    train_idx = [idx for idx, sample in enumerate(samples) if sample.prompt_index not in val_prompts]
    val_idx = [idx for idx, sample in enumerate(samples) if sample.prompt_index in val_prompts]
    return train_idx, val_idx


# -----------------------------------------------------------------------------
# Variable-length flattened LFD / Navi helpers
# -----------------------------------------------------------------------------

def _align_flat_steps(array: np.ndarray, num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Align one flattened feature sample to ``num_steps`` without resampling.

    Returns
    -------
    aligned:
        ``[num_steps, D]``. Missing steps are zero-filled.
    lengths:
        ``[num_steps]``. Each present step has its original flattened width D;
        missing steps have length 0. These lengths later create chunk masks.
    """
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected flattened [step, feature], got {array.shape}")

    width = int(array.shape[1])
    aligned = np.zeros((num_steps, width), dtype=np.float32)
    lengths = np.zeros((num_steps,), dtype=np.int64)
    use_steps = min(num_steps, int(array.shape[0]))
    if use_steps > 0 and width > 0:
        aligned[:use_steps] = np.nan_to_num(
            array[:use_steps], nan=0.0, posinf=0.0, neginf=0.0
        )
        lengths[:use_steps] = width
    return aligned, lengths


# -----------------------------------------------------------------------------
# Trace / image groups
# -----------------------------------------------------------------------------

def _step_dict_layout(samples, attribute):
    return sorted({name for sample in samples for values in getattr(sample, attribute).values() for name in values})


def _step_dict_array(samples, attribute, names, steps, normalize_trace=False):
    output = np.full((len(samples), len(steps), len(names)), np.nan, dtype=np.float32)
    name_index = {name: idx for idx, name in enumerate(names)}
    step_index = {step: idx for idx, step in enumerate(steps)}
    for sample_idx, sample in enumerate(samples):
        for step, values in getattr(sample, attribute).items():
            if step not in step_index:
                continue
            for name, value in values.items():
                if name in name_index:
                    output[sample_idx, step_index[step], name_index[name]] = value
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    if normalize_trace:
        for family in TRACE_FAMILIES:
            indices = [idx for idx, name in enumerate(names) if name.startswith(family + "_")]
            if indices:
                block = output[:, :, indices]
                norm = np.linalg.norm(block.reshape(len(samples), -1), axis=1)
                output[:, :, indices] = block / np.maximum(norm[:, None, None], 1e-12)
    return output


# -----------------------------------------------------------------------------
# Dataset and dynamic batch padding
# -----------------------------------------------------------------------------

class StepFeatureDataset(Dataset):
    """Keep LFD/Navi variable-length until collation to avoid global padding."""

    def __init__(self, samples, indices, trace, image, labels, num_steps):
        self.samples = samples
        self.indices = list(indices)
        self.trace = trace
        self.image = image
        self.labels = labels
        self.num_steps = num_steps

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = self.indices[item]
        sample = self.samples[idx]
        lfd, lfd_lengths = _align_flat_steps(sample.lfd, self.num_steps)
        navi, navi_lengths = _align_flat_steps(sample.navi, self.num_steps)
        return (
            lfd, lfd_lengths,
            self.trace[idx], self.image[idx],
            navi, navi_lengths,
            int(self.labels[idx]),
        )


def _pad_flat_group(arrays, lengths):
    """Pad variable ``[step, D_i]`` arrays only to this batch's maximum D."""
    batch = len(arrays)
    steps = arrays[0].shape[0]
    width = max(array.shape[1] for array in arrays)
    data = torch.zeros((batch, steps, width), dtype=torch.float32)
    length_tensor = torch.zeros((batch, steps), dtype=torch.long)
    for i, (array, row_lengths) in enumerate(zip(arrays, lengths)):
        w = array.shape[1]
        if w:
            data[i, :, :w] = torch.from_numpy(array).float()
        length_tensor[i] = torch.as_tensor(row_lengths, dtype=torch.long)
    return data, length_tensor


def collate_variable_features(batch):
    lfd, lfd_lengths, trace, image, navi, navi_lengths, labels = zip(*batch)
    lfd, lfd_lengths = _pad_flat_group(lfd, lfd_lengths)
    navi, navi_lengths = _pad_flat_group(navi, navi_lengths)
    trace = torch.from_numpy(np.stack(trace)).float()
    image = torch.from_numpy(np.stack(image)).float()
    labels = torch.as_tensor(labels, dtype=torch.long)
    return lfd, lfd_lengths, trace, image, navi, navi_lengths, labels


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class VariableLengthFeatureEncoder(nn.Module):
    """Chunk a variable-width flat vector and reduce it to one token per step."""

    def __init__(self, chunk_dim: int, token_dim: int, layers: int,
                 nhead: int, dropout: float, ff_mult: int = 4):
        super().__init__()
        if chunk_dim <= 0:
            raise ValueError("chunk_dim must be positive")
        if token_dim % nhead:
            raise ValueError("token_dim must be divisible by chunk_nhead")
        self.chunk_dim = int(chunk_dim)
        self.token_dim = int(token_dim)

        self.chunk_projection = nn.Sequential(
            nn.LayerNorm(self.chunk_dim),
            nn.Linear(self.chunk_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=nhead,
            dim_feedforward=self.token_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.pool_score = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, 1),
        )
        self.output_norm = nn.LayerNorm(self.token_dim)

    @staticmethod
    def _sinusoidal_position(num_tokens: int, dim: int, device, dtype):
        """Deterministic chunk-position encoding; no model-specific boundaries."""
        position = torch.arange(num_tokens, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(dim, 1))
        )
        pe = torch.zeros((num_tokens, dim), device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        return pe.to(dtype=dtype)

    def forward(self, x: torch.Tensor, feature_lengths: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            ``[batch, step, D_batch_max]``. D is padded only within the batch.
        feature_lengths:
            ``[batch, step]`` with the true flattened width for each step.

        Returns
        -------
        ``[batch, step, token_dim]``.
        """
        b, s, d = x.shape
        p = self.chunk_dim
        if feature_lengths.shape != (b, s):
            raise ValueError(
                f"feature_lengths must have shape {(b, s)}, got {tuple(feature_lengths.shape)}"
            )
        if feature_lengths.numel() and int(feature_lengths.max()) > d:
            raise ValueError(
                f"A feature length exceeds the batch padded width D={d}: "
                f"max_length={int(feature_lengths.max())}"
            )

        # Only zero-pad to a multiple of chunk_dim; no interpolation/resampling.
        padded_dim = max(p, ((d + p - 1) // p) * p)
        if padded_dim > d:
            x = torch.nn.functional.pad(x, (0, padded_dim - d))
        num_tokens = padded_dim // p

        # [B,S,D] -> [B*S,L,P] -> [B*S,L,token_dim]
        chunks = x.reshape(b * s, num_tokens, p)
        encoded = self.chunk_projection(chunks)
        encoded = encoded + self._sinusoidal_position(
            num_tokens, self.token_dim, encoded.device, encoded.dtype
        ).unsqueeze(0)

        # A chunk is valid when its index is below ceil(original_length / P).
        valid_tokens = torch.div(
            feature_lengths.reshape(-1) + p - 1,
            p,
            rounding_mode="floor",
        )
        token_ids = torch.arange(num_tokens, device=x.device).unsqueeze(0)
        valid = token_ids < valid_tokens.unsqueeze(1)
        padding_mask = ~valid

        # Avoid all-masked attention for genuinely missing steps.
        all_empty = ~valid.any(dim=1)
        safe_padding_mask = padding_mask.clone()
        if all_empty.any():
            safe_padding_mask[all_empty, 0] = False
            encoded = encoded.clone()
            encoded[all_empty, 0] = 0.0

        encoded = self.transformer(encoded, src_key_padding_mask=safe_padding_mask)

        scores = self.pool_score(encoded).squeeze(-1)
        scores = scores.masked_fill(safe_padding_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        pooled = self.output_norm(pooled)
        pooled[all_empty] = 0.0
        return pooled.reshape(b, s, self.token_dim)


class ArchitectureAgnosticDetector(nn.Module):
    """Concatenate four step groups, then apply 1D CNN -> Transformer -> MLP."""

    def __init__(self, trace_dim, image_dim, num_steps,
                 chunk_dim=256, token_dim=256,
                 chunk_transformer_layers=2, chunk_nhead=4,
                 cnn_dim=128, transformer_layers=2, nhead=4,
                 mlp_dim=128, dropout=0.2):
        super().__init__()
        if cnn_dim % nhead:
            raise ValueError("cnn_dim must be divisible by nhead")

        self.num_steps = num_steps
        self.token_dim = token_dim
        self.trace_dim = trace_dim
        self.image_dim = image_dim
        self.concat_dim = token_dim + trace_dim + image_dim + token_dim

        # LFD and Navi share the same output width but keep independent weights.
        self.lfd_encoder = VariableLengthFeatureEncoder(
            chunk_dim, token_dim, chunk_transformer_layers, chunk_nhead, dropout
        )
        self.navi_encoder = VariableLengthFeatureEncoder(
            chunk_dim, token_dim, chunk_transformer_layers, chunk_nhead, dropout
        )

        # Trace/image enter this CNN at their original scalar dimensions.
        self.cnn = nn.Sequential(
            nn.Conv1d(self.concat_dim, cnn_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, cnn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(cnn_dim, cnn_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, cnn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(cnn_dim, cnn_dim, kernel_size=1),
            nn.GroupNorm(1, cnn_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.step_embedding = nn.Parameter(torch.zeros(1, num_steps, cnn_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=cnn_dim, nhead=nhead, dim_feedforward=cnn_dim * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cnn_dim))
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(cnn_dim), nn.Linear(cnn_dim, mlp_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(mlp_dim, 2),
        )

    def encode_groups(self, lfd, lfd_lengths, trace, image, navi, navi_lengths):
        """Return encoded LFD/Navi and unchanged Trace/Image step features."""
        lfd_token = self.lfd_encoder(lfd, lfd_lengths)    # [B,S,token_dim]
        navi_token = self.navi_encoder(navi, navi_lengths) # [B,S,token_dim]
        return lfd_token, trace, image, navi_token

    def forward(self, lfd, lfd_lengths, trace, image, navi, navi_lengths):
        groups = self.encode_groups(lfd, lfd_lengths, trace, image, navi, navi_lengths)
        # [B,step,d_total], where d_total=token_dim+trace_dim+image_dim+token_dim.
        concatenated = torch.cat(groups, dim=-1)
        encoded = self.cnn(concatenated.transpose(1, 2)).transpose(1, 2)
        encoded = encoded + self.step_embedding[:, :encoded.shape[1]]
        cls = self.cls_token.expand(encoded.shape[0], -1, -1)
        encoded = self.transformer(torch.cat([cls, encoded], dim=1))
        return self.classifier(encoded[:, 0])


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def _move_batch(batch, device):
    return [value.to(device) for value in batch]


def _evaluate(model, loader, device, criterion):
    model.eval()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for batch in loader:
            lfd, lfd_lengths, trace, image, navi, navi_lengths, labels = _move_batch(batch, device)
            logits = model(lfd, lfd_lengths, trace, image, navi, navi_lengths)
            losses.append(float(criterion(logits, labels)))
            correct += int((logits.argmax(1) == labels).sum())
            total += labels.numel()
    return float(np.mean(losses)), correct / max(total, 1)


def _build_model_from_config(config, device):
    model = ArchitectureAgnosticDetector(
        trace_dim=config["trace_dim"],
        image_dim=config["image_dim"],
        num_steps=len(config["steps"]),
        chunk_dim=config["chunk_dim"],
        token_dim=config["token_dim"],
        chunk_transformer_layers=config["chunk_transformer_layers"],
        chunk_nhead=config["chunk_nhead"],
        cnn_dim=config["cnn_dim"],
        transformer_layers=config["transformer_layers"],
        nhead=config["nhead"],
        mlp_dim=config["mlp_dim"],
        dropout=config["dropout"],
    ).to(device)
    return model


def _validate_prediction_layout(samples, checkpoint):
    """Reject incompatible step counts or missing trained scalar features."""
    steps = set(checkpoint["steps"])
    expected_trace = set(checkpoint["trace_names"])
    expected_image = set(checkpoint["image_names"])
    errors = []
    for sample in samples:
        if sample.lfd.shape[0] != len(steps):
            errors.append(f"prompt={sample.prompt_index}: LFD steps={sample.lfd.shape[0]}")
        if sample.navi.shape[0] != len(steps):
            errors.append(f"prompt={sample.prompt_index}: Navi steps={sample.navi.shape[0]}")
        trace_steps = set(sample.trace)
        image_steps = set(sample.image)
        if not trace_steps.issubset(steps):
            errors.append(f"prompt={sample.prompt_index}: unexpected trace steps={sorted(trace_steps - steps)}")
        if not image_steps.issubset(steps):
            errors.append(f"prompt={sample.prompt_index}: unexpected image steps={sorted(image_steps - steps)}")
        present_trace = {name for values in sample.trace.values() for name in values}
        present_image = {name for values in sample.image.values() for name in values}
        if expected_trace - present_trace:
            errors.append(f"prompt={sample.prompt_index}: missing trace fields={sorted(expected_trace - present_trace)}")
        if expected_image - present_image:
            errors.append(f"prompt={sample.prompt_index}: missing image fields={sorted(expected_image - present_image)}")
        if len(errors) >= 20:
            break
    if errors:
        raise ValueError(
            "Prediction feature layout does not match the checkpoint:\n  "
            + "\n  ".join(errors)
        )


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    steps = [int(value) for value in args.steps.split(",") if value.strip()]
    samples = load_samples(args, require_label=True)
    train_idx, val_idx = split_by_prompt(samples, args.val_ratio, args.seed)
    train_samples = [samples[idx] for idx in train_idx]

    trace_names = _step_dict_layout(train_samples, "trace")
    image_names = _step_dict_layout(train_samples, "image")
    if not trace_names or not image_names:
        raise ValueError("Training split has no trace or image features")

    trace = _step_dict_array(samples, "trace", trace_names, steps, normalize_trace=True)
    image = _step_dict_array(samples, "image", image_names, steps)
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)

    lfd_widths = sorted({sample.lfd.shape[1] for sample in samples})
    navi_widths = sorted({sample.navi.shape[1] for sample in samples})

    print(
        "Prepared variable-length groups:\n"
        f"  LFD flattened widths  = {lfd_widths}\n"
        f"  Navi flattened widths = {navi_widths}\n"
        f"  Trace = {trace.shape}\n"
        f"  Image = {image.shape}\n"
        f"  chunk_dim = {args.chunk_dim}, post-encoder token_dim = {args.token_dim}\n"
        f"  CNN d_total = {2 * args.token_dim + trace.shape[-1] + image.shape[-1]} "
        f"(LFD {args.token_dim} + Trace {trace.shape[-1]} + "
        f"Image {image.shape[-1]} + Navi {args.token_dim})"
    )

    train_ds = StepFeatureDataset(samples, train_idx, trace, image, labels, len(steps))
    val_ds = StepFeatureDataset(samples, val_idx, trace, image, labels, len(steps))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_variable_features,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_variable_features,
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_config = {
        "steps": steps, "trace_dim": trace.shape[-1], "image_dim": image.shape[-1],
        "chunk_dim": args.chunk_dim, "token_dim": args.token_dim,
        "chunk_transformer_layers": args.chunk_transformer_layers,
        "chunk_nhead": args.chunk_nhead, "cnn_dim": args.cnn_dim,
        "transformer_layers": args.transformer_layers, "nhead": args.nhead,
        "mlp_dim": args.mlp_dim, "dropout": args.dropout,
    }
    model = _build_model_from_config(model_config, device)

    train_labels = labels[train_idx]
    counts = np.bincount(train_labels, minlength=2)
    weights = counts.sum() / np.maximum(counts, 1)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state, best_loss, stale = None, float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            lfd_b, lfd_lengths_b, trace_b, image_b, navi_b, navi_lengths_b, labels_b = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(lfd_b, lfd_lengths_b, trace_b, image_b, navi_b, navi_lengths_b)
            loss = criterion(logits, labels_b)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))

        val_loss, val_acc = _evaluate(model, val_loader, device, criterion)
        print(
            f"epoch={epoch:03d} train_loss={np.mean(losses):.5f} "
            f"val_loss={val_loss:.5f} val_acc={val_acc:.4f}"
        )
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "feature_layout": "architecture_agnostic_step_concat_chunk_masked_v4",
        "model_state": model.state_dict(),
        "steps": steps,
        "trace_names": trace_names,
        "image_names": image_names,
        "trace_dim": trace.shape[-1],
        "image_dim": image.shape[-1],
        "concat_dim": 2 * args.token_dim + trace.shape[-1] + image.shape[-1],
        "chunk_dim": args.chunk_dim,
        "token_dim": args.token_dim,
        "chunk_transformer_layers": args.chunk_transformer_layers,
        "chunk_nhead": args.chunk_nhead,
        "cnn_dim": args.cnn_dim,
        "transformer_layers": args.transformer_layers,
        "nhead": args.nhead,
        "mlp_dim": args.mlp_dim,
        "dropout": args.dropout,
        "train_prompt_indices": sorted({samples[idx].prompt_index for idx in train_idx}),
        "val_prompt_indices": sorted({samples[idx].prompt_index for idx in val_idx}),
        "train_prompt_texts": sorted({samples[idx].original_prompt for idx in train_idx}),
        "val_prompt_texts": sorted({samples[idx].original_prompt for idx in val_idx}),
    }, args.model_out)
    print(f"Saved model to {args.model_out}")


def predict(args):
    """Predict test samples while ignoring any labels stored in feature CSVs."""
    checkpoint = _torch_load(args.model_in)
    if checkpoint.get("feature_layout") != "architecture_agnostic_step_concat_chunk_masked_v4":
        raise ValueError("Checkpoint feature layout is not supported by this script")

    samples = load_samples(args, require_label=False)
    _validate_prediction_layout(samples, checkpoint)
    steps = list(checkpoint["steps"])
    trace = _step_dict_array(
        samples, "trace", list(checkpoint["trace_names"]), steps,
        normalize_trace=True,
    )
    image = _step_dict_array(samples, "image", list(checkpoint["image_names"]), steps)
    dummy_labels = np.zeros(len(samples), dtype=np.int64)
    dataset = StepFeatureDataset(
        samples, range(len(samples)), trace, image, dummy_labels, len(steps)
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_variable_features,
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _build_model_from_config(checkpoint, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch in loader:
            lfd, lfd_lengths, trace_b, image_b, navi, navi_lengths, _labels = _move_batch(batch, device)
            logits = model(lfd, lfd_lengths, trace_b, image_b, navi, navi_lengths)
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())

    args.predict_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.predict_csv.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "prompt_index", "original_prompt", "predicted_label",
            "predicted_tag", "backdoor_probability",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for sample, probability in zip(samples, probabilities):
            prediction = int(probability >= args.threshold)
            writer.writerow({
                "prompt_index": sample.prompt_index,
                "original_prompt": sample.original_prompt,
                "predicted_label": prediction,
                "predicted_tag": "backdoor" if prediction else "clean",
                "backdoor_probability": float(probability),
            })
    print(f"Wrote {len(samples)} predictions to {args.predict_csv}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Step-aligned architecture-agnostic detector with flat chunk masked Transformers"
    )
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--dataset_manifest", type=Path, required=True)
    parser.add_argument(
        "--model_out", type=Path,
        default=Path("artifacts/detector_checkpoints/architecture_agnostic_4.pt")
    )
    parser.add_argument(
        "--model_in", type=Path,
        default=Path("artifacts/detector_checkpoints/architecture_agnostic_4.pt")
    )
    parser.add_argument(
        "--predict_csv", type=Path,
        default=Path("artifacts/paper_results/architecture_agnostic_predictions.csv")
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--steps", default="0,10,20,30,40,49")

    # LFD/Navi: flatten non-step axes, then chunk the variable-width vector.
    parser.add_argument("--chunk_dim", type=int, default=256)
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--chunk_transformer_layers", type=int, default=2)
    parser.add_argument("--chunk_nhead", type=int, default=4)

    # Global four-group fusion network.
    parser.add_argument("--cnn_dim", type=int, default=128)
    #parser.add_argument("--cnn_token_bins", type=int, default=8)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.mode == "train":
        train(parsed_args)
    else:
        predict(parsed_args)
