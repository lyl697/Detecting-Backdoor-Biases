"""Assemble aligned feature records and PyTorch datasets for classifiers.

Feature layout:
- LFD: [step, lfd_blocks, hidden_dim], e.g. [6, 24, hidden_dim]
- latent trajectory/reference latent/image similarity/activation difference:
  merged as [step, 1, extra_dim]

For every denoising step, the dataset exposes all LFD block tokens plus one
extra token containing the selected non-LFD features.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from feature_data import (
    LABELS,
    SampleRecord,
    _dwt_probability_blocks,
    _read_csv,
    _split_paths,
    fit_trace_preprocessor,
    load_activation_difference_features,
    load_numeric_feature_csv,
    load_trace_features,
    parse_steps,
    transform_trace,
)


def _load_lfd_block_tensor(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data)
    data = data.detach().float().cpu()
    data = _dwt_probability_blocks(data)

    if data.dim() == 1:
        return data.reshape(1, 1, -1)
    if data.dim() == 2:
        return data.unsqueeze(0)
    if data.dim() == 3:
        return data
    return data.flatten(start_dim=2)


def _pad_lfd_blocks(tensor: torch.Tensor, steps: int, blocks: int, hidden_dim: int) -> torch.Tensor:
    out = torch.zeros(steps, blocks, hidden_dim, dtype=torch.float32)
    src_steps = min(steps, tensor.shape[0])
    src_blocks = min(blocks, tensor.shape[1])
    src_dim = min(hidden_dim, tensor.shape[2])
    out[:src_steps, :src_blocks, :src_dim] = tensor[:src_steps, :src_blocks, :src_dim]
    return out


def _pad_extra_block(array: np.ndarray, steps: int, extra_dim: int) -> np.ndarray:
    out = np.zeros((steps, extra_dim), dtype=np.float32)
    if array.size == 0:
        return out
    src_steps = min(steps, array.shape[0])
    src_dim = min(extra_dim, array.shape[1])
    out[:src_steps, :src_dim] = array[:src_steps, :src_dim]
    return out


def _mean_lfd_block_tensors(paths: list[Path]) -> torch.Tensor | None:
    tensors = [_load_lfd_block_tensor(path) for path in paths]
    if not tensors:
        return None
    steps = max(tensor.shape[0] for tensor in tensors)
    blocks = max(tensor.shape[1] for tensor in tensors)
    hidden_dim = max(tensor.shape[2] for tensor in tensors)
    padded = [_pad_lfd_blocks(tensor, steps, blocks, hidden_dim) for tensor in tensors]
    return torch.stack(padded, dim=0).mean(dim=0)


def load_block_records(
    lfd_csv: Path,
    latent_cosine_csv: Path | None,
    latent_norm_csv: Path | None,
    latent_update_norm_csv: Path | None,
    reference_latent_csv,
    image_similarity_csv,
    activation_difference_csv,
    steps: list[int],
    require_label: bool,
) -> list[SampleRecord]:
    trace_features = load_trace_features(latent_cosine_csv, latent_norm_csv, latent_update_norm_csv)
    reference_latent_features = load_numeric_feature_csv(reference_latent_csv, ("modelcross_",))
    image_similarity_features = load_numeric_feature_csv(
        image_similarity_csv,
        ("perturb_", "image_similarity_"),
    )
    activation_difference_features = load_activation_difference_features(activation_difference_csv, steps)

    records: list[SampleRecord] = []
    for row in _read_csv(lfd_csv):
        prompt_index = int(row["prompt_index"])
        for column, value in row.items():
            if not column.endswith("_orig_feature_paths") or not value:
                continue
            model_tag = column.removesuffix("_orig_feature_paths")
            label = LABELS.get(model_tag)
            if require_label and label is None:
                continue

            hidden = _mean_lfd_block_tensors(_split_paths(value))
            if hidden is None:
                continue

            tabular_features = {}
            tabular_features.update(trace_features.get((prompt_index, model_tag), {}))
            tabular_features.update(reference_latent_features.get((prompt_index, model_tag), {}))
            tabular_features.update(image_similarity_features.get((prompt_index, model_tag), {}))
            tabular_features.update(activation_difference_features.get((prompt_index, model_tag), {}))
            records.append(
                SampleRecord(
                    prompt_index=prompt_index,
                    model_tag=model_tag,
                    label=label,
                    hidden=hidden,
                    trace=tabular_features,
                )
            )
    return records


class ExtraBlockDataset(Dataset):
    def __init__(
        self,
        records,
        extra_array: np.ndarray,
        steps: list[int],
        hidden_blocks: int,
        hidden_dim: int,
    ):
        self.records = records
        self.extra_array = extra_array
        self.steps = steps
        self.num_steps = len(steps)
        self.hidden_blocks = hidden_blocks
        self.hidden_dim = hidden_dim
        self.extra_dim = extra_array.shape[2] if extra_array.ndim == 3 else 0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        hidden = _pad_lfd_blocks(record.hidden, self.num_steps, self.hidden_blocks, self.hidden_dim)
        extra = _pad_extra_block(self.extra_array[idx], self.num_steps, self.extra_dim)
        label = -1 if record.label is None else int(record.label)
        return (
            hidden.float(),
            torch.from_numpy(extra).float(),
            torch.tensor(label, dtype=torch.long),
        )



def hidden_shape(records) -> tuple[int, int]:
    hidden_blocks = max(record.hidden.shape[1] for record in records)
    hidden_dim = max(record.hidden.shape[2] for record in records)
    return hidden_blocks, hidden_dim


def evaluate(model, loader, device):
    model.eval()
    losses = []
    preds = []
    labels = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for hidden, extra, label in loader:
            hidden = hidden.to(device)
            extra = extra.to(device)
            label = label.to(device)
            logits = model(hidden, extra)
            losses.append(float(criterion(logits, label).item()))
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(label.cpu().tolist())
    if not labels:
        return float("nan"), float("nan")
    return float(np.mean(losses)), float(np.mean(np.array(preds) == np.array(labels)))


def write_extra_block_feature_csv(records, extra_array, extra_names, steps, output_csv: Path | None):
    if output_csv is None:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    def encode(value) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_index",
                "model_tag",
                "label",
                "steps",
                "extra_block_feature_names",
                "extra_block",
            ],
        )
        writer.writeheader()
        for record, feature in zip(records, extra_array):
            writer.writerow(
                {
                    "prompt_index": record.prompt_index,
                    "model_tag": record.model_tag,
                    "label": "" if record.label is None else int(record.label),
                    "steps": encode(steps),
                    "extra_block_feature_names": encode(extra_names),
                    "extra_block": encode(feature.astype(float).tolist()),
                }
            )
    print(f"Wrote extra-block features to {output_csv}")


def build_records_and_extra(args, require_label: bool, checkpoint=None):
    steps = checkpoint.get("steps", parse_steps(args.steps)) if checkpoint else parse_steps(args.steps)
    records = load_block_records(
        args.lfd_csv,
        args.latent_cosine_csv,
        args.latent_norm_csv,
        args.latent_update_norm_csv,
        args.reference_latent_csv,
        args.image_similarity_csv,
        args.activation_difference_csv,
        require_label=require_label,
        steps=steps,
    )
    if not records:
        raise ValueError("No records were loaded")

    if checkpoint is None:
        extra_names, extra_mins, extra_ranges, extra_array = fit_trace_preprocessor(records, steps)
    else:
        extra_names = checkpoint["extra_names"]
        extra_mins = checkpoint["extra_mins"]
        extra_ranges = checkpoint["extra_ranges"]
        extra_array = transform_trace(records, steps, extra_names, extra_mins, extra_ranges)

    return records, steps, extra_names, extra_mins, extra_ranges, extra_array


