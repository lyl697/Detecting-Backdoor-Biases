#!/usr/bin/env python3
"""
Automatic feature-ablation suite for the block-channel CNN+Transformer detector.

Five conceptual groups:
  LFD   : Local Feature Dynamics block features
  LTF   : latent-trajectory features
  CROSS : reference latent discrepancy features (legacy CSV prefix: modelcross_)
  IMAGE : reference image-similarity features
  ADF   : reference activation-difference features (legacy tensor prefix: navi_)

Default variants trained in one run:
  full           = LFD + LTF + CROSS + IMAGE + ADF
  self_only      = LFD + LTF
  reference_only = CROSS + IMAGE + ADF
  wo_lfd         = LTF + CROSS + IMAGE + ADF
  wo_ltf         = LFD + CROSS + IMAGE + ADF
  wo_cross       = LFD + LTF + IMAGE + ADF
  wo_image       = LFD + LTF + CROSS + ADF
  wo_adf         = LFD + LTF + CROSS + IMAGE

The non-LFD features selected for each variant are still concatenated into ONE
extra block and projected by the same extra_proj as the original classifier.
Only the feature subset changes; the CNN/Transformer structure and training
hyperparameters remain the same.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from feature_data import _load_checkpoint, split_train_val
from classifier_data import (
    ExtraBlockDataset,
    build_records_and_extra,
    evaluate,
    hidden_shape,
)


FEATURE_GROUPS = ("ltf", "cross", "image", "adf")
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


@dataclass(frozen=True)
class AblationSpec:
    name: str
    use_lfd: bool
    extra_groups: frozenset[str]
    description: str


ABLATION_SPECS: Dict[str, AblationSpec] = {
    "full": AblationSpec(
        "full", True,
        frozenset({"ltf", "cross", "image", "adf"}),
        "LFD + LTF + CROSS + IMAGE + ADF",
    ),
    "self_only": AblationSpec(
        "self_only", True,
        frozenset({"ltf"}),
        "LFD + LTF",
    ),
    "reference_only": AblationSpec(
        "reference_only", False,
        frozenset({"cross", "image", "adf"}),
        "CROSS + IMAGE + ADF",
    ),
    "wo_lfd": AblationSpec(
        "wo_lfd", False,
        frozenset({"ltf", "cross", "image", "adf"}),
        "LTF + CROSS + IMAGE + ADF",
    ),
    "wo_ltf": AblationSpec(
        "wo_ltf", True,
        frozenset({"cross", "image", "adf"}),
        "LFD + CROSS + IMAGE + ADF",
    ),
    "wo_cross": AblationSpec(
        "wo_cross", True,
        frozenset({"ltf", "image", "adf"}),
        "LFD + LTF + IMAGE + ADF",
    ),
    "wo_image": AblationSpec(
        "wo_image", True,
        frozenset({"ltf", "cross", "adf"}),
        "LFD + LTF + CROSS + ADF",
    ),
    "wo_adf": AblationSpec(
        "wo_adf", True,
        frozenset({"ltf", "cross", "image"}),
        "LFD + LTF + CROSS + IMAGE",
    ),
}


def feature_group(name: str) -> str:
    """Map one extra-feature name to one conceptual feature group."""
    name = str(name)
    if name.startswith("trace_"):
        return "ltf"
    if name.startswith("modelcross_"):
        return "cross"
    if name.startswith(("perturb_", "image_similarity_")):
        return "image"
    if name.startswith("navi_"):
        return "adf"
    return "unknown"


def build_feature_group_map(extra_names: Sequence[str]) -> Dict[str, List[int]]:
    groups = {"ltf": [], "cross": [], "image": [], "adf": [], "unknown": []}
    for idx, name in enumerate(extra_names):
        groups[feature_group(name)].append(idx)
    return groups


def print_feature_group_summary(extra_names: Sequence[str]):
    groups = build_feature_group_map(extra_names)
    print("\n" + "=" * 80)
    print("Loaded non-LFD feature groups")
    print("=" * 80)
    for group in ("ltf", "cross", "image", "adf", "unknown"):
        indices = groups[group]
        print(f"{group:>8s}: {len(indices)} dims")
        for idx in indices[:5]:
            print(f"          - {extra_names[idx]}")
        if len(indices) > 5:
            print(f"          ... ({len(indices) - 5} more)")

    if groups["unknown"]:
        names = [extra_names[i] for i in groups["unknown"]]
        raise ValueError(
            "Unknown extra feature prefixes. Update feature_group() so the "
            "ablation never silently drops a feature:\n  " + "\n  ".join(names)
        )

    empty = [g for g in FEATURE_GROUPS if not groups[g]]
    if empty:
        print("[warning] Empty groups: " + ", ".join(empty))
    return groups


def _slice_stat_vector(values, indices):
    arr = np.asarray(values)
    if arr.size == 0:
        return arr
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D preprocessing stats, got {arr.shape}")
    return arr[indices]


def select_extra_features(
    extra_names,
    extra_array,
    stat_a,
    stat_b,
    selected_groups,
):
    indices = [
        i for i, name in enumerate(extra_names)
        if feature_group(name) in selected_groups
    ]
    names = [extra_names[i] for i in indices]

    if extra_array.ndim == 3:
        array = extra_array[:, :, indices]
    elif extra_array.ndim == 2:
        array = extra_array[:, indices]
    else:
        raise ValueError(f"Unexpected extra_array shape: {extra_array.shape}")

    return (
        names,
        array.astype(np.float32, copy=False),
        _slice_stat_vector(stat_a, indices),
        _slice_stat_vector(stat_b, indices),
        indices,
    )



from model import AblationBlockChannelDetector

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_train_loader(dataset, args):
    generator = torch.Generator().manual_seed(args.random_state)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )


def parse_variant_names(text: str) -> List[str]:
    names = [x.strip() for x in str(text).split(",") if x.strip()]
    if not names:
        names = list(ALL_VARIANTS)
    invalid = [x for x in names if x not in ABLATION_SPECS]
    if invalid:
        raise ValueError(
            "Unknown variants: " + ", ".join(invalid)
            + ". Available: " + ", ".join(ALL_VARIANTS)
        )
    return list(dict.fromkeys(names))


def train_one_variant(
    args,
    spec,
    records,
    steps,
    full_extra_names,
    full_extra_array,
    full_stat_a,
    full_stat_b,
    train_idx,
    val_idx,
    hidden_blocks,
    hidden_dim,
    device,
):
    print("\n" + "#" * 90)
    print(f"ABLATION: {spec.name}")
    print(f"FEATURES: {spec.description}")
    print("#" * 90)

    names, extra_array, stat_a, stat_b, selected_indices = select_extra_features(
        full_extra_names,
        full_extra_array,
        full_stat_a,
        full_stat_b,
        spec.extra_groups,
    )

    if len(names) == 0 and not spec.use_lfd:
        raise ValueError(f"{spec.name}: no features remain")

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    train_ds = ExtraBlockDataset(
        train_records, extra_array[train_idx], steps, hidden_blocks, hidden_dim
    )
    val_ds = ExtraBlockDataset(
        val_records, extra_array[val_idx], steps, hidden_blocks, hidden_dim
    )

    # Identical split and deterministic loader seed for every variant.
    set_random_seed(args.random_state)
    train_loader = make_train_loader(train_ds, args)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = AblationBlockChannelDetector(
        hidden_dim=hidden_dim,
        hidden_blocks=hidden_blocks,
        extra_dim=len(names),
        num_steps=len(steps),
        use_lfd=spec.use_lfd,
        token_dim=args.token_dim,
        cnn_dim=args.cnn_dim,
        cnn_token_bins=args.cnn_token_bins,
        transformer_layers=args.transformer_layers,
        nhead=args.nhead,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
    ).to(device)

    labels = np.array([int(r.label) for r in train_records])
    counts = np.bincount(labels, minlength=2)
    weights = counts.sum() / np.maximum(counts, 1)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for hidden, extra, label in train_loader:
            hidden = hidden.to(device)
            extra = extra.to(device)
            label = label.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(hidden, extra), label)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_loss, val_acc = evaluate(model, val_loader, device)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        print(
            f"[{spec.name}] epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_loss, final_val_acc = evaluate(model, val_loader, device)

    variant_dir = args.ablation_output_dir / spec.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    model_path = variant_dir / "detector.pt"

    torch.save(
        {
            "feature_layout": "lfd_plus_extra_block_channel_ablation_v1",
            "ablation_variant": spec.name,
            "ablation_description": spec.description,
            "use_lfd": bool(spec.use_lfd),
            "selected_extra_groups": sorted(spec.extra_groups),
            "selected_extra_indices": list(selected_indices),
            "model_state": model.state_dict(),
            "steps": steps,
            "hidden_dim": hidden_dim,
            "hidden_blocks": hidden_blocks,
            "extra_names": names,
            # Legacy field names are deliberately retained so
            # build_records_and_extra(checkpoint=...) can transform test data.
            "extra_mins": stat_a,
            "extra_ranges": stat_b,
            "token_dim": args.token_dim,
            "cnn_dim": args.cnn_dim,
            "cnn_token_bins": args.cnn_token_bins,
            "transformer_layers": args.transformer_layers,
            "nhead": args.nhead,
            "mlp_dim": args.mlp_dim,
            "dropout": args.dropout,
            "best_epoch": best_epoch,
            "best_val_loss": float(final_val_loss),
            "best_val_acc": float(final_val_acc),
            "random_state": int(args.random_state),
            "train_record_count": len(train_idx),
            "val_record_count": len(val_idx),
        },
        model_path,
    )

    with (variant_dir / "feature_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "variant": spec.name,
                "description": spec.description,
                "use_lfd": bool(spec.use_lfd),
                "selected_extra_groups": sorted(spec.extra_groups),
                "extra_dim": len(names),
                "extra_names": names,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[saved] {spec.name}: {model_path}")
    return {
        "variant": spec.name,
        "description": spec.description,
        "use_lfd": int(spec.use_lfd),
        "extra_groups": "+".join(sorted(spec.extra_groups)),
        "extra_dim": len(names),
        "best_epoch": best_epoch,
        "best_val_loss": float(final_val_loss),
        "best_val_acc": float(final_val_acc),
        "model_path": str(model_path),
    }


def train_all(args):
    set_random_seed(args.random_state)
    (
        records,
        steps,
        full_extra_names,
        full_stat_a,
        full_stat_b,
        full_extra_array,
    ) = build_records_and_extra(args, require_label=True)

    if len({r.label for r in records}) < 2:
        raise ValueError("Need both clean and backdoor records for training")

    print_feature_group_summary(full_extra_names)
    hidden_blocks, hidden_dim = hidden_shape(records)

    # Critical for fair ablation: generate ONE split and reuse it everywhere.
    train_idx, val_idx = split_train_val(records, args.val_ratio, args.random_state)

    print("\n" + "=" * 80)
    print("Shared train/validation split")
    print("=" * 80)
    print(f"train records = {len(train_idx)}")
    print(f"val records   = {len(val_idx)}")
    print(f"steps         = {steps}")
    print(f"LFD blocks    = {hidden_blocks}")
    print(f"LFD dim       = {hidden_dim}")
    print(f"full extra    = {len(full_extra_names)}")

    args.ablation_output_dir.mkdir(parents=True, exist_ok=True)
    with (args.ablation_output_dir / "full_feature_names.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(list(full_extra_names), f, ensure_ascii=False, indent=2)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    summary_rows = []

    for name in parse_variant_names(args.variants):
        row = train_one_variant(
            args=args,
            spec=ABLATION_SPECS[name],
            records=records,
            steps=steps,
            full_extra_names=full_extra_names,
            full_extra_array=full_extra_array,
            full_stat_a=full_stat_a,
            full_stat_b=full_stat_b,
            train_idx=train_idx,
            val_idx=val_idx,
            hidden_blocks=hidden_blocks,
            hidden_dim=hidden_dim,
            device=device,
        )
        summary_rows.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = args.ablation_output_dir / "ablation_training_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "variant", "description", "use_lfd", "extra_groups", "extra_dim",
            "best_epoch", "best_val_loss", "best_val_acc", "model_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 90)
    print("ALL ABLATION TRAINING FINISHED")
    print("=" * 90)
    for row in summary_rows:
        print(
            f"{row['variant']:>15s}: val_acc={row['best_val_acc']:.4f}, "
            f"val_loss={row['best_val_loss']:.4f}"
        )
    print(f"Summary: {summary_path}")


@torch.no_grad()
def collect_probabilities(model, loader, device):
    probs = []
    for hidden, extra, _ in loader:
        logits = model(hidden.to(device), extra.to(device))
        probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
    return np.asarray(probs, dtype=np.float32)


def predict_one_checkpoint(args, checkpoint_path: Path, output_csv: Path):
    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint.get("feature_layout") != "lfd_plus_extra_block_channel_ablation_v1":
        raise ValueError(f"Not an ablation-suite checkpoint: {checkpoint_path}")

    records, steps, extra_names, _, _, extra_array = build_records_and_extra(
        args, require_label=False, checkpoint=checkpoint
    )
    dataset = ExtraBlockDataset(
        records,
        extra_array,
        steps,
        checkpoint["hidden_blocks"],
        checkpoint["hidden_dim"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = AblationBlockChannelDetector(
        hidden_dim=checkpoint["hidden_dim"],
        hidden_blocks=checkpoint["hidden_blocks"],
        extra_dim=len(extra_names),
        num_steps=len(steps),
        use_lfd=bool(checkpoint["use_lfd"]),
        token_dim=checkpoint["token_dim"],
        cnn_dim=checkpoint["cnn_dim"],
        cnn_token_bins=checkpoint["cnn_token_bins"],
        transformer_layers=checkpoint["transformer_layers"],
        nhead=checkpoint["nhead"],
        mlp_dim=checkpoint["mlp_dim"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    probs = collect_probabilities(model, loader, device)
    preds = (probs >= args.threshold).astype(np.int64)
    prompt_texts = (
        args.prompt_txt.read_text(encoding="utf-8").splitlines()
        if args.prompt_txt and args.prompt_txt.is_file() else []
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_id", "architecture", "prompt_index", "original_prompt", "true_label",
                "model_tag", "predicted_label", "predicted_tag",
                "backdoor_probability",
            ],
        )
        writer.writeheader()
        for record, pred, prob in zip(records, preds, probs):
            writer.writerow(
                {
                    "model_id": args.model_id or record.model_tag,
                    "architecture": args.architecture or "",
                    "prompt_index": record.prompt_index,
                    "original_prompt": (
                        prompt_texts[record.prompt_index]
                        if 0 <= record.prompt_index < len(prompt_texts) else ""
                    ),
                    "true_label": (
                        args.true_label if args.true_label is not None
                        else ("" if record.label is None else int(record.label))
                    ),
                    "model_tag": record.model_tag,
                    "predicted_label": int(pred),
                    "predicted_tag": "backdoor" if int(pred) else "clean",
                    "backdoor_probability": float(prob),
                }
            )

    return {
        "variant": checkpoint["ablation_variant"],
        "num_prompts": len(probs),
        "num_pred_clean": int((preds == 0).sum()),
        "num_pred_backdoor": int((preds == 1).sum()),
        "mean_backdoor_probability": float(probs.mean()) if len(probs) else float("nan"),
        "threshold": float(args.threshold),
        "prediction_csv": str(output_csv),
    }


def predict_all(args):
    args.predict_output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in parse_variant_names(args.variants):
        ckpt = args.ablation_output_dir / name / "detector.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        print("\n" + "#" * 90)
        print(f"PREDICT: {name}")
        print("#" * 90)
        row = predict_one_checkpoint(
            args, ckpt, args.predict_output_dir / f"{name}.csv"
        )
        rows.append(row)
        print(
            f"{name}: clean={row['num_pred_clean']}, "
            f"backdoor={row['num_pred_backdoor']}, "
            f"mean_prob={row['mean_backdoor_probability']:.6f}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = args.predict_output_dir / "ablation_prediction_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "variant", "num_prompts", "num_pred_clean", "num_pred_backdoor",
            "mean_backdoor_probability", "threshold", "prediction_csv",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prediction summary: {summary}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Automatic feature ablation for block-channel detector"
    )
    p.add_argument(
        "--mode", choices=["train_all", "predict_all"], default="train_all"
    )

    p.add_argument("--lfd_csv", "--hidden_csv", dest="lfd_csv", type=Path, required=True)
    p.add_argument("--latent_cosine_csv", "--trace_angle_csv", dest="latent_cosine_csv", type=Path, required=True)
    p.add_argument("--latent_norm_csv", "--trace_norm_csv", dest="latent_norm_csv", type=Path, required=True)
    p.add_argument("--latent_update_norm_csv", "--trace_delta_csv", dest="latent_update_norm_csv", type=Path, required=True)
    p.add_argument("--reference_latent_csv", "--modelcross_csv", dest="reference_latent_csv", type=str, required=True, help="One or more reference-latent CSVs, separated by comma or semicolon")
    p.add_argument("--image_similarity_csv", type=str, required=True, help="One or more image-similarity CSVs, separated by comma or semicolon")
    p.add_argument("--activation_difference_csv", "--navi_hidden_csv", dest="activation_difference_csv", type=str, required=True, help="One or more activation-difference CSVs, separated by comma or semicolon")

    p.add_argument(
        "--ablation_output_dir", type=Path,
        default=Path("artifacts/checkpoints/feature_ablation")
    )
    p.add_argument(
        "--predict_output_dir", type=Path,
        default=Path("artifacts/predictions/feature_ablation")
    )
    p.add_argument(
        "--variants", type=str, default=",".join(ALL_VARIANTS),
        help="Comma-separated subset of: " + ",".join(ALL_VARIANTS)
    )

    # Required by build_records_and_extra(), retained for compatibility.
    p.add_argument("--extra_feature_csv_out", type=str, default="")
    p.add_argument("--steps", type=str, default="0,10,20,30,40,49")

    # Same defaults as the current block-channel classifier.
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--prompt_txt", type=Path, default=None,
                   help="Prompt text file used to populate original_prompt in prediction CSVs")
    p.add_argument("--model_id", type=str, default=None,
                   help="Stable suspect-model ID written to prediction rows")
    p.add_argument("--architecture", choices=["sd14", "sd2", "sd35", "flux"], default=None)
    p.add_argument("--true_label", type=int, choices=[0, 1], default=None)
    p.add_argument("--token_dim", type=int, default=256)
    p.add_argument("--cnn_dim", type=int, default=128)
    p.add_argument("--cnn_token_bins", type=int, default=8)
    p.add_argument("--transformer_layers", type=int, default=2)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--mlp_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "train_all":
        train_all(args)
    else:
        predict_all(args)


if __name__ == "__main__":
    main()
