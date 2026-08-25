"""Shared utilities for intrinsic LFD and latent-trajectory extraction.

The generation pass is shared, but outputs keep the old layout:
- Local Feature Dynamics (LFD) CSV plus saved .pt tensors
- latent-trajectory cosine, norm, and update-norm CSVs
"""

import csv
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F


CURRENT_DIR = Path(__file__).resolve().parent
CLASSIFY_DIR = CURRENT_DIR.parents[1]
sys.path.insert(0, str(CLASSIFY_DIR))

from utils.latent_trajectory import (  # noqa: E402
    SampleTrace,
    _parse_steps,
    _project_step_metrics,
    compute_adjacent_z_metrics,
    ensure_dir,
    generate_with_step_trace,
    get_flux_pipeline,
    get_sd2_pipeline,
    get_sd35_pipeline,
    load_trace_files,
    load_prompts,
    save_image,
    save_trace_files,
)


PROMPT_RANGES = [(0, 1000)]


def build_prompt_subset(input_txt: str) -> List[str]:
    prompts_all = load_prompts(Path(input_txt))
    prompts: List[str] = []
    for start, end in PROMPT_RANGES:
        if start < len(prompts_all):
            prompts.extend(prompts_all[start:min(end, len(prompts_all))])
    return prompts if prompts else prompts_all


def release_pipeline(pipe) -> None:
    try:
        del pipe
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_selected_pipeline(args, model_id: str, lora_id: str | None = None):
    hf_token = os.environ.get("HF_TOKEN")
    if args.model_family == "sd2":
        return get_sd2_pipeline(model_id, args.device, hf_token)
    if args.model_family == "sd35":
        return get_sd35_pipeline(model_id, args.device, hf_token, lora_id=lora_id)
    if args.model_family == "flux":
        return get_flux_pipeline(model_id, args.device, hf_token, lora_id=lora_id)
    if args.model_family == "sd14":
        return get_sd2_pipeline(model_id, args.device, hf_token)
    raise ValueError(f"Unknown model_family: {args.model_family}")


def _last_tensor_from_output(out):
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "sample") and isinstance(out.sample, torch.Tensor):
        return out.sample
    if isinstance(out, dict):
        tensors = [value for value in out.values() if isinstance(value, torch.Tensor)]
        return tensors[-1] if tensors else None
    if isinstance(out, (list, tuple)):
        tensors = [value for value in out if isinstance(value, torch.Tensor)]
        return tensors[-1] if tensors else None
    return None


def _register_transformer_hooks(pipe, make_hook):
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        raise AttributeError("Expected pipeline to expose pipe.transformer")

    blocks = None
    for attr in ("transformer_blocks", "joint_transformer_blocks", "blocks"):
        candidate = getattr(transformer, attr, None)
        if candidate is not None and len(candidate) > 0:
            blocks = candidate
            break
    if blocks is None:
        raise AttributeError("Could not find transformer blocks on pipe.transformer")

    return [block.register_forward_hook(make_hook(f"transformer_{idx:02d}")) for idx, block in enumerate(blocks)]


def _register_unet_hooks(pipe, make_hook):
    unet = getattr(pipe, "unet", None)
    if unet is None:
        raise AttributeError("Expected SD2 pipeline to expose pipe.unet")

    handles = []
    for idx, block in enumerate(getattr(unet, "down_blocks", [])):
        handles.append(block.register_forward_hook(make_hook(f"down_{idx:02d}")))
    if hasattr(unet, "mid_block"):
        handles.append(unet.mid_block.register_forward_hook(make_hook("mid")))
    for idx, block in enumerate(getattr(unet, "up_blocks", [])):
        handles.append(block.register_forward_hook(make_hook(f"up_{idx:02d}")))
    if not handles:
        raise AttributeError("Could not find hookable UNet blocks")
    return handles


def _register_feature_hooks(pipe, make_hook):
    if hasattr(pipe, "transformer"):
        return _register_transformer_hooks(pipe, make_hook)
    return _register_unet_hooks(pipe, make_hook)


def _pool_hidden_activations(collected: Dict[str, List[torch.Tensor]], export_steps: List[int]) -> torch.Tensor:
    if not collected:
        raise RuntimeError("No hidden activations were collected")

    layer_step_vectors = []
    for name in sorted(collected):
        stacked = torch.stack([torch.as_tensor(x) for x in collected[name]], dim=0)
        if stacked.dim() == 5:
            pooled = stacked.mean(dim=(-2, -1)).mean(dim=1)
        elif stacked.dim() == 4:
            pooled = stacked.mean(dim=(1, 2))
        elif stacked.dim() == 3:
            pooled = stacked.mean(dim=1)
        elif stacked.dim() == 2:
            pooled = stacked
        else:
            pooled = stacked.flatten(start_dim=1)

        if max(export_steps) >= pooled.shape[0]:
            raise ValueError(
                f"Requested export step {max(export_steps)} but layer {name} "
                f"only collected {pooled.shape[0]} forwards"
            )
        layer_step_vectors.append(pooled[export_steps])

    max_len = max(item.shape[-1] for item in layer_step_vectors)
    padded = []
    for item in layer_step_vectors:
        if item.shape[-1] < max_len:
            item = F.pad(item, (0, max_len - item.shape[-1]))
        padded.append(item)
    return torch.stack(padded, dim=1).detach().cpu()


def generate_sample_features(
    pipe,
    prompt: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    export_steps: List[int],
) -> tuple[SampleTrace, torch.Tensor]:
    collected: Dict[str, List[torch.Tensor]] = {}

    def make_hook(name):
        def hook(module, inp, out):
            tensor_out = _last_tensor_from_output(out)
            if tensor_out is not None:
                collected.setdefault(name, []).append(tensor_out.detach().cpu().clone())
        return hook

    handles = _register_feature_hooks(pipe, make_hook)
    try:
        sample = generate_with_step_trace(
            pipe,
            prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            export_steps=export_steps,
        )
    finally:
        for handle in handles:
            handle.remove()

    hidden = _pool_hidden_activations(collected, export_steps)
    return sample, hidden


def save_hidden_features(features: List[torch.Tensor], base_dir: Path, prompt_idx: int, suffix: str) -> List[str]:
    ensure_dir(base_dir)
    paths = []
    for idx, feature in enumerate(features):
        path = base_dir / f"prompt_{prompt_idx}_img_{idx + 1}_{suffix}.pt"
        torch.save(feature.detach().cpu(), path)
        paths.append(str(path))
    return paths


def load_done_prompt_indices(csv_path: Path) -> set[int]:
    done = set()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return done
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = row.get("prompt_index", "")
            if str(value).isdigit():
                done.add(int(value))
    return done


def result_specs(args):
    return {
        "angle": {
            "path": Path(args.latent_cosine_csv),
            "metric_keys": ["mean_angle_deg"],
        },
        "norm": {
            "path": Path(args.latent_norm_csv),
            "metric_keys": ["mean_prev_norm", "mean_curr_norm"],
        },
        "delta_norm": {
            "path": Path(args.latent_update_norm_csv),
            "metric_keys": ["mean_delta_norm"],
        },
    }


def write_trace_metric_rows(
    writers: Dict[str, csv.DictWriter],
    prompt_index: int,
    prompt: str,
    metric_by_column: Dict[str, Dict[int, Dict[str, float]]],
    specs,
) -> None:
    for name, spec in specs.items():
        row = {"prompt_index": prompt_index, "original_prompt": prompt}
        for column, metrics in metric_by_column.items():
            row[column] = json.dumps(_project_step_metrics(metrics, spec["metric_keys"]), ensure_ascii=False)
        writers[name].writerow(row)


def generate_model_outputs(
    args,
    pipe,
    prompts: List[str],
    pending_indices: List[int],
    model_tag: str,
    hidden_dir: Path,
    trace_dir: Path,
) -> Dict[int, Dict[str, object]]:
    outputs: Dict[int, Dict[str, object]] = {}
    for prompt_idx in pending_indices:
        prompt = prompts[prompt_idx]
        samples = []
        hidden_features = []
        for image_idx in range(args.num_images_orig):
            sample, hidden = generate_sample_features(
                pipe,
                prompt,
                seed=args.seed + prompt_idx + image_idx,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                export_steps=args.export_steps,
            )
            samples.append(sample)
            hidden_features.append(hidden)

            img_path = trace_dir / "images" / f"{model_tag}_orig" / f"prompt_{prompt_idx}_img_{image_idx + 1}.png"
            save_image(sample.image, img_path)
            save_trace_files(sample.traces, trace_dir, prompt_idx, model_tag, "orig", image_idx)

        feature_paths = save_hidden_features(
            hidden_features,
            hidden_dir / f"{model_tag}_orig_features",
            prompt_idx,
            "orig",
        )
        metrics = compute_adjacent_z_metrics([sample.traces for sample in samples], args.export_steps)
        outputs[prompt_idx] = {
            "feature_paths": feature_paths,
            "metrics": metrics,
        }
        print(f"[done] {model_tag} prompt_index={prompt_idx}")
    return outputs


def load_cached_model_outputs(
    args,
    pending_indices: List[int],
    model_tag: str,
    hidden_dir: Path,
    trace_dir: Path,
) -> tuple[Dict[int, Dict[str, object]], List[int]]:
    """Recover complete per-prompt outputs already written before an interruption."""

    outputs: Dict[int, Dict[str, object]] = {}
    missing_indices = []
    for prompt_idx in pending_indices:
        feature_paths = [
            hidden_dir
            / f"{model_tag}_orig_features"
            / f"prompt_{prompt_idx}_img_{image_idx + 1}_orig.pt"
            for image_idx in range(args.num_images_orig)
        ]
        if not all(path.exists() for path in feature_paths):
            missing_indices.append(prompt_idx)
            continue

        sample_traces = []
        try:
            for image_idx in range(args.num_images_orig):
                sample_traces.append(
                    load_trace_files(
                        base_dir=trace_dir,
                        prompt_idx=prompt_idx,
                        model_tag=model_tag,
                        group_tag="orig",
                        sample_idx=image_idx,
                        export_steps=args.export_steps,
                    )
                )
        except (FileNotFoundError, OSError, RuntimeError, EOFError):
            missing_indices.append(prompt_idx)
            continue

        outputs[prompt_idx] = {
            "feature_paths": [str(path) for path in feature_paths],
            "metrics": compute_adjacent_z_metrics(sample_traces, args.export_steps),
        }
        print(f"[resume] reuse {model_tag} prompt_index={prompt_idx}")
    return outputs, missing_indices


def prepare_common(args):
    args.export_steps = _parse_steps(args.export_steps, args.num_inference_steps)
    prompts = build_prompt_subset(args.input_txt)
    args.device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(Path(args.lfd_tensor_dir))
    ensure_dir(Path(args.latent_trajectory_dir))
    ensure_dir(Path(args.lfd_output_csv).parent)
    for spec in result_specs(args).values():
        ensure_dir(spec["path"].parent)
    return prompts
