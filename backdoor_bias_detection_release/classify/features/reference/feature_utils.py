"""Reference-assisted feature utilities.

This combines:
- decoded-image similarity
- reference latent discrepancy
- reference activation difference

The implementation computes final reference-vs-target features directly and
does not keep raw activation caches.
"""

import csv
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn


CURRENT_DIR = Path(__file__).resolve().parent
CLASSIFY_DIR = CURRENT_DIR.parents[1]
sys.path.insert(0, str(CLASSIFY_DIR))

from utils.image_similarity import (  # noqa: E402
    PERTURB_METRICS,
    _decode_latents_to_image,
    build_scorers,
    compute_pair_metrics,
    save_step_images,
)
from utils.reference_latent import (  # noqa: E402
    MODEL_CROSS_METRICS,
    SampleTrace,
    compute_cross_model_trace_metrics,
    flatten_metrics,
    generate_with_step_trace,
    load_selected_pipeline,
    parse_steps,
    _get_prompt_embeds,
)
from utils.activation_difference import (  # noqa: E402
    common_fieldnames,
    save_vector,
)


PROMPT_RANGES = [(0, 1000)]


@dataclass
class LightOutput:
    images_by_step: Dict[int, object]
    traces: Dict[int, Dict[str, torch.Tensor]]
    activations: Dict[str, torch.Tensor]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_prompts(txt_path: Path) -> List[str]:
    with txt_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_prompt_subset(input_txt: str) -> tuple[List[int], List[str]]:
    all_prompts = load_prompts(Path(input_txt))
    prompts: List[str] = []
    for start, end in PROMPT_RANGES:
        if start < len(all_prompts):
            prompts.extend(all_prompts[start:min(end, len(all_prompts))])
    if not prompts:
        prompts = all_prompts
    return list(range(len(prompts))), prompts


def release_pipeline(pipe):
    if pipe is None:
        return
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def _feature_root(pipe):
    if getattr(pipe, "_lfd_model_family", "") == "sd2" or hasattr(pipe, "unet"):
        return "unet", pipe.unet
    if hasattr(pipe, "transformer"):
        return "transformer", pipe.transformer
    raise AttributeError("Pipeline does not expose unet or transformer")


def _register_summary_hooks(pipe, collected, layer_start=None, layer_end=None):
    root_name, root = _feature_root(pipe)
    block_containers = []
    for attr in ("transformer_blocks", "joint_transformer_blocks", "blocks", "down_blocks", "up_blocks"):
        candidate = getattr(root, attr, None)
        if candidate is not None and len(candidate) > 0:
            block_containers.append((attr, candidate))
    if hasattr(root, "mid_block"):
        block_containers.append(("mid_block", [root.mid_block]))
    if not block_containers:
        block_containers = [("root", [root])]

    def make_hook(name):
        def hook(module, inp, out):
            tensor = _last_tensor_from_output(out)
            if tensor is None:
                return
            tensor = tensor.detach().float()
            if tensor.dim() <= 1:
                value = tensor.reshape(1).mean()
            else:
                value = tensor.flatten(start_dim=1).mean(dim=1)
            collected.setdefault(name, []).append(value.cpu())
        return hook

    handles = []
    for container_name, blocks in block_containers:
        start = 0 if layer_start is None else layer_start
        end = len(blocks) - 1 if layer_end is None else min(layer_end, len(blocks) - 1)
        if start >= len(blocks) or end < start:
            continue
        for block_idx in range(start, end + 1):
            block = blocks[block_idx]
            wrapped_prefixes = []
            for module_name, module in block.named_modules():
                if not module_name:
                    continue
                if any(module_name == p or module_name.startswith(f"{p}.") for p in wrapped_prefixes):
                    continue
                if isinstance(module, (nn.ModuleList, nn.ModuleDict, nn.Sequential)):
                    continue
                base_layer = getattr(module, "base_layer", None)
                if isinstance(base_layer, nn.Module):
                    wrapped_prefixes.append(module_name)
                elif any(True for _ in module.children()):
                    continue
                full_name = f"{root_name}.{container_name}.{block_idx}.{module_name}"
                handles.append(module.register_forward_hook(make_hook(full_name)))
    if not handles:
        raise RuntimeError("No hookable modules were found for navi summaries")
    return handles


def _summary_to_activation_dict(collected: Dict[str, List[torch.Tensor]], export_steps: List[int]):
    activations = {}
    for name, values in sorted(collected.items()):
        if max(export_steps) >= len(values):
            raise ValueError(
                f"Requested export step {max(export_steps)} but module {name} "
                f"was called only {len(values)} times"
            )
        activations[name] = torch.stack([values[step] for step in export_steps], dim=0)
    return activations


def generate_light_output(args, pipe, prompt: str, seed: int) -> LightOutput:
    collected: Dict[str, List[torch.Tensor]] = {}
    handles = _register_summary_hooks(
        pipe,
        collected,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
    )
    try:
        sample = generate_with_step_trace(
            pipe,
            prompt,
            seed=seed,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            export_steps=args.export_steps,
        )
    finally:
        for handle in handles:
            handle.remove()

    images_by_step = {
        step: _decode_latents_to_image(pipe, sample.traces[step]["z_t"], args.height, args.width)
        for step in args.export_steps
        if step in sample.traces and "z_t" in sample.traces[step]
    }
    activations = _summary_to_activation_dict(collected, args.export_steps)
    return LightOutput(images_by_step=images_by_step, traces=sample.traces, activations=activations)


def _summary_to_activation_dicts(
    collected: Dict[str, List[torch.Tensor]],
    export_steps: List[int],
    batch_size: int,
    do_cfg: bool,
) -> List[Dict[str, torch.Tensor]]:
    activations = [dict() for _ in range(batch_size)]
    for name, values in sorted(collected.items()):
        if max(export_steps) >= len(values):
            raise ValueError(
                f"Requested export step {max(export_steps)} but module {name} "
                f"was called only {len(values)} times"
            )
        stacked = torch.stack([values[step] for step in export_steps], dim=0)
        if stacked.dim() == 1:
            stacked = stacked.reshape(len(export_steps), 1)

        for sample_idx in range(batch_size):
            if do_cfg and stacked.shape[1] >= batch_size * 2:
                sample_values = stacked[:, [sample_idx, sample_idx + batch_size]]
            elif stacked.shape[1] >= batch_size:
                sample_values = stacked[:, sample_idx : sample_idx + 1]
            else:
                sample_values = stacked
            activations[sample_idx][name] = sample_values.float()
    return activations


def _make_batched_latents(
    batch_size: int,
    in_channels: int,
    latent_h: int,
    latent_w: int,
    seeds: List[int],
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    latents = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        latents.append(
            torch.randn(
                (1, in_channels, latent_h, latent_w),
                generator=generator,
                device=device,
                dtype=dtype,
            )
        )
    return torch.cat(latents, dim=0)


def generate_light_outputs_batch(args, pipe, prompt_indices: List[int], prompts: List[str]) -> Dict[int, LightOutput]:
    """Generate a batch of light outputs. FLUX currently falls back to single-prompt generation."""

    if not prompts:
        return {}
    if pipe.__class__.__name__.lower().startswith("flux"):
        return {
            prompt_idx: generate_light_output(args, pipe, prompt, args.seed + prompt_idx)
            for prompt_idx, prompt in zip(prompt_indices, prompts)
        }

    batch_size = len(prompts)
    seeds = [args.seed + prompt_idx for prompt_idx in prompt_indices]
    collected: Dict[str, List[torch.Tensor]] = {}
    handles = _register_summary_hooks(
        pipe,
        collected,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
    )

    try:
        sample = _generate_with_step_trace_sd_batch(
            pipe,
            prompts,
            seeds,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            export_steps=args.export_steps,
            height=args.height,
            width=args.width,
        )
    finally:
        for handle in handles:
            handle.remove()

    do_cfg = args.guidance_scale > 1.0
    activations_by_sample = _summary_to_activation_dicts(
        collected,
        args.export_steps,
        batch_size,
        do_cfg,
    )
    outputs = {}
    for sample_idx, prompt_idx in enumerate(prompt_indices):
        sample_traces = {}
        for step, payload in sample.traces.items():
            sample_payload = {}
            for name, tensor in payload.items():
                sample_payload[name] = tensor[sample_idx : sample_idx + 1].contiguous()
            sample_traces[step] = sample_payload
        images_by_step = {
            step: _decode_latents_to_image(pipe, sample_traces[step]["z_t"], args.height, args.width)
            for step in args.export_steps
            if step in sample_traces and "z_t" in sample_traces[step]
        }
        outputs[prompt_idx] = LightOutput(
            images_by_step=images_by_step,
            traces=sample_traces,
            activations=activations_by_sample[sample_idx],
        )
    return outputs


def _generate_with_step_trace_sd_batch(
    pipe,
    prompts: List[str],
    seeds: List[int],
    num_inference_steps: int,
    guidance_scale: float,
    export_steps: List[int],
    height: int,
    width: int,
) -> SampleTrace:
    device = str(pipe.device)
    do_cfg = guidance_scale > 1.0
    denoiser = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
    if denoiser is None:
        raise RuntimeError("Pipeline does not expose transformer or unet")
    is_sd3 = getattr(pipe, "_is_sd3", False) or hasattr(pipe, "transformer")
    batch_size = len(prompts)

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = _get_prompt_embeds(pipe, prompts, device, do_cfg)

        vae_scale = int(getattr(pipe, "vae_scale_factor", 8))
        latent_h = int(height // vae_scale)
        latent_w = int(width // vae_scale)
        in_channels = int(getattr(denoiser.config, "in_channels", 4))
        latents = _make_batched_latents(
            batch_size,
            in_channels,
            latent_h,
            latent_w,
            seeds,
            device,
            denoiser.dtype,
        )

        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = latents * getattr(pipe.scheduler, "init_noise_sigma", 1.0)

        traces: Dict[int, Dict[str, torch.Tensor]] = {}
        export_set = set(export_steps)

        for step_idx, t in enumerate(pipe.scheduler.timesteps):
            latent_model_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
            if hasattr(pipe.scheduler, "scale_model_input"):
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            timestep_input = t
            if is_sd3:
                timestep_input = torch.as_tensor(t, device=device).reshape(1).repeat(latent_model_input.shape[0])

            if is_sd3:
                noise_pred = denoiser(
                    hidden_states=latent_model_input,
                    timestep=timestep_input,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
            else:
                noise_pred = denoiser(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if step_idx in export_set:
                traces[step_idx] = {
                    "z_t": latents.detach().float().cpu(),
                    "eps_hat": noise_pred.detach().float().cpu(),
                }

            step_out = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)
            if isinstance(step_out, tuple):
                latents = step_out[0]
            elif isinstance(step_out, dict) and "prev_sample" in step_out:
                latents = step_out["prev_sample"]
            else:
                latents = step_out

        vae_config = getattr(getattr(pipe, "vae", None), "config", None)
        scaling_factor = float(getattr(vae_config, "scaling_factor", 1.0))
        decoded = pipe.vae.decode(latents / scaling_factor, return_dict=False)
        image_tensor = decoded[0] if isinstance(decoded, (tuple, list)) else decoded

        if hasattr(pipe, "image_processor") and hasattr(pipe.image_processor, "postprocess"):
            images = pipe.image_processor.postprocess(image_tensor, output_type="pil")
            image = images[0] if images else None
        else:
            image = None

    return SampleTrace(image=image, traces=traces)


def generate_outputs_for_prompts(args, model_id, lora_id, prompt_indices, prompts, tag):
    pipe = load_selected_pipeline(args.model_family, model_id, args.device, lora_id=lora_id)
    outputs = {}
    try:
        batch_size = max(1, int(getattr(args, "prompt_batch_size", 1)))
        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            batch_indices = prompt_indices[start:end]
            batch_prompts = prompts[start:end]
            outputs.update(generate_light_outputs_batch(args, pipe, batch_indices, batch_prompts))
            print(f"[features] {tag} prompt_index={batch_indices[0]}-{batch_indices[-1]}")
    finally:
        release_pipeline(pipe)
    return outputs


def image_fieldnames(export_steps):
    fields = []
    for ref_name in ("clean_ref", "backdoor_ref"):
        for step in export_steps:
            fields.extend([f"perturb_{ref_name}_step_{step:03d}_{metric}" for metric in PERTURB_METRICS])
    return fields


def reference_latent_fieldnames(export_steps):
    fields = []
    for ref_name in ("clean_ref", "backdoor_ref"):
        for step in export_steps:
            fields.extend([f"modelcross_{ref_name}_step_{step:03d}_{metric}" for metric in MODEL_CROSS_METRICS])
    return fields


def write_rows(output_csv: Path, fieldnames: List[str], rows: List[dict]):
    ensure_dir(output_csv.parent)
    existing_rows = read_rows_by_prompt(output_csv)
    for row in rows:
        prompt_idx = _prompt_index_from_row(row)
        if prompt_idx is None:
            continue
        merged = {field: "" for field in fieldnames}
        if prompt_idx in existing_rows:
            merged.update(existing_rows[prompt_idx])
        merged.update(row)
        existing_rows[prompt_idx] = merged

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for prompt_idx in sorted(existing_rows):
            row = existing_rows[prompt_idx]
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def save_activation_difference_row_vectors(args, output_dir: Path, prompt_idx: int, clean_vector, backdoor_vector):
    clean_path = save_vector(clean_vector, output_dir / "difference_vectors" / "clean_ref", prompt_idx, "clean_ref")
    backdoor_path = save_vector(backdoor_vector, output_dir / "difference_vectors" / "backdoor_ref", prompt_idx, "backdoor_ref")
    return clean_path, backdoor_path


def compute_summary_difference_vector(reference: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
    """Compute [step, module] abs differences from per-module step summaries."""

    reference_names = set(reference)
    target_names = set(target)
    if reference_names != target_names:
        raise ValueError(
            "Reference/target intermediate module names do not match: "
            f"missing_target={sorted(reference_names - target_names)}, "
            f"missing_reference={sorted(target_names - reference_names)}"
        )

    layer_names = sorted(reference_names)
    values = []
    for name in layer_names:
        ref_tensor = reference[name].float()
        target_tensor = target[name].float()
        if ref_tensor.shape != target_tensor.shape:
            raise ValueError(
                f"Activation summary shape mismatch for {name}: "
                f"reference={tuple(ref_tensor.shape)}, target={tuple(target_tensor.shape)}"
            )

        diff = (target_tensor - ref_tensor).abs()
        if diff.dim() == 0:
            per_step = diff.reshape(1)
        elif diff.dim() == 1:
            per_step = diff
        else:
            per_step = diff.flatten(start_dim=1).mean(dim=1)
        values.append(per_step)

    return layer_names, torch.stack(values, dim=1).float()


def init_reference_rows(args, prompt_idx: int, prompt: str, target_tag: str, label):
    image_row = {
        "prompt_index": prompt_idx,
        "original_prompt": prompt,
        "model_tag": target_tag,
        "label": "" if label is None else int(label),
    }
    reference_latent_row = {
        "prompt_index": prompt_idx,
        "original_prompt": prompt,
        "model_tag": target_tag,
        "label": "" if label is None else int(label),
    }
    activation_difference_row = {
        "prompt_index": prompt_idx,
        "original_prompt": prompt,
        "model_tag": target_tag,
        "label": "" if label is None else int(label),
    }
    return image_row, reference_latent_row, activation_difference_row


def update_rows_with_reference(
    args,
    prompt_idx: int,
    target,
    reference,
    ref_name: str,
    image_row: dict,
    reference_latent_row: dict,
    activation_difference_row: dict,
    scorers,
):
    image_base_dir = Path(args.image_similarity_image_dir)
    activation_difference_tensor_dir = Path(args.activation_difference_tensor_dir)

    if "target_paths" not in image_row:
        target_paths = save_step_images(target.images_by_step, image_base_dir / "target", prompt_idx, "target")
        image_row["target_paths"] = ";".join(target_paths[step] for step in args.export_steps)

    ref_paths = save_step_images(reference.images_by_step, image_base_dir / ref_name, prompt_idx, ref_name)
    image_row[f"{ref_name}_paths"] = ";".join(ref_paths[step] for step in args.export_steps)

    for step in args.export_steps:
        metrics = compute_pair_metrics(
            [target.images_by_step[step]],
            [reference.images_by_step[step]],
            scorers,
            args.device,
        )
        for metric, value in metrics.items():
            image_row[f"perturb_{ref_name}_step_{step:03d}_{metric}"] = value

    cross_metrics = compute_cross_model_trace_metrics([reference.traces], [target.traces], args.export_steps)
    reference_latent_row.update(flatten_metrics(ref_name, cross_metrics))

    layer_names, diff_vector = compute_summary_difference_vector(reference.activations, target.activations)
    vector_path = save_vector(
        diff_vector,
        activation_difference_tensor_dir / "difference_vectors" / ref_name,
        prompt_idx,
        ref_name,
    )
    activation_difference_row[f"{ref_name}_difference_vector_path"] = vector_path
    activation_difference_row[f"{ref_name}_summed_intermediate_difference"] = float(diff_vector.sum().item())
    existing_names = activation_difference_row.get("intermediate_layer_names")
    layer_names_json = json.dumps(layer_names)
    if existing_names not in (None, layer_names_json):
        raise ValueError("Reference layer names do not match")
    activation_difference_row["intermediate_layer_names"] = layer_names_json


def compute_rows_for_target(
    args,
    prompt_indices: List[int],
    prompts: List[str],
    target_outputs,
    clean_outputs,
    backdoor_outputs,
    target_tag: str,
    label,
    scorers=None,
):
    if scorers is None:
        scorers = build_scorers(
            args.device,
            getattr(args, "openclip_model", ""),
            getattr(args, "openclip_pretrained", ""),
            getattr(args, "openclip_cache_dir", None),
            getattr(args, "openclip_revision", None),
        )
    image_rows = []
    reference_latent_rows = []
    activation_difference_rows = []
    image_base_dir = Path(args.image_similarity_image_dir)
    activation_difference_tensor_dir = Path(args.activation_difference_tensor_dir)

    for prompt_idx, prompt in zip(prompt_indices, prompts):
        target = target_outputs[prompt_idx]
        clean_ref = clean_outputs[prompt_idx]
        backdoor_ref = backdoor_outputs[prompt_idx]

        image_row = {
            "prompt_index": prompt_idx,
            "original_prompt": prompt,
            "model_tag": target_tag,
            "label": "" if label is None else int(label),
        }
        target_paths = save_step_images(target.images_by_step, image_base_dir / "target", prompt_idx, "target")
        clean_paths = save_step_images(clean_ref.images_by_step, image_base_dir / "clean_ref", prompt_idx, "clean_ref")
        backdoor_paths = save_step_images(backdoor_ref.images_by_step, image_base_dir / "backdoor_ref", prompt_idx, "backdoor_ref")
        image_row["target_paths"] = ";".join(target_paths[step] for step in args.export_steps)
        image_row["clean_ref_paths"] = ";".join(clean_paths[step] for step in args.export_steps)
        image_row["backdoor_ref_paths"] = ";".join(backdoor_paths[step] for step in args.export_steps)

        for step in args.export_steps:
            clean_metrics = compute_pair_metrics(
                [target.images_by_step[step]],
                [clean_ref.images_by_step[step]],
                scorers,
                args.device,
            )
            backdoor_metrics = compute_pair_metrics(
                [target.images_by_step[step]],
                [backdoor_ref.images_by_step[step]],
                scorers,
                args.device,
            )
            for metric, value in clean_metrics.items():
                image_row[f"perturb_clean_ref_step_{step:03d}_{metric}"] = value
            for metric, value in backdoor_metrics.items():
                image_row[f"perturb_backdoor_ref_step_{step:03d}_{metric}"] = value
        image_rows.append(image_row)

        clean_cross = compute_cross_model_trace_metrics([clean_ref.traces], [target.traces], args.export_steps)
        backdoor_cross = compute_cross_model_trace_metrics([backdoor_ref.traces], [target.traces], args.export_steps)
        reference_latent_row = {
            "prompt_index": prompt_idx,
            "original_prompt": prompt,
            "model_tag": target_tag,
            "label": "" if label is None else int(label),
        }
        reference_latent_row.update(flatten_metrics("clean_ref", clean_cross))
        reference_latent_row.update(flatten_metrics("backdoor_ref", backdoor_cross))
        reference_latent_rows.append(reference_latent_row)

        clean_names, clean_vector = compute_difference_vector(clean_ref.activations, target.activations)
        backdoor_names, backdoor_vector = compute_difference_vector(backdoor_ref.activations, target.activations)
        if clean_names != backdoor_names:
            raise ValueError("Clean-reference and backdoor-reference layer names do not match")
        clean_path, backdoor_path = save_activation_difference_row_vectors(
            args,
            activation_difference_tensor_dir,
            prompt_idx,
            clean_vector,
            backdoor_vector,
        )
        activation_difference_rows.append(
            {
                "prompt_index": prompt_idx,
                "original_prompt": prompt,
                "model_tag": target_tag,
                "label": "" if label is None else int(label),
                "clean_ref_difference_vector_path": clean_path,
                "backdoor_ref_difference_vector_path": backdoor_path,
                "intermediate_layer_names": json.dumps(clean_names),
                "clean_ref_summed_intermediate_difference": float(clean_vector.sum().item()),
                "backdoor_ref_summed_intermediate_difference": float(backdoor_vector.sum().item()),
            }
        )

    return image_rows, reference_latent_rows, activation_difference_rows


def prepare_args(args):
    args.export_steps = parse_steps(args.export_steps, args.num_inference_steps)
    args.device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    for path in (
        Path(args.image_similarity_output_csv).parent,
        Path(args.reference_latent_output_csv).parent,
        Path(args.activation_difference_output_csv).parent,
        Path(args.image_similarity_image_dir),
        Path(args.activation_difference_tensor_dir),
    ):
        ensure_dir(path)
    return args

def completed_prompt_indices(output_csv: str | Path) -> set[int]:
    output_csv = Path(output_csv)
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return set()
    done = set()
    with output_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = row.get("prompt_index", "")
            if str(value).isdigit():
                done.add(int(value))
    return done


def _prompt_index_from_row(row: dict) -> int | None:
    value = row.get("prompt_index", "")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read_rows_by_prompt(output_csv: str | Path) -> Dict[int, dict]:
    output_csv = Path(output_csv)
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return {}
    rows = {}
    with output_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prompt_idx = _prompt_index_from_row(row)
            if prompt_idx is not None:
                rows[prompt_idx] = row
    return rows


def _row_has_values(row: dict | None, columns: List[str]) -> bool:
    if not row:
        return False
    return all(str(row.get(column, "")).strip() for column in columns)


def clean_joint_prompt_indices(args) -> set[int]:
    image_rows = read_rows_by_prompt(args.image_similarity_output_csv)
    reference_latent_rows = read_rows_by_prompt(args.reference_latent_output_csv)
    activation_difference_rows = read_rows_by_prompt(args.activation_difference_output_csv)
    clean_reference_latent_columns = [
        f"modelcross_clean_ref_step_{step:03d}_{metric}"
        for step in args.export_steps
        for metric in MODEL_CROSS_METRICS
    ]

    done = set(image_rows) & set(reference_latent_rows) & set(activation_difference_rows)
    return {
        prompt_idx
        for prompt_idx in done
        if _row_has_values(image_rows.get(prompt_idx), ["target_paths", "clean_ref_paths"])
        and _row_has_values(reference_latent_rows.get(prompt_idx), clean_reference_latent_columns)
        and _row_has_values(activation_difference_rows.get(prompt_idx), ["clean_ref_difference_vector_path"])
    }


def completed_joint_prompt_indices(args) -> set[int]:
    image_rows = read_rows_by_prompt(args.image_similarity_output_csv)
    reference_latent_rows = read_rows_by_prompt(args.reference_latent_output_csv)
    activation_difference_rows = read_rows_by_prompt(args.activation_difference_output_csv)
    backdoor_reference_latent_columns = [
        f"modelcross_backdoor_ref_step_{step:03d}_{metric}"
        for step in args.export_steps
        for metric in MODEL_CROSS_METRICS
    ]

    done = set(image_rows) & set(reference_latent_rows) & set(activation_difference_rows)
    return {
        prompt_idx
        for prompt_idx in done
        if _row_has_values(image_rows.get(prompt_idx), ["target_paths", "clean_ref_paths", "backdoor_ref_paths"])
        and _row_has_values(reference_latent_rows.get(prompt_idx), backdoor_reference_latent_columns)
        and _row_has_values(
            activation_difference_rows.get(prompt_idx),
            ["clean_ref_difference_vector_path", "backdoor_ref_difference_vector_path"],
        )
    }



def write_all_outputs(args, image_rows, reference_latent_rows, activation_difference_rows):
    image_fields = [
        "prompt_index",
        "original_prompt",
        "model_tag",
        "label",
        "target_paths",
        "clean_ref_paths",
        "backdoor_ref_paths",
    ] + image_fieldnames(args.export_steps)
    reference_latent_fields = [
        "prompt_index",
        "original_prompt",
        "model_tag",
        "label",
    ] + reference_latent_fieldnames(args.export_steps)
    write_rows(Path(args.image_similarity_output_csv), image_fields, image_rows)
    write_rows(Path(args.reference_latent_output_csv), reference_latent_fields, reference_latent_rows)
    write_rows(Path(args.activation_difference_output_csv), common_fieldnames(), activation_difference_rows)
