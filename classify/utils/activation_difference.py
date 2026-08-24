"""Utilities for first-step hidden/intermediate activation difference features.

The feature is a per-module absolute mean difference between a reference model
and a target model at the first denoising step. No layer averaging is applied.
"""

import csv
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


ActivationDict = Dict[str, torch.Tensor]

PROMPT_RANGES = [(0, 10), (100, 110), (200, 210), (300, 310), (400, 410), (500, 510)]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_prompts(txt_path: Path) -> List[str]:
    with txt_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_prompt_subset(input_txt: str) -> Tuple[List[int], List[str]]:
    all_prompts = load_prompts(Path(input_txt))
    prompts: List[str] = []
    for start, end in PROMPT_RANGES:
        prompts.extend(all_prompts[start:min(end, len(all_prompts))])
    return list(range(len(prompts))), prompts


def parse_export_steps(step_text: str) -> List[int]:
    steps = sorted({int(x.strip()) for x in step_text.split(",") if x.strip()})
    if not steps:
        raise ValueError("No valid export steps were provided")
    return steps


def _dtype_for_device(device: str):
    return torch.float16 if device.startswith("cuda") else torch.float32


def _finish_pipeline_setup(pipe, device: str, model_family: str):
    pipe = pipe.to(device)
    pipe._lfd_model_family = model_family
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    return pipe


def get_sd2_pipeline(model_id: str, device: str, hf_token: str | None = None):
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=_dtype_for_device(device),
        token=hf_token if hf_token else None,
    )
    return _finish_pipeline_setup(pipe, device, "sd2")


def get_sd35_pipeline(model_id: str, device: str, hf_token: str | None = None, lora_id: str | None = None):
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=_dtype_for_device(device),
        token=hf_token if hf_token else None,
    )
    if lora_id:
        pipe.load_lora_weights(lora_id)
    return _finish_pipeline_setup(pipe, device, "sd35")


def get_flux_pipeline(model_id: str, device: str, hf_token: str | None = None, lora_id: str | None = None):
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=_dtype_for_device(device),
        token=hf_token if hf_token else None,
    )
    if lora_id:
        pipe.load_lora_weights(lora_id)
    return _finish_pipeline_setup(pipe, device, "flux")


def load_selected_pipeline(model_family: str, model_id: str, device: str, lora_id: str | None = None):
    hf_token = os.environ.get("HF_TOKEN")
    if model_family == "sd2":
        return get_sd2_pipeline(model_id, device, hf_token)
    if model_family == "sd35":
        return get_sd35_pipeline(model_id, device, hf_token, lora_id=lora_id)
    if model_family == "flux":
        return get_flux_pipeline(model_id, device, hf_token, lora_id=lora_id)
    raise ValueError(f"Unknown model family: {model_family}")


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
        root = getattr(pipe, "unet", None)
        if root is None:
            raise AttributeError("Expected SD2 pipeline to expose pipe.unet")
        return "unet", root

    root = getattr(pipe, "transformer", None)
    if root is None:
        raise AttributeError("Expected SD3.5/FLUX pipeline to expose pipe.transformer")
    return "transformer", root


def _register_intermediate_hooks(pipe, make_hook, layer_start: int | None = None, layer_end: int | None = None):
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

                is_inside_wrapper = any(
                    module_name == prefix or module_name.startswith(f"{prefix}.")
                    for prefix in wrapped_prefixes
                )
                if is_inside_wrapper:
                    continue

                if isinstance(module, (nn.ModuleList, nn.ModuleDict, nn.Sequential)):
                    continue

                # PEFT/LoRA replaces Linear layers with wrapper modules that contain
                # base_layer/lora_A/lora_B children. Hook the wrapper itself so clean,
                # backdoor, and target models expose the same logical module names.
                base_layer = getattr(module, "base_layer", None)
                is_lora_wrapper = isinstance(base_layer, nn.Module)
                if is_lora_wrapper:
                    wrapped_prefixes.append(module_name)
                elif any(True for _ in module.children()):
                    continue

                full_name = f"{root_name}.{container_name}.{block_idx}.{module_name}"
                handles.append(module.register_forward_hook(make_hook(full_name)))

    if not handles:
        raise RuntimeError("No hookable intermediate modules were found")
    return handles


def extract_step_activations(
    pipe,
    prompt: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    export_steps: List[int],
    layer_start: int | None = None,
    layer_end: int | None = None,
) -> ActivationDict:
    collected: Dict[str, List[torch.Tensor]] = {}
    if max(export_steps) >= num_inference_steps:
        raise ValueError(f"export step {max(export_steps)} must be < num_inference_steps={num_inference_steps}")

    def make_hook(name):
        def hook(module, inp, out):
            tensor_out = _last_tensor_from_output(out)
            if tensor_out is not None:
                collected.setdefault(name, []).append(tensor_out.detach().cpu().to(torch.float16))
        return hook

    handles = _register_intermediate_hooks(pipe, make_hook, layer_start, layer_end)
    pipe_device = getattr(pipe, "device", getattr(pipe, "_execution_device", "cpu"))
    generator = torch.Generator(device=pipe_device).manual_seed(seed)
    try:
        kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "generator": generator,
            "output_type": "latent",
        }
        try:
            pipe(**kwargs)
        except TypeError:
            kwargs.pop("output_type", None)
            pipe(**kwargs)
    finally:
        for handle in handles:
            handle.remove()

    if not collected:
        raise RuntimeError("No intermediate activations were collected")

    selected: ActivationDict = {}
    for name, tensors in sorted(collected.items()):
        if max(export_steps) >= len(tensors):
            raise ValueError(
                f"Requested export step {max(export_steps)} but module {name} "
                f"was called only {len(tensors)} times"
            )
        selected[name] = torch.stack([tensors[step] for step in export_steps], dim=0)
    return selected


def extract_first_step_activations(
    pipe,
    prompt: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    layer_start: int | None = None,
    layer_end: int | None = None,
) -> ActivationDict:
    return extract_step_activations(
        pipe,
        prompt,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        export_steps=[0],
        layer_start=layer_start,
        layer_end=layer_end,
    )


def generate_activation_table(args, prompt_indices: List[int], prompts: List[str], model_id: str, lora_id: str | None):
    pipe = load_selected_pipeline(args.model_family, model_id, args.device, lora_id=lora_id)
    activations_by_prompt: Dict[int, ActivationDict] = {}
    try:
        for prompt_idx, prompt in zip(prompt_indices, prompts):
            activations_by_prompt[prompt_idx] = extract_first_step_activations(
                pipe,
                prompt,
                seed=args.seed + prompt_idx,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                layer_start=args.layer_start,
                layer_end=args.layer_end,
            )
            print(f"[features] prompt_index={prompt_idx}")
    finally:
        release_pipeline(pipe)
    return activations_by_prompt


def save_activation_dict(activations: ActivationDict, base_dir: Path, prompt_idx: int, model_tag: str) -> str:
    """Save one prompt's activation tensors as shards to avoid keeping all prompts in RAM."""

    prompt_dir = base_dir / model_tag / f"prompt_{prompt_idx}"
    ensure_dir(prompt_dir)
    manifest = {}
    for tensor_idx, (name, tensor) in enumerate(sorted(activations.items())):
        shard_name = f"{tensor_idx:04d}.pt"
        torch.save(tensor.detach().cpu().to(torch.float16), prompt_dir / shard_name)
        manifest[name] = shard_name
    manifest_path = prompt_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f)
    return str(manifest_path)


def load_activation_dict(manifest_path: str | Path) -> ActivationDict:
    """Load one prompt's activation shard manifest."""

    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    return {
        name: torch.load(manifest_path.parent / shard_name, map_location="cpu")
        for name, shard_name in manifest.items()
    }


def delete_activation_manifest(manifest_path: str | Path):
    """Delete one prompt cache directory created by save_activation_dict()."""

    shutil.rmtree(Path(manifest_path).parent, ignore_errors=True)


def cleanup_activation_cache(output_dir: str | Path):
    """Delete the temporary activation cache directory."""

    shutil.rmtree(Path(output_dir) / "activation_cache", ignore_errors=True)


def generate_activation_cache(
    args,
    prompt_indices: List[int],
    prompts: List[str],
    model_id: str,
    lora_id: str | None,
    model_tag: str,
) -> Dict[int, str]:
    """Extract activations one prompt at a time and save them to disk."""

    cache_dir = Path(args.output_dir) / "activation_cache"
    pipe = load_selected_pipeline(args.model_family, model_id, args.device, lora_id=lora_id)
    manifest_by_prompt: Dict[int, str] = {}
    try:
        for prompt_idx, prompt in zip(prompt_indices, prompts):
            activations = extract_step_activations(
                pipe,
                prompt,
                seed=args.seed + prompt_idx,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                export_steps=args.export_steps,
                layer_start=args.layer_start,
                layer_end=args.layer_end,
            )
            manifest_by_prompt[prompt_idx] = save_activation_dict(
                activations,
                cache_dir,
                prompt_idx,
                model_tag,
            )
            del activations
            gc.collect()
            print(f"[features] {model_tag} prompt_index={prompt_idx}")
    finally:
        release_pipeline(pipe)
    return manifest_by_prompt


def compute_difference_vector(reference: ActivationDict, target: ActivationDict) -> Tuple[List[str], torch.Tensor]:
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
                f"Activation shape mismatch for {name}: "
                f"reference={tuple(ref_tensor.shape)}, target={tuple(target_tensor.shape)}"
            )
        diff = (target_tensor - ref_tensor).abs()
        if diff.dim() <= 1:
            per_step = diff.reshape(1).mean().reshape(1)
        else:
            per_step = diff.flatten(start_dim=1).mean(dim=1)
        values.append(per_step)
    return layer_names, torch.stack(values, dim=1).float()


def save_vector(vector: torch.Tensor, base_dir: Path, prompt_idx: int, prefix: str) -> str:
    ensure_dir(base_dir)
    path = base_dir / f"prompt_{prompt_idx}_{prefix}_step_abs_mean.pt"
    torch.save(vector.detach().cpu(), path)
    return str(path)


def write_rows(output_csv: Path, fieldnames: List[str], rows: List[dict]):
    ensure_dir(output_csv.parent)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_rows(output_csv: Path, fieldnames: List[str], rows: List[dict]):
    """Append feature rows and create the CSV header when needed."""

    if not rows:
        return
    ensure_dir(output_csv.parent)
    file_exists = output_csv.exists() and output_csv.stat().st_size > 0
    if file_exists:
        with output_csv.open("r", newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header != fieldnames:
            raise ValueError(
                f"Existing CSV header does not match expected fields: {output_csv}"
            )

    with output_csv.open("a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def read_feature_rows(output_csv: Path) -> List[dict]:
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return []
    with output_csv.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def completed_prompt_indices(output_csv: Path) -> set[int]:
    done = set()
    for row in read_feature_rows(output_csv):
        value = row.get("prompt_index", "")
        if str(value).isdigit():
            done.add(int(value))
    return done


def common_fieldnames() -> List[str]:
    return [
        "prompt_index",
        "original_prompt",
        "model_tag",
        "label",
        "clean_ref_difference_vector_path",
        "backdoor_ref_difference_vector_path",
        "intermediate_layer_names",
        "clean_ref_summed_intermediate_difference",
        "backdoor_ref_summed_intermediate_difference",
    ]
