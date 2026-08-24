"""Generate diffusion traces and summarize adjacent-step latent changes.

The script runs clean, backdoor, and test models on the same prompt set. For
selected denoising steps it saves the latent `z_t` and noise prediction
`eps_hat`, then writes CSV summaries of adjacent-step cosine/norm changes.
"""

import argparse
import csv
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from PIL import Image


@dataclass
class SampleTrace:
    """A generated image plus the intermediate tensors captured during denoising."""

    image: Image.Image
    traces: Dict[int, Dict[str, torch.Tensor]]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_prompts(txt_path: Path) -> List[str]:
    """Read non-empty prompt lines from a UTF-8 text file."""

    lines: List[str] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
    return lines


def _dtype_for_device(device: str) -> torch.dtype:
    """Use fp16 on CUDA and fp32 on CPU."""

    return torch.float16 if device.startswith("cuda") else torch.float32


def _finish_pipeline_setup(pipe, device: str, is_sd3: bool):
    """Move a loaded pipeline to device and attach metadata used by tracing."""

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    setattr(pipe, "_is_sd3", is_sd3)
    return pipe


def get_sd2_pipeline(model_id: str, device: str, hf_token: str | None = None):
    """Load a Stable Diffusion 1.x/2.x model."""

    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=_dtype_for_device(device),
        token=hf_token if hf_token else None,
    )
    return _finish_pipeline_setup(pipe, device, is_sd3=False)


def get_sd35_pipeline(
    model_id: str,
    device: str,
    hf_token: str | None = None,
    lora_id: str | None = None,
):
    """Load an SD3/SD3.5 base model, optionally with LoRA weights."""

    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=_dtype_for_device(device),
        token=hf_token if hf_token else None,
    )
    if lora_id:
        pipe.load_lora_weights(lora_id)
    return _finish_pipeline_setup(pipe, device, is_sd3=True)


def get_flux_pipeline(
    model_id: str,
    device: str,
    hf_token: str | None = None,
    lora_id: str | None = None,
):
    """Load a FLUX.1 model, optionally with LoRA weights."""

    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=_dtype_for_device(device),
        token=hf_token if hf_token else None,
    )
    if lora_id:
        pipe.load_lora_weights(lora_id)
    return _finish_pipeline_setup(pipe, device, is_sd3=False)

def _parse_steps(step_text: str, num_inference_steps: int) -> List[int]:
    """Parse and validate comma-separated export step indices."""

    raw = [x.strip() for x in step_text.split(",") if x.strip()]
    if not raw:
        return [num_inference_steps - 1]

    steps = sorted({int(x) for x in raw})
    valid = [s for s in steps if 0 <= s < num_inference_steps]
    if not valid:
        raise ValueError("No valid export steps found in --export_steps")
    return valid



def _get_prompt_embeds(pipe, prompt: str, device: str, do_cfg: bool):
    """Return token and pooled prompt embeddings across diffusers variants.

    Different pipeline versions expose slightly different `encode_prompt`
    signatures and return tuple layouts. This helper strips unsupported keyword
    arguments and normalizes the outputs used by SD/SD3/Flux denoising calls.
    """

    import re

    def _align_and_cat(neg, pos):
        if neg is None:
            return pos
        if pos is None:
            return neg
        if neg.ndim == pos.ndim and neg.shape[-1] == pos.shape[-1]:
            return torch.cat([neg, pos], dim=0)
        # 不可安全拼接时直接报错给上层决定是否降级
        raise RuntimeError(
            f"Cannot concat embeds: neg.shape={tuple(neg.shape)} vs pos.shape={tuple(pos.shape)}"
        )

    def try_encode(kwargs):
        try:
            return pipe.encode_prompt(**kwargs)
        except TypeError as e:
            msg = str(e)
            m = re.search(r"unexpected keyword argument '([^']+)'", msg)
            if m:
                bad = m.group(1)
                kwargs2 = dict(kwargs)
                kwargs2.pop(bad, None)
                return try_encode(kwargs2)

            for k in [
                "do_classifier_free_guidance",
                "negative_prompt",
                "prompt_2",
                "negative_prompt_2",
                "prompt_3",
                "negative_prompt_3",
                "max_sequence_length",
                "pooled_projections",
                "device",
            ]:
                if k in kwargs:
                    kwargs2 = dict(kwargs)
                    kwargs2.pop(k, None)
                    try:
                        return pipe.encode_prompt(**kwargs2)
                    except TypeError:
                        continue

            return pipe.encode_prompt(prompt)

    encoded = None
    if hasattr(pipe, "encode_prompt"):
        encode_kwargs = dict(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt="",
            max_sequence_length=256,
        )
        if hasattr(pipe, "text_encoder_2"):
            encode_kwargs["prompt_2"] = prompt
            encode_kwargs["negative_prompt_2"] = ""
        if hasattr(pipe, "text_encoder_3"):
            encode_kwargs["prompt_3"] = prompt
            encode_kwargs["negative_prompt_3"] = ""

        encoded = try_encode(encode_kwargs)

    if encoded is not None:
        if not isinstance(encoded, tuple):
            return encoded, None

        # 通用解析：
        # SD3 常见: (prompt, neg, pooled, neg_pooled)
        # FLUX 常见: (prompt, pooled, ...)
        if len(encoded) >= 4:
            prompt_embeds = encoded[0]
            negative_prompt_embeds = encoded[1]
            pooled_prompt_embeds = encoded[2]
            negative_pooled_prompt_embeds = encoded[3]

            if do_cfg:
                prompt_embeds = _align_and_cat(negative_prompt_embeds, prompt_embeds)
                if (
                    negative_pooled_prompt_embeds is not None
                    and pooled_prompt_embeds is not None
                    and negative_pooled_prompt_embeds.ndim == pooled_prompt_embeds.ndim
                    and negative_pooled_prompt_embeds.shape[-1] == pooled_prompt_embeds.shape[-1]
                ):
                    pooled_prompt_embeds = _align_and_cat(
                        negative_pooled_prompt_embeds, pooled_prompt_embeds
                    )
            return prompt_embeds, pooled_prompt_embeds

        if len(encoded) == 3:
            # 对 FLUX：通常第2个是 pooled，不是 negative
            prompt_embeds = encoded[0]
            pooled_prompt_embeds = encoded[1]
            return prompt_embeds, pooled_prompt_embeds

        if len(encoded) == 2:
            a, b = encoded[0], encoded[1]
            # 只有在 a/b 可安全视作 neg/pos 时才做 CFG 拼接
            if do_cfg and isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                if a.ndim == b.ndim and a.shape[-1] == b.shape[-1]:
                    # 这里按 (pos, neg) 更稳妥地判断：通常 pos batch=1, neg batch=1
                    # 不强行假设顺序，若失败就降级为不拼接
                    try:
                        prompt_embeds = _align_and_cat(b, a)
                        return prompt_embeds, None
                    except Exception:
                        pass
            # 不能当 neg/pos 用时，把第二个当 pooled
            return a, b

        return encoded[0], None

    if hasattr(pipe, "_encode_prompt"):
        prompt_embeds = pipe._encode_prompt(
            prompt,
            device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt="",
        )
        return prompt_embeds, None

    if hasattr(pipe, "text_encoder") and hasattr(pipe, "tokenizer"):
        tokens = pipe.tokenizer(
            prompt, return_tensors="pt", padding="max_length", truncation=True
        ).to(device)
        embeds = pipe.text_encoder(**tokens).last_hidden_state
        if do_cfg:
            neg = torch.zeros_like(embeds)
            embeds = torch.cat([neg, embeds], dim=0)
        return embeds, None

    raise RuntimeError("Unable to obtain prompt embeddings from pipeline")

def _pack_flux_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack Flux 4D latents into the sequence layout expected by transformer."""

    # Shape: (B, C, H, W) -> (B, H/2 * W/2, C*4).
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), c * 4)
    return latents


def _unpack_flux_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int) -> torch.Tensor:
    """Restore Flux packed sequence latents to image-like 4D latent maps."""

    # Shape: (B, H/2*W/2, C*4) -> (B, C, H, W).
    b, hw, c4 = latents.shape
    h = int(height // vae_scale_factor)
    w = int(width // vae_scale_factor)
    h2 = h // 2
    w2 = w // 2
    c = c4 // 4
    latents = latents.view(b, h2, w2, c, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)
    return latents


def _extract_flux_prompt_embeds(encoded):
    """Pick token-level and pooled tensors from Flux `encode_prompt` output."""

    if not isinstance(encoded, tuple):
        return encoded, None

    tensors = [x for x in encoded if isinstance(x, torch.Tensor)]
    if not tensors:
        raise RuntimeError("Flux encode_prompt() returned no tensor outputs")

    # token-level embeds 通常是 3D
    prompt_embeds = next((t for t in tensors if t.ndim == 3), tensors[0])
    # pooled 通常是 2D
    pooled_prompt_embeds = next((t for t in tensors if t.ndim == 2), None)
    return prompt_embeds, pooled_prompt_embeds


def generate_with_step_trace_flux(
    pipe,
    prompt: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    export_steps: List[int],
) -> SampleTrace:
    """Run a Flux pipeline manually and capture selected denoising steps."""

    device = str(pipe.device)

    with torch.no_grad():
        # Flux 不走你之前那套手工 CFG 拼接
        try:
            encoded = pipe.encode_prompt(prompt=prompt, max_sequence_length=256)
        except TypeError:
            encoded = pipe.encode_prompt(prompt=prompt)
        prompt_embeds, pooled_prompt_embeds = _extract_flux_prompt_embeds(encoded)

        target_height = 768
        target_width = 768
        vae_scale = int(getattr(pipe, "vae_scale_factor", 8))

        latent_h = target_height // vae_scale
        latent_w = target_width // vae_scale
        num_channels_latents = int(getattr(pipe.vae.config, "latent_channels", 16))

        gen = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(
            (1, num_channels_latents, latent_h, latent_w),
            generator=gen,
            device=device,
            dtype=pipe.transformer.dtype,
        )

        scheduler = pipe.scheduler
        use_dynamic_shifting = bool(getattr(getattr(scheduler, "config", None), "use_dynamic_shifting", False))
        if use_dynamic_shifting:
            try:
                from diffusers.pipelines.flux.pipeline_flux import calculate_shift

                image_seq_len = (latent_h // 2) * (latent_w // 2)
                cfg = scheduler.config
                mu = float(
                    calculate_shift(
                        image_seq_len,
                        int(getattr(cfg, "base_image_seq_len", 256)),
                        int(getattr(cfg, "max_image_seq_len", 4096)),
                        float(getattr(cfg, "base_shift", 0.5)),
                        float(getattr(cfg, "max_shift", 1.15)),
                    )
                )
            except Exception:
                mu = float(getattr(getattr(scheduler, "config", None), "base_shift", 1.0))
            scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device)

        latents = latents * getattr(scheduler, "init_noise_sigma", 1.0)

        if hasattr(pipe, "_pack_latents"):
            latents = pipe._pack_latents(latents, 1, num_channels_latents, latent_h, latent_w)
        else:
            latents = _pack_flux_latents(latents)

        traces: Dict[int, Dict[str, torch.Tensor]] = {}
        export_set = set(export_steps)

        # text token ids for Flux rotary embedding path
        # prompt_embeds: [B, T, D]
        txt_seq_len = int(prompt_embeds.shape[1])
        txt_ids = torch.zeros((txt_seq_len, 3), device=device, dtype=latents.dtype)

        # image token ids for Flux rotary embedding path
        if hasattr(pipe, "_prepare_latent_image_ids"):
            # packed latent grid is (latent_h//2, latent_w//2)
            img_ids = pipe._prepare_latent_image_ids(
                batch_size=1,
                height=latent_h // 2,
                width=latent_w // 2,
                device=device,
                dtype=latents.dtype,
            )
        else:
            # fallback: simple zero ids with correct sequence length
            img_seq_len = int(latents.shape[1])  # packed sequence length
            img_ids = torch.zeros((img_seq_len, 3), device=device, dtype=latents.dtype)

        for step_idx, t in enumerate(scheduler.timesteps):
            latent_model_input = latents
            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            t_tensor = torch.as_tensor(t, device=device, dtype=latent_model_input.dtype).reshape(1)
            timestep_input = t_tensor.repeat(latent_model_input.shape[0]) / 1000.0

            guidance = None
            if bool(getattr(pipe.transformer.config, "guidance_embeds", False)):
                guidance = torch.full(
                    (latent_model_input.shape[0],),
                    float(guidance_scale),
                    device=device,
                    dtype=latent_model_input.dtype,
                )

            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep_input,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]

            if step_idx in export_set:
                # 存储时转回 4D latent，便于你后续统计逻辑统一
                if latents.ndim == 3:
                    if hasattr(pipe, "_unpack_latents"):
                        z_t = pipe._unpack_latents(latents, target_height, target_width, vae_scale)
                    else:
                        z_t = _unpack_flux_latents(latents, target_height, target_width, vae_scale)
                else:
                    z_t = latents
                traces[step_idx] = {
                    "z_t": z_t.detach().float().cpu(),
                    "eps_hat": noise_pred.detach().float().cpu(),
                }

            step_out = scheduler.step(noise_pred, t, latents, return_dict=False)
            latents = step_out[0] if isinstance(step_out, tuple) else step_out

        if latents.ndim == 3:
            if hasattr(pipe, "_unpack_latents"):
                latents = pipe._unpack_latents(latents, target_height, target_width, vae_scale)
            else:
                latents = _unpack_flux_latents(latents, target_height, target_width, vae_scale)

        vae_config = pipe.vae.config
        scaling_factor = float(getattr(vae_config, "scaling_factor", 1.0))
        shift_factor = float(getattr(vae_config, "shift_factor", 0.0))
        latents_for_decode = (latents / scaling_factor) + shift_factor

        decoded = pipe.vae.decode(latents_for_decode, return_dict=False)
        image_tensor = decoded[0] if isinstance(decoded, (tuple, list)) else decoded

        if hasattr(pipe, "image_processor") and hasattr(pipe.image_processor, "postprocess"):
            image = pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]
        else:
            import torchvision.transforms.functional as TF
            img = image_tensor.detach().cpu()[0]
            if img.min() < 0:
                img = (img + 1.0) / 2.0
            image = TF.to_pil_image(img.clamp(0, 1))

    return SampleTrace(image=image, traces=traces)


def generate_with_step_trace_sd(
    pipe,
    prompt: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    export_steps: List[int],
) -> SampleTrace:
    """Run an SD/SD3 pipeline manually and capture selected denoising steps."""

    device = str(pipe.device)
    do_cfg = guidance_scale > 1.0
    denoiser = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
    if denoiser is None:
        raise RuntimeError("Pipeline does not expose transformer or unet")
    is_sd3 = getattr(pipe, "_is_sd3", False) or hasattr(pipe, "transformer")

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = _get_prompt_embeds(pipe, prompt, device, do_cfg)

        target_height = 768
        target_width = 768
        vae_scale = getattr(pipe, "vae_scale_factor", 8)
        latent_h = int(target_height // int(vae_scale))
        latent_w = int(target_width // int(vae_scale))

        gen = torch.Generator(device=device).manual_seed(seed)
        in_channels = int(getattr(denoiser.config, "in_channels", 4))
        latents = torch.randn(
            (1, in_channels, latent_h, latent_w),
            generator=gen,
            device=device,
            dtype=denoiser.dtype,
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
            image = pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]
        else:
            import torchvision.transforms.functional as TF
            img = image_tensor.detach().cpu()[0]
            if img.min() < 0:
                img = (img + 1.0) / 2.0
            image = TF.to_pil_image(img.clamp(0, 1))

    return SampleTrace(image=image, traces=traces)


def generate_with_step_trace(
    pipe,
    prompt: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    export_steps: List[int],
) -> SampleTrace:
    """Dispatch to the Flux or SD trace generator based on pipeline class."""

    is_flux = pipe.__class__.__name__.lower().startswith("flux")
    if is_flux:
        return generate_with_step_trace_flux(
            pipe, prompt, seed, num_inference_steps, guidance_scale, export_steps
        )
    return generate_with_step_trace_sd(
        pipe, prompt, seed, num_inference_steps, guidance_scale, export_steps
    )


def save_trace_files(
    traces: Dict[int, Dict[str, torch.Tensor]],
    base_dir: Path,
    prompt_idx: int,
    model_tag: str,
    group_tag: str,
    sample_idx: int,
) -> Dict[int, str]:
    """Persist each exported step as an individual `.pt` trace file."""

    out: Dict[int, str] = {}
    for step, payload in traces.items():
        step_dir = base_dir / "traces" / f"prompt_{prompt_idx}" / group_tag / model_tag / f"sample_{sample_idx}"
        ensure_dir(step_dir)
        file_path = step_dir / f"step_{step}.pt"
        torch.save(payload, file_path)
        out[step] = str(file_path)
    return out


def save_image(img: Image.Image, out_path: Path) -> str:
    ensure_dir(out_path.parent)
    img.save(out_path)
    return str(out_path)


def compute_adjacent_z_metrics(
    traces: List[Dict[int, Dict[str, torch.Tensor]]],
    export_steps: List[int],
) -> Dict[int, Dict[str, float]]:
    """Compute adjacent-step statistics for captured latent `z_t` tensors.

    `mean_angle_deg` is kept as a legacy CSV key, but the stored value is the
    cosine similarity between consecutive exported latent vectors.
    """

    metrics: Dict[int, Dict[str, float]] = {}
    if len(export_steps) < 2:
        return metrics

    eps = 1e-12

    for idx in range(1, len(export_steps)):
        prev_step = export_steps[idx - 1]
        curr_step = export_steps[idx]
        angle_list: List[float] = []
        prev_norm_list: List[float] = []
        curr_norm_list: List[float] = []
        delta_norm_list: List[float] = []

        for sample_idx, sample_traces in enumerate(traces):
            if prev_step not in sample_traces or curr_step not in sample_traces:
                continue

            prev_z = sample_traces[prev_step]["z_t"].detach().float().reshape(-1)
            curr_z = sample_traces[curr_step]["z_t"].detach().float().reshape(-1)

            if prev_z.numel() != curr_z.numel():
                warnings.warn(
                    f"Skip step pair {prev_step}->{curr_step} at sample {sample_idx}: "
                    f"vector length mismatch {prev_z.numel()} vs {curr_z.numel()}",
                    RuntimeWarning,
                )
                continue

            prev_norm = float(torch.linalg.norm(prev_z).item())
            curr_norm = float(torch.linalg.norm(curr_z).item())
            if prev_norm < eps or curr_norm < eps:
                continue

            cosine = float(torch.dot(prev_z, curr_z).item() / (prev_norm * curr_norm))
            delta_norm = float(torch.linalg.norm(curr_z - prev_z).item())

            angle_list.append(cosine)
            prev_norm_list.append(prev_norm)
            curr_norm_list.append(curr_norm)
            delta_norm_list.append(delta_norm)

        if angle_list:
            metrics[curr_step] = {
                "from_step": float(prev_step),
                "to_step": float(curr_step),
                "mean_angle_deg": float(np.mean(angle_list)),
                "mean_prev_norm": float(np.mean(prev_norm_list)),
                "mean_curr_norm": float(np.mean(curr_norm_list)),
                "mean_delta_norm": float(np.mean(delta_norm_list)),
                "num_pairs": float(len(angle_list)),
            }
        else:
            metrics[curr_step] = {
                "from_step": float(prev_step),
                "to_step": float(curr_step),
                "mean_angle_deg": float("nan"),
                "mean_prev_norm": float("nan"),
                "mean_curr_norm": float("nan"),
                "mean_delta_norm": float("nan"),
                "num_pairs": 0.0,
            }

    return metrics


def _project_step_metrics(
    metrics: Dict[int, Dict[str, float]],
    metric_keys: List[str],
) -> Dict[int, Dict[str, float]]:
    """Keep only the metric fields needed by the current output CSV."""

    projected: Dict[int, Dict[str, float]] = {}
    for step, payload in metrics.items():
        row = {
            "from_step": payload.get("from_step", float("nan")),
            "to_step": payload.get("to_step", float("nan")),
            "num_pairs": payload.get("num_pairs", 0.0),
        }
        for key in metric_keys:
            row[key] = payload.get(key, float("nan"))
        projected[step] = row
    return projected

def _load_done_prompt_indices(csv_path: Path) -> set[int]:
    """Read completed prompt indices so interrupted runs can resume safely."""

    done = set()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return done
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = row.get("prompt_index", "")
            if str(v).isdigit():
                done.add(int(v))
    return done
def load_trace_files(
    base_dir: Path,
    prompt_idx: int,
    model_tag: str,
    group_tag: str,
    sample_idx: int,
    export_steps: List[int],
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Load all requested step trace files for one prompt/model/sample."""

    traces: Dict[int, Dict[str, torch.Tensor]] = {}
    sample_dir = base_dir / "traces" / f"prompt_{prompt_idx}" / group_tag / model_tag / f"sample_{sample_idx}"
    for step in export_steps:
        file_path = sample_dir / f"step_{step}.pt"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing trace file: {file_path}")
        traces[step] = torch.load(file_path, map_location="cpu")
    return traces

def load_sample_trace_from_disk(
    base_dir: Path,
    prompt_idx: int,
    model_tag: str,
    group_tag: str,
    sample_idx: int,
    export_steps: List[int],
) -> SampleTrace:
    """Reconstruct a `SampleTrace` from previously saved image and trace files."""

    traces = load_trace_files(
        base_dir=base_dir,
        prompt_idx=prompt_idx,
        model_tag=model_tag,
        group_tag=group_tag,
        sample_idx=sample_idx,
        export_steps=export_steps,
    )
    img_path = base_dir / "images" / f"{model_tag}_{group_tag}" / f"prompt_{prompt_idx}_img_{sample_idx + 1}.png"
    image = Image.open(img_path).convert("RGB")
    return SampleTrace(image=image, traces=traces)

def _build_prompt_subset(prompts: List[str]) -> List[str]:
    """Use the same stratified 60-prompt subset as the original experiment."""

    picks = [
        (0, 10),
        (100, 110),
        (200, 210),
        (300, 310),
        (400, 410),
        (500, 510),

    ]
    """ picks = [
        (0, 60),
    ] """
    out: List[str] = []
    for s, e in picks:
        if s < len(prompts):
            out.extend(prompts[s:min(e, len(prompts))])
    return out if out else prompts

def run(args):
    """Main experiment loop: prepare prompts, generate traces, and write CSVs."""

    prompts_all = load_prompts(Path(args.input_txt))
    prompts = _build_prompt_subset(prompts_all)
    export_steps = _parse_steps(args.export_steps, args.num_inference_steps)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # Split large JSON metric payloads into separate CSVs for easier analysis.
    result_specs = {
        "angle": {
            "path": Path(args.output_csv_angle),
            "metric_keys": ["mean_angle_deg"],
        },
        "norm": {
            "path": Path(args.output_csv_norm),
            "metric_keys": ["mean_prev_norm", "mean_curr_norm"],
        },
        "delta_norm": {
            "path": Path(args.output_csv_delta_norm),
            "metric_keys": ["mean_delta_norm"],
        },
    }

    for spec in result_specs.values():
        ensure_dir(spec["path"].parent)

    # Resume support: skip prompt rows already present in every requested CSV.
    done_by_metric = {name: _load_done_prompt_indices(spec["path"]) for name, spec in result_specs.items()}
    file_exists_by_metric = {
        name: spec["path"].exists() and spec["path"].stat().st_size > 0
        for name, spec in result_specs.items()
    }
    mode_by_metric = {name: ("a" if exists else "w") for name, exists in file_exists_by_metric.items()}

    fieldnames = [
        "prompt_index",
        "original_prompt",
        "clean_orig",
        "backdoor_orig",
        #"test_orig",
    ]

    def _trace_file_map(model_tag: str, group_tag: str, prompt_idx: int, sample_idx: int) -> Dict[int, str]:
        return {
            step: str(
                output_dir
                / "traces"
                / f"prompt_{prompt_idx}"
                / group_tag
                / model_tag
                / f"sample_{sample_idx}"
                / f"step_{step}.pt"
            )
            for step in export_steps
        }

    def _collect_outputs(
        samples: List[SampleTrace],
        model_tag: str,
        group_tag: str,
        prompt_idx: int,
        persist: bool,
    ):
        paths: List[str] = []
        trace_files: Dict[int, Dict[int, str]] = {}

        for j, sample in enumerate(samples):
            img_path = output_dir / "images" / f"{model_tag}_{group_tag}" / f"prompt_{prompt_idx}_img_{j + 1}.png"
            if persist:
                paths.append(save_image(sample.image, img_path))
                trace_files[j] = save_trace_files(sample.traces, output_dir, prompt_idx, model_tag, group_tag, j)
            else:
                paths.append(str(img_path))
                trace_files[j] = _trace_file_map(model_tag, group_tag, prompt_idx, j)

        return paths, trace_files

    def _write_metric_row(
        writer: csv.DictWriter,
        prompt_index: int,
        prompt: str,
        clean_orig_metrics: Dict[int, Dict[str, float]],
        backdoor_orig_metrics: Dict[int, Dict[str, float]],
        #test_orig_metrics: Dict[int, Dict[str, float]],
        metric_keys: List[str],
    ) -> None:
        writer.writerow(
            {
                "prompt_index": prompt_index,
                "original_prompt": prompt,
                "clean_orig": json.dumps(_project_step_metrics(clean_orig_metrics, metric_keys), ensure_ascii=False),
                "backdoor_orig": json.dumps(_project_step_metrics(backdoor_orig_metrics, metric_keys), ensure_ascii=False),
                #"test_orig": json.dumps(_project_step_metrics(test_orig_metrics, metric_keys), ensure_ascii=False),
            }
        )

    if args.dry_run:
        # Dry run validates prompt selection and output schemas without loading models.
        with (
            result_specs["angle"]["path"].open(mode_by_metric["angle"], newline="", encoding="utf-8") as f_angle,
            result_specs["norm"]["path"].open(mode_by_metric["norm"], newline="", encoding="utf-8") as f_norm,
            result_specs["delta_norm"]["path"].open(mode_by_metric["delta_norm"], newline="", encoding="utf-8") as f_delta,
        ):
            writer_angle = csv.DictWriter(f_angle, fieldnames=fieldnames)
            writer_norm = csv.DictWriter(f_norm, fieldnames=fieldnames)
            writer_delta = csv.DictWriter(f_delta, fieldnames=fieldnames)

            if not file_exists_by_metric["angle"]:
                writer_angle.writeheader()
            if not file_exists_by_metric["norm"]:
                writer_norm.writeheader()
            if not file_exists_by_metric["delta_norm"]:
                writer_delta.writeheader()

            for i, prompt in enumerate(prompts):
                if i in done_by_metric["angle"] and i in done_by_metric["norm"] and i in done_by_metric["delta_norm"]:
                    continue

                empty_metrics = {}
                empty_row = {
                    "prompt_index": i,
                    "original_prompt": prompt,
                    "clean_orig": json.dumps(empty_metrics, ensure_ascii=False),
                    "backdoor_orig": json.dumps(empty_metrics, ensure_ascii=False),
                    #"test_orig": json.dumps(empty_metrics, ensure_ascii=False),
                }

                writer_angle.writerow(empty_row)
                writer_norm.writerow(empty_row)
                writer_delta.writerow(empty_row)

                f_angle.flush()
                f_norm.flush()
                f_delta.flush()

        print(
            f"Dry run done. Results at {result_specs['angle']['path']}, {result_specs['norm']['path']} and {result_specs['delta_norm']['path']}"
        )
        return

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    hf_token = os.environ.get("HF_TOKEN", None)

    # Models are loaded one at a time, but prompts are exhausted for the current
    # model before switching to the next one.

    with (
        result_specs["angle"]["path"].open(mode_by_metric["angle"], newline="", encoding="utf-8") as f_angle,
        result_specs["norm"]["path"].open(mode_by_metric["norm"], newline="", encoding="utf-8") as f_norm,
        result_specs["delta_norm"]["path"].open(mode_by_metric["delta_norm"], newline="", encoding="utf-8") as f_delta,
    ):
        writer_angle = csv.DictWriter(f_angle, fieldnames=fieldnames)
        writer_norm = csv.DictWriter(f_norm, fieldnames=fieldnames)
        writer_delta = csv.DictWriter(f_delta, fieldnames=fieldnames)

        if not file_exists_by_metric["angle"]:
            writer_angle.writeheader()
        if not file_exists_by_metric["norm"]:
            writer_norm.writeheader()
        if not file_exists_by_metric["delta_norm"]:
            writer_delta.writeheader()

        prompt_records = []
        for i, prompt in enumerate(prompts):
            need_angle = i not in done_by_metric["angle"]
            need_norm = i not in done_by_metric["norm"]
            need_delta = i not in done_by_metric["delta_norm"]
            if not (need_angle or need_norm or need_delta):
                print(f"[skip] prompt_index={i} already exists in all result files")
                continue

            prompt_records.append(
                {
                    "prompt_index": i,
                    "prompt": prompt,
                    "need_angle": need_angle,
                    "need_norm": need_norm,
                    "need_delta": need_delta,
                }
            )

        persist_outputs = not args.read_existing

        def _load_orig_samples(prompt_idx: int, model_tag: str) -> List[SampleTrace]:
            return [
                load_sample_trace_from_disk(
                    base_dir=output_dir,
                    prompt_idx=prompt_idx,
                    model_tag=model_tag,
                    group_tag="orig",
                    sample_idx=j,
                    export_steps=export_steps,
                )
                for j in range(args.num_images_orig)
            ]

        def _write_all_metric_rows(
            record: Dict[str, object],
            clean_orig_metrics: Dict[int, Dict[str, float]],
            backdoor_orig_metrics: Dict[int, Dict[str, float]],
            test_orig_metrics: Dict[int, Dict[str, float]],
        ) -> None:
            prompt_index = int(record["prompt_index"])
            prompt = str(record["prompt"])

            if record["need_angle"]:
                _write_metric_row(
                    writer_angle,
                    prompt_index=prompt_index,
                    prompt=prompt,
                    clean_orig_metrics=clean_orig_metrics,
                    backdoor_orig_metrics=backdoor_orig_metrics,
                    #test_orig_metrics=test_orig_metrics,
                    metric_keys=result_specs["angle"]["metric_keys"],
                )
                f_angle.flush()

            if record["need_norm"]:
                _write_metric_row(
                    writer_norm,
                    prompt_index=prompt_index,
                    prompt=prompt,
                    clean_orig_metrics=clean_orig_metrics,
                    backdoor_orig_metrics=backdoor_orig_metrics,
                    #test_orig_metrics=test_orig_metrics,
                    metric_keys=result_specs["norm"]["metric_keys"],
                )
                f_norm.flush()

            if record["need_delta"]:
                _write_metric_row(
                    writer_delta,
                    prompt_index=prompt_index,
                    prompt=prompt,
                    clean_orig_metrics=clean_orig_metrics,
                    backdoor_orig_metrics=backdoor_orig_metrics,
                    #test_orig_metrics=test_orig_metrics,
                    metric_keys=result_specs["delta_norm"]["metric_keys"],
                )
                f_delta.flush()

        if args.read_existing:
            # Recompute CSV metrics from traces already saved on disk.
            for record in prompt_records:
                i = record["prompt_index"]
                clean_orig_samples = _load_orig_samples(i, "clean")
                backdoor_orig_samples = _load_orig_samples(i, "backdoor")
                #test_orig_samples = _load_orig_samples(i, "test")

                clean_orig_metrics = compute_adjacent_z_metrics([x.traces for x in clean_orig_samples], export_steps)
                backdoor_orig_metrics = compute_adjacent_z_metrics([x.traces for x in backdoor_orig_samples], export_steps)
                #test_orig_metrics = compute_adjacent_z_metrics([x.traces for x in test_orig_samples], export_steps)

                _write_all_metric_rows(record, clean_orig_metrics, backdoor_orig_metrics, None)

                print(f"[done] prompt_index={i}")
        else:
            model_map = [
                ("clean", args.clean_model_id),
                ("backdoor", args.backdoor_model_id),
                #("test", args.test_model_id),
            ]

            for model_tag, model_id in model_map:
                # Keep only one model in GPU memory at a time.
                lora_id = None if model_tag == "clean" else model_id
                base_model_id = model_id if lora_id is None else args.clean_model_id
                #pipe = get_sd35_pipeline(base_model_id, device, hf_token, lora_id=lora_id)
                pipe = get_sd2_pipeline(model_id, device, hf_token)

                # To switch experiments manually:
                # SD2 full model:   pipe = get_sd2_pipeline(model_id, device, hf_token)
                # FLUX base + LoRA: pipe = get_flux_pipeline(args.clean_model_id, device, hf_token, lora_id=model_id)

                for record in prompt_records:
                    i = int(record["prompt_index"])
                    prompt = str(record["prompt"])
                    orig_seed_base = args.seed + i

                    orig_samples = []
                    for j in range(args.num_images_orig):
                        orig_samples.append(
                            generate_with_step_trace(
                                pipe,
                                prompt,
                                seed=orig_seed_base + j,
                                num_inference_steps=args.num_inference_steps,
                                guidance_scale=args.guidance_scale,
                                export_steps=export_steps,
                            )
                        )

                    _collect_outputs(orig_samples, model_tag, "orig", i, persist_outputs)

                try:
                    del pipe
                except Exception:
                    pass
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

            for record in prompt_records:
                i = int(record["prompt_index"])
                clean_orig_samples = _load_orig_samples(i, "clean")
                backdoor_orig_samples = _load_orig_samples(i, "backdoor")
                #test_orig_samples = _load_orig_samples(i, "test")

                clean_orig_metrics = compute_adjacent_z_metrics([x.traces for x in clean_orig_samples], export_steps)
                backdoor_orig_metrics = compute_adjacent_z_metrics([x.traces for x in backdoor_orig_samples], export_steps)
                #test_orig_metrics = compute_adjacent_z_metrics([x.traces for x in test_orig_samples], export_steps)

                _write_all_metric_rows(record, clean_orig_metrics, backdoor_orig_metrics, None)
                print(f"[done] prompt_index={i}")

    print(
        f"Done. Results at {result_specs['angle']['path']}, {result_specs['norm']['path']} and {result_specs['delta_norm']['path']}"
    )

def _legacy_parse_args():
    parser = argparse.ArgumentParser(
        description="Generate samples for clean, backdoor, and test models, then compute adjacent-z statistics."
    )
    parser.add_argument("--input_txt", type=str, required=True, help="Input prompts txt")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base folder for images and traces",
    )
    parser.add_argument(
        "--output_csv_angle",
        type=str,
        required=True,
        help="CSV output for adjacent-z angles",
    )
    parser.add_argument(
        "--output_csv_norm",
        type=str,
        required=True,
        help="CSV output for adjacent-z norms",
    )
    parser.add_argument(
        "--output_csv_delta_norm",
        type=str,
        required=True,
        help="CSV output for adjacent-z difference norms",
    )
    parser.add_argument(
        "--read_existing",
        action="store_true",
        help="Read existing images/traces from output_dir and only compute metrics, skip generation",
    )

    parser.add_argument(
        "--clean_model_id",
        type=str,
        required=True,
        help="Clean model path or HF repo id",
    )
    parser.add_argument(
        "--backdoor_model_id",
        type=str,
        required=True,
        help="Backdoor model path or HF repo id",
    )
    parser.add_argument(
        "--test_model_id",
        type=str,
        required=True,
        help="Test model path or HF repo id",
    )

    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cuda:N / cpu")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="DDIM/PNDM sampling steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG guidance scale")
    parser.add_argument(
        "--export_steps",
        type=str,
        default="0, 10, 20, 30, 40, 49",
        help="Comma-separated zero-based step indices to export",
    )
    parser.add_argument("--num_images_orig", type=int, default=1, help="Images per original prompt per model")
    parser.add_argument("--dry_run", action="store_true", help="Only write prompt rows, skip generation")
    return parser.parse_args()



if __name__ == "__main__":
    raise SystemExit("This is a library module; use classify/generate_features.py")

#swd/swdkl 修改后

