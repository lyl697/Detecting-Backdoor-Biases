"""Utilities for cross-model denoising-trace feature extraction."""

import csv
import gc
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from dataclasses import dataclass
from PIL import Image


@dataclass
class SampleTrace:
    """A generated image plus the intermediate tensors captured during denoising."""

    image: Image.Image
    traces: Dict[int, Dict[str, torch.Tensor]]

TraceDict = Dict[int, Dict[str, torch.Tensor]]
MetricDict = Dict[int, Dict[str, float]]
MODEL_CROSS_METRICS = [
    "mean_cosine",
    #"mean_angle_deg",
    "mean_reference_norm",
    "mean_test_norm",
    "mean_delta_norm",
]

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

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_prompts(txt_path: Path) -> List[str]:
    with txt_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_steps(step_text: str, num_inference_steps: int) -> List[int]:
    steps = sorted({int(x.strip()) for x in step_text.split(",") if x.strip()})
    valid = [step for step in steps if 0 <= step < num_inference_steps]
    if not valid:
        raise ValueError("No valid export steps")
    return valid


def load_selected_pipeline(model_family: str, model_id: str, device: str, lora_id: str | None = None):
    hf_token = os.environ.get("HF_TOKEN")
    if model_family == "sd14":
        return get_sd2_pipeline(model_id, device, hf_token)
    if model_family == "sd2":
        return get_sd2_pipeline(model_id, device, hf_token)
    if model_family == "sd35":
        return get_sd35_pipeline(model_id, device, hf_token, lora_id=lora_id)
    if model_family == "flux":
        return get_flux_pipeline(model_id, device, hf_token, lora_id=lora_id)
    raise ValueError(f"Unknown model family: {model_family}")


def release_pipeline(pipe):
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_model_traces(
    args,
    prompt_indices: List[int],
    prompts: List[str],
    model_id: str,
    lora_id: str | None,
    model_tag: str,
    export_steps: List[int],
) -> Dict[int, List[TraceDict]]:
    pipe = load_selected_pipeline(args.model_family, model_id, args.device, lora_id=lora_id)
    traces_by_prompt: Dict[int, List[TraceDict]] = {}
    try:
        for prompt_idx, prompt in zip(prompt_indices, prompts):
            samples = []
            for sample_idx in range(args.num_images):
                sample = generate_with_step_trace(
                    pipe,
                    prompt,
                    seed=args.seed + prompt_idx,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    export_steps=export_steps,
                )
                samples.append(sample.traces)
                image_path = Path(args.output_dir) / "images" / model_tag / f"prompt_{prompt_idx}_img_{sample_idx + 1}.png"
                save_image(sample.image, image_path)
                save_trace_files(sample.traces, Path(args.output_dir), prompt_idx, model_tag, "orig", sample_idx)
            traces_by_prompt[prompt_idx] = samples
    finally:
        release_pipeline(pipe)
    return traces_by_prompt


def compute_cross_model_trace_metrics(
    reference_traces: List[TraceDict],
    test_traces: List[TraceDict],
    export_steps: List[int],
) -> MetricDict:
    metrics: MetricDict = {}
    eps = 1e-12
    n_pairs = min(len(reference_traces), len(test_traces))
    if n_pairs == 0:
        raise ValueError("No trace pairs are available for cross-model comparison")

    for step in export_steps:
        cosines = []
        angles = []
        reference_norms = []
        test_norms = []
        delta_norms = []

        for sample_idx in range(n_pairs):
            reference_sample = reference_traces[sample_idx]
            test_sample = test_traces[sample_idx]
            if step not in reference_sample or step not in test_sample:
                continue
            if "z_t" not in reference_sample[step] or "z_t" not in test_sample[step]:
                continue

            reference_value = reference_sample[step]["z_t"].detach().float().reshape(-1)
            test_value = test_sample[step]["z_t"].detach().float().reshape(-1)
            if reference_value.numel() != test_value.numel():
                continue

            reference_norm = float(torch.linalg.norm(reference_value).item())
            test_norm = float(torch.linalg.norm(test_value).item())
            if reference_norm < eps or test_norm < eps:
                continue

            cosine = float(torch.dot(reference_value, test_value).item() / (reference_norm * test_norm))
            cosine = float(np.clip(cosine, -1.0, 1.0))
            cosines.append(cosine)
            angles.append(float(np.degrees(np.arccos(cosine))))
            reference_norms.append(reference_norm)
            test_norms.append(test_norm)
            delta_norms.append(float(torch.linalg.norm(test_value - reference_value).item()))

        metrics[step] = {
            "step": float(step),
            "num_pairs": float(len(cosines)),
            "mean_cosine": float(np.mean(cosines)) if cosines else float("nan"),
            #"mean_angle_deg": float(np.mean(angles)) if angles else float("nan"),
            "mean_reference_norm": float(np.mean(reference_norms)) if reference_norms else float("nan"),
            "mean_test_norm": float(np.mean(test_norms)) if test_norms else float("nan"),
            "mean_delta_norm": float(np.mean(delta_norms)) if delta_norms else float("nan"),
        }
    return metrics


def flatten_metrics(prefix: str, metrics: MetricDict) -> Dict[str, float]:
    features = {}
    for step, payload in metrics.items():
        for metric_name in MODEL_CROSS_METRICS:
            features[f"modelcross_{prefix}_step_{step:03d}_{metric_name}"] = payload.get(metric_name, float("nan"))
    return features


def feature_fieldnames(export_steps: List[int], prefixes: List[str]) -> List[str]:
    fields = []
    for prefix in prefixes:
        for step in export_steps:
            for metric_name in MODEL_CROSS_METRICS:
                fields.append(f"modelcross_{prefix}_step_{step:03d}_{metric_name}")
    return fields


def write_rows(output_csv: Path, fieldnames: List[str], rows: List[dict]):
    ensure_dir(output_csv.parent)
    file_exists = output_csv.exists() and output_csv.stat().st_size > 0
    with output_csv.open("a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
