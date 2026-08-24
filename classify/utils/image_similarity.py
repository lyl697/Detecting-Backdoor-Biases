"""Utilities for perturbation-image feature extraction.

The exported CSV rows are keyed by (prompt_index, model_tag), so they can be
merged with hidden/trace features in the fusion detector.
"""

import csv
import gc
import math
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    from skimage.feature import hog as sk_hog
except Exception:  # pragma: no cover - optional dependency
    sk_hog = None


PERTURB_METRICS = ["clip_sim", "lpips", "mse", "ssim", "hog_pearson", "hog_dtw"]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_prompts(txt_path: Path) -> List[str]:
    with txt_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _dtype_for_device(device: str):
    return torch.float16 if device.startswith("cuda") else torch.float32


def _finish_pipeline_setup(pipe, device: str, model_family: str):
    pipe = pipe.to(device)
    pipe._lfd_model_family = model_family
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    return pipe


def get_sd2_pipeline(model_id: str, device: str, hf_token: str | None = None):
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=_dtype_for_device(device),
        token=hf_token if hf_token else None,
    )
    return _finish_pipeline_setup(pipe, device, "sd2")


def get_sd35_pipeline(
    model_id: str,
    device: str,
    hf_token: str | None = None,
    lora_id: str | None = None,
):
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=_dtype_for_device(device),
        token=hf_token if hf_token else None,
    )
    if lora_id:
        pipe.load_lora_weights(lora_id)
    return _finish_pipeline_setup(pipe, device, "sd35")


def get_flux_pipeline(
    model_id: str,
    device: str,
    hf_token: str | None = None,
    lora_id: str | None = None,
):
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
    """Manual switch point for SD2, SD3.5, and FLUX.1."""

    hf_token = os.environ.get("HF_TOKEN")
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


def generate_images(pipe, prompts: List[str], seed: int, num_steps: int, guidance_scale: float, height: int, width: int):
    images = []
    for idx, prompt in enumerate(prompts):
        generator = torch.Generator(device=pipe.device)
        generator.manual_seed(seed + idx)
        out = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )
        images.append(out.images[0])
    return images


def parse_steps(step_text: str) -> List[int]:
    steps = sorted({int(x.strip()) for x in step_text.split(",") if x.strip()})
    if not steps:
        raise ValueError("No valid export steps were provided")
    return steps


def _pipeline_device(pipe):
    return getattr(pipe, "device", getattr(pipe, "_execution_device", "cpu"))


def _decode_latents_to_image(pipe, latents: torch.Tensor, height: int, width: int) -> Image.Image:
    latents = latents.detach().to(_pipeline_device(pipe))

    # FLUX stores latents in a packed sequence layout. Diffusers exposes an
    # unpack helper on FluxPipeline in recent versions.
    if latents.dim() == 3 and hasattr(pipe, "_unpack_latents"):
        latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)

    latents = latents[:1].to(dtype=getattr(pipe.vae, "dtype", latents.dtype))
    scaling_factor = getattr(pipe.vae.config, "scaling_factor", 1.0)
    shift_factor = getattr(pipe.vae.config, "shift_factor", None)
    if shift_factor is not None:
        latents = (latents / scaling_factor) + shift_factor
    else:
        latents = latents / scaling_factor

    with torch.no_grad():
        image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def generate_step_images(
    pipe,
    prompts: List[str],
    seed: int,
    num_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    export_steps: List[int],
) -> List[Dict[int, Image.Image]]:
    """Generate and decode intermediate images at selected denoising steps."""

    if max(export_steps) >= num_steps:
        raise ValueError(f"export step {max(export_steps)} must be < num_steps={num_steps}")

    outputs: List[Dict[int, Image.Image]] = []
    export_set = set(export_steps)
    for idx, prompt in enumerate(prompts):
        generator = torch.Generator(device=_pipeline_device(pipe)).manual_seed(seed + idx)
        captured: Dict[int, torch.Tensor] = {}

        def on_step_end(pipeline, step_idx, timestep, callback_kwargs):
            latents = callback_kwargs.get("latents")
            if step_idx in export_set and latents is not None:
                captured[step_idx] = latents.detach().cpu()
            return callback_kwargs

        pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            callback_on_step_end=on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        missing = sorted(export_set - set(captured))
        if missing:
            raise RuntimeError(f"Missing intermediate latents for steps {missing}")
        outputs.append(
            {
                step: _decode_latents_to_image(pipe, captured[step], height, width)
                for step in export_steps
            }
        )
    return outputs


def save_images(images: List[Image.Image], base_dir: Path, prompt_idx: int, tag: str) -> List[str]:
    ensure_dir(base_dir)
    paths = []
    for idx, image in enumerate(images):
        path = base_dir / f"prompt_{prompt_idx}_{tag}_{idx + 1}.png"
        image.save(path)
        paths.append(str(path))
    return paths


def save_step_images(step_images: Dict[int, Image.Image], base_dir: Path, prompt_idx: int, tag: str) -> Dict[int, str]:
    ensure_dir(base_dir)
    paths = {}
    for step, image in step_images.items():
        path = base_dir / f"prompt_{prompt_idx}_{tag}_step_{step:03d}.png"
        image.save(path)
        paths[step] = str(path)
    return paths


def get_clip_image_encoder(
    device: str,
    model_name: str,
    pretrained: str,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
):
    """Build the paper's OpenCLIP encoder without changing its legacy defaults.

    The original method used ViT-L-14 without a pretrained tag, which creates
    random weights. That behavior is available only through the explicit
    ``NONE_RANDOM_INITIALIZATION`` sentinel; omission is never a silent
    fallback. ``revision`` is recorded and printed because OpenCLIP's public
    constructor selects weights through ``pretrained`` rather than a separate
    revision argument.
    """
    import open_clip

    if not model_name or not pretrained:
        raise ValueError("OpenCLIP model and pretrained selection must be explicit")
    todo_marker = "TODO" + "(USER)"
    if todo_marker in model_name or todo_marker in pretrained or (
        revision is not None and todo_marker in revision
    ):
        raise ValueError("Resolve the user-supplied OpenCLIP configuration before extraction")
    random_initialization = pretrained == "NONE_RANDOM_INITIALIZATION"
    kwargs = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if not random_initialization:
        kwargs["pretrained"] = pretrained
    print(
        "[OpenCLIP] "
        f"model={model_name} pretrained={pretrained} revision={revision or 'UNSPECIFIED'} "
        f"cache_dir={cache_dir or 'DEFAULT'}"
    )
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        **kwargs,
    )
    model = model.to(device)
    model.eval()
    return model, preprocess


def get_lpips_fn(device: str):
    import lpips

    return lpips.LPIPS(net="alex").to(device).eval()


def get_ssim_fn(device: str):
    from torchmetrics import StructuralSimilarityIndexMeasure

    return StructuralSimilarityIndexMeasure(data_range=1.0).to(device)


def tensorize_for_lpips(image: Image.Image, device: str):
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return to_tensor(image).unsqueeze(0).to(device)


def image_embeddings(model, preprocess, images: List[Image.Image], device: str) -> torch.Tensor:
    with torch.no_grad():
        batch = torch.stack([preprocess(image) for image in images]).to(device)
        feats = model.encode_image(batch)
        return feats / feats.norm(dim=-1, keepdim=True)


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    if n < 2:
        return float("nan")
    a = a[:n]
    b = b[:n]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    dp = np.full((len(a) + 1, len(b) + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i, ai in enumerate(a, start=1):
        for j, bj in enumerate(b, start=1):
            dp[i, j] = abs(ai - bj) + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[len(a), len(b)] / (len(a) + len(b)))


def hog_pearson_dtw(a_img: Image.Image, b_img: Image.Image, bins: int = 64):
    if cv2 is None or sk_hog is None:
        return float("nan"), float("nan")

    a = cv2.cvtColor(np.array(a_img.convert("RGB").resize((256, 256), Image.BICUBIC)), cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(np.array(b_img.convert("RGB").resize((256, 256), Image.BICUBIC)), cv2.COLOR_RGB2GRAY)
    fa = sk_hog(a, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    fb = sk_hog(b, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    hist_a, _ = np.histogram(fa, bins=bins, density=True)
    hist_b, _ = np.histogram(fb, bins=bins, density=True)
    return _pearson_corr(hist_a, hist_b), _dtw_distance(hist_a, hist_b)


def compute_pair_metrics(images_a: List[Image.Image], images_b: List[Image.Image], scorers: dict, device: str) -> Dict[str, float]:
    if len(images_a) != len(images_b):
        raise ValueError("Image lists must have the same length")
    if not images_a:
        return {name: float("nan") for name in PERTURB_METRICS}

    clip_model = scorers["clip_model"]
    clip_preprocess = scorers["clip_preprocess"]
    lpips_fn = scorers["lpips_fn"]
    ssim_fn = scorers["ssim_fn"]

    with torch.no_grad():
        feats_a = image_embeddings(clip_model, clip_preprocess, images_a, device)
        feats_b = image_embeddings(clip_model, clip_preprocess, images_b, device)
        clip_sim = float((feats_a * feats_b).sum(dim=1).mean().item())

    to_tensor = transforms.ToTensor()
    lpips_scores = []
    mse_scores = []
    ssim_scores = []
    hog_pearsons = []
    hog_dtws = []
    with torch.no_grad():
        for a_img, b_img in zip(images_a, images_b):
            lpips_scores.append(float(lpips_fn(tensorize_for_lpips(a_img, device), tensorize_for_lpips(b_img, device)).item()))
            a = to_tensor(a_img).unsqueeze(0).to(device)
            b = to_tensor(b_img).unsqueeze(0).to(device)
            mse_scores.append(float(F.mse_loss(a, b, reduction="mean").item()))
            ssim_scores.append(float(ssim_fn(a, b).item()))
            hog_p, hog_d = hog_pearson_dtw(a_img, b_img)
            hog_pearsons.append(hog_p)
            hog_dtws.append(hog_d)

    return {
        "clip_sim": clip_sim,
        "lpips": float(np.nanmean(lpips_scores)),
        "mse": float(np.nanmean(mse_scores)),
        "ssim": float(np.nanmean(ssim_scores)),
        "hog_pearson": float(np.nanmean(hog_pearsons)),
        "hog_dtw": float(np.nanmean(hog_dtws)),
    }


def build_scorers(
    device: str,
    openclip_model: str,
    openclip_pretrained: str,
    openclip_cache_dir: str | Path | None = None,
    openclip_revision: str | None = None,
):
    clip_model, clip_preprocess = get_clip_image_encoder(
        device,
        model_name=openclip_model,
        pretrained=openclip_pretrained,
        cache_dir=openclip_cache_dir,
        revision=openclip_revision,
    )
    return {
        "clip_model": clip_model,
        "clip_preprocess": clip_preprocess,
        "lpips_fn": get_lpips_fn(device),
        "ssim_fn": get_ssim_fn(device),
    }


def _load_done_prompt_indices(csv_path: Path) -> set[int]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return {int(row["prompt_index"]) for row in csv.DictReader(f) if row.get("prompt_index")}


def write_rows(output_csv: Path, fieldnames: List[str], rows: List[dict]):
    ensure_dir(output_csv.parent)
    file_exists = output_csv.exists() and output_csv.stat().st_size > 0
    with output_csv.open("a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
