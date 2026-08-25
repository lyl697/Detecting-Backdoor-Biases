"""Generate paper reference-assisted training features."""

import argparse
import copy
import gc
import json
from pathlib import Path

import torch

from feature_utils import (
    build_prompt_subset,
    build_scorers,
    clean_reference_prompt_indices,
    completed_reference_prompt_indices,
    generate_light_outputs_batch,
    init_reference_rows,
    load_selected_pipeline,
    prepare_args,
    read_rows_by_prompt,
    release_pipeline,
    update_rows_with_reference,
    write_all_outputs,
)
from utils.config_paths import expand_runtime_variables


TARGET_SPECS = (
    ("backdoor1", "odd", "backdoor"),
    ("backdoor2", "even", "backdoor"),
    ("clean1", "odd", "clean"),
    ("clean2", "even", "clean"),
)


def _load_models(path):
    with Path(path).open("r", encoding="utf-8") as f:
        models = expand_runtime_variables(json.load(f))
    missing = {name for name, _, _ in TARGET_SPECS} - set(models)
    if missing:
        raise ValueError(f"models_json is missing: {', '.join(sorted(missing))}")
    return models


def _clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _iter_batches(items, batch_size):
    batch_size = max(1, int(batch_size))
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _load_shared_adapter_pipeline(args, models, clean_ref_id, backdoor_ref_id):
    model_ids = {str(clean_ref_id), str(backdoor_ref_id)} | {
        str(models[name]["model_id"]) for name, _, _ in TARGET_SPECS
    }
    if len(model_ids) != 1:
        raise ValueError(
            "Shared-adapter mode requires clean_ref, backdoor_ref, and all four "
            "targets to use the same base model_id"
        )
    pipe = load_selected_pipeline(
        args.model_family,
        next(iter(model_ids)),
        args.device,
        lora_id=None,
    )
    role_loras = {
        "clean_ref": args.clean_ref_lora_id,
        "backdoor_ref": args.backdoor_ref_lora_id,
        **{name: models[name].get("lora_id") for name, _, _ in TARGET_SPECS},
    }
    loaded = set()
    for role, lora_id in role_loras.items():
        if not lora_id:
            continue
        pipe.load_lora_weights(lora_id, adapter_name=role)
        loaded.add(role)
        print(f"[adapter] loaded {role}: {lora_id}")
    return pipe, loaded


def _activate_role(pipe, loaded_adapters, role):
    if loaded_adapters is None:
        return
    if role in loaded_adapters:
        if hasattr(pipe, "enable_lora"):
            pipe.enable_lora()
        pipe.set_adapters(role)
    elif hasattr(pipe, "disable_lora"):
        pipe.disable_lora()
    else:
        raise RuntimeError(f"Cannot activate base-model role without disable_lora(): {role}")


def _output_args(args, label):
    name = f"{label}_{args.dataset_suffix}"
    current = copy.copy(args)
    current.target_label = label
    current.image_output_csv = str(
        args.output_root / "image_similarity" / name / "results.csv"
    )
    current.image_output_dir = str(
        args.output_root / "image_similarity" / name / "images"
    )
    current.modelcross_output_csv = str(
        args.output_root / "reference_latent" / name / "results.csv"
    )
    current.navi_output_csv = str(
        args.output_root / "activation_difference" / name / "results.csv"
    )
    current.activation_difference_tensor_dir = str(
        args.output_root / "activation_difference" / name
    )
    return prepare_args(current)


def _initial_rows(args):
    return (
        read_rows_by_prompt(args.image_similarity_output_csv),
        read_rows_by_prompt(args.reference_latent_output_csv),
        read_rows_by_prompt(args.activation_difference_output_csv),
    )


def _compute_reference_rows(
    args,
    items,
    target_outputs,
    reference_pipe,
    reference_role,
    scorers,
    existing_rows,
    label_value,
    loaded_adapters=None,
):
    indices = [item[0] for item in items]
    prompts = [item[1] for item in items]
    _activate_role(reference_pipe, loaded_adapters, reference_role)
    reference_outputs = generate_light_outputs_batch(args, reference_pipe, indices, prompts)
    output_rows = [[], [], []]
    for prompt_index, prompt in items:
        rows = init_reference_rows(args, prompt_index, prompt, args.target_label, label_value)
        for row, existing in zip(rows, existing_rows):
            row.update(existing.get(prompt_index, {}))
        update_rows_with_reference(
            args,
            prompt_index,
            target_outputs[prompt_index],
            reference_outputs[prompt_index],
            reference_role,
            *rows,
            scorers,
        )
        for collected, row, existing in zip(output_rows, rows, existing_rows):
            existing[prompt_index] = row
            collected.append(row)
    write_all_outputs(args, *output_rows)
    del reference_outputs
    _clear_memory()


def run(args):
    models = _load_models(args.models_json)
    prompt_indices, prompts = build_prompt_subset(args.input_txt)
    if not prompts:
        raise ValueError("No prompts selected")

    clean_ref_id = args.clean_ref_model_id or args.base_model_id
    backdoor_ref_id = args.backdoor_ref_model_id or args.base_model_id
    if not clean_ref_id or not backdoor_ref_id:
        raise ValueError("clean_ref_model_id and backdoor_ref_model_id are required")

    scorers = build_scorers(
        args.device,
        args.openclip_model,
        args.openclip_pretrained,
        args.openclip_cache_dir,
        args.openclip_revision,
    )
    shared_mode = args.model_family in {"sd35", "flux"}
    shared_pipe = None
    loaded_adapters = None
    if shared_mode:
        print("[load] shared base pipeline and six role adapters")
        shared_pipe, loaded_adapters = _load_shared_adapter_pipeline(
            args, models, clean_ref_id, backdoor_ref_id
        )
        clean_ref_pipe = shared_pipe
        backdoor_ref_pipe = shared_pipe
    else:
        print(f"[load] clean_ref: {clean_ref_id}")
        clean_ref_pipe = load_selected_pipeline(
            args.model_family, clean_ref_id, args.device, lora_id=args.clean_ref_lora_id or None
        )
        print(f"[load] backdoor_ref: {backdoor_ref_id}")
        backdoor_ref_pipe = load_selected_pipeline(
            args.model_family,
            backdoor_ref_id,
            args.device,
            lora_id=args.backdoor_ref_lora_id or None,
        )

    try:
        for target_name, parity, label in TARGET_SPECS:
            current = _output_args(args, label)
            done = completed_reference_prompt_indices(current)
            parity_value = 1 if parity == "odd" else 0
            pending = [
                (idx, prompt)
                for idx, prompt in zip(prompt_indices, prompts)
                if (idx + 1) % 2 == parity_value and idx not in done
            ]
            if not pending:
                print(f"[resume] {target_name}: no pending prompts")
                continue

            existing_rows = _initial_rows(current)
            clean_done = clean_reference_prompt_indices(current)
            target = models[target_name]
            if not target.get("model_id"):
                raise ValueError(f"Missing model_id for {target_name}")
            if shared_mode:
                target_pipe = shared_pipe
            else:
                print(f"[load] {target_name}: {target['model_id']}")
                target_pipe = load_selected_pipeline(
                    args.model_family,
                    target["model_id"],
                    args.device,
                    lora_id=target.get("lora_id") or None,
                )
            label_value = 0 if label == "clean" else 1
            try:
                for batch in _iter_batches(pending, args.prompt_batch_size):
                    indices = [item[0] for item in batch]
                    batch_prompts = [item[1] for item in batch]
                    print(f"[{target_name}] prompt_index={indices[0]}-{indices[-1]}")
                    _activate_role(target_pipe, loaded_adapters, target_name)
                    target_outputs = generate_light_outputs_batch(
                        current, target_pipe, indices, batch_prompts
                    )
                    clean_items = [item for item in batch if item[0] not in clean_done]
                    if clean_items:
                        _compute_reference_rows(
                            current,
                            clean_items,
                            target_outputs,
                            clean_ref_pipe,
                            "clean_ref",
                            scorers,
                            existing_rows,
                            label_value,
                            loaded_adapters,
                        )
                    _compute_reference_rows(
                        current,
                        batch,
                        target_outputs,
                        backdoor_ref_pipe,
                        "backdoor_ref",
                        scorers,
                        existing_rows,
                        label_value,
                        loaded_adapters,
                    )
                    del target_outputs
                    _clear_memory()
            finally:
                if not shared_mode:
                    release_pipeline(target_pipe)
    finally:
        if shared_mode:
            release_pipeline(shared_pipe)
        else:
            release_pipeline(backdoor_ref_pipe)
            release_pipeline(clean_ref_pipe)

    print("Done. Wrote reference-assisted training features.")


def parse_args():
    parser = argparse.ArgumentParser(description="Paper reference-assisted training-feature generation")
    parser.add_argument(
        "--models_json",
        type=Path,
        required=True,
    )
    parser.add_argument("--input_txt", required=True)
    parser.add_argument("--model_family", choices=["sd2", "sd35", "flux", "sd14"], required=True)
    parser.add_argument("--base_model_id", type=str, required=True)
    parser.add_argument("--clean_ref_model_id",default=None)
    parser.add_argument("--backdoor_ref_model_id",default=None)
    parser.add_argument("--clean_ref_lora_id", default=None)
    parser.add_argument("--backdoor_ref_lora_id", default=None)
    parser.add_argument("--output_root", type=Path, default=Path("artifacts/features/train"))
    parser.add_argument("--dataset_suffix", default="stage1_train_5biases4clean")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--export_steps", default="0,10,20,30,40,49")
    parser.add_argument("--layer_start", type=int, default=None)
    parser.add_argument("--layer_end", type=int, default=None)
    parser.add_argument("--prompt_batch_size", type=int, default=8)
    parser.add_argument(
        "--openclip_model",
        type=str,
        required=True,
        help="OpenCLIP architecture; must match the paper run",
    )
    parser.add_argument(
        "--openclip_pretrained",
        type=str,
        required=True,
        help="OpenCLIP pretrained tag/path, or explicit NONE_RANDOM_INITIALIZATION",
    )
    parser.add_argument(
        "--openclip_cache_dir",
        type=str,
        default=None,
        help="Explicit OpenCLIP cache directory",
    )
    parser.add_argument(
        "--openclip_revision",
        type=str,
        required=True,
        help="Exact checkpoint revision identifier, or explicit UNSPECIFIED",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
