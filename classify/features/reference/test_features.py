"""Joint test feature generation for reference-target features."""

import argparse
import gc

import torch

from feature_utils import (
    build_scorers,
    build_prompt_subset,
    clean_joint_prompt_indices,
    completed_joint_prompt_indices,
    generate_light_outputs_batch,
    init_reference_rows,
    load_selected_pipeline,
    prepare_args,
    read_rows_by_prompt,
    release_pipeline,
    update_rows_with_reference,
    write_all_outputs,
)


def _iter_batches(prompt_indices, prompts, batch_size: int):
    batch_size = max(1, int(batch_size))
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        yield prompt_indices[start:end], prompts[start:end]


def _load_shared_adapter_pipeline(args):
    """Load one SD3.5/FLUX/FLUX base pipeline and attach role LoRAs as named adapters."""

    model_ids = {
        str(args.test_model_id),
        str(args.clean_ref_model_id),
        str(args.backdoor_ref_model_id),
    }
    if len(model_ids) != 1:
        raise ValueError(
            "The shared SD3.5/FLUX pipeline requires test/clean_ref/backdoor_ref to use "
            "the same base model_id. Use different LoRA paths to distinguish them."
        )

    pipe = load_selected_pipeline(
        args.model_family,
        args.test_model_id,
        args.device,
        lora_id=None,
    )
    role_loras = {
        "test": args.test_lora_id,
        "clean_ref": args.clean_ref_lora_id,
        "backdoor_ref": args.backdoor_ref_lora_id,
    }
    loaded_adapters = set()
    for adapter_name, lora_path in role_loras.items():
        if not lora_path:
            continue
        pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        loaded_adapters.add(adapter_name)
        print(f"[{args.model_family}] loaded adapter {adapter_name}: {lora_path}")
    return pipe, loaded_adapters


def _activate_shared_role(pipe, loaded_adapters, role: str):
    if role in loaded_adapters:
        if hasattr(pipe, "enable_lora"):
            pipe.enable_lora()
        pipe.set_adapters(role)
    else:
        if hasattr(pipe, "disable_lora"):
            pipe.disable_lora()
        elif loaded_adapters:
            raise RuntimeError(
                f"Cannot select base SD3.5/FLUX for role={role!r}: this diffusers version "
                "does not provide disable_lora()."
            )


def _run_shared(args):
    args = prepare_args(args)
    prompt_indices, prompts = build_prompt_subset(args.input_txt)
    if not prompts:
        raise ValueError("No prompts selected")
    done_indices = completed_joint_prompt_indices(args)
    if done_indices:
        print(f"[resume] skip {len(done_indices)} completed prompts")
    pending_items = [
        (prompt_idx, prompt)
        for prompt_idx, prompt in zip(prompt_indices, prompts)
        if prompt_idx not in done_indices
    ]
    if not pending_items:
        print("Done. No pending prompts.")
        return
    prompt_indices = [item[0] for item in pending_items]
    prompts = [item[1] for item in pending_items]
    clean_done_indices = clean_joint_prompt_indices(args)
    if clean_done_indices:
        print(f"[resume] reuse {len(clean_done_indices)} clean_ref rows")
    existing_image_rows = read_rows_by_prompt(args.image_similarity_output_csv)
    existing_reference_latent_rows = read_rows_by_prompt(args.reference_latent_output_csv)
    existing_activation_difference_rows = read_rows_by_prompt(args.activation_difference_output_csv)

    scorers = build_scorers(
        args.device,
        args.openclip_model,
        args.openclip_pretrained,
        args.openclip_cache_dir,
        args.openclip_revision,
    )

    pipe, loaded_adapters = _load_shared_adapter_pipeline(args)
    try:
        for batch_indices, batch_prompts in _iter_batches(
            prompt_indices,
            prompts,
            args.prompt_batch_size,
        ):
            print(f"[test] prompt_index={batch_indices[0]}-{batch_indices[-1]}")
            _activate_shared_role(pipe, loaded_adapters, "test")
            target_outputs = generate_light_outputs_batch(
                args,
                pipe,
                batch_indices,
                batch_prompts,
            )

            clean_items = [
                (prompt_idx, prompt)
                for prompt_idx, prompt in zip(batch_indices, batch_prompts)
                if prompt_idx not in clean_done_indices
            ]
            if clean_items:
                clean_indices = [item[0] for item in clean_items]
                clean_prompts = [item[1] for item in clean_items]
                print(f"[clean_ref] prompt_index={clean_indices[0]}-{clean_indices[-1]}")
                _activate_shared_role(pipe, loaded_adapters, "clean_ref")
                clean_outputs = generate_light_outputs_batch(
                    args,
                    pipe,
                    clean_indices,
                    clean_prompts,
                )
                image_rows = []
                reference_latent_rows = []
                activation_difference_rows = []
                for prompt_idx, prompt in clean_items:
                    image_row, reference_latent_row, activation_difference_row = init_reference_rows(
                        args,
                        prompt_idx,
                        prompt,
                        "test",
                        None,
                    )
                    image_row.update(existing_image_rows.get(prompt_idx, {}))
                    reference_latent_row.update(existing_reference_latent_rows.get(prompt_idx, {}))
                    activation_difference_row.update(existing_activation_difference_rows.get(prompt_idx, {}))
                    update_rows_with_reference(
                        args,
                        prompt_idx,
                        target_outputs[prompt_idx],
                        clean_outputs[prompt_idx],
                        "clean_ref",
                        image_row,
                        reference_latent_row,
                        activation_difference_row,
                        scorers,
                    )
                    existing_image_rows[prompt_idx] = image_row
                    existing_reference_latent_rows[prompt_idx] = reference_latent_row
                    existing_activation_difference_rows[prompt_idx] = activation_difference_row
                    image_rows.append(image_row)
                    reference_latent_rows.append(reference_latent_row)
                    activation_difference_rows.append(activation_difference_row)
                write_all_outputs(args, image_rows, reference_latent_rows, activation_difference_rows)
                del clean_outputs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print(f"[backdoor_ref] prompt_index={batch_indices[0]}-{batch_indices[-1]}")
            _activate_shared_role(pipe, loaded_adapters, "backdoor_ref")
            backdoor_outputs = generate_light_outputs_batch(args, pipe, batch_indices, batch_prompts)
            image_rows = []
            reference_latent_rows = []
            activation_difference_rows = []
            for prompt_idx, prompt in zip(batch_indices, batch_prompts):
                image_row, reference_latent_row, activation_difference_row = init_reference_rows(
                    args,
                    prompt_idx,
                    prompt,
                    "test",
                    None,
                )
                image_row.update(existing_image_rows.get(prompt_idx, {}))
                reference_latent_row.update(existing_reference_latent_rows.get(prompt_idx, {}))
                activation_difference_row.update(existing_activation_difference_rows.get(prompt_idx, {}))
                update_rows_with_reference(
                    args,
                    prompt_idx,
                    target_outputs[prompt_idx],
                    backdoor_outputs[prompt_idx],
                    "backdoor_ref",
                    image_row,
                    reference_latent_row,
                    activation_difference_row,
                    scorers,
                )
                existing_image_rows[prompt_idx] = image_row
                existing_reference_latent_rows[prompt_idx] = reference_latent_row
                existing_activation_difference_rows[prompt_idx] = activation_difference_row
                image_rows.append(image_row)
                reference_latent_rows.append(reference_latent_row)
                activation_difference_rows.append(activation_difference_row)
            write_all_outputs(args, image_rows, reference_latent_rows, activation_difference_rows)

            del target_outputs
            del backdoor_outputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        release_pipeline(pipe)

    print("Done. Wrote joint reference test features.")


def _clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_separate(args):
    args = prepare_args(args)
    prompt_indices, prompts = build_prompt_subset(args.input_txt)
    if not prompts:
        raise ValueError("No prompts selected")

    done_indices = completed_joint_prompt_indices(args)
    pending = [
        (idx, prompt)
        for idx, prompt in zip(prompt_indices, prompts)
        if idx not in done_indices
    ]
    if not pending:
        print("Done. No pending prompts.")
        return
    prompt_indices = [item[0] for item in pending]
    prompts = [item[1] for item in pending]

    clean_done = clean_joint_prompt_indices(args)
    existing_image = read_rows_by_prompt(args.image_similarity_output_csv)
    existing_reference_latent = read_rows_by_prompt(args.reference_latent_output_csv)
    existing_activation_difference = read_rows_by_prompt(args.activation_difference_output_csv)
    scorers = build_scorers(
        args.device,
        args.openclip_model,
        args.openclip_pretrained,
        args.openclip_cache_dir,
        args.openclip_revision,
    )

    print(f"[load] {args.model_family} test model")
    test_pipe = load_selected_pipeline(
        args.model_family, args.test_model_id, args.device, lora_id=None
    )
    print(f"[load] {args.model_family} clean reference model")
    clean_ref_pipe = load_selected_pipeline(
        args.model_family, args.clean_ref_model_id, args.device, lora_id=None
    )
    print(f"[load] {args.model_family} backdoor reference model")
    backdoor_ref_pipe = load_selected_pipeline(
        args.model_family, args.backdoor_ref_model_id, args.device, lora_id=None
    )

    try:
        for batch_indices, batch_prompts in _iter_batches(
            prompt_indices, prompts, args.prompt_batch_size
        ):
            print(f"[test] prompt_index={batch_indices[0]}-{batch_indices[-1]}")
            target_outputs = generate_light_outputs_batch(
                args, test_pipe, batch_indices, batch_prompts
            )

            clean_items = [
                (idx, prompt)
                for idx, prompt in zip(batch_indices, batch_prompts)
                if idx not in clean_done
            ]
            if clean_items:
                clean_indices = [item[0] for item in clean_items]
                clean_prompts = [item[1] for item in clean_items]
                print(f"[clean_ref] prompt_index={clean_indices[0]}-{clean_indices[-1]}")
                clean_outputs = generate_light_outputs_batch(
                    args, clean_ref_pipe, clean_indices, clean_prompts
                )
                rows = [[], [], []]
                for prompt_idx, prompt in clean_items:
                    image_row, reference_latent_row, activation_difference_row = init_reference_rows(
                        args, prompt_idx, prompt, "test", None
                    )
                    image_row.update(existing_image.get(prompt_idx, {}))
                    reference_latent_row.update(existing_reference_latent.get(prompt_idx, {}))
                    activation_difference_row.update(existing_activation_difference.get(prompt_idx, {}))
                    update_rows_with_reference(
                        args,
                        prompt_idx,
                        target_outputs[prompt_idx],
                        clean_outputs[prompt_idx],
                        "clean_ref",
                        image_row,
                        reference_latent_row,
                        activation_difference_row,
                        scorers,
                    )
                    existing_image[prompt_idx] = image_row
                    existing_reference_latent[prompt_idx] = reference_latent_row
                    existing_activation_difference[prompt_idx] = activation_difference_row
                    for output_rows, row in zip(rows, (image_row, reference_latent_row, activation_difference_row)):
                        output_rows.append(row)
                write_all_outputs(args, *rows)
                del clean_outputs
                _clear_memory()

            print(f"[backdoor_ref] prompt_index={batch_indices[0]}-{batch_indices[-1]}")
            backdoor_outputs = generate_light_outputs_batch(
                args, backdoor_ref_pipe, batch_indices, batch_prompts
            )
            rows = [[], [], []]
            for prompt_idx, prompt in zip(batch_indices, batch_prompts):
                image_row, reference_latent_row, activation_difference_row = init_reference_rows(
                    args, prompt_idx, prompt, "test", None
                )
                image_row.update(existing_image.get(prompt_idx, {}))
                reference_latent_row.update(existing_reference_latent.get(prompt_idx, {}))
                activation_difference_row.update(existing_activation_difference.get(prompt_idx, {}))
                update_rows_with_reference(
                    args,
                    prompt_idx,
                    target_outputs[prompt_idx],
                    backdoor_outputs[prompt_idx],
                    "backdoor_ref",
                    image_row,
                    reference_latent_row,
                    activation_difference_row,
                    scorers,
                )
                existing_image[prompt_idx] = image_row
                existing_reference_latent[prompt_idx] = reference_latent_row
                existing_activation_difference[prompt_idx] = activation_difference_row
                for output_rows, row in zip(rows, (image_row, reference_latent_row, activation_difference_row)):
                    output_rows.append(row)
            write_all_outputs(args, *rows)
            del target_outputs, backdoor_outputs
            _clear_memory()
    finally:
        release_pipeline(backdoor_ref_pipe)
        release_pipeline(clean_ref_pipe)
        release_pipeline(test_pipe)

    print(f"Done. Wrote {args.model_family} joint reference test features.")



def run(args):
    if args.model_family in {"sd14", "sd2"}:
        return _run_separate(args)
    return _run_shared(args)


def parse_args():
    parser = argparse.ArgumentParser(description="Joint reference-target test features")
    parser.add_argument("--input_txt", type=str, required=True)
    parser.add_argument("--model_family", choices=["sd14", "sd2", "sd35", "flux"], required=True)
    parser.add_argument("--base_model_id", type=str, required=True)
    parser.add_argument("--clean_ref_model_id", type=str, default=None)
    parser.add_argument("--backdoor_ref_model_id", type=str, default=None)
    parser.add_argument("--test_model_id", type=str, default=None)
    parser.add_argument("--clean_ref_lora_id", type=str, default=None)
    parser.add_argument("--backdoor_ref_lora_id", type=str, default=None)
    parser.add_argument("--test_lora_id", type=str, default=None)
    parser.add_argument("--image_similarity_output_csv", "--image_output_csv", dest="image_similarity_output_csv", type=str, required=True)
    parser.add_argument("--image_similarity_image_dir", "--image_output_dir", dest="image_similarity_image_dir", type=str, required=True)
    parser.add_argument("--reference_latent_output_csv", "--modelcross_output_csv", dest="reference_latent_output_csv", type=str, required=True)
    parser.add_argument("--activation_difference_output_csv", "--navi_output_csv", dest="activation_difference_output_csv", type=str, required=True)
    parser.add_argument("--activation_difference_tensor_dir", "--activation_difference_tensor_dir", dest="activation_difference_tensor_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--export_steps", type=str, default="0,10,20,30,40,49")
    parser.add_argument("--layer_start", type=int, default=None)
    parser.add_argument("--layer_end", type=int, default=None)
    parser.add_argument("--prompt_batch_size", type=int, default=4)
    parser.add_argument(
        "--openclip_model",
        type=str,
        required=True,
        help="OpenCLIP architecture, e.g. ViT-L-14; must match the paper run",
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
    args = parser.parse_args()
    args.clean_ref_model_id = args.clean_ref_model_id or args.base_model_id
    args.backdoor_ref_model_id = args.backdoor_ref_model_id or args.base_model_id
    args.test_model_id = args.test_model_id or args.base_model_id
    return args


if __name__ == "__main__":
    run(parse_args())
