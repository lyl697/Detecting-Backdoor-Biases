"""Direct odd/even joint LFD-hidden and adjacent-trace generation."""

import argparse
import csv
import json
from pathlib import Path

from feature_utils import (
    generate_model_outputs,
    load_cached_model_outputs,
    load_done_prompt_indices,
    load_selected_pipeline,
    prepare_common,
    release_pipeline,
    result_specs,
    write_trace_metric_rows,
)
from utils.config_paths import expand_runtime_variables


PAIR_SPECS = (
    ("odd", "clean1", "backdoor1"),
    ("even", "clean2", "backdoor2"),
)


def _load_models(path):
    with Path(path).open("r", encoding="utf-8") as f:
        models = expand_runtime_variables(json.load(f))
    required = {name for _, clean_name, backdoor_name in PAIR_SPECS for name in (clean_name, backdoor_name)}
    missing = required - set(models)
    if missing:
        raise ValueError(f"models_json is missing: {', '.join(sorted(missing))}")
    for name in required:
        if not str(models[name].get("model_id", "")).strip():
            raise ValueError(f"Missing model_id for {name}")
    return models


def _configure_output_paths(args):
    # output_root already identifies train/<architecture>; do not append a
    # second architecture component.
    hidden_base = args.output_root / "lfd" / args.dataset_name
    trace_base = args.output_root / "latent_trajectory" / args.dataset_name
    args.lfd_output_csv = str(hidden_base / "results.csv")
    args.lfd_tensor_dir = str(hidden_base / "features_out")
    args.latent_trajectory_dir = str(trace_base / "images_and_traces")
    args.latent_cosine_csv = str(trace_base / "adjacent_z_cos.csv")
    args.latent_norm_csv = str(trace_base / "adjacent_z_norm.csv")
    args.latent_update_norm_csv = str(trace_base / "adjacent_z_delta_norm.csv")


def _pending_indices(prompts, hidden_csv, specs, parity):
    done_hidden = load_done_prompt_indices(hidden_csv)
    done_trace = {
        name: load_done_prompt_indices(spec["path"])
        for name, spec in specs.items()
    }
    parity_value = 1 if parity == "odd" else 0
    return [
        idx
        for idx in range(len(prompts))
        if (idx + 1) % 2 == parity_value
        and (idx not in done_hidden or any(idx not in done for done in done_trace.values()))
    ]


def _load_or_generate(args, prompts, indices, model, model_tag, hidden_dir, trace_dir):
    outputs, missing = load_cached_model_outputs(
        args,
        indices,
        model_tag,
        hidden_dir,
        trace_dir,
    )
    if not missing:
        return outputs

    print(f"[generate] {model_tag}: {len(missing)} prompts from {model['model_id']}")
    pipe = load_selected_pipeline(
        args,
        model["model_id"],
        lora_id=model.get("lora_id") or None,
    )
    try:
        outputs.update(
            generate_model_outputs(
                args,
                pipe,
                prompts,
                missing,
                model_tag,
                hidden_dir,
                trace_dir,
            )
        )
    finally:
        release_pipeline(pipe)
    return outputs


def _append_pair_rows(args, prompts, indices, clean_outputs, backdoor_outputs, specs):
    hidden_csv = Path(args.lfd_output_csv)
    hidden_exists = hidden_csv.exists() and hidden_csv.stat().st_size > 0
    trace_exists = {
        name: spec["path"].exists() and spec["path"].stat().st_size > 0
        for name, spec in specs.items()
    }
    hidden_fields = [
        "prompt_index",
        "original_prompt",
        "clean_orig_feature_paths",
        "backdoor_orig_feature_paths",
    ]
    trace_fields = ["prompt_index", "original_prompt", "clean_orig", "backdoor_orig"]

    with hidden_csv.open("a" if hidden_exists else "w", newline="", encoding="utf-8") as hidden_file:
        hidden_writer = csv.DictWriter(hidden_file, fieldnames=hidden_fields)
        if not hidden_exists:
            hidden_writer.writeheader()
        trace_files = {
            name: spec["path"].open(
                "a" if trace_exists[name] else "w",
                newline="",
                encoding="utf-8",
            )
            for name, spec in specs.items()
        }
        try:
            trace_writers = {
                name: csv.DictWriter(handle, fieldnames=trace_fields)
                for name, handle in trace_files.items()
            }
            for name, writer in trace_writers.items():
                if not trace_exists[name]:
                    writer.writeheader()

            for idx in indices:
                hidden_writer.writerow(
                    {
                        "prompt_index": idx,
                        "original_prompt": prompts[idx],
                        "clean_orig_feature_paths": ";".join(clean_outputs[idx]["feature_paths"]),
                        "backdoor_orig_feature_paths": ";".join(backdoor_outputs[idx]["feature_paths"]),
                    }
                )
                hidden_file.flush()
                write_trace_metric_rows(
                    trace_writers,
                    idx,
                    prompts[idx],
                    {
                        "clean_orig": clean_outputs[idx]["metrics"],
                        "backdoor_orig": backdoor_outputs[idx]["metrics"],
                    },
                    specs,
                )
                for handle in trace_files.values():
                    handle.flush()
        finally:
            for handle in trace_files.values():
                handle.close()


def run(args):
    models = _load_models(args.models_json)
    _configure_output_paths(args)
    prompts = prepare_common(args)
    if not prompts:
        raise ValueError("No prompts selected")

    hidden_csv = Path(args.lfd_output_csv)
    hidden_dir = Path(args.lfd_tensor_dir)
    trace_dir = Path(args.latent_trajectory_dir)
    specs = result_specs(args)

    for parity, clean_name, backdoor_name in PAIR_SPECS:
        indices = _pending_indices(prompts, hidden_csv, specs, parity)
        if not indices:
            print(f"[resume] {parity}: no pending prompts")
            continue
        print(
            f"[odd-even LFD/trace] {parity}: {clean_name} + {backdoor_name}, "
            f"prompts={len(indices)}"
        )
        clean_outputs = _load_or_generate(
            args,
            prompts,
            indices,
            models[clean_name],
            "clean",
            hidden_dir,
            trace_dir,
        )
        backdoor_outputs = _load_or_generate(
            args,
            prompts,
            indices,
            models[backdoor_name],
            "backdoor",
            hidden_dir,
            trace_dir,
        )
        _append_pair_rows(
            args,
            prompts,
            indices,
            clean_outputs,
            backdoor_outputs,
            specs,
        )
        del clean_outputs, backdoor_outputs

    print(f"Done. LFD CSV: {args.lfd_output_csv}")
    print(
        "Latent-trajectory CSVs: "
        f"{args.latent_cosine_csv}, {args.latent_norm_csv}, "
        f"{args.latent_update_norm_csv}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Direct odd/even joint LFD/trace generation")
    parser.add_argument(
        "--models_json",
        type=Path,
        required=True,
    )
    parser.add_argument("--input_txt", required=True)
    parser.add_argument("--model_family", choices=["sd2", "sd35", "flux", "sd14"], required=True)
    parser.add_argument("--output_root", type=Path, default=Path("artifacts/features/train"))
    parser.add_argument("--dataset_name", default="odd_even_joint_train_5biases4clean")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--export_steps", default="0,10,20,30,40,49")
    parser.add_argument("--num_images_orig", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
