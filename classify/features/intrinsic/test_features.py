"""Generate test LFD and latent-trajectory features in one pass."""

import argparse
import csv
from pathlib import Path

from feature_utils import (
    generate_model_outputs,
    load_done_prompt_indices,
    load_selected_pipeline,
    prepare_common,
    release_pipeline,
    result_specs,
    write_trace_metric_rows,
)


def run(args):
    prompts = prepare_common(args)
    hidden_csv = Path(args.lfd_output_csv)
    specs = result_specs(args)

    done_hidden = load_done_prompt_indices(hidden_csv)
    done_trace = {name: load_done_prompt_indices(spec["path"]) for name, spec in specs.items()}
    pending_indices = [
        idx
        for idx in range(len(prompts))
        if idx not in done_hidden or any(idx not in done for done in done_trace.values())
    ]
    if not pending_indices:
        print("Done. No pending prompts.")
        return

    hidden_file_exists = hidden_csv.exists() and hidden_csv.stat().st_size > 0
    trace_file_exists = {
        name: spec["path"].exists() and spec["path"].stat().st_size > 0
        for name, spec in specs.items()
    }

    hidden_fieldnames = ["prompt_index", "original_prompt", "test_orig_feature_paths"]
    trace_fieldnames = ["prompt_index", "original_prompt", "test_orig"]

    test_lora = args.test_lora_id if args.test_lora_id else None
    test_model_id = args.base_model_id if test_lora else args.test_model_id
    test_pipe = load_selected_pipeline(args, test_model_id, lora_id=test_lora)
    try:
        test_outputs = generate_model_outputs(
            args,
            test_pipe,
            prompts,
            pending_indices,
            "test",
            Path(args.lfd_tensor_dir),
            Path(args.latent_trajectory_dir),
        )
    finally:
        release_pipeline(test_pipe)

    with hidden_csv.open("a" if hidden_file_exists else "w", newline="", encoding="utf-8") as f_hidden:
        hidden_writer = csv.DictWriter(f_hidden, fieldnames=hidden_fieldnames)
        if not hidden_file_exists:
            hidden_writer.writeheader()

        trace_files = {
            name: spec["path"].open("a" if trace_file_exists[name] else "w", newline="", encoding="utf-8")
            for name, spec in specs.items()
        }
        try:
            trace_writers = {
                name: csv.DictWriter(handle, fieldnames=trace_fieldnames)
                for name, handle in trace_files.items()
            }
            for name, writer in trace_writers.items():
                if not trace_file_exists[name]:
                    writer.writeheader()

            for idx in pending_indices:
                prompt = prompts[idx]
                hidden_writer.writerow(
                    {
                        "prompt_index": idx,
                        "original_prompt": prompt,
                        "test_orig_feature_paths": ";".join(test_outputs[idx]["feature_paths"]),
                    }
                )
                f_hidden.flush()

                write_trace_metric_rows(
                    trace_writers,
                    idx,
                    prompt,
                    {"test_orig": test_outputs[idx]["metrics"]},
                    specs,
                )
                for handle in trace_files.values():
                    handle.flush()
        finally:
            for handle in trace_files.values():
                handle.close()

    print(f"Done. LFD CSV: {args.lfd_output_csv}")
    print(
        "Latent-trajectory CSVs: "
        f"{args.latent_cosine_csv}, {args.latent_norm_csv}, {args.latent_update_norm_csv}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Intrinsic test-feature generation")
    parser.add_argument("--input_txt", type=str, required=True)
    parser.add_argument("--model_family", choices=["sd14", "sd2", "sd35", "flux"], required=True)
    parser.add_argument("--base_model_id", type=str, required=True)
    parser.add_argument("--test_model_id", type=str, required=True)
    parser.add_argument("--test_lora_id", type=str, default=None, help="Use base_model_id as base and load this LoRA when set")
    parser.add_argument("--lfd_output_csv", "--hidden_output_csv", dest="lfd_output_csv", type=str, required=True)
    parser.add_argument("--lfd_tensor_dir", "--hidden_output_dir", dest="lfd_tensor_dir", type=str, required=True)
    parser.add_argument("--latent_trajectory_dir", "--trace_output_dir", dest="latent_trajectory_dir", type=str, required=True)
    parser.add_argument("--latent_cosine_csv", "--trace_output_csv_angle", dest="latent_cosine_csv", type=str, required=True)
    parser.add_argument("--latent_norm_csv", "--trace_output_csv_norm", dest="latent_norm_csv", type=str, required=True)
    parser.add_argument("--latent_update_norm_csv", "--trace_output_csv_delta_norm", dest="latent_update_norm_csv", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--export_steps", type=str, default="0,10,20,30,40,49")
    parser.add_argument("--num_images_orig", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
