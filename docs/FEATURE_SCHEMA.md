# Feature terminology and compatibility

Public commands, configuration files, directories and documentation use the
paper concepts below. Legacy CLI spellings remain accepted so downloaded server
commands continue to run.

| Paper concept | Canonical CLI/config stem | Legacy CLI stem |
|---|---|---|
| Local Feature Dynamics (LFD) | `lfd` | `hidden` |
| latent-trajectory cosine | `latent_cosine` | `trace_angle` |
| latent norm | `latent_norm` | `trace_norm` |
| latent update norm | `latent_update_norm` | `trace_delta` |
| reference latent discrepancy | `reference_latent` | `modelcross` |
| decoded-image similarity | `image_similarity` | `image_output` |
| reference activation difference | `activation_difference` | `navi_hidden`, `navi_output` |

The following serialized names are intentionally unchanged:

- LFD CSV path columns ending in `_orig_feature_paths`;
- latent-trajectory CSV payloads and the historical `mean_angle_deg` key;
- reference-latent columns beginning with `modelcross_`;
- activation-difference tensor-derived classifier names beginning with `navi_`;
- checkpoint state-dict keys.

Changing those serialized names would prevent old feature files and checkpoints
from being consumed directly. They are compatibility schema, not the preferred
terminology for new commands.

## Prompt prediction CSV

New classifier prediction files use:

`model_id,architecture,prompt_index,original_prompt,true_label,model_tag,predicted_label,predicted_tag,backdoor_probability`

`model_tag` is retained as a compatibility alias of `model_id`. Missing prompt
text or ground truth may be empty in inference-only runs. Paper model-level
aggregation consumes `backdoor_probability`, never the hard
`predicted_label` ratio.

## Canonical artifact paths

Training features are stored below
`artifacts/features/train/<architecture>/<feature_group>/...`; test features
are stored below
`artifacts/features/test/<architecture>/<feature_group>/<model_id>/...`.
The `feature_ablation.py` command-line arguments define the input paths, and
`configs/feature_jobs.json` supplies generator outputs. Generator and
classifier configurations must resolve to the same paths before execution.
