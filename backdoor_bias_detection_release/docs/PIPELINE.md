# Method-to-code and experiment pipeline

## Controlled generation

For every suspect/reference comparison, preserve the exact prompt text and order, `seed + prompt_index + image_index` rule used by the relevant generator, scheduler and scheduler configuration, number of inference steps, guidance scale, height/width, dtype, model revision and LoRA revision. Selected Stage-I steps are `0,10,20,30,40,49` in the detector configuration; this must agree with extraction outputs.

## Stage I

`classify/generate_features.py` is the only public feature-generation launcher. It reads `configs/feature_jobs.paper.json`; each job selects a feature group and supplies the model family plus complete arguments. One job and many jobs use the same execution path and produce a timestamped command/log summary.

Training jobs must select the odd/even backends. The ordinary training generators are not part of the paper pipeline.

1. `features/intrinsic/` jointly records coarse block activations and adjacent latent traces in one generation pass.
2. `features/reference/` produces reference-assisted activation, latent, and decoded-image comparisons in one generation pass.
3. Shared formulas and generation helpers live in `utils/`; the redundant standalone LFD, adjacent-latent, image-similarity, and activation-difference entry-point directories were removed from the public release.
4. The seven `feature_ablation.py` CSV arguments must contain the same prompt keys and compatible step layout. Referenced tensor paths must remain valid.
5. `classifier/feature_ablation.py --mode train_all` performs the original seed-42 prompt-level train/validation split and trains the declared feature variants.
6. `--mode predict_all` writes prompt probabilities for every selected variant. `batch_feature_ablation.py` is the supported multi-model prediction entry point, and `evaluation/model_aggregation.py` computes the paper model score.

Prompt diagnostic labels use `>=` and do not enter the paper decision. The paper model score is `mean(backdoor_probability)` and its decision uses the strict comparison `P(M) > tau`.

## Stage II

`stage2/stage2_reference_dispersion_measurement.py` generates/reuses reference images, obtains localized SAM crops (full images for scene domains), encodes them with DINOv2 or CLIP, applies PCA and HDBSCAN, and writes decision-free per-domain measurements. Batch mode reuses reference generations after the first case. `stage2/stage2_perturbation_response_verifier.py` then reads the single-model JSON or batch summary and computes the paper perturbation-response decision. Keep the first case and reference directories fixed and recorded because ordering affects what is reused.

Stage II must receive only models flagged by the fixed Stage-I rule. Its thresholds must be selected on validation models, frozen, and then applied once to test models.

## File contracts

- CSVs are UTF-8 with headers; prompt identity is carried by `prompt_index` and `original_prompt`.
- Tensor feature paths are semicolon-separated in `*_feature_paths` columns and point to `torch.save` files.
- Detector checkpoints are dictionaries containing `feature_layout`, model state, feature names/dimensions, architecture parameters, and train/validation prompt membership.
- Prediction CSV probability is class-1 softmax output. Final model prediction is not interchangeable with a prompt label.

Do not rename columns, reorder prompts, recompute labels, change tensor serialization, or normalize features outside the existing detector; those actions would change the method contract.
