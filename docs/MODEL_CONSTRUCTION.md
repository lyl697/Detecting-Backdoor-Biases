# Model construction scope

This repository contributes the detection framework. Fine-tuned benign and
backdoored diffusion checkpoints and dedicated batch backdoor-construction
scripts are not redistributed. Model construction follows the existing B² /
Backdooring Bias poisoning methodology and standard public fine-tuning
frameworks.

## SD1.4

- Base model directory name: `stable-diffusion-v1-4` (revision not specified).
- Full text-to-image fine-tuning with EMA: 512 px, batch 16, accumulation 4,
  50 epochs, fp16, learning rate 1e-5, constant schedule, no warmup,
  max-gradient norm 1, gradient checkpointing, checkpoint every 2000 steps.
- Seed and optimizer were not specified by the launcher.

## SD2

- Base model: `stable-diffusion-2` (author-confirmed; revision not specified).
- The shared launcher defines the same training options as SD1.4.
- Its supplied SD2 job group is empty. The exact poison jobs/sizes and seed
  therefore cannot be recovered from that launcher.

## SD3.5

- Base model directory name: `stable-diffusion-3.5-medium` (revision not specified).
- LoRA: 1024 px, batch 1, accumulation 16, 1000 steps, fp16, learning rate
  1e-4, constant schedule, no warmup, max-gradient norm 1, gradient
  checkpointing, checkpoint at 1200 steps.
- LoRA rank and seed were not specified by the launcher.

## FLUX.1

- Base model directory name: `FLUX.1-Krea-dev` (revision not specified).
- LoRA rank/alpha 16, dropout 0; 512 px, batch 1, accumulation 4, 20 epochs,
  bf16, learning rate 1e-4, constant schedule, no warmup, max-gradient norm 1,
  gradient checkpointing, 8-bit Adam, TF32, seed 42.
- Guidance scale 3.5, no weighting scheme, maximum sequence length 128,
  precomputed text embeddings (batch 4), checkpoint every 1000 steps and at
  most three checkpoints.
