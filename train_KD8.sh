#!/usr/bin/env bash
# Train the 8-layer KD8 teacher from scratch (no distillation).
python train.py \
  --config recipes/KD8/KD8.yaml \
  --exp_folder exp \
  --exp_name RT-SEMamba_KD8
