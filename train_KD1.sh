#!/usr/bin/env bash
# Distill the 1-layer KD1 student from the KD8 teacher.
# Requires the teacher checkpoint referenced by
# recipes/KD1/KD1.yaml -> training_cfg.kd.teacher_checkpoint (ckpts/g_00259000.pth).
python train.py \
  --config recipes/KD1/KD1.yaml \
  --exp_folder exp \
  --exp_name RT-SEMamba_KD1
