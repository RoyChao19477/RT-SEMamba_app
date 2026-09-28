#!/usr/bin/env bash
# Enhance a folder of wavs with the 8-layer KD8 teacher (offline, full-utterance).
# Add --streaming True --streaming_mode realtime for the streaming comparison.
PYTORCH_ENABLE_MPS_FALLBACK=1 python inference.py \
   --input_folder /path/to/noisy_testset_wav_16k/ \
   --output_folder results_KD8 \
   --checkpoint_file ckpts/g_00259000.pth \
   --config recipes/KD8/KD8.yaml \
   --post_processing_PCS False \
   --streaming False
