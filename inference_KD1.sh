#!/usr/bin/env bash
# Enhance a folder of wavs with the 1-layer KD1 student, in frame-by-frame
# streaming mode (the mode the model is designed for).
PYTORCH_ENABLE_MPS_FALLBACK=1 python inference.py \
   --input_folder /path/to/noisy_testset_wav_16k/ \
   --output_folder results_KD1 \
   --checkpoint_file ckpts/g_00766000.pth \
   --config recipes/KD1/KD1.yaml \
   --post_processing_PCS False \
   --streaming True \
   --streaming_mode realtime \
   --chunk_size 1600
