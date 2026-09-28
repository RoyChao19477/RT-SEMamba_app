#!/usr/bin/env bash
# Regenerate data/*.json for your local VCTK-DEMAND (16 kHz) copy.
# --prefix_path must contain: clean_trainset_28spk_wav_16k/, noisy_trainset_28spk_wav_16k/,
#                             clean_testset_wav_16k/, noisy_testset_wav_16k/
python data/make_dataset_json.py \
    --prefix_path /path/to/your/Corpora/noisy_vctk/
