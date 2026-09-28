#!/usr/bin/env bash
# Real-time enhancement app (GUI): live mic -> RT-SEMamba -> recordings/ (+ optional monitor).
#   start the GUI:            sh run_realtime.sh
#   start with KD8 teacher:   sh run_realtime.sh --model kd8
#   list audio devices:       sh run_realtime.sh --list-devices
#   headless check on a wav:  sh run_realtime.sh --input-file noisy.wav --verify
PYTORCH_ENABLE_MPS_FALLBACK=1 python rt_enhance.py "$@"
