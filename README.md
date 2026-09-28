# RT-SEMamba: Real-Time Speech Enhancement Mamba via Progressive Knowledge Distillation

[![Interspeech 2026](https://img.shields.io/badge/Interspeech%202026-Oral-b31b1b)](https://github.com/RoyChao19477/RT-SEMamba)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **RT-SEMamba**, accepted to **Interspeech 2026 (oral)**.

> [!NOTE]
> This is the **real-time demo app** of RT-SEMamba: a desktop GUI that enhances your
> microphone live and records the result, with a fused Metal kernel for Apple Silicon
> ([Real-time app](#real-time-app)). Original RT-SEMamba repository:
> **https://github.com/RoyChao19477/RT-SEMamba**

Rong Chao<sup>1,2</sup>, Sung-Feng Huang<sup>5</sup>, Moreno La Quatra<sup>3</sup>, Sabato Marco Siniscalchi<sup>4</sup>, Wen-Huang Cheng<sup>2</sup>, Szu-Wei Fu<sup>5</sup>, Yu Tsao<sup>1</sup>

<sup>1</sup>Academia Sinica, Taiwan &nbsp;·&nbsp; <sup>2</sup>National Taiwan University, Taiwan &nbsp;·&nbsp; <sup>3</sup>Kore University of Enna, Italy &nbsp;·&nbsp; <sup>4</sup>University of Palermo, Italy &nbsp;·&nbsp; <sup>5</sup>NVIDIA

---

RT-SEMamba is a **fully causal** speech enhancement model built on causal time–frequency
Mamba (**cTF-Mamba**) blocks. Unlike Transformer-based architectures that rely on a growing
key–value cache, Mamba propagates a **fixed-size recurrent state per layer**, so memory and
bandwidth stay constant regardless of utterance length.

The model emits one enhanced frame per input frame with **no lookahead**. Its **algorithm
latency** is bounded by a single analysis window — **25 ms** at 16 kHz — and no extra spectral
normalization is applied. Wall-clock throughput, of course, depends on your hardware; the
algorithm latency does not.

We distill an 8-layer teacher into a 1-layer student with a progressive KD scheme, lifting
the 1-layer model from **3.06 → 3.18 PESQ** on VoiceBank-DEMAND at no additional streaming
cost.

---

## Quick start — real-time app on Apple Silicon

```bash
git clone https://github.com/RoyChao19477/RT-SEMamba_app.git && cd RT-SEMamba_app

conda create -n rtsemamba python=3.11 -y && conda activate rtsemamba
pip install torch==2.7.0 torchaudio==2.7.0
pip install -r requirements.txt
pip install -e ./mamba_install_mps --no-deps

sh run_realtime.sh          # opens the GUI → press ● Start and speak
```

Recordings land in `recordings/`. See [Real-time app](#real-time-app) for every option.

---

## Results

VCTK-DEMAND (16 kHz). The two released checkpoints are in bold: the **8-layer teacher**
and the **`KD1` student** (8-layer→1-layer distilled).

| Model | PESQ | CSIG | CBAK | COVL | STOI | Params | Algorithm latency |
|-------|:----:|:----:|:----:|:----:|:----:|:------:|:------------:|
| Noisy | 1.97 | 3.34 | 2.44 | 2.63 | 0.92 | — | — |
| 1-layer (no KD) | 3.06 | 4.38 | 3.61 | 3.79 | 0.94 | 1.05 M | 25 ms |
| 2-layer (no KD) | 3.19 | 4.51 | 3.66 | 3.93 | 0.95 | 1.29 M | 25 ms |
| 3-layer (no KD) | 3.20 | 4.56 | 3.75 | 3.96 | 0.95 | 1.53 M | 25 ms |
| 4-layer (no KD) | 3.29 | 4.59 | 3.76 | 4.03 | 0.95 | 1.77 M | 25 ms |
| 5-layer (no KD) | 3.27 | 4.56 | 3.75 | 4.00 | 0.95 | 2.01 M | 25 ms |
| **8-layer teacher** | **3.32** | 4.64 | 3.72 | 4.08 | 0.95 | 2.74 M | 25 ms |
| 8-layer→2-layer | 3.22 | 4.55 | 3.69 | 3.97 | 0.95 | 1.29 M | 25 ms |
| **8-layer→1-layer — `KD1` student** | **3.18** | 4.43 | 3.68 | 3.89 | 0.95 | **1.05 M** | 25 ms |

Distillation recovers 46.2% / 19.2% / 63.6% / 34.5% of the teacher–student gap in PESQ /
CSIG / CBAK / COVL, while the student keeps the runtime of a plain 1-layer model.

### Streaming cost — selected operating points

Per-second complexity and steady-state RTF **as reported in the paper**, measured there on a
single NVIDIA RTX 5090 after warm-up. Depths 3, 5, 6, and 7 are omitted here; params, MACs,
and RTF all scale close to linearly between the rows shown.

| cTF-Mamba layers | Params (M) | MACs (G/s) | RTF |
|:----------------:|:----------:|:----------:|:---:|
| 1 (`KD1` student) | 1.05 | 20.56 | **0.11** |
| 2 | 1.29 | 24.39 | 0.13 |
| 4 | 1.77 | 32.04 | 0.19 |
| 8 (teacher) | 2.74 | 47.35 | 0.29 |

The distilled 1-layer student runs at 0.11 RTF against the teacher's 0.29 — ≈2.6× faster —
while recovering much of its quality. Distillation shifts the operating point upward in
quality without increasing runtime. Algorithm latency is a fixed 25 ms at every depth.

> RTF is hardware-dependent: these are the paper's measurements on an NVIDIA RTX 5090, not a
> guarantee for your device. Expect very different numbers on CPU or Apple Silicon. What is
> device-independent is the causal structure and the 25 ms algorithm latency bound.

---

## Released models

| Name | Paper row | cTF-Mamba blocks | Params | Config | Checkpoint |
|------|-----------|:----------------:|:------:|--------|-----------|
| **8-layer teacher** | 8-layer | 8 | 2.74 M | `recipes/KD8/KD8.yaml` | `ckpts/g_00259000.pth` |
| **KD1** | 8-layer→1-layer | 1 | 1.05 M | `recipes/KD1/KD1.yaml` | `ckpts/g_00766000.pth` |

The 8-layer model is the teacher (trained without distillation); KD1 is the distilled
student and the model intended for deployment.

---

## Causal design

Everything below is what makes the model streamable in one frame / out one frame.

| Component | Change from non-causal SEMamba |
|-----------|--------------------------------|
| STFT / iSTFT | `center=False`, `W=400`, `H=100` for both analysis and synthesis → algorithm latency bounded by one 25 ms window |
| Convolutions | Asymmetric causal padding: `K-1` zeros on the left, none on the right (`_get_causal_padding_2d`) |
| Normalization | `InstanceNorm2d` → channel-wise `ChannelLayerNorm2d`, causal along time |
| Time Mamba | Uni-directional over `t = 1…T`, no lookahead |
| Frequency Mamba | Bidirectional — the frequency axis is not a causal axis |
| Per block | An extra MLP after each cTF-Mamba block, to improve per-frame modeling |

At inference the model carries three small states across frames, all fixed-size:

- a **frame buffer** holding the previous `K-1` frames needed by the causal temporal convolutions,
- a **`conv_state`** buffer for the depthwise 1-D convolution preceding the SSM (`d_conv - 1` steps),
- the **`ssm_state`** recurrent hidden state `h_t`.

`SEMamba.set_streaming(True)` switches the encoder, every cTF-Mamba block, and both decoders
into this cached mode; `reset_streaming_state()` clears the states between utterances. Because
nothing grows with `T`, per-frame compute and memory are effectively independent of utterance
length.

---

## Requirements

- Python >= 3.10 (3.11 recommended and tested)
- PyTorch >= 2.2; the real-time app's fused Metal kernel needs PyTorch >= 2.6 (tested: 2.7.0)

| Task | CUDA | Apple Silicon (MPS) | CPU |
|------|:----:|:-------------------:|:---:|
| Training (`train.py`) | ✅ CUDA >= 12.0 | ❌ | ❌ |
| Inference (`inference.py`) | ✅ | ✅ | ✅ |
| Real-time app (`rt_enhance.py`) | ✅ | ✅ (fused Metal scan, PyTorch ≥ 2.6) | ✅ (slow) |

⚠️ Training requires an NVIDIA GPU — `train.py` raises `Mamba needs GPU acceleration`
otherwise. Inference runs anywhere, and the released checkpoints let you use both models
without training.

## Installation

### Apple Silicon (real-time app, inference)

The Mamba setup follows [SEMamba-Apple-Silicon](https://github.com/RoyChao19477/SEMamba-Apple-Silicon).

```bash
conda create -n rtsemamba python=3.11 -y
conda activate rtsemamba

pip install torch==2.7.0 torchaudio==2.7.0     # install first, so requirements.txt keeps this version
pip install -r requirements.txt
pip install -e ./mamba_install_mps --no-deps  # --no-deps: skip transformers etc., not needed here
```

`mamba_install_mps` is a patched `mamba_ssm` that skips CUDA compilation and uses pure-PyTorch
reference implementations for the selective scan, layernorm, and state-update kernels — see
`mamba_install_mps/README_MLX.md`. The real-time app additionally replaces the selective scan
with a fused Metal kernel (`models/mps_kernels.py`). On macOS, run with
`PYTORCH_ENABLE_MPS_FALLBACK=1` (the provided `.sh` scripts and `rt_enhance.py` already set it).

Check the install:

```bash
python -c "import torch, mamba_ssm; print(torch.__version__, torch.backends.mps.is_available())"
# 2.7.0 True
```

### NVIDIA / CUDA (training, inference)

```bash
conda create -n rtsemamba python=3.11 -y
conda activate rtsemamba
```

Install PyTorch from the [official instructions](https://pytorch.org/get-started/locally/)
for your CUDA version, then:

```bash
pip install -r requirements.txt
cd mamba_install && pip install .
```

📌 If you hit numpy errors, `pip install numpy==1.26.4`.

---

## Real-time app

`rt_enhance.py` is a desktop GUI (tkinter — nothing extra to install) that enhances your
microphone live with RT-SEMamba and records both the raw and the enhanced audio.

### Run it

```bash
sh run_realtime.sh                  # or: python rt_enhance.py
```

1. **Model** — `KD1 · 1-layer student` (fast, default) or `8-layer teacher` (stronger, needs
   more compute).
2. **Chunk** — STFT frames per model call (1 frame = 6.25 ms). Smaller = lower latency, more
   calls per second. Defaults: 4 frames (25 ms) for KD1, 8 frames (50 ms) for the 8-layer
   teacher. Chunks too small for the selected model are marked *(too slow)*.
3. **Mic** — input device; **Gain** — linear input gain (default 1.0).
4. Optionally tick **Monitor to** and pick an output to hear the enhanced stream live.
   **Use headphones**, otherwise the speaker feeds back into the mic.
5. Press **● Start** and speak; press **■ Stop** to finish. The first time, macOS asks for
   microphone permission for your terminal app.
6. Compare with **▶ Noisy** / **▶ Enhanced**. Both files are in `recordings/`:
   `noisy_<time>.wav` and `enhanced_<time>_<model>.wav` (16 kHz, float32).

**Enhance file…** enhances a 16 kHz wav file with the selected model and saves
`recordings/<name>_enhanced_<model>.wav`.

While recording, the GUI shows live noisy / enhanced waveforms and:

| Indicator | Meaning |
|-----------|---------|
| Model RTF | model time ÷ audio time; must stay < 1 (green) to keep up |
| Last call | wall time of the last processing step |
| App latency | chunk + 18.75 ms overlap-add + processing (audio devices add their own buffering) |
| Backlog | captured audio not yet processed; grows (red) if the chunk is too small for the model |
| Mic overflows | input samples dropped by the audio device; should stay 0 |
| Monitor underruns | monitor output ran dry (audible gaps in the monitor only, not in the files) |

### Command-line options

| Option | Default | Meaning |
|--------|---------|---------|
| `--model {kd1,kd8}` | `kd1` | Initial model: `kd1` = 1-layer student, `kd8` = 8-layer teacher |
| `--device {mps,cuda,cpu}` | auto | Torch device |
| `--list-devices` | | Print audio device indices and exit |
| `--input-file WAV` | | Headless: enhance a 16 kHz wav instead of opening the GUI |
| `--verify` | | With `--input-file`: compare the streaming output to the offline forward pass |
| `--chunk-frames N` | model default | Chunk size for `--input-file` |
| `--input-gain G` | `1.0` | Linear input gain for `--input-file` |
| `--out-dir DIR` | `recordings` | Output folder for `--input-file` |

```bash
python rt_enhance.py --model kd8                                   # GUI, start with the 8-layer teacher
python rt_enhance.py --input-file noisy.wav --verify               # headless KD1 + equivalence check
python rt_enhance.py --model kd8 --input-file noisy.wav --chunk-frames 16
```

`--verify` prints e.g. `Verify vs offline: max|diff| 2.87e-05  SNR 111.9 dB → PASS`.

### How it works

**Same output as offline.** The app runs the model in *stateful chunked streaming*: each
forward call processes `N` STFT frames and carries every causal state (temporal-conv
caches, Mamba `conv_state` and `ssm_state`) into the next call. Because the model is
causal, the result equals the offline whole-utterance forward pass for any `N` — only
float32 rounding differs (≥ 110 dB SNR in our checks). `--verify` checks this on
any 16 kHz wav.

**Apple Silicon acceleration.** With the vendored `mamba_install_mps`, the selective scan is a
pure-PyTorch loop that issues several MPS kernels per sequence step; the bidirectional
frequency Mamba alone cost ~17 ms per call regardless of chunk size. `models/mps_kernels.py`
replaces it with one fused Metal kernel — adapted from
[SpeechLens](https://github.com/faraday/SpeechLens) — that keeps the 16 SSM states per
channel in registers and fuses softplus, `-exp(A_log)`, the `D` skip, and the `SiLU(z)` gate.
We added reading/writing the SSM state from a caller-owned buffer so the same kernel also
serves the stateful time Mamba. Audio capture and monitoring run in a separate process
(`utils/audio_worker.py`), so GUI or model work cannot starve the audio callbacks.

Model time per call on an Apple M4 Pro (macOS 26.6, PyTorch 2.7.0, MPS). Your numbers will
differ; RTF < 1 means real time.

| Model | Chunk | Pure-PyTorch scan | Fused Metal scan |
|-------|:-----:|:-----------------:|:----------------:|
| KD1 | 4 frames (25 ms) | 27.3 ms (RTF 1.09) | **6.4 ms (RTF 0.26)** |
| KD1 | 16 frames (100 ms) | 29.7 ms (RTF 0.30) | **7.0 ms (RTF 0.07)** |
| 8-layer teacher | 8 frames (50 ms) | — | **19.5 ms (RTF 0.39)** |
| 8-layer teacher | 16 frames (100 ms) | — | **19.8 ms (RTF 0.20)** |

Chunk size trades latency for throughput; the GUI marks chunks too small for the selected
model. Latency inside the app ≈ chunk + 18.75 ms (overlap-add of the 25 ms window) +
processing time; the audio devices add their own buffering on top. The 25 ms algorithm
latency of the model itself is unchanged.

---

## Dataset

[VoiceBank-DEMAND (VCTK-DEMAND)](https://datashare.ed.ac.uk/handle/10283/2791), resampled to
16 kHz: 11,572 noisy–clean training pairs from 28 speakers, 824 test utterances from 2 unseen
speakers.

The `data/*.json` files are lists of absolute paths. Point them at your copy:

```bash
sh make_dataset.sh   # edit --prefix_path inside first
```

`--prefix_path` must contain `clean_trainset_28spk_wav_16k/`, `noisy_trainset_28spk_wav_16k/`,
`clean_testset_wav_16k/`, and `noisy_testset_wav_16k/`.

---

## Training and batch inference

### Step 1 — Train the 8-layer teacher

```bash
sh train_KD8.sh
```

### Step 2 — Distill the 1-layer student

```bash
sh train_KD1.sh
```

The student reads the teacher from `recipes/KD1/KD1.yaml`. It defaults to the released
`ckpts/g_00259000.pth`; point it at your own Step-1 result to distill from scratch:

```yaml
training_cfg:
  kd:
    teacher_config: recipes/KD8/KD8.yaml
    teacher_checkpoint: exp/RT-SEMamba_KD8/g_00259000.pth
```

Set `kd.enabled: false` to train the 1-layer model without distillation (the *1-layer (no KD)*
row above).

📌 Checkpoints and logs land in `exp/<exp_name>/`; watch training with
`tensorboard --logdir exp/RT-SEMamba_KD1/logs`.

### Step 3 — Enhance

```bash
sh inference_KD1.sh   # 1-layer student, frame-by-frame streaming
sh inference_KD8.sh   # 8-layer teacher
```

Edit `--input_folder` / `--output_folder` first. Key flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--streaming` | `False` | Enable streaming STFT/iSTFT and cached model state |
| `--streaming_mode` | `realtime` | `realtime` = one frame per forward; `chunked` = whole chunk per forward |
| `--chunk_size` | `1600` | Input block size in samples (100 ms @ 16 kHz) |
| `--post_processing_PCS` | `False` | Apply Perceptual Contrast Stretching to the output |

---

## Distillation objective

The teacher is frozen; the student's encoder, decoders, and single cTF-Mamba block are
initialized from it. On top of the standard SEMamba task loss we add output-level and
intermediate-feature distillation, introduced **progressively** via a ramp
`γ(k) = min(k / K_ramp, 1)` with `K_ramp` = 10% of total steps:

```
L_total = L_task + γ(k) · ( λ_out · L_KD_out + λ_feat · L_KD_feat )
```

- `L_KD_out` — magnitude, phase, and complex MSE against the teacher, weighted `1.0 / 0.3 / 0.5`
- `L_KD_feat` — MSE at the cTF-Mamba tap point against the mean of all 8 teacher block
  outputs, with per-sample normalization on both sides
- `λ_out = 0.5`, `λ_feat = 0.1`

All of this is configurable under `training_cfg.kd` in `recipes/KD1/KD1.yaml`.

---

## Repository layout

```
train.py                 Training entry point (teacher from scratch, student with KD)
inference.py             Batch enhancement — offline, chunked, or frame-by-frame streaming
rt_enhance.py            Real-time app: GUI, live mic enhancement + recording, file mode

models/
  generator.py           Encoder -> cTF-Mamba stack -> mask/phase decoders
  mamba_block.py         cTF-Mamba block (causal time + bidirectional frequency) and MLP
  transformer_block.py   Optional cTF-Transformer block (teacher-architecture study, §4.4)
  codec_module.py        Causal dense encoder, magnitude decoder, phase decoder
  discriminator.py       PESQ metric discriminator
  loss.py                Phase losses and PESQ scoring
  stfts.py               Causal (center=False) magnitude/phase STFT and iSTFT
  mps_kernels.py         Fused Metal selective-scan kernel for Apple Silicon (from SpeechLens)
  lsigmoid.py            Learnable sigmoid
  pcs400.py              Perceptual Contrast Stretching

utils/
  streaming.py           StreamingSTFT / StreamingISTFT ring buffers
  util.py                Config loading, checkpointing, distributed setup
  audio_worker.py        Mic capture / monitor playback in a separate process

dataloaders/             VCTK-DEMAND dataset
data/                    Dataset JSON file lists + generator script
recipes/KD1, recipes/KD8 Model configs (KD1 student, 8-layer teacher)
ckpts/                   Released KD1 and 8-layer teacher checkpoints
mamba_install/           Vendored mamba_ssm (CUDA)
mamba_install_mps/       Vendored mamba_ssm patched for Apple Silicon / CPU
```

---

## Evaluation

PESQ, CSIG, CBAK, COVL, and STOI are computed with the
[CMGAN evaluation script](https://github.com/ruizhecao96/CMGAN/blob/main/src/tools/compute_metrics.py).

## Acknowledgements

Built on [SEMamba](https://github.com/RoyChao19477/SEMamba). We also thank the authors of
[MP-SENet](https://github.com/yxlu-0102/MP-SENet),
[CMGAN](https://github.com/ruizhecao96/CMGAN),
[HiFi-GAN](https://github.com/jik876/hifi-gan), and
[NSPP](https://github.com/YangAi520/NSPP).
The PCS implementation is from [PCS400](https://github.com/RoyChao19477/PCS/tree/main/PCS400).

The Apple Silicon (MPS) acceleration in `models/mps_kernels.py` is adapted from the Metal
selective-scan kernel of [SpeechLens](https://github.com/faraday/SpeechLens)
(Copyright 2026 Çağatay Çallı, Apache License 2.0); see [NOTICE](NOTICE).

## Citation

```bibtex
@inproceedings{chao2026rtsemamba,
  title={RT-SEMamba: Real-Time Speech Enhancement Mamba via Progressive Knowledge Distillation},
  author={Chao, Rong and Huang, Sung-Feng and La Quatra, Moreno and Siniscalchi, Sabato Marco and Cheng, Wen-Huang and Fu, Szu-Wei and Tsao, Yu},
  booktitle={Proc. Interspeech},
  year={2026}
}
```

If you also use the underlying SEMamba architecture, please cite:

```bibtex
@inproceedings{chao2024investigation,
  title={An Investigation of Incorporating Mamba for Speech Enhancement},
  author={Chao, Rong and Cheng, Wen-Huang and La Quatra, Moreno and Siniscalchi, Sabato Marco and Yang, Chao-Han Huck and Fu, Szu-Wei and Tsao, Yu},
  booktitle={Proc. IEEE SLT},
  year={2024}
}
```

## License

MIT — see [LICENSE](LICENSE).
