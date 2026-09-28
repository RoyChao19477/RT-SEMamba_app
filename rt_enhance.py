#!/usr/bin/env python3
"""
rt_enhance.py — Real-time RT-SEMamba speech enhancement with a desktop GUI.

    Mic ─► [audio process] ─► StreamingSTFT ─► RT-SEMamba (stateful, N frames / call) ─► overlap-add
                                                                                  │
                         noisy_<ts>.wav + enhanced_<ts>_<model>.wav  ◄────────────┤
                         (optional) live monitor to headphones        ◄───────────┘

* Stateful chunked streaming: every forward call processes N STFT frames and carries the
  Mamba conv/SSM states and the causal-conv caches into the next call, so the output is
  numerically identical to the offline (whole-utterance) forward pass for any N.
* Apple Silicon: the Mamba selective scan runs as one fused Metal kernel
  (models/mps_kernels.py, adapted from SpeechLens), ~4× faster per call than the
  pure-PyTorch reference scan, so KD1 runs in real time with 25 ms chunks.
* Audio capture and monitor playback run in their own process (utils/audio_worker.py),
  so GUI or model work cannot starve the audio callbacks.

Usage:
    python rt_enhance.py                                        # GUI
    python rt_enhance.py --model kd8                            # GUI, start with the KD8 teacher
    python rt_enhance.py --list-devices
    python rt_enhance.py --input-file noisy.wav --verify        # headless; checks vs offline forward
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import time
import queue
import threading
import argparse
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch
import yaml

from models.generator import SEMamba
from models.mps_kernels import accelerate
from models.stfts import mag_phase_stft, mag_phase_istft
from utils.audio_worker import AudioWorker
from utils.streaming import StreamingSTFT

SAMPLE_RATE   = 16_000
BLOCK_SAMPLES = 800         # block size used by the headless file mode


def select_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(config_path: str, ckpt_path: str, device: torch.device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model = SEMamba(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state.get("generator", state))
    model.eval()
    return model, cfg


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Incremental overlap-add (same window/normalisation as mag_phase_istft, center=False)
# ---------------------------------------------------------------------------
class OverlapAdd:
    """Emits each output sample as soon as every frame covering it has been added."""

    def __init__(self, n_fft: int, hop_size: int, win_size: int, compress_factor: float):
        self.n_fft, self.hop, self.win, self.compress = n_fft, hop_size, win_size, compress_factor
        window = (torch.hann_window(win_size) + 1e-3).numpy()
        self.window, self.window_sq = window, window * window
        self.acc = np.zeros(win_size, dtype=np.float64)
        self.wsum = np.zeros(win_size, dtype=np.float64)

    def add(self, mag: torch.Tensor, pha: torch.Tensor) -> np.ndarray:
        """mag/pha: (1, F, T). Returns T × hop completed samples."""
        mag = torch.pow(torch.clamp(mag, min=1e-10), 1.0 / self.compress)
        com = torch.polar(mag, pha).squeeze(0).T.cpu()           # (T, F)
        frames = torch.fft.irfft(com, n=self.n_fft, dim=-1)[:, : self.win].numpy() * self.window
        out = np.empty(len(frames) * self.hop, dtype=np.float32)
        for i, frame in enumerate(frames):
            self.acc += frame
            self.wsum += self.window_sq
            out[i * self.hop:(i + 1) * self.hop] = self.acc[: self.hop] / self.wsum[: self.hop]
            self.acc = np.concatenate([self.acc[self.hop:], np.zeros(self.hop)])
            self.wsum = np.concatenate([self.wsum[self.hop:], np.zeros(self.hop)])
        return out

    def flush(self) -> np.ndarray:
        n = self.win - self.hop
        out = (self.acc[:n] / np.maximum(self.wsum[:n], 1e-8)).astype(np.float32)
        self.acc[:] = 0.0
        self.wsum[:] = 0.0
        return out


# ---------------------------------------------------------------------------
# Stateful chunked enhancer
# ---------------------------------------------------------------------------
class StreamingEnhancer:
    def __init__(self, model, cfg, device: torch.device, chunk_frames: int = 4):
        self.model, self.device, self.chunk_frames = model, device, chunk_frames
        s = cfg["stft_cfg"]
        self.n_fft, self.hop, self.win = s["n_fft"], s["hop_size"], s["win_size"]
        self.compress = cfg["model_cfg"]["compress_factor"]
        self.model_ms = 0.0
        self.reset()

    def reset(self) -> None:
        self.model.set_streaming(True)           # also resets all streaming state
        # STFT runs on CPU: MPS FFTs are noticeably less precise, and this part is cheap.
        self.stft = StreamingSTFT(self.n_fft, self.hop, self.win, self.compress, torch.device("cpu"))
        self.ola = OverlapAdd(self.n_fft, self.hop, self.win, self.compress)
        self._pending_mag: list[torch.Tensor] = []
        self._pending_pha: list[torch.Tensor] = []
        self._pending = 0

    @torch.no_grad()
    def _run(self, mag: torch.Tensor, pha: torch.Tensor) -> np.ndarray:
        t0 = time.perf_counter()
        enh_mag, enh_pha, _ = self.model(mag, pha)
        out = self.ola.add(enh_mag, enh_pha)     # .cpu() inside syncs the device
        self.model_ms += (time.perf_counter() - t0) * 1000
        return out

    def _drain(self, force: bool) -> list[np.ndarray]:
        outs = []
        while self._pending >= self.chunk_frames or (force and self._pending > 0):
            mag = torch.cat(self._pending_mag, dim=-1)
            pha = torch.cat(self._pending_pha, dim=-1)
            n = min(self.chunk_frames, mag.shape[-1])
            outs.append(self._run(mag[..., :n], pha[..., :n]))
            self._pending_mag, self._pending_pha = [mag[..., n:]], [pha[..., n:]]
            self._pending -= n
        return outs

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Push raw samples; returns whatever enhanced samples became final."""
        out = self.stft.process(torch.from_numpy(samples))
        if out is not None:
            mag, pha, _ = out
            self._pending_mag.append(mag.to(self.device))
            self._pending_pha.append(pha.to(self.device))
            self._pending += mag.shape[-1]
        outs = self._drain(force=False)
        return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)

    def finish(self) -> np.ndarray:
        """Process leftover frames (< chunk) and flush the OLA tail.

        Like the offline center=False STFT, trailing samples that do not fill a whole
        window are not analysed.
        """
        outs = self._drain(force=True)
        outs.append(self.ola.flush())
        return np.concatenate(outs)


# ---------------------------------------------------------------------------
# File mode (for testing without a mic, and for --verify)
# ---------------------------------------------------------------------------
def run_file(enhancer: StreamingEnhancer, model, cfg, device, args, enhanced_path: str) -> None:
    audio, sr = sf.read(args.input_file, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        sys.exit(f"Input sample rate {sr} != {SAMPLE_RATE}")
    audio = audio[:, 0] * args.input_gain

    t0 = time.perf_counter()
    outs = [enhancer.process(audio[i:i + BLOCK_SAMPLES]) for i in range(0, len(audio), BLOCK_SAMPLES)]
    outs.append(enhancer.finish())
    elapsed = time.perf_counter() - t0
    enhanced = np.concatenate(outs)[: len(audio)]
    sf.write(enhanced_path, enhanced, SAMPLE_RATE, subtype="FLOAT")

    dur = len(audio) / SAMPLE_RATE
    print(f"Processed {dur:.2f} s in {elapsed:.2f} s  →  RTF {elapsed / dur:.3f} "
          f"(model {enhancer.model_ms / 1000 / dur:.3f})")
    print(f"Saved: {enhanced_path}")

    if args.verify:
        s = cfg["stft_cfg"]
        cf = cfg["model_cfg"]["compress_factor"]
        with torch.no_grad():
            model.set_streaming(False)
            x = torch.from_numpy(audio).unsqueeze(0)
            mag, pha, _ = mag_phase_stft(x, s["n_fft"], s["hop_size"], s["win_size"], cf)
            om, oph, _ = model(mag.to(device), pha.to(device))
            ref = mag_phase_istft(om.cpu(), oph.cpu(), s["n_fft"], s["hop_size"], s["win_size"], cf,
                                  length=len(audio)).squeeze(0).cpu().numpy()
        # Offline iSTFT leaves the un-analysed tail as zeros; compare the covered region only.
        n = (mag.shape[-1] - 1) * s["hop_size"] + s["win_size"]
        err = ref[:n] - enhanced[:n]
        snr = 10 * np.log10(np.sum(ref[:n] ** 2) / max(np.sum(err ** 2), 1e-20))
        print(f"Verify vs offline: max|diff| {np.abs(err).max():.2e}  SNR {snr:.1f} dB  "
              f"→ {'PASS' if snr > 60 else 'FAIL'}")


# key: (label, config, checkpoint, default chunk, minimum recommended chunk)
MODELS = {
    "kd1": ("KD1 · 1-layer student (fast)", "recipes/KD1/KD1.yaml", "ckpts/g_00766000.pth", 4, 2),
    "kd8": ("KD8 · 8-layer teacher (stronger)", "recipes/KD8/KD8.yaml", "ckpts/g_00259000.pth", 8, 8),
}
CHUNK_CHOICES  = [1, 2, 4, 8, 16, 32]
DISPLAY_SEC    = 6.0        # seconds of waveform shown in the GUI
MONITOR_MAX_MS = 120.0      # monitor buffer cap; older audio is skipped to bound latency


def build_model(config: str, ckpt: str, device: torch.device):
    model, cfg = load_model(config, ckpt, device)
    if device.type == "mps":
        if hasattr(torch.mps, "compile_shader"):
            accelerate(model)
        else:  # older PyTorch: fall back to the (slower, equally exact) pure-PyTorch scan
            print("torch.mps.compile_shader unavailable (PyTorch < 2.6?) — fused Metal scan disabled")
    return model, cfg


def warmup(model, cfg, device, chunk_frames: int) -> None:
    F = cfg["stft_cfg"]["n_fft"] // 2 + 1
    model.set_streaming(True)
    with torch.no_grad():
        for _ in range(5):
            model(torch.rand(1, F, chunk_frames, device=device), torch.rand(1, F, chunk_frames, device=device))
    sync(device)


class Ring:
    """Fixed-length history of the most recent samples, for display."""

    def __init__(self, n: int):
        self.data = np.zeros(n, dtype=np.float32)
        self.lock = threading.Lock()

    def push(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        with self.lock:
            x = x[-len(self.data):]
            self.data = np.roll(self.data, -len(x))
            self.data[-len(x):] = x

    def snapshot(self) -> np.ndarray:
        with self.lock:
            return self.data.copy()


# ---------------------------------------------------------------------------
# Real-time engine (no GUI code)
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, device: torch.device):
        self.device = device
        self._models: dict[str, tuple] = {}
        n = int(DISPLAY_SEC * SAMPLE_RATE)
        self.noisy_ring, self.enh_ring = Ring(n), Ring(n)
        self.running = False
        self.worker: AudioWorker | None = None
        self.paths: tuple[str, str] | None = None

    def get_model(self, key: str):
        if key not in self._models:
            _, config, ckpt, _, _ = MODELS[key]
            self._models[key] = build_model(config, ckpt, self.device)
        return self._models[key]

    def ensure_worker(self) -> None:
        if self.worker is None or not self.worker.proc.is_alive():
            self.worker = AudioWorker(SAMPLE_RATE)

    def start(self, model_key: str, chunk_frames: int, input_device, output_device, monitor: bool,
              out_dir: str, input_gain: float = 1.0) -> None:
        self.model, self.cfg = self.get_model(model_key)
        self.model_key = model_key
        self.hop = self.cfg["stft_cfg"]["hop_size"]
        self.win = self.cfg["stft_cfg"]["win_size"]
        warmup(self.model, self.cfg, self.device, chunk_frames)
        self.enhancer = StreamingEnhancer(self.model, self.cfg, self.device, chunk_frames=chunk_frames)
        self.chunk_frames, self.block = chunk_frames, chunk_frames * self.hop
        self.input_gain, self.monitor = input_gain, monitor
        self.processed = 0
        self.last_call_ms = 0.0
        self.noisy_ring.push(np.zeros(len(self.noisy_ring.data), np.float32))
        self.enh_ring.push(np.zeros(len(self.enh_ring.data), np.float32))

        self.ensure_worker()
        self.worker.start(input_device, output_device, monitor, self.block,
                          int(MONITOR_MAX_MS / 1000 * SAMPLE_RATE))

        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.paths = (os.path.join(out_dir, f"noisy_{stamp}.wav"),
                      os.path.join(out_dir, f"enhanced_{stamp}_{model_key}.wav"))
        self.noisy_f = sf.SoundFile(self.paths[0], "w", SAMPLE_RATE, 1, subtype="FLOAT")
        self.enh_f = sf.SoundFile(self.paths[1], "w", SAMPLE_RATE, 1, subtype="FLOAT")

        self.running = True
        self.t_start = time.perf_counter()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="enhance")
        self.thread.start()

    def _loop(self) -> None:
        while True:
            try:
                block = self.worker.data_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if block is None:  # capture stopped and fully drained
                return
            block = block * self.input_gain
            t0 = time.perf_counter()
            enhanced = self.enhancer.process(block)
            self.last_call_ms = (time.perf_counter() - t0) * 1000
            self.noisy_f.write(block)
            self.noisy_ring.push(block)
            if enhanced.size:
                self.enh_f.write(enhanced)
                self.enh_ring.push(enhanced)
                if self.monitor:
                    self.worker.monitor(enhanced)
            self.processed += len(block)

    def stop(self) -> tuple[str, str]:
        self.worker.stop_capture()
        self.thread.join()
        self.running = False
        self.worker.stop_monitor()
        tail = self.enhancer.finish()
        self.enh_f.write(tail)
        self.enh_ring.push(tail)
        self.noisy_f.close()
        self.enh_f.close()
        self.model.set_streaming(False)
        return self.paths

    def close(self) -> None:
        if self.running:
            self.stop()
        if self.worker is not None:
            self.worker.close()

    def stats(self) -> dict:
        audio_s = max(self.processed / SAMPLE_RATE, 1e-6)
        rtf = self.enhancer.model_ms / 1000 / audio_s
        block_ms = self.block / SAMPLE_RATE * 1000
        ola_ms = (self.win - self.hop) / SAMPLE_RATE * 1000
        return {
            "elapsed": time.perf_counter() - self.t_start,
            "rtf": rtf,
            "call_ms": self.last_call_ms,
            "backlog_ms": (self.worker.captured - self.processed) / SAMPLE_RATE * 1000,
            # Buffering inside this app only; the audio devices add their own latency on top.
            "latency_ms": block_ms + ola_ms + rtf * block_ms,
            "overflows": self.worker.overflows,
            "underruns": self.worker.underruns,
        }


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class App:
    BG, PANEL, FG, MUTED = "#15171c", "#1f2229", "#e8e8ea", "#8a8f99"
    NOISY, ENH, GOOD, BAD = "#e0a458", "#4fb3bf", "#6cc070", "#e06c6c"

    def __init__(self, engine: Engine, device: torch.device, model_key: str = "kd1"):
        import tkinter as tk
        from tkinter import ttk
        import sounddevice as sd

        self.tk, self.engine, self.device = tk, engine, device
        self.root = tk.Tk()
        self.root.title("RT-SEMamba")
        self.root.configure(bg=self.BG)
        self.root.minsize(820, 600)

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=self.BG, foreground=self.FG, fieldbackground=self.PANEL)
        style.configure("TLabel", background=self.BG, foreground=self.FG)
        style.configure("Muted.TLabel", foreground=self.MUTED)
        style.configure("Stat.TLabel", font=("Menlo", 13))
        style.configure("TCheckbutton", background=self.BG, foreground=self.FG)
        style.configure("TButton", padding=6)
        style.configure("Accent.TButton", padding=6, font=("Helvetica", 13, "bold"))

        devices = sd.query_devices()
        self.inputs = [(i, d["name"]) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
        self.outputs = [(i, d["name"]) for i, d in enumerate(devices) if d["max_output_channels"] > 0]
        default_in, default_out = sd.default.device

        # --- controls ---
        top = ttk.Frame(self.root, padding=(16, 12, 16, 4))
        top.pack(fill="x")
        ttk.Label(top, text="Model").grid(row=0, column=0, sticky="w")
        self.model_labels = {v[0]: k for k, v in MODELS.items()}
        self.model_var = tk.StringVar(value=MODELS[model_key][0])
        self.model_box = ttk.Combobox(top, textvariable=self.model_var, values=list(self.model_labels),
                                      state="readonly", width=34)
        self.model_box.grid(row=0, column=1, sticky="we", padx=(6, 16))
        self.model_box.bind("<<ComboboxSelected>>", lambda e: self.on_model_change())
        ttk.Label(top, text="Chunk").grid(row=0, column=2, sticky="w")
        self.chunk_var = tk.StringVar()
        self.chunk_box = ttk.Combobox(top, textvariable=self.chunk_var, state="readonly", width=26)
        self.chunk_box.grid(row=0, column=3, sticky="w", padx=(6, 0))

        ttk.Label(top, text="Mic").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.in_var = tk.StringVar(value=self._label(self.inputs, default_in))
        ttk.Combobox(top, textvariable=self.in_var, values=[self._fmt(d) for d in self.inputs],
                     state="readonly", width=34).grid(row=1, column=1, sticky="we", padx=(6, 16), pady=(8, 0))
        ttk.Label(top, text="Gain").grid(row=1, column=2, sticky="w", pady=(8, 0))
        self.gain_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(top, from_=0.1, to=10.0, increment=0.1, textvariable=self.gain_var, width=6
                    ).grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(8, 0))

        self.mon_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Monitor to", variable=self.mon_var).grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.out_var = tk.StringVar(value=self._label(self.outputs, default_out))
        ttk.Combobox(top, textvariable=self.out_var, values=[self._fmt(d) for d in self.outputs],
                     state="readonly", width=34).grid(row=2, column=1, sticky="we", padx=(6, 16), pady=(8, 0))
        ttk.Label(top, text="(use headphones)", style="Muted.TLabel").grid(row=2, column=2, columnspan=2,
                                                                         sticky="w", pady=(8, 0))
        top.columnconfigure(1, weight=1)

        btns = ttk.Frame(self.root, padding=(16, 8, 16, 4))
        btns.pack(fill="x")
        self.start_btn = ttk.Button(btns, text="●  Start", style="Accent.TButton", command=self.toggle)
        self.start_btn.pack(side="left")
        self.file_btn = ttk.Button(btns, text="Enhance file…", command=self.enhance_file)
        self.file_btn.pack(side="left", padx=(8, 0))
        self.play_noisy = ttk.Button(btns, text="▶ Noisy", command=lambda: self.play(0), state="disabled")
        self.play_enh = ttk.Button(btns, text="▶ Enhanced", command=lambda: self.play(1), state="disabled")
        self.stop_play = ttk.Button(btns, text="■", width=3, command=sd.stop, state="disabled")
        self.stop_play.pack(side="right")
        self.play_enh.pack(side="right", padx=(8, 4))
        self.play_noisy.pack(side="right")

        # --- stats ---
        stats = ttk.Frame(self.root, padding=(16, 6, 16, 2))
        stats.pack(fill="x")
        self.stat_vars = {}
        for col, (key, title) in enumerate([("time", "Recorded"), ("rtf", "Model RTF"), ("call", "Last call"),
                                            ("lat", "App latency"), ("backlog", "Backlog"),
                                            ("ovf", "Mic overflows"), ("und", "Monitor underruns")]):
            ttk.Label(stats, text=title, style="Muted.TLabel").grid(row=0, column=col, sticky="w", padx=(0, 18))
            v = tk.StringVar(value="—")
            lbl = ttk.Label(stats, textvariable=v, style="Stat.TLabel")
            lbl.grid(row=1, column=col, sticky="w", padx=(0, 18))
            self.stat_vars[key] = (v, lbl)

        # --- waveforms ---
        self.canvases = []
        for title, color in [("Noisy input", self.NOISY), ("Enhanced (stateful, = offline)", self.ENH)]:
            frame = ttk.Frame(self.root, padding=(16, 8, 16, 0))
            frame.pack(fill="both", expand=True)
            head = ttk.Frame(frame)
            head.pack(fill="x")
            ttk.Label(head, text=title).pack(side="left")
            level = tk.Canvas(head, width=160, height=10, bg=self.PANEL, highlightthickness=0)
            level.pack(side="right", pady=3)
            c = tk.Canvas(frame, bg=self.PANEL, highlightthickness=0, height=140)
            c.pack(fill="both", expand=True, pady=(4, 0))
            self.canvases.append((c, level, color))

        self.status = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status, style="Muted.TLabel", padding=(16, 8)).pack(fill="x")

        self.last_paths = None
        self.on_model_change()
        self.status.set(f"Ready · device {str(device).upper()} · fused Metal scan "
                        f"{'on' if device.type == 'mps' and hasattr(torch.mps, 'compile_shader') else 'off'} · audio I/O in separate process")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(50, self.refresh)
        # start the audio process in the background so the first Start is quick
        threading.Thread(target=self.engine.ensure_worker, daemon=True).start()

    # --- helpers ---
    @staticmethod
    def _fmt(d):
        return f"{d[0]}: {d[1]}"

    def _label(self, devs, idx):
        for d in devs:
            if d[0] == idx:
                return self._fmt(d)
        return self._fmt(devs[0]) if devs else ""

    @staticmethod
    def _dev_index(label):
        return int(label.split(":", 1)[0]) if label else None

    def _model_key(self):
        return self.model_labels[self.model_var.get()]

    def _chunk_label(self, c):
        min_ok = MODELS[self._model_key()][4]
        note = "  (too slow)" if c < min_ok else ""
        return f"{c} frame{'s' if c > 1 else ''} ({c * 6.25:g} ms){note}"

    def _chunk_value(self):
        return int(self.chunk_var.get().split(" ", 1)[0])

    def on_model_change(self):
        key = self._model_key()
        self.chunk_box.configure(values=[self._chunk_label(c) for c in CHUNK_CHOICES])
        self.chunk_var.set(self._chunk_label(MODELS[key][3]))

    def _set_busy(self, busy: bool):
        state = "disabled" if busy else "readonly"
        self.model_box.configure(state=state)
        self.chunk_box.configure(state=state)
        self.file_btn.configure(state="disabled" if busy else "normal")
        for b in (self.play_noisy, self.play_enh, self.stop_play):
            b.configure(state="disabled" if busy or not self.last_paths else "normal")

    # --- actions ---
    def toggle(self):
        if self.engine.running:
            self.start_btn.configure(state="disabled")
            self.status.set("Stopping … finishing queued audio")
            self.root.update_idletasks()
            s = self.engine.stats()
            self.last_paths = self.engine.stop()
            self.start_btn.configure(text="●  Start", state="normal")
            self._set_busy(False)
            self.status.set(f"Saved {self.last_paths[0]} · {os.path.basename(self.last_paths[1])} "
                            f"· mic overflows {s['overflows']}")
            return
        key, chunk = self._model_key(), self._chunk_value()
        self.status.set(f"Loading {MODELS[key][0]} and warming up …")
        self.root.update_idletasks()
        try:
            self.engine.start(key, chunk, self._dev_index(self.in_var.get()), self._dev_index(self.out_var.get()),
                              self.mon_var.get(), "recordings", float(self.gain_var.get()))
        except Exception as e:  # device errors etc.
            self.status.set(f"Could not start: {e}")
            return
        self.start_btn.configure(text="■  Stop")
        self._set_busy(True)
        mon = " · monitoring" if self.mon_var.get() else ""
        self.status.set(f"Recording · {MODELS[key][0]} · chunk {chunk} frames ({chunk * 6.25:g} ms){mon}")

    def play(self, which: int):
        import sounddevice as sd
        if not self.last_paths:
            return
        data, sr = sf.read(self.last_paths[which], dtype="float32")
        sd.play(data, sr, device=self._dev_index(self.out_var.get()))

    def enhance_file(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("WAV", "*.wav"), ("All", "*.*")])
        if not path:
            return
        key = self._model_key()
        self._set_busy(True)
        self.start_btn.configure(state="disabled")
        self.status.set(f"Enhancing {os.path.basename(path)} with {MODELS[key][0]} …")
        gain = float(self.gain_var.get())

        def work():
            try:
                audio, sr = sf.read(path, dtype="float32", always_2d=True)
                if sr != SAMPLE_RATE:
                    raise ValueError(f"needs {SAMPLE_RATE} Hz audio, got {sr} Hz")
                audio = audio[:, 0] * gain
                eng = self.engine
                model, cfg = eng.get_model(key)
                enh = StreamingEnhancer(model, cfg, eng.device, chunk_frames=32)
                step = 32 * cfg["stft_cfg"]["hop_size"]
                t0 = time.perf_counter()
                outs = []
                for i in range(0, len(audio), step):
                    outs.append(enh.process(audio[i:i + step]))
                    eng.noisy_ring.push(audio[i:i + step])
                    eng.enh_ring.push(outs[-1])
                outs.append(enh.finish())
                out = np.concatenate(outs)[: len(audio)]
                model.set_streaming(False)
                os.makedirs("recordings", exist_ok=True)
                dst = os.path.join("recordings",
                                   os.path.splitext(os.path.basename(path))[0] + f"_enhanced_{key}.wav")
                sf.write(dst, out, SAMPLE_RATE, subtype="FLOAT")
                el = time.perf_counter() - t0
                dur = len(audio) / SAMPLE_RATE
                self.last_paths = (path, dst)
                msg = f"Saved {dst} · {dur:.1f} s in {el:.2f} s (RTF {el / dur:.3f})"
            except Exception as e:
                msg = f"Failed: {e}"
            self.root.after(0, lambda: (self.status.set(msg), self._set_busy(False),
                                        self.start_btn.configure(state="normal")))

        threading.Thread(target=work, daemon=True).start()

    # --- drawing ---
    def refresh(self):
        eng = self.engine
        if eng.running:
            s = eng.stats()
            self._stat("time", f"{s['elapsed']:6.1f} s")
            self._stat("rtf", f"{s['rtf']:.3f}", self.GOOD if s["rtf"] < 0.8 else self.BAD)
            self._stat("call", f"{s['call_ms']:5.1f} ms")
            self._stat("lat", f"≈{s['latency_ms']:4.0f} ms")
            self._stat("backlog", f"{s['backlog_ms']:4.0f} ms", self.GOOD if s["backlog_ms"] < 200 else self.BAD)
            self._stat("ovf", f"{s['overflows']}", self.GOOD if s["overflows"] == 0 else self.BAD)
            self._stat("und", f"{s['underruns']}" if eng.monitor else "—",
                       self.GOOD if s["underruns"] == 0 else self.BAD)
        for (canvas, level, color), ring in zip(self.canvases, (eng.noisy_ring, eng.enh_ring)):
            data = ring.snapshot()
            self._draw_wave(canvas, data, color)
            self._draw_level(level, data[-1600:], color)
        self.root.after(50, self.refresh)

    def _stat(self, key, text, color=None):
        var, lbl = self.stat_vars[key]
        var.set(text)
        lbl.configure(foreground=color or self.FG)

    def _draw_wave(self, c, data, color):
        w, h = max(c.winfo_width(), 10), max(c.winfo_height(), 10)
        c.delete("all")
        mid = h / 2
        c.create_line(0, mid, w, mid, fill="#2c3038")
        for sec in range(1, int(DISPLAY_SEC)):
            x = w * (1 - sec / DISPLAY_SEC)
            c.create_line(x, 0, x, h, fill="#262a31")
            c.create_text(x + 3, h - 8, text=f"-{sec}s", fill="#555b66", anchor="w", font=("Helvetica", 9))
        cols = data[: len(data) // w * w].reshape(w, -1)
        peak = max(float(np.abs(data).max()), 0.05)
        top = mid - cols.max(axis=1) / peak * (mid - 4)
        bot = mid - cols.min(axis=1) / peak * (mid - 4)
        pts = np.empty((w, 4))
        pts[:, 0] = pts[:, 2] = np.arange(w)
        pts[:, 1], pts[:, 3] = top, bot
        c.create_line(*pts.ravel().tolist(), fill=color, width=1)

    def _draw_level(self, c, data, color):
        c.delete("all")
        rms = float(np.sqrt(np.mean(data ** 2))) if data.size else 0.0
        db = 20 * np.log10(max(rms, 1e-6))
        frac = min(max((db + 60) / 60, 0), 1)
        c.create_rectangle(0, 0, 160 * frac, 10, fill=color, width=0)

    def on_close(self):
        self.engine.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="RT-SEMamba real-time enhancement",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--model", choices=list(MODELS), default="kd1")
    parser.add_argument("--device", default=None, help="mps / cuda / cpu (default: auto)")
    parser.add_argument("--chunk-frames", type=int, default=None, help="headless file mode only")
    parser.add_argument("--input-file", default=None, help="headless: enhance a wav file instead of the GUI")
    parser.add_argument("--verify", action="store_true", help="with --input-file: compare to offline")
    parser.add_argument("--input-gain", type=float, default=1.0)
    parser.add_argument("--out-dir", default="recordings")
    parser.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    device = select_device(args.device)
    engine = Engine(device)

    if args.input_file:
        _, _, _, default_chunk, _ = MODELS[args.model]
        chunk = args.chunk_frames or default_chunk
        print(f"Torch device: {device} · model {MODELS[args.model][0]} · chunk {chunk}", flush=True)
        model, cfg = engine.get_model(args.model)
        warmup(model, cfg, device, chunk)
        enhancer = StreamingEnhancer(model, cfg, device, chunk_frames=chunk)
        os.makedirs(args.out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.input_file))[0]
        run_file(enhancer, model, cfg, device, args,
                 os.path.join(args.out_dir, f"{base}_enhanced_{args.model}.wav"))
        return

    print(f"Torch device: {device} · loading {MODELS[args.model][0]} …", flush=True)
    engine.get_model(args.model)
    App(engine, device, args.model).run()


if __name__ == "__main__":
    main()
