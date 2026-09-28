"""Audio capture / monitor playback in a separate process.

PortAudio callbacks need the Python GIL. When capture shares a process with the
model and the GUI, any stall of the main process (GUI drawing, window dragging,
Python-side model code) delays the input callback and CoreAudio drops samples
(input overflow), which is heard as clicks. Running the streams in their own
process gives the callbacks their own interpreter and GIL.

This module only imports numpy and sounddevice so the child starts quickly.
"""

import multiprocessing as mp
import queue
import threading

import numpy as np

_STOP = "stop"
_QUIT = "quit"


def _worker(cmd_q, data_q, mon_q, counters, sample_rate):
    import sounddevice as sd

    overflows, underruns, captured, skipped = counters
    in_stream = out_stream = None
    mon_buf = np.zeros(0, dtype=np.float32)
    mon_lock = threading.Lock()
    mon_cap = 0
    mon_run = threading.Event()

    def in_cb(indata, frames, time_info, status):
        if status.input_overflow:
            with overflows.get_lock():
                overflows.value += 1
        data_q.put(indata[:, 0].copy())
        with captured.get_lock():
            captured.value += frames

    def out_cb(outdata, frames, time_info, status):
        nonlocal mon_buf
        with mon_lock:
            k = min(frames, len(mon_buf))
            outdata[:k, 0] = mon_buf[:k]
            outdata[k:, 0] = 0.0
            mon_buf = mon_buf[k:]
        if k < frames and captured.value > sample_rate // 2:
            with underruns.get_lock():
                underruns.value += 1

    def mon_feeder():
        nonlocal mon_buf
        while mon_run.is_set():
            try:
                x = mon_q.get(timeout=0.05)
            except queue.Empty:
                continue
            with mon_lock:
                mon_buf = np.concatenate([mon_buf, x])
                drop = len(mon_buf) - mon_cap
                if drop > 0:  # bound monitor latency: skip the oldest audio
                    mon_buf = mon_buf[drop:]
                    with skipped.get_lock():
                        skipped.value += drop

    feeder = None
    while True:
        cmd = cmd_q.get()
        if cmd[0] == "start":
            _, in_dev, out_dev, monitor, block, cap = cmd
            try:
                if monitor:
                    mon_cap = cap
                    mon_buf = np.zeros(0, dtype=np.float32)
                    mon_run.set()
                    feeder = threading.Thread(target=mon_feeder, daemon=True)
                    feeder.start()
                    out_stream = sd.OutputStream(samplerate=sample_rate, blocksize=block, channels=1,
                                                 dtype="float32", device=out_dev, callback=out_cb,
                                                 latency="low")
                    out_stream.start()
                # 'high' input latency = a larger host buffer, more tolerance to scheduling hiccups.
                in_stream = sd.InputStream(samplerate=sample_rate, blocksize=block, channels=1,
                                           dtype="float32", device=in_dev, callback=in_cb, latency="high")
                in_stream.start()
                data_q.put(("started", None))
            except Exception as e:  # report device errors to the parent
                data_q.put(("error", str(e)))
        elif cmd[0] == _STOP:
            if in_stream is not None:
                in_stream.stop()
                in_stream.close()
                in_stream = None
            data_q.put(None)  # end-of-capture marker
        elif cmd[0] == "stop_monitor":
            mon_run.clear()
            if feeder is not None:
                feeder.join()
                feeder = None
            if out_stream is not None:
                out_stream.stop()
                out_stream.close()
                out_stream = None
        elif cmd[0] == _QUIT:
            return


class AudioWorker:
    """Parent-side handle. Input blocks arrive on `data_q` as float32 arrays."""

    def __init__(self, sample_rate: int):
        ctx = mp.get_context("spawn")
        self.cmd_q, self.data_q, self.mon_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
        self.counters = tuple(ctx.Value("q", 0) for _ in range(4))
        self.proc = ctx.Process(target=_worker, daemon=True,
                                args=(self.cmd_q, self.data_q, self.mon_q, self.counters, sample_rate))
        self.proc.start()

    def start(self, input_device, output_device, monitor: bool, block: int, monitor_cap: int) -> None:
        for c in self.counters:
            c.value = 0
        self.cmd_q.put(("start", input_device, output_device, monitor, block, monitor_cap))
        kind, msg = self.data_q.get(timeout=10)
        if kind == "error":
            self.cmd_q.put(("stop_monitor",))
            raise RuntimeError(msg)

    def stop_capture(self) -> None:
        self.cmd_q.put((_STOP,))

    def stop_monitor(self) -> None:
        self.cmd_q.put(("stop_monitor",))

    def monitor(self, samples: np.ndarray) -> None:
        self.mon_q.put(samples)

    @property
    def overflows(self): return self.counters[0].value
    @property
    def underruns(self): return self.counters[1].value
    @property
    def captured(self): return self.counters[2].value
    @property
    def monitor_skipped(self): return self.counters[3].value

    def close(self) -> None:
        if self.proc.is_alive():
            self.cmd_q.put((_QUIT,))
            self.proc.join(timeout=2)
            if self.proc.is_alive():
                self.proc.terminate()
