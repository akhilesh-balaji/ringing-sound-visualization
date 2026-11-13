"""Usage:
    python viz.py live   [--palette macchiato] [--ringing 0.7] [--sensitivity 2] ...
    python viz.py render song.mp3 -o out.mp4 [--duration 25] [--fps 60] [...]

Every VizParams field is a flag: --ripple-scale, --gain, --floor, --post-sigma, --max-side, ...
"""
import argparse
import subprocess
import sys
import time
from dataclasses import fields

import cv2
import numpy as np

from waveviz import Analyzer, VizParams, Visualizer


def add_param_flags(ap):
    for f in fields(VizParams):
        d, flag = f.default, "--" + f.name.replace("_", "-")
        if isinstance(d, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=d)
        else:
            ap.add_argument(flag, type=type(d), default=d)


def params_from(args):
    return VizParams(**{f.name: getattr(args, f.name) for f in fields(VizParams)})


# ------------------------------------------------------------------ live
def run_live(p, device=None):
    import pyaudio
    sr = 44100
    ring = np.zeros(p.win, np.float32)

    def cb(in_data, frame_count, time_info, status):
        nonlocal ring
        x = np.frombuffer(in_data, np.float32)
        ring = np.concatenate((ring[len(x):], x))        # new array -> atomic swap
        return None, pyaudio.paContinue

    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=sr, input=True,
                     input_device_index=device, frames_per_buffer=512, stream_callback=cb)
    stream.start_stream()
    viz, an = Visualizer(p), Analyzer(p, sr)
    cv2.namedWindow("viz", cv2.WINDOW_AUTOSIZE)
    period = 1.0 / p.fps
    last = time.perf_counter()
    try:
        while True:
            t0 = time.perf_counter()
            dt, last = t0 - last, t0                      # real frame time keeps envelopes correct if fps drops
            cv2.imshow("viz", viz.render(an.update(ring, dt)))
            wait = max(1, int((period - (time.perf_counter() - t0)) * 1000))
            if cv2.waitKey(wait) & 0xFF in (27, ord("q")):
                break
    finally:
        stream.stop_stream(); stream.close(); pa.terminate()
        cv2.destroyAllWindows()


# ------------------------------------------------------------------ offline (render + mux in one pass)
def run_render(p, audio, out, duration=None, start=0.0):
    import librosa
    x, sr = librosa.load(audio, sr=None, mono=True, offset=start, duration=duration)
    x = x.astype(np.float32)
    viz, an = Visualizer(p), Analyzer(p, sr)
    W, H = viz.size
    n = int(len(x) / sr * p.fps)
    xp = np.concatenate((np.zeros(p.win, np.float32), x))   # so early frames have a full window

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(p.fps), "-i", "-"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", audio, "-map", "0:v", "-map", "1:a"]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    t0 = time.perf_counter()
    try:
        for i in range(n):
            end = p.win + int(round((i + 1) * sr / p.fps))
            end = min(end, len(xp))
            ff.stdin.write(viz.render(an.update(xp[end - p.win:end])).tobytes())
            if i % 60 == 0:
                el = time.perf_counter() - t0
                print(f"\r{i / n * 100:5.1f}%  {i / max(el, 1e-9):5.1f} fps", end="", file=sys.stderr)
    finally:
        ff.stdin.close()
        ff.wait()
    print(f"\rdone -> {out}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    a = sub.add_parser("live")
    a.add_argument("--device", type=int, default=None, help="PyAudio input device index")
    add_param_flags(a)

    b = sub.add_parser("render")
    b.add_argument("audio")
    b.add_argument("-o", "--out", default="output.mp4")
    b.add_argument("--duration", type=float, default=None)
    b.add_argument("--start", type=float, default=0.0)
    add_param_flags(b)

    args = ap.parse_args()
    params = params_from(args)
    if args.mode == "live":
        run_live(params, args.device)
    else:
        run_render(params, args.audio, args.out, args.duration, args.start)
