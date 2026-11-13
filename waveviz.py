"""waveviz: audio-reactive 'standing wave' visualizer via regularized deconvolution.

    from waveviz import VizParams, Analyzer, Visualizer
    p   = VizParams(img_path="rect5.png", palette="macchiato", ringing=0.7)
    an  = Analyzer(p, sr)             # audio -> State(log_f, loud, low, mid, high)
    viz = Visualizer(p)               # State -> BGR uint8 frame
    state = an.update(last_win_samples, dt)
    frame = viz.render(state)

Filter:  H(f) = K / (K^2 + eps) * post,   K = exp(-2 pi^2 sigma^2 |f|^2)
"""
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np
import scipy.fft as sfft
from PIL import Image

# ------------------------------------------------------------------ palettes
PALETTES = {
    "macchiato": [(0.00, "#24273a"), (0.25, "#494d64"), (0.50, "#8aadf4"),
                  (0.70, "#7dc4e4"), (0.85, "#8bd5ca"), (1.00, "#a6da95")],
    "macchiato-glow": [(0.00, "#24273a"), (0.22, "#494d64"), (0.45, "#8aadf4"),
                       (0.62, "#7dc4e4"), (0.78, "#8bd5ca"), (0.90, "#a6da95"),
                       (1.00, "#cad3f5")],
    "macchiato-seagreen": [
        (0.0, "#1e2030"),
        (0.047, "#24273a"),
        (0.17, "#1a3b51"),
        (0.3, "#0a5559"),
        (0.43, "#1f7059"),
        (0.56, "#44885e"),
        (0.68, "#309b96"),
        (0.79, "#57a5d0"),
        (0.88, "#8aadf4"),
        (1.0, "#b7bdf8")],
    "macchiato-seagreen-light": [      # as seagreen, but passes through the pastel Green -> Teal -> Sapphire -> Blue
        (0.000, "#1e2030"), (0.047, "#24273a"), (0.150, "#1a3b51"), (0.270, "#0a5559"),
        (0.380, "#1f7059"), (0.480, "#44885e"), (0.550, "#6fb283"), (0.620, "#a6da95"),
        (0.720, "#8bd5ca"), (0.800, "#7dc4e4"), (0.880, "#8aadf4"), (1.000, "#b7bdf8")],
    "macchiato-viridis-cool": [      # viridis-like sweep in Macchiato pastels: Base > blue > teal > green > mint
        (0.000, "#1e2030"), (0.047, "#24273a"), (0.080, "#282d46"), (0.151, "#313c60"),
        (0.222, "#374b7c"), (0.292, "#385b8d"), (0.363, "#376b9d"), (0.434, "#327da9"),
        (0.505, "#2d90af"), (0.575, "#30a2b3"), (0.646, "#3db3b4"), (0.717, "#53c4b2"),
        (0.788, "#6dd1ad"), (0.858, "#8bdaa4"), (0.929, "#a8dc9a"), (1.000, "#c3e8b4")],
    "macchiato-viridis": [      # same, but the top end lifts from Green to Macchiato Yellow like viridis
        (0.000, "#1e2030"), (0.047, "#24273a"), (0.080, "#282d46"), (0.151, "#313c60"),
        (0.222, "#374b7c"), (0.292, "#385b8d"), (0.363, "#376b9d"), (0.434, "#327da9"),
        (0.505, "#2d90af"), (0.575, "#30a2b3"), (0.646, "#3db3b4"), (0.717, "#53c4b2"),
        (0.788, "#6dd1ad"), (0.858, "#8bdaa4"), (0.929, "#b1d992"), (1.000, "#eed49f")],
    "aqua": [(0.00, "#020b1f"), (0.25, "#05345c"), (0.50, "#0a7a8c"),
             (0.75, "#2fd3c0"), (1.00, "#e6fff4")],
}


def _hex(h):
    h = h.strip().lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
_M1i, _M2i = np.linalg.inv(_M1), np.linalg.inv(_M2)


def _to_lch(rgb):
    """sRGB (N,3) in 0..1 -> OKLCH (L, C, H in degrees)."""
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    lab = np.cbrt(lin @ _M1.T) @ _M2.T
    return lab[:, 0], np.hypot(lab[:, 1], lab[:, 2]), np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))


def _from_lch(L, C, H):
    """OKLCH -> sRGB (N,3) in 0..1; out-of-gamut colors keep L and H and lose chroma."""
    for _ in range(40):
        lab = np.stack([L, C * np.cos(np.radians(H)), C * np.sin(np.radians(H))], 1)
        lin = ((lab @ _M2i.T) ** 3) @ _M1i.T
        bad = ((lin < -1e-4) | (lin > 1 + 1e-4)).any(1)
        if not bad.any():
            break
        C = np.where(bad, C * 0.95, C)
    s = np.clip(lin, 0, 1)
    return np.where(s <= 0.0031308, 12.92 * s, 1.055 * s ** (1 / 2.4) - 0.055)


def make_lut(spec="macchiato", reverse=False, hue_rotate=0.0, chroma_scale=1.0,
             pin_base="", pin_width=0.3):
    """spec: preset name | matplotlib cmap name | 'hex,hex,...' | [hex,...] | [(stop, hex),...]
    hue_rotate: degrees, rotates hue in OKLCH (perceptually even; lightness is preserved).
    chroma_scale: multiplies colorfulness (<1 mutes, >1 boosts).
    pin_base: hex; the dark end is blended to exactly this color over the first `pin_width` of the ramp.
    Returns a (256, 3) uint8 BGR lookup table."""
    xs = np.linspace(0, 1, 256)
    if isinstance(spec, str) and "," in spec:
        spec = [s for s in spec.split(",") if s.strip()]
    if isinstance(spec, str) and spec in PALETTES:
        spec = PALETTES[spec]
    if isinstance(spec, str):                                   # matplotlib colormap
        import matplotlib
        rgb = matplotlib.colormaps[spec](xs)[:, :3] * 255
    else:
        spec = list(spec)
        if isinstance(spec[0], str):
            stops, cols = np.linspace(0, 1, len(spec)), [_hex(c) for c in spec]
        else:
            stops, cols = [s for s, _ in spec], [_hex(c) for _, c in spec]
        cols = np.array(cols, float)
        rgb = np.stack([np.interp(xs, stops, cols[:, i]) for i in range(3)], 1)
    if reverse:
        rgb = rgb[::-1]
    if hue_rotate or chroma_scale != 1.0:
        L, C, H = _to_lch(rgb / 255.0)
        rgb = _from_lch(L, C * chroma_scale, H + hue_rotate) * 255.0
    if pin_base:
        w = np.clip(xs / max(pin_width, 1e-6), 0, 1)
        w = (w * w * (3 - 2 * w))[:, None]                       # smoothstep: pure pin at 0, pure ramp at pin_width
        rgb = np.array(_hex(pin_base), float) * (1 - w) + rgb * w
    return np.ascontiguousarray(np.clip(rgb, 0, 255)[:, ::-1].astype(np.uint8))


# ------------------------------------------------------------------ params
@dataclass
class VizParams:
    # image / output
    img_path: str = "./rect5.png"
    max_side: int = 1024         # processing resolution (long side); never upscaled
    display_scale: float = 1.0   # output size = processed size * this
    pad: int = 24                # reflect padding against FFT wrap-around
    ref_width: float = 512.0     # sigma/post_sigma are specified at this width
    # color
    color_mode: str = "palette"  # 'palette': luminance -> palette | 'rgb': deconvolve R,G,B separately, keep image colors
    rgb_gain: float = 2.0        # rgb mode: soft-clip strength for overshoot (endpoints 0 and 1 are preserved)
    palette: str = "macchiato"
    reverse_palette: bool = False
    hue_rotate: float = 0.0      # degrees, OKLCH hue rotation of the palette (viridis -> Macchiato Base is about -41)
    chroma_scale: float = 1.0    # palette colorfulness multiplier
    pin_base: str = ""           # hex: force the dark end to exactly this color (e.g. '#24273a')
    pin_width: float = 0.3       # fraction of the ramp over which the pin fades out
    bg: str = "auto"             # image background level 0..1, or 'auto' (median)
    gain: float = 2.5            # contrast of features and ripples
    floor: float = 1.5           # how deep the background sits (1 lifted .. 2.5 pure base)
    # ringing / ripple character
    ripple_scale: float = 300.0  # sigma = ripple_scale / f_ref; smaller = finer ripples
    ringing: float = 0.5         # 0 smooth .. 1 strong; eps *= 10**(1 - 2*ringing)
    eps_quiet: float = 1e-2      # regularization at silence
    eps_loud: float = 1e-3       # regularization at full loudness
    post_sigma: float = 0.45     # lowpass on the filter; larger kills finest ripples
    sigma_min: float = 0.6
    sigma_max: float = 8.0
    # loudness mapping (dB domain, optional auto-gain)
    sensitivity: float = 1.0     # multiplies normalized loudness / highs (>1 reacts earlier)
    level_curve: float = 0.6     # exponent on normalized loudness (<1 = more responsive to quiet)
    agc: bool = True             # normalize to the recent peak so any input level works
    agc_decay: float = 8.0       # seconds for the peak reference to relax downward
    agc_min_db: float = -50.0    # AGC never amplifies below this level (silence stays silent)
    ref_db: float = -26.0        # loudness treated as 'full' when agc is off
    dyn_db: float = 40.0         # dB range mapped to 0..1
    # frequency mapping
    low_hz: float = 250.0        # low band: 40..low_hz
    high_hz: float = 2500.0      # high band: high_hz..high_max
    high_max: float = 12000.0
    tilt: float = 4.0            # dB/octave boost so highs (naturally weaker) can reach full scale
    centroid_scale: float = 0.3  # spectral centroid is ~3-4x the pitch; this puts them on one scale
    pitch_weight: float = 0.6    # 0 = brightness (centroid) only, 1 = pitch only (when voiced)
    freq_tau: float = 0.08       # seconds to follow frequency changes at 200 Hz; faster for higher f
    hf_fine: float = 0.4         # highs make ripples finer: sigma /= 1 + hf_fine * highs
    hf_ringing: float = 0.8      # highs add ringing: eps *= 10**(-hf_ringing * highs)  (decades)
    hf_speed: float = 4.0        # highs react this many times faster than the loudness envelope
    bass_punch: float = 0.5      # bass pulses contrast: gain *= 1 + bass_punch * low
    tau_attack: float = 0.010    # seconds, envelope attack (loudness; bass is 2x slower, highs hf_speed x faster)
    tau_release: float = 0.150   # seconds, envelope release
    # analysis
    fps: float = 60.0
    win: int = 4096              # pitch analysis window (samples)
    band_win: int = 1024         # band/centroid window (samples); shorter = more 'now'
    fmin: float = 65.4
    fmax: float = 2093.0
    conf: float = 0.5            # min pitch confidence
    gate: float = 3e-3           # rms below this: hold frequency
    median_len: int = 5


# ------------------------------------------------------------------ audio -> state
class State(NamedTuple):
    log_f: float   # smoothed log reference frequency (pitch blended with brightness)
    loud: float    # 0..1 loudness (dB, normalized)
    low: float     # 0..1 low-band energy (slow)
    mid: float     # 0..1 mid-band energy
    high: float    # 0..1 high-band energy (fast)


class Analyzer:
    """Audio window -> State: loudness, three band envelopes, and a reference frequency."""

    def __init__(self, p: VizParams, sr: int):
        if p.band_win > p.win:
            raise ValueError("band_win must be <= win")
        self.p, self.sr, self.win = p, sr, p.win
        self.hann = np.hanning(self.win).astype(np.float32)
        self._wac = sfft.irfft(np.abs(sfft.rfft(self.hann, 2 * self.win)) ** 2)[:self.win] + 1e-9
        self.lo, self.hi = int(sr / p.fmax), min(int(sr / p.fmin), self.win - 2)

        bw = p.band_win
        self.hann_b = np.hanning(bw).astype(np.float32)
        self.scale_b = 2.0 / float(self.hann_b.sum())               # full-scale sine -> amplitude 1
        fr = sfft.rfftfreq(bw, 1.0 / sr)
        edges = (40.0, p.low_hz, p.high_hz, min(p.high_max, sr / 2.0 - 1))
        self.masks = [(fr >= edges[i]) & (fr < edges[i + 1]) for i in range(3)]
        centers = np.sqrt([edges[i] * edges[i + 1] for i in range(3)])
        self.offs = p.tilt * np.log2(centers / centers[0])          # spectral tilt compensation, dB
        self.cmask = (fr >= 80.0) & (fr <= min(8000.0, sr / 2.0))
        self.clogf = np.log(fr[self.cmask])

        self.speed = np.array([1.0, 0.5, 1.0, p.hf_speed])          # loud, low, mid, high
        self.hist = deque(maxlen=p.median_len)
        self.log_f = float(np.log(200.0))
        self.env = np.zeros(4)
        self.ceil = np.array([p.ref_db, p.ref_db], float)           # AGC reference [total, bands]; starts sane, not at the first sound

    def _pitch(self, x):
        X = sfft.rfft((x - x.mean()) * self.hann, 2 * self.win)
        ac = sfft.irfft(X.real ** 2 + X.imag ** 2)[:self.win] / self._wac
        ac /= ac[0] + 1e-12
        seg = ac[self.lo:self.hi]
        m = seg.max()
        pk = np.flatnonzero((seg[1:-1] >= seg[:-2]) & (seg[1:-1] > seg[2:]) & (seg[1:-1] >= 0.85 * m))
        k = float(pk[0] + 1) if len(pk) else float(seg.argmax())   # first strong peak: no octave drops
        ki = int(k)
        if 0 < ki < len(seg) - 1:
            a, b, c = seg[ki - 1], seg[ki], seg[ki + 1]
            d = a - 2 * b + c
            if d != 0:
                k = ki + 0.5 * (a - c) / d
        return self.sr / (k + self.lo), m

    def update(self, x, dt=None):
        """x: most recent >= win samples (float32). dt: seconds since last call (default 1/fps)."""
        p = self.p
        dt = float(np.clip(dt if dt else 1.0 / p.fps, 1e-3, 0.25))
        x = x[-self.win:]
        rms = float(np.sqrt(np.mean(x[-1024:] ** 2)))

        Xb = sfft.rfft(x[-p.band_win:] * self.hann_b) * self.scale_b
        P = Xb.real ** 2 + Xb.imag ** 2
        db = np.empty(4)
        db[0] = 10 * np.log10(2 * rms * rms + 1e-12)                # total, sine-equivalent power
        for i, m in enumerate(self.masks):
            db[i + 1] = 10 * np.log10(P[m].sum() + 1e-12) + self.offs[i]

        # --- dB -> 0..1 with a slowly relaxing peak reference (shared by the 3 bands so balance is kept)
        if p.agc:
            relax = 1 - np.exp(-dt / p.agc_decay)
            for j, pk in enumerate((db[0], db[1:].max())):
                c = self.ceil[j]
                self.ceil[j] = pk if pk > c else c + (pk - c) * relax
            cl = np.maximum(self.ceil, p.agc_min_db)
            ceil = np.array([cl[0], cl[1], cl[1], cl[1]])
        else:
            ceil = np.full(4, p.ref_db)
        norm = np.clip((db - (ceil - p.dyn_db)) / p.dyn_db, 0.0, 1.0)

        # --- envelopes: fast attack / slower release, with the highs much quicker than the lows
        k = dt * self.speed
        a = np.where(norm > self.env, 1 - np.exp(-k / p.tau_attack), 1 - np.exp(-k / p.tau_release))
        self.env += (norm - self.env) * a

        # --- reference frequency: pitch blended with spectral centroid (so noise/percussion still moves it)
        if rms > p.gate:
            mag = np.sqrt(P[self.cmask])
            tot = float(mag.sum())
            if tot > 1e-9:
                logfc = float((mag * self.clogf).sum() / tot) + float(np.log(p.centroid_scale))
                f0, c = self._pitch(x)
                tgt = p.pitch_weight * np.log(f0) + (1 - p.pitch_weight) * logfc if c > p.conf else logfc
                self.hist.append(float(tgt))
        if self.hist:
            med = float(np.median(self.hist))
            tau = float(np.clip(p.freq_tau * (200.0 / np.exp(self.log_f)) ** 0.5, 0.01, 0.4))
            self.log_f += (med - self.log_f) * (1 - np.exp(-dt / tau))   # higher frequency -> follows faster

        return State(self.log_f, *map(float, self.env))


# ------------------------------------------------------------------ state -> frame
class Visualizer:
    def __init__(self, p: VizParams):
        self.p = p
        if p.color_mode not in ("palette", "rgb"):
            raise ValueError("color_mode must be 'palette' or 'rgb'")
        self.rgb = p.color_mode == "rgb"
        im = Image.open(p.img_path)
        img = np.asarray(im.convert("RGB" if self.rgb else "L"), np.float32) / 255.0
        h, w = img.shape[:2]
        s = p.max_side / max(h, w)
        if s < 1:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        if img.ndim == 2:
            img = img[..., None]                                  # (h, w, C), C = 1 or 3
        self.h, self.w = h, w
        gray = img.mean(-1)
        self.bg = float(np.median(gray)) if p.bg == "auto" else float(p.bg)
        self.light_bg = self.bg > 0.5

        Hp = sfft.next_fast_len(h + 2 * p.pad, real=True)
        Wp = sfft.next_fast_len(w + 2 * p.pad, real=True)
        self.Hp, self.Wp = Hp, Wp
        self.t, self.l = (Hp - h) // 2, (Wp - w) // 2
        padded = np.pad(img, ((self.t, Hp - h - self.t), (self.l, Wp - w - self.l), (0, 0)), mode="reflect")
        padded = np.ascontiguousarray(padded.transpose(2, 0, 1))  # (C, Hp, Wp)
        self.Fb = sfft.rfft2(padded, axes=(-2, -1), workers=-1)   # all channels in one batched FFT
        self.rgb_norm = 0.5 / np.tanh(0.5 * p.rgb_gain)

        fy = sfft.fftfreq(Hp).astype(np.float32)
        fx = sfft.rfftfreq(Wp).astype(np.float32)
        self.f2 = fy[:, None] ** 2 + fx[None, :] ** 2
        self.g = np.empty_like(self.f2)
        self.den = np.empty_like(self.f2)
        self.spec = np.empty_like(self.Fb)
        self.C = 2 * np.pi ** 2
        self.size_scale = w / p.ref_width
        self.post = np.exp(self.f2 * (-self.C * (p.post_sigma * self.size_scale) ** 2))
        self.lut = make_lut(p.palette, p.reverse_palette, p.hue_rotate, p.chroma_scale, p.pin_base, p.pin_width)

        self.size = (max(2, int(w * p.display_scale) // 2 * 2),     # even: yuv420p-safe
                     max(2, int(h * p.display_scale) // 2 * 2))

    def eps_for(self, s):
        p = self.p
        a = float(np.clip(s.loud * p.sensitivity, 0, 1)) ** p.level_curve
        hf = float(np.clip(s.high * p.sensitivity, 0, 1))
        k = 10.0 ** (1.0 - 2.0 * p.ringing)
        lq, ll = np.log(p.eps_quiet * k), np.log(p.eps_loud * k)
        eps = np.exp(lq + a * (ll - lq)) * 10.0 ** (-p.hf_ringing * hf)   # loudness, then highs, add ringing
        return float(np.clip(eps, 1e-5, 1.0))

    def render(self, s):
        """s: a State from Analyzer.update."""
        p = self.p
        hf = float(np.clip(s.high * p.sensitivity, 0, 1))
        sigma = float(np.clip(p.ripple_scale / np.exp(s.log_f) / (1.0 + p.hf_fine * hf),
                              p.sigma_min, p.sigma_max)) * self.size_scale
        eps = self.eps_for(s)
        punch = 1.0 + p.bass_punch * s.low

        g, den = self.g, self.den
        np.multiply(self.f2, -self.C * sigma * sigma, out=g)
        np.exp(g, out=g)
        np.multiply(g, g, out=den); den += eps
        np.divide(g, den, out=g); g *= self.post
        np.multiply(self.Fb, g, out=self.spec)
        out = sfft.irfft2(self.spec, s=(self.Hp, self.Wp), axes=(-2, -1), workers=-1, overwrite_x=True)
        view = out[:, self.t:self.t + self.h, self.l:self.l + self.w]      # (C, h, w)

        if self.rgb:
            # undo DC gain so 0 and 1 stay put, then soft-clip overshoot symmetrically around 0.5
            r = view * (1.0 + eps)
            kg = p.rgb_gain * punch
            r -= 0.5; r *= kg
            np.tanh(r, out=r)
            r *= 0.5 / np.tanh(0.5 * kg); r += 0.5
            np.clip(r, 0.0, 1.0, out=r)
            r *= 255.0
            frame = np.ascontiguousarray(r.astype(np.uint8).transpose(1, 2, 0)[..., ::-1])   # RGB -> BGR
        else:
            # anchor the background (DC gain 1/(1+eps)) to the bottom of the ramp
            ref = self.bg / (1.0 + eps)
            d = (ref - view[0]) if self.light_bg else (view[0] - ref)
            v = np.tanh(d * (p.gain * punch) - p.floor)
            v *= 127.5; v += 127.5
            frame = self.lut[v.astype(np.uint8)]
        if (self.w, self.h) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_CUBIC)
        return frame
