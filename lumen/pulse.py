"""Remote pulse (rPPG) from facial skin colour, using the POS method.

Wang, den Brinker, Stuijk, de Haan (2017), "Algorithmic Principles of Remote
PPG", IEEE TBME. POS projects RGB onto a plane orthogonal to skin tone, which
cancels most lighting/motion changes and leaves the blood-volume pulse.

LUMEN uses this as a *liveness* signal (is there blood flow under this
"skin"?), not as a medical measurement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt

FS = 30.0                 # resampling rate (Hz)
BAND = (0.7, 3.0)         # 42-180 bpm
MIN_SECONDS = 6.0


@dataclass
class PulseResult:
    bpm: float | None
    snr_db: float | None
    quality: str              # "acquiring" | "none" | "weak" | "strong"
    waveform: np.ndarray      # filtered pulse, normalised to [-1, 1]
    freqs: np.ndarray
    power: np.ndarray


def resample(t: np.ndarray, rgb: np.ndarray, fs: float = FS) -> np.ndarray:
    """Webcams drop frames; put samples on an even time grid first."""
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    return np.stack([np.interp(grid, t, rgb[:, c]) for c in range(3)], axis=1)


def pos(rgb: np.ndarray, fs: float = FS, window: float = 1.6) -> np.ndarray:
    n, l = len(rgb), int(window * fs)
    proj = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    h = np.zeros(n)
    for s in range(0, n - l + 1):
        c = rgb[s:s + l].T
        cn = c / (c.mean(axis=1, keepdims=True) + 1e-9)
        x = proj @ cn
        p = x[0] + (x[0].std() / (x[1].std() + 1e-9)) * x[1]
        h[s:s + l] += p - p.mean()
    return h


def bandpass(x: np.ndarray, fs: float = FS, band=BAND, order: int = 3) -> np.ndarray:
    b, a = butter(order, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def spectrum_snr(x: np.ndarray, fs: float = FS, band=BAND):
    """Peak frequency and SNR (dB): power at the peak + first harmonic vs the
    rest of the band (de Haan & Jeanne, 2013)."""
    nfft = max(4096, 1 << int(np.ceil(np.log2(len(x)))))
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft)) ** 2
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    f0 = freqs[in_band][np.argmax(power[in_band])]
    sig = (np.abs(freqs - f0) <= 0.1) | (np.abs(freqs - 2 * f0) <= 0.2)
    noise = in_band & ~sig
    snr = 10 * np.log10(power[sig & (freqs <= 2 * band[1])].sum() / (power[noise].sum() + 1e-12))
    return f0, float(snr), freqs[in_band], power[in_band]


def estimate(t, rgb, strong_db: float = 3.0, weak_db: float = 0.0) -> PulseResult:
    """t: seconds, rgb: (N, 3) mean skin colour in R, G, B order."""
    t, rgb = np.asarray(t, float), np.asarray(rgb, float)
    empty = np.zeros(0)
    if len(t) < 2 or t[-1] - t[0] < MIN_SECONDS:
        return PulseResult(None, None, "acquiring", empty, empty, empty)
    x = resample(t, rgb)
    wave = bandpass(pos(x))
    f0, snr, freqs, power = spectrum_snr(wave)
    quality = "strong" if snr >= strong_db else "weak" if snr >= weak_db else "none"
    norm = wave / (np.max(np.abs(wave)) + 1e-12)
    return PulseResult(f0 * 60.0, snr, quality, norm, freqs, power)
