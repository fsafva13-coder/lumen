"""Rolling buffer of skin colour over time. This raw trace is what the pulse
(rPPG) module will turn into a heart-rate signal in the next milestone."""
from __future__ import annotations

from collections import deque

import numpy as np


class SkinSignal:
    def __init__(self, seconds: float = 12.0, max_fps: int = 60):
        self.t: deque[float] = deque(maxlen=int(seconds * max_fps))
        self.rgb: deque[np.ndarray] = deque(maxlen=int(seconds * max_fps))
        self.seconds = seconds

    def push(self, t: float, bgr: np.ndarray) -> None:
        self.t.append(t)
        self.rgb.append(bgr[::-1].copy())  # store as R, G, B
        while self.t and self.t[-1] - self.t[0] > self.seconds:
            self.t.popleft()
            self.rgb.popleft()

    def reset(self) -> None:
        self.t.clear()
        self.rgb.clear()

    @property
    def fps(self) -> float:
        if len(self.t) < 2:
            return 0.0
        return (len(self.t) - 1) / max(self.t[-1] - self.t[0], 1e-6)

    @property
    def duration(self) -> float:
        return self.t[-1] - self.t[0] if len(self.t) > 1 else 0.0

    def green_trace(self, smooth: int = 3) -> np.ndarray:
        """Detrended, normalised green channel (green carries most of the
        blood-volume signal). For display only at this stage."""
        if len(self.rgb) < 10:
            return np.zeros(0)
        g = np.array(self.rgb)[:, 1]
        g = g / (g.mean() + 1e-6) - 1.0
        k = max(len(g) // 8, 5)
        trend = np.convolve(g, np.ones(k) / k, mode="same")
        g = g - trend
        if smooth > 1:
            g = np.convolve(g, np.ones(smooth) / smooth, mode="same")
        peak = np.max(np.abs(g)) + 1e-9
        return g / peak

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """(t, rgb) as numpy arrays, rgb in R, G, B order."""
        return np.array(self.t), np.array(self.rgb)
