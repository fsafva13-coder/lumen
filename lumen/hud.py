"""Sci-fi scanning overlay drawn with OpenCV primitives."""
from __future__ import annotations

import cv2 as cv
import numpy as np

# BGR palette: champagne gold, rose, pearl, alert
GOLD = (120, 200, 240)
ROSE = (180, 130, 255)
PEARL = (235, 235, 245)
ALERT = (80, 80, 255)
DIM = (120, 120, 130)
FONT = cv.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.5, color=PEARL, thick=1):
    cv.putText(img, s, (org[0] + 1, org[1] + 1), FONT, scale, (0, 0, 0), thick, cv.LINE_AA)
    cv.putText(img, s, org, FONT, scale, color, thick, cv.LINE_AA)


def brackets(img, box, color=GOLD, length=0.22, thick=2):
    x, y, w, h = box
    lx, ly = int(w * length), int(h * length)
    for (cx, cy, dx, dy) in [(x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)]:
        cv.line(img, (cx, cy), (cx + dx * lx, cy), color, thick, cv.LINE_AA)
        cv.line(img, (cx, cy), (cx, cy + dy * ly), color, thick, cv.LINE_AA)


def scanline(img, box, t, color=ROSE, period=2.0):
    """A glowing line sweeping down the face."""
    x, y, w, h = box
    H, W = img.shape[:2]
    x0, x1 = max(x, 0), min(x + w, W)
    if x1 <= x0:
        return
    phase = (t % period) / period
    sy = int(y + phase * h)
    for i, alpha in enumerate([0.08, 0.15, 0.3]):
        band = 14 - i * 5
        y0, y1 = max(sy - band, 0), min(sy + band, H)
        if y1 > y0:
            roi = img[y0:y1, x0:x1]
            overlay = np.full_like(roi, color)
            cv.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)
    if 0 <= sy < H:
        cv.line(img, (x0, sy), (x1, sy), color, 1, cv.LINE_AA)


def landmarks(img, pts, color=GOLD):
    for p in pts:
        c = (int(p[0]), int(p[1]))
        cv.circle(img, c, 2, color, -1, cv.LINE_AA)
        cv.circle(img, c, 6, color, 1, cv.LINE_AA)


def roi_boxes(img, rois: dict, color=ROSE):
    for name, (x, y, w, h) in rois.items():
        cv.rectangle(img, (x, y), (x + w, y + h), color, 1, cv.LINE_AA)
        _text(img, name.upper(), (x, y - 4), 0.35, color)


def panel(img, lines: list[tuple[str, tuple]], origin=(14, 14), width=300):
    """Translucent status panel; lines are (text, colour)."""
    x, y = origin
    h = 28 + 22 * len(lines)
    H, W = img.shape[:2]
    x1, y1 = min(x + width, W), min(y + h, H)
    roi = img[y:y1, x:x1]
    cv.addWeighted(np.zeros_like(roi), 0.55, roi, 0.45, 0, dst=roi)
    cv.line(img, (x, y), (x1, y), GOLD, 2)
    _text(img, "LUMEN // PROOF-OF-LIFE", (x + 10, y + 20), 0.5, GOLD, 1)
    for i, (s, c) in enumerate(lines):
        _text(img, s, (x + 10, y + 44 + 22 * i), 0.45, c)


def waveform(img, values: np.ndarray, rect, label="SKIN SIGNAL (RAW)", color=ROSE):
    x, y, w, h = rect
    H, W = img.shape[:2]
    x1, y1 = min(x + w, W), min(y + h, H)
    roi = img[y:y1, x:x1]
    cv.addWeighted(np.zeros_like(roi), 0.55, roi, 0.45, 0, dst=roi)
    _text(img, label, (x + 8, y + 16), 0.4, DIM)
    if len(values) < 2:
        _text(img, "acquiring...", (x + 8, y + h // 2 + 6), 0.45, DIM)
        return
    xs = np.linspace(x + 6, x1 - 6, len(values))
    mid, amp = y + h / 2 + 6, (h - 30) / 2
    pts = np.stack([xs, mid - values * amp], axis=1).astype(np.int32)
    cv.polylines(img, [pts], False, color, 1, cv.LINE_AA)


def readout(img, label: str, value: str, unit: str, color=GOLD, anchor="tr", margin=14):
    """Large numeric readout, e.g. the BPM display."""
    H, W = img.shape[:2]
    (vw, vh), _ = cv.getTextSize(value, FONT, 1.6, 2)
    box_w, box_h = max(vw + 80, 180), vh + 46
    x = W - box_w - margin if anchor == "tr" else margin
    y = margin
    roi = img[y:y + box_h, x:x + box_w]
    cv.addWeighted(np.zeros_like(roi), 0.55, roi, 0.45, 0, dst=roi)
    cv.line(img, (x, y), (x + box_w, y), color, 2)
    _text(img, label, (x + 10, y + 18), 0.4, DIM)
    _text(img, value, (x + 10, y + 18 + vh + 14), 1.6, color, 2)
    _text(img, unit, (x + 18 + vw, y + 18 + vh + 14), 0.5, color)


def quality_bar(img, snr_db, rect, lo=-6.0, hi=9.0):
    """Signal-quality meter from SNR (dB)."""
    x, y, w, h = rect
    cv.rectangle(img, (x, y), (x + w, y + h), DIM, 1, cv.LINE_AA)
    if snr_db is None:
        return
    frac = float(np.clip((snr_db - lo) / (hi - lo), 0, 1))
    color = GOLD if snr_db >= 3 else ROSE if snr_db >= 0 else ALERT
    cv.rectangle(img, (x + 2, y + 2), (x + 2 + int((w - 4) * frac), y + h - 2), color, -1)
