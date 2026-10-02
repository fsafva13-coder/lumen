"""Face detection and skin regions of interest (ROIs) using OpenCV 5 YuNet."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "face_detection_yunet_2023mar.onnx"

Box = tuple[int, int, int, int]  # x, y, w, h


@dataclass
class Face:
    box: Box
    landmarks: np.ndarray  # (5, 2): right eye, left eye, nose tip, right mouth, left mouth
    score: float
    raw_landmarks: np.ndarray | None = None  # unsmoothed copy, for motion analysis

    @property
    def eye_distance(self) -> float:
        return float(np.linalg.norm(self.landmarks[0] - self.landmarks[1]))

    def skin_rois(self, frame_shape) -> dict[str, Box]:
        """Forehead and cheek patches: the skin areas where blood-flow colour
        changes are strongest and least affected by eyes/mouth movement."""
        h, w = frame_shape[:2]
        d = self.eye_distance
        re, le, nose = self.landmarks[0], self.landmarks[1], self.landmarks[2]
        eye_mid = (re + le) / 2

        def box(cx, cy, bw, bh) -> Box:
            x0 = int(np.clip(cx - bw / 2, 0, w - 1))
            y0 = int(np.clip(cy - bh / 2, 0, h - 1))
            x1 = int(np.clip(cx + bw / 2, 0, w))
            y1 = int(np.clip(cy + bh / 2, 0, h))
            return x0, y0, max(x1 - x0, 1), max(y1 - y0, 1)

        cheek_y = (eye_mid[1] + nose[1]) / 2 + 0.25 * d
        return {
            "forehead": box(eye_mid[0], eye_mid[1] - 0.6 * d, 0.9 * d, 0.35 * d),
            "cheek_r": box(re[0] - 0.05 * d, cheek_y, 0.35 * d, 0.3 * d),
            "cheek_l": box(le[0] + 0.05 * d, cheek_y, 0.35 * d, 0.3 * d),
        }


class FaceTracker:
    def __init__(self, model_path: Path = MODEL_PATH, score_threshold: float = 0.8):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Face model not found at {model_path}. Run: python scripts/download_models.py"
            )
        self._det = cv.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold, 0.3, 5000)
        self._size: tuple[int, int] | None = None
        self._smooth: np.ndarray | None = None

    def detect(self, frame: np.ndarray) -> list[Face]:
        """All faces in the frame, largest first."""
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self._det.setInputSize((w, h))
            self._size = (w, h)
        _, rows = self._det.detect(frame)
        if rows is None:
            return []
        faces = [
            Face(
                box=tuple(int(v) for v in r[:4]),
                landmarks=r[4:14].reshape(5, 2).astype(np.float32),
                score=float(r[14]),
                raw_landmarks=r[4:14].reshape(5, 2).astype(np.float32),
            )
            for r in rows
        ]
        faces = sorted(faces, key=lambda f: f.box[2] * f.box[3], reverse=True)
        self._stabilise(faces)
        return faces

    def _stabilise(self, faces: list[Face]) -> None:
        """Smooth the main face's landmarks over time. The detector jitters by
        a pixel or two per frame, which would otherwise leak into the skin
        colour signal as noise. Large jumps (real movement) reset the filter."""
        if not faces:
            self._smooth = None
            return
        cur = faces[0].landmarks
        if self._smooth is None or np.abs(cur - self._smooth).max() > 25:
            self._smooth = cur.copy()
        else:
            self._smooth = 0.8 * self._smooth + 0.2 * cur
        faces[0].landmarks = self._smooth.copy()


def mean_color(frame: np.ndarray, roi: Box) -> np.ndarray:
    """Mean B, G, R of a region."""
    x, y, w, h = roi
    return np.array(cv.mean(frame[y:y + h, x:x + w])[:3], dtype=np.float64)


def skin_color(frame: np.ndarray, rois: dict[str, Box]) -> np.ndarray:
    """Area-weighted mean B, G, R over all skin ROIs (more pixels = less noise)."""
    total, acc = 0, np.zeros(3)
    for x, y, w, h in rois.values():
        acc += mean_color(frame, (x, y, w, h)) * (w * h)
        total += w * h
    return acc / max(total, 1)


def inner_face_box(face: Face, frame_shape) -> Box:
    """Brow line to just below the mouth, about 1.5 eye-distances wide."""
    h, w = frame_shape[:2]
    re, le, _, mr, ml = face.landmarks
    d = face.eye_distance
    eye_mid, mouth_mid = (re + le) / 2, (mr + ml) / 2
    top, bottom = eye_mid[1] - 0.6 * d, mouth_mid[1] + 0.2 * d
    x0 = int(np.clip(eye_mid[0] - 0.75 * d, 0, w - 1))
    x1 = int(np.clip(eye_mid[0] + 0.75 * d, 1, w))
    y0 = int(np.clip(top, 0, h - 1))
    y1 = int(np.clip(bottom, 1, h))
    return x0, y0, max(x1 - x0, 1), max(y1 - y0, 1)


class SkinSampler:
    """Mean skin colour of the inner face, ignoring hair, fabric, eyes and
    pixels too dark or blown out to carry a signal.

    The skin mask is averaged over time into soft weights. A hard per-frame
    mask flickers at its edges, and that flicker is larger than the pulse.
    """

    SIZE = 64

    def __init__(self, rate: float = 0.05, min_skin: float = 0.15):
        self.rate, self.min_skin = rate, min_skin
        self._w: np.ndarray | None = None

    def reset(self) -> None:
        self._w = None

    def sample(self, frame: np.ndarray, face: Face) -> tuple[np.ndarray | None, float, Box]:
        """Returns (mean B, G, R or None if too little skin, skin fraction, box)."""
        box = inner_face_box(face, frame.shape)
        x, y, w, h = box
        patch = cv.resize(frame[y:y + h, x:x + w], (self.SIZE, self.SIZE), interpolation=cv.INTER_AREA)
        ycc = cv.cvtColor(patch, cv.COLOR_BGR2YCrCb)
        mask = cv.inRange(ycc, (0, 133, 77), (255, 173, 127)) & cv.inRange(patch, (20, 20, 20), (245, 245, 245))
        m = mask.astype(np.float32) / 255.0
        self._w = m if self._w is None else (1 - self.rate) * self._w + self.rate * m
        frac = float(self._w.mean())
        if frac < self.min_skin:
            return None, frac, box
        wsum = float(self._w.sum())
        color = (patch.astype(np.float32) * self._w[..., None]).sum(axis=(0, 1)) / wsum
        return color.astype(np.float64), frac, box
