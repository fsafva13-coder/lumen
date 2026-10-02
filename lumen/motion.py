"""Head-turn challenge: does the nose move like a 3D nose?

Eyes and mouth corners lie roughly on one plane; the nose tip sticks out of
it. When a real head turns, the nose slides sideways relative to that plane
(parallax). On a printed photo or a screen everything is on one plane, so the
nose stays exactly where the plane says it should, however the photo is
tilted, bent toward the camera, or moved.

Method: take a frontal reference frame, map its eye/mouth points onto the
current frame with a homography (cv.getPerspectiveTransform), predict where
the nose would be if the face were flat, and measure how far the real nose is
from that prediction, in units of eye distance.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

PLANE = [0, 1, 3, 4]   # right eye, left eye, right mouth corner, left mouth corner
NOSE = 2


@dataclass
class MotionResult:
    ok: bool
    reason: str = ""
    parallax: float = 0.0      # peak nose displacement off the face plane (eye distances)
    nose_shift: float = 0.0    # signed peak sideways nose movement (eye distances); + = image right
    foreshorten: float = 1.0   # smallest eye-distance / face-height ratio vs reference (1 = no turn)
    face_px: float = 0.0       # eye distance in pixels at reference
    frames: int = 0

    def features(self) -> dict:
        return {k: round(float(getattr(self, k)), 4) for k in
                ("parallax", "nose_shift", "foreshorten", "face_px", "frames")}


def _shape(lm: np.ndarray):
    eye_mid = (lm[0] + lm[1]) / 2
    mouth_mid = (lm[3] + lm[4]) / 2
    iod = np.linalg.norm(lm[1] - lm[0])
    height = np.linalg.norm(mouth_mid - eye_mid)
    return eye_mid, iod, height


def parallax_series(ref: np.ndarray, landmarks: np.ndarray):
    """Per-frame (parallax, nose offset, eye-distance/height ratio)."""
    ref = ref.astype(np.float32)
    out = []
    for lm in landmarks.astype(np.float32):
        h = cv.getPerspectiveTransform(ref[PLANE], lm[PLANE])
        pred = cv.perspectiveTransform(ref[NOSE].reshape(1, 1, 2), h)[0, 0]
        eye_mid, iod, height = _shape(lm)
        out.append((np.linalg.norm(lm[NOSE] - pred) / iod, (lm[NOSE][0] - eye_mid[0]) / iod, iod / height))
    return np.array(out)


def analyze(times, landmarks, ref_seconds: float = 0.8) -> MotionResult:
    """times: seconds from the start of the 'look straight' phase.
    landmarks: (n, 5, 2) raw YuNet landmarks."""
    t, lm = np.asarray(times, float), np.asarray(landmarks, float)
    ref_mask = t <= ref_seconds
    if ref_mask.sum() < 8 or (~ref_mask).sum() < 20:
        return MotionResult(False, "face not visible for long enough")
    ref = np.median(lm[ref_mask], axis=0)
    _, ref_iod, ref_height = _shape(ref)
    ref_offset = (ref[NOSE][0] - (ref[0][0] + ref[1][0]) / 2) / ref_iod
    s = parallax_series(ref, lm[~ref_mask])
    par = np.convolve(s[:, 0], np.ones(5) / 5, mode="same")     # ride out single-frame jitter
    shift = np.convolve(s[:, 1] - ref_offset, np.ones(5) / 5, mode="same")
    peak_shift = shift[np.argmax(np.abs(shift))]
    return MotionResult(True, "", float(np.percentile(par, 90)), float(peak_shift),
                        float(np.percentile(s[:, 2], 5) / (ref_iod / ref_height)),
                        float(ref_iod), int(len(t)))


# Provisional thresholds from the geometry of an average face; re-fit on data.
MIN_EYE_PX = 45        # eye distance in pixels (about a 120 px wide face)
TURN_SHIFT = 0.10      # nose moved sideways by 10% of eye distance, or...
TURN_FORESHORTEN = 0.92  # ...the face narrowed by 8%: something turned
LIVE_PARALLAX = 0.10
FLAT_PARALLAX = 0.06


def verdict(res: MotionResult) -> tuple[str, str]:
    """Returns (label, reason). label: live | fake | inconclusive."""
    if not res.ok:
        return "inconclusive", res.reason
    if res.face_px < MIN_EYE_PX:
        return "inconclusive", "face too small - move closer to the camera"
    if res.parallax >= LIVE_PARALLAX:
        return "live", "nose moves off the face plane like a 3D head"
    turned = abs(res.nose_shift) >= TURN_SHIFT or res.foreshorten <= TURN_FORESHORTEN
    if not turned:
        return "inconclusive", "no head turn seen - turn further"
    if res.parallax < FLAT_PARALLAX:
        return "fake", "turned, but the nose stayed on the face plane (flat surface)"
    return "inconclusive", "turn was too small to judge depth"
