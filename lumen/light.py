"""Light challenge: interrogate a face with random screen colours.

The screen flashes a random red/green/blue code that an attacker cannot
predict. A real face in front of the screen reflects that code back:
  * in time   - skin brightness follows the code (a pre-recorded or injected
                video cannot react to a code it has never seen);
  * in colour - a red flash makes the face redder, not just brighter;
  * in shape  - a 3D face reflects unevenly (nose, cheeks, eye sockets),
                while a flat photo or screen reflects almost uniformly.

The idea follows "Face Flashing" (Tang et al., NDSS 2018). All analysis is
plain least-squares on OpenCV-extracted face patches.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

from .face import Face, FaceTracker, inner_face_box

PATCH = 48          # face patches are resampled to PATCH x PATCH
MAX_LAG = 0.45      # seconds: screen-to-camera delay search range


@dataclass
class Challenge:
    seed: int
    step_seconds: float
    codes: np.ndarray  # (steps, 3) of 0/1, in R, G, B order

    @property
    def duration(self) -> float:
        return len(self.codes) * self.step_seconds

    def emitted(self, t: np.ndarray) -> np.ndarray:
        """Screen colour (R, G, B in 0/1) at times t; black outside the code."""
        t = np.atleast_1d(np.asarray(t, float))
        idx = np.floor(t / self.step_seconds).astype(int)
        inside = (idx >= 0) & (idx < len(self.codes))
        out = np.zeros((len(t), 3))
        out[inside] = self.codes[idx[inside]]
        return out

    def bgr_at(self, t: float) -> tuple[int, int, int]:
        r, g, b = self.emitted(t)[0]
        return int(b * 255), int(g * 255), int(r * 255)


def make_challenge(steps: int = 14, step_seconds: float = 0.4, seed: int | None = None) -> Challenge:
    """Random, balanced code. Each channel changes at most every 0.4 s, which
    keeps flashing well under the 3-per-second photosensitivity guideline."""
    seed = secrets.randbits(32) if seed is None else seed
    rng = np.random.default_rng(seed)
    while True:
        codes = rng.integers(0, 2, (steps, 3))
        ones = codes.mean(axis=0)
        flips = np.abs(np.diff(codes, axis=0)).sum(axis=0)
        if ((ones < 0.35) | (ones > 0.65)).any() or (flips < 4).any():
            continue
        cc = np.corrcoef(codes.T)[np.triu_indices(3, 1)]
        if np.abs(cc).max() < 0.5:
            return Challenge(seed, step_seconds, codes)


@dataclass
class Capture:
    """Face patches gathered while the challenge played."""
    times: list[float] = field(default_factory=list)      # seconds since challenge start
    patches: list[np.ndarray] = field(default_factory=list)  # PATCH x PATCH x 3, BGR float32
    background: list[np.ndarray] = field(default_factory=list)  # mean BGR outside the face
    face_widths: list[int] = field(default_factory=list)   # detector box width, pixels
    centres: list[tuple[float, float]] = field(default_factory=list)  # face box centre
    frames_total: int = 0

    def add(self, t: float, frame: np.ndarray, face: Face | None) -> None:
        self.frames_total += 1
        if face is None:
            return
        x, y, w, h = inner_face_box(face, frame.shape)
        patch = cv.resize(frame[y:y + h, x:x + w], (PATCH, PATCH), interpolation=cv.INTER_AREA)
        fx, fy, fw, fh = face.box
        total = np.array(cv.sumElems(frame)[:3])
        fx0, fy0 = max(fx, 0), max(fy, 0)
        inside = frame[fy0:fy + fh, fx0:fx + fw]
        n_out = frame.shape[0] * frame.shape[1] - inside.shape[0] * inside.shape[1]
        bg = (total - np.array(cv.sumElems(inside)[:3])) / max(n_out, 1)
        self.face_widths.append(int(fw))
        self.centres.append((fx + fw / 2, fy + fh / 2))
        self.times.append(t)
        self.patches.append(patch.astype(np.float32))
        self.background.append(bg)


@dataclass
class LightResult:
    ok: bool
    reason: str = ""
    lag: float = 0.0            # screen-to-camera delay found (s)
    r2: float = 0.0             # how well face brightness follows the code (0-1)
    gain: float = 0.0           # relative brightness change caused by the screen
    purity: float = 0.0         # colour fidelity: red flash -> red response (0-1)
    relief: float = 0.0         # spatial unevenness of the reflection (3D shape cue)
    bg_gain: float = 0.0        # same as gain, for the background
    frames_used: int = 0
    face_px: float = 0.0        # median face width in pixels
    motion: float = 0.0         # face wander during the challenge, in face widths
    lost: int = 0               # frames where no face was found
    ratio_map: np.ndarray | None = None  # PATCH x PATCH reflection map
    skin: np.ndarray | None = None       # PATCH x PATCH skin weights

    def features(self) -> dict:
        return {k: round(float(getattr(self, k)), 4) for k in
                ("lag", "r2", "gain", "purity", "relief", "bg_gain", "frames_used", "face_px", "motion", "lost")}


def _design(t: np.ndarray, e: np.ndarray) -> np.ndarray:
    tc = (t - t.mean()) / (np.ptp(t) + 1e-9)
    return np.column_stack([np.ones_like(t), tc, e])  # offset, drift, R, G, B


def _fit(x: np.ndarray, y: np.ndarray):
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    return coef, 1.0 - (resid ** 2).sum(axis=0) / ss_tot


def analyze(cap: Capture, ch: Challenge) -> LightResult:
    fps = cap.frames_total / max(ch.duration + 1.6, 1e-6)
    if cap.frames_total and fps < 12:
        return LightResult(False, f"camera too slow ({fps:.0f} fps) - add more light")
    if len(cap.times) < 40:
        return LightResult(False, "face not visible for long enough")
    t = np.array(cap.times)
    p = np.stack(cap.patches)[..., ::-1]          # -> R, G, B
    bg = np.array(cap.background)[:, ::-1]

    # Skin weights from the time-averaged patch (robust to the coloured light).
    mean_bgr = np.clip(p.mean(axis=0)[..., ::-1], 0, 255).astype(np.uint8)
    ycc = cv.cvtColor(mean_bgr, cv.COLOR_BGR2YCrCb)
    skin = (cv.inRange(ycc, (0, 130, 75), (255, 178, 130)) > 0)
    if skin.mean() < 0.15:
        skin = np.ones_like(skin)
    face_mean = p[:, skin].mean(axis=1)           # (n, 3)

    # Screen-to-camera delay: pick the lag where the code explains the face best.
    best = (-1.0, 0.0)
    for lag in np.arange(0.0, MAX_LAG, 0.02):
        _, r2 = _fit(_design(t, ch.emitted(t - lag)), face_mean)
        if r2.mean() > best[0]:
            best = (float(r2.mean()), float(lag))
    lag = best[1]

    # Drop frames exposed across a colour change.
    e = ch.emitted(t - lag)
    steady = (ch.emitted(t - lag - 0.04) == e).all(axis=1)
    if steady.sum() < 30:
        return LightResult(False, "too few steady frames")
    t, p, bg, e, face_mean = t[steady], p[steady], bg[steady], e[steady], face_mean[steady]
    x = _design(t, e)

    coef, r2 = _fit(x, face_mean)
    base, mix = coef[0], coef[2:5]                # mix[c_emitted, c_camera]
    gain = float(np.mean(np.diag(mix) / (base + 1e-6)))
    purity = float(np.trace(mix) / (np.abs(mix).sum() + 1e-9))

    coef_bg, _ = _fit(x, bg)
    bg_gain = float(np.mean(np.diag(coef_bg[2:5]) / (coef_bg[0] + 1e-6)))

    face_px = float(np.median(cap.face_widths))
    c = np.array(cap.centres)
    motion = float(np.hypot(c[:, 0].std(), c[:, 1].std()) / max(face_px, 1.0))

    # Per-pixel reflection map: brightness gained from the screen / ambient brightness.
    lum = p.sum(axis=-1).reshape(len(t), -1)
    ratio = _ratio_map(x, lum)

    # Relief: what is left after removing the best-fitting plane. A flat
    # photo or screen is (nearly) a plane; a face is not. To keep camera noise
    # from looking like relief, the map is computed twice from alternating
    # code steps and only the part that shows up in BOTH halves is counted.
    half = np.floor((t - lag) / ch.step_seconds).astype(int) % 2 == 0
    if min(half.sum(), (~half).sum()) < 15:
        return LightResult(False, "too few frames per half")
    ra, med_a = _deplane(_ratio_map(x[half], lum[half]), skin)
    rb, med_b = _deplane(_ratio_map(x[~half], lum[~half]), skin)
    shared = max(float(np.mean(ra * rb)), 0.0)
    relief = float(np.sqrt(shared) / (abs(med_a + med_b) / 2 + 1e-6))

    return LightResult(True, "", lag, float(r2.mean()), gain, purity, relief, bg_gain,
                       int(steady.sum()), face_px, motion, cap.frames_total - len(cap.times),
                       ratio, skin.astype(np.float32))


def _ratio_map(x: np.ndarray, lum: np.ndarray) -> np.ndarray:
    c, _ = _fit(x, lum)
    amb = np.maximum(c[0], 24.0)
    r = (c[2:5].sum(axis=0) / amb).reshape(PATCH, PATCH).astype(np.float32)
    return cv.GaussianBlur(r, (5, 5), 0)


def _deplane(ratio: np.ndarray, skin: np.ndarray) -> tuple[np.ndarray, float]:
    ys, xs = np.nonzero(skin)
    a = np.column_stack([np.ones(len(xs)), xs, ys])
    vals = ratio[ys, xs]
    plane, *_ = np.linalg.lstsq(a, vals, rcond=None)
    return vals - a @ plane, float(np.median(vals))


# --- Decision ---------------------------------------------------------------
# Version 2 (evaluation round 1, 26 Oct). Round 1 on the final camera setup
# showed the 3D-relief test rejecting real faces (relief 0.19-0.31 vs a 0.26
# cut-off) and a 0.85 code-match cut-off rejecting one real face at 0.84.
# Relief had already failed to stop a steady printed photo in the pilot, and
# the mandatory head turn catches prints, so relief is now REPORTED but no
# longer rejects. The light challenge's job is to reject what cannot react
# to the random code at all: screen replays and injected video.
MIN_FACE_PX = 120     # below this the face is too small to judge
MIN_R2 = 0.80         # face brightness must follow the random code
MIN_GAIN = 0.10       # and by a meaningful amount
MAX_MOTION = 0.06     # face may wander at most 6% of its width...
MAX_LOST = 3          # ...and vanish for at most 3 frames


def verdict(res: LightResult) -> tuple[str, str]:
    """Returns (label, reason). label: live | fake | inconclusive."""
    if not res.ok:
        return "inconclusive", res.reason
    if res.face_px < MIN_FACE_PX:
        return "inconclusive", "face too small - move closer to the camera"
    if res.motion > MAX_MOTION or res.lost > MAX_LOST:
        return "inconclusive", "too much movement - hold still and run again"
    if res.r2 < MIN_R2 or res.gain < MIN_GAIN:
        return "fake", "does not reflect the code (screen replay or injected video)"
    return "live", "reflects the random light code"


def capture_from_frames(frames, times, tracker: FaceTracker) -> Capture:
    """Rebuild a Capture from saved frames (offline analysis)."""
    cap = Capture()
    for f, t in zip(frames, times):
        faces = tracker.detect(f)
        cap.add(t, f, faces[0] if faces else None)
    return cap
