"""The interactive challenges, as reusable steps for a verification session.

Each function drives the screen and camera for one challenge and returns the
analysis result. Frames are analysed in memory and never written to disk.
"""
from __future__ import annotations

import time

import cv2 as cv
import numpy as np

from . import hud
from .camera import lock_and_calibrate, unlock_exposure
from .face import FaceTracker
from .light import Capture, LightResult, analyze as analyze_light, make_challenge
from .motion import MotionResult, analyze as analyze_turn
from .voice import Voice

CW, CH = 1280, 720
BG = (24, 20, 20)


def _stage(view: np.ndarray | None, title: str, line: str, colour, sub: str = "") -> np.ndarray:
    """Dark full-screen canvas with the (mirrored) camera view in the middle."""
    canvas = np.full((CH, CW, 3), BG, np.uint8)
    if view is not None:
        h, w = view.shape[:2]
        x, y = (CW - w) // 2, 90
        canvas[y:y + h, x:x + w] = view
    hud._text(canvas, f"LUMEN // {title}", (40, 50), 0.9, hud.GOLD, 2)
    hud._text(canvas, line, (40, CH - 80), 0.9, colour, 2)
    if sub:
        hud._text(canvas, sub, (40, CH - 45), 0.55, hud.PEARL)
    return canvas


def wait_ready(cap, tracker: FaceTracker, win: str, title: str, sub: str = "") -> bool:
    """Preview until SPACE (with exactly one face in view). False = quit."""
    while True:
        ok, frame = cap.read()
        if not ok:
            return False
        view = cv.flip(frame, 1)
        faces = tracker.detect(view)
        for f in faces[:1]:
            hud.brackets(view, f.box)
        ready = len(faces) == 1
        cv.imshow(win, _stage(view, title, "SPACE to start" if ready else "Show exactly one face",
                              hud.GOLD if ready else hud.ALERT, sub))
        key = cv.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv.getWindowProperty(win, cv.WND_PROP_VISIBLE) < 1:
            return False
        if key == ord(" ") and ready:
            return True


def light_challenge(cap, tracker: FaceTracker, win: str, voice: Voice,
                    steps: int = 14, step_seconds: float = 0.4) -> LightResult:
    return analyze_light(*light_capture(cap, tracker, win, voice, steps, step_seconds))


def light_capture(cap, tracker: FaceTracker, win: str, voice: Voice,
                  steps: int = 14, step_seconds: float = 0.4):
    """Play the flashes and collect face patches. Returns (Capture, Challenge)."""
    ch = make_challenge(steps, step_seconds)
    voice.say("Hold still. Scanning with light.")
    # Exposure is frozen only for the flashes, where a camera that re-adjusts
    # its brightness would cancel out the very signal being measured. A dark
    # picture is fine here: the measurement is screen light / room light.
    cv.imshow(win, _stage(None, "LIGHT", "Hold still...", hud.PEARL))
    cv.waitKey(1)
    lock_and_calibrate(cap, settle=10)
    capt = Capture()
    canvas = np.zeros((CH, CW, 3), np.uint8)
    t0 = time.perf_counter() + 1.0          # one second of black first
    boost = None
    while True:
        t = time.perf_counter() - t0
        if t > ch.duration + 0.6:
            break
        canvas[:] = ch.bgr_at(t)
        cv.imshow(win, canvas)
        cv.waitKey(1)
        ok, frame = cap.read()
        if not ok:
            break
        # Find the face on a brightened copy (a locked, dim picture can hide it
        # from the detector); measure the light on the original pixels.
        if boost is None:
            boost = float(np.clip(110.0 / max(frame.mean(), 1.0), 1.0, 12.0))
        faces = tracker.detect(cv.convertScaleAbs(frame, alpha=boost) if boost > 1.2 else frame)
        capt.add(time.perf_counter() - t0, frame, faces[0] if faces else None)
    unlock_exposure(cap)
    return capt, ch


STRAIGHT = 2.0


def head_turn(cap, tracker: FaceTracker, win: str, voice: Voice, direction: str) -> MotionResult:
    times, lms = head_turn_capture(cap, tracker, win, voice, direction)
    return analyze_turn(times, lms, ref_seconds=STRAIGHT - 0.3)


def head_turn_capture(cap, tracker: FaceTracker, win: str, voice: Voice, direction: str,
                      straight: float = STRAIGHT, turn: float = 3.5, back: float = 1.0):
    """Prompt the turn and collect landmarks. Returns (times, landmarks)."""
    times, lms, spoken = [], [], None
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        t = time.perf_counter() - t0
        if not ok or t > straight + turn + back:
            break
        faces = tracker.detect(frame)
        if faces:
            times.append(t)
            lms.append(faces[0].raw_landmarks)
        phase = "straight" if t < straight else "turn" if t < straight + turn else "back"
        text = {"straight": "Look straight at the camera",
                "turn": f"Turn your head to YOUR {direction.upper()} until you face sideways",
                "back": "Back to centre"}[phase]
        if phase != spoken:
            voice.say(text.replace("YOUR", "your").replace(direction.upper(), direction) + ".")
            spoken = phase
        cv.imshow(win, _stage(cv.flip(frame, 1), "HEAD TURN", text,
                              hud.GOLD if phase == "turn" else hud.PEARL))
        cv.waitKey(1)
    return times, lms


def message(win: str, title: str, line: str, colour, sub: str = "", seconds: float = 1.6) -> None:
    cv.imshow(win, _stage(None, title, line, colour, sub))
    cv.waitKey(int(seconds * 1000))
