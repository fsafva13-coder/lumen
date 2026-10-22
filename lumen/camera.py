"""Webcam helper tuned for pulse work.

Two things destroy the faint pulse signal before OpenCV ever sees it:
  1. the camera compressing frames (MJPG) - so we ask for raw YUY2 frames;
  2. auto-exposure constantly re-adjusting brightness - so it can be locked.
"""
import os
import sys

import cv2 as cv


def open_camera(index: int = 0, width: int = 640, height: int = 480, fps: int = 30,
                raw: bool = True, lock_exposure: bool = False) -> cv.VideoCapture:
    backend = cv.CAP_DSHOW if os.name == "nt" else cv.CAP_ANY
    cap = cv.VideoCapture(index, backend)
    if raw:
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"YUY2"))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv.CAP_PROP_FPS, fps)
    ok = cap.isOpened() and cap.read()[0]
    if not ok and raw:  # camera refused raw mode: fall back to its default
        cap.release()
        return open_camera(index, width, height, fps, raw=False, lock_exposure=lock_exposure)
    if not ok:
        sys.exit(f"Could not open camera {index}. Close other apps using the webcam or try --camera 1")
    if lock_exposure:
        lock_and_calibrate(cap)
    return cap


def _brightness(cap: cv.VideoCapture, settle: int = 4) -> float:
    """Mean brightness of the centre of the picture, after the camera settles."""
    for _ in range(settle):
        cap.read()
    ok, frame = cap.read()
    if not ok:
        return -1.0
    h, w = frame.shape[:2]
    centre = frame[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    return float(cv.cvtColor(centre, cv.COLOR_BGR2GRAY).mean())


def lock_and_calibrate(cap: cv.VideoCapture, target: float = 105.0, settle: int = 45) -> None:
    """Freeze exposure, but at a sensible brightness.

    Switching a webcam to manual exposure often drops it to a driver default
    that is far too dark. So after locking, try each manual exposure step and
    keep the one whose picture is closest to `target` (out of 255). The target
    leaves headroom so the light challenge's white flash does not saturate.
    Exposure is relative for the light challenge (screen light / room light),
    so a brighter picture does not change its measurements, only its noise.
    """
    for _ in range(settle):  # let auto-exposure settle on the face first
        cap.read()
    auto_level = _brightness(cap, settle=0)
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25 if os.name == "nt" else 1)  # manual mode
    cap.set(cv.CAP_PROP_AUTO_WB, 0)
    if os.name != "nt":
        return  # exposure units differ by platform; only Windows is calibrated
    # Longest exposure allowed is 2**-5 s (1/32 s). Anything longer and the
    # camera drops below 30 frames per second and each frame smears across
    # several colour flashes, which ruins the light challenge.
    # Take the longest safe exposure, stepping shorter only if the picture
    # would be too bright to leave headroom for the white flash. 2**-4 s
    # (about 15 fps) is the longest allowed: still several frames per flash.
    best = None
    for exp in range(-4, -12, -1):  # DirectShow units: 2**exp seconds
        cap.set(cv.CAP_PROP_EXPOSURE, exp)
        level = _brightness(cap)
        if level < 0:
            continue
        best = (exp, level)
        if level <= 1.3 * target:
            break
    if best is None:
        cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.75)
        print("Camera: manual exposure unavailable, using auto")
        return
    cap.set(cv.CAP_PROP_EXPOSURE, best[0])
    print(f"Camera: exposure locked at step {best[0]}, gain {cap.get(cv.CAP_PROP_GAIN):.0f} "
          f"(brightness {best[1]:.0f}, auto was {auto_level:.0f})")


def describe(cap: cv.VideoCapture) -> str:
    code = int(cap.get(cv.CAP_PROP_FOURCC))
    fourcc = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)) if code > 0 else "?"
    return (f"{int(cap.get(cv.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))} "
            f"{fourcc.strip() or '?'} @ {cap.get(cv.CAP_PROP_FPS):.0f}fps")


def unlock_exposure(cap: cv.VideoCapture, settle: int = 20) -> None:
    """Hand exposure back to the camera (for previews and the head turn,
    which need a well-lit picture more than a steady one)."""
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.75 if os.name == "nt" else 3)
    cap.set(cv.CAP_PROP_AUTO_WB, 1)
    for _ in range(settle):
        cap.read()
