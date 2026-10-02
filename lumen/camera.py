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
        for _ in range(45):  # let auto-exposure settle on the face first
            cap.read()
        cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25 if os.name == "nt" else 1)  # manual mode
        cap.set(cv.CAP_PROP_AUTO_WB, 0)
    return cap


def describe(cap: cv.VideoCapture) -> str:
    code = int(cap.get(cv.CAP_PROP_FOURCC))
    fourcc = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)) if code > 0 else "?"
    return (f"{int(cap.get(cv.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))} "
            f"{fourcc.strip() or '?'} @ {cap.get(cv.CAP_PROP_FPS):.0f}fps")
