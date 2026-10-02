"""Milestone 3: the light challenge.

The screen flashes a random colour code for about 6 seconds and LUMEN measures
how the face reflects it.

Run:   python scripts/03_light_challenge.py --label real --subject p01
Keys:  SPACE = start / run again     q / Esc = quit

Setup: screen brightness at maximum, face 30-45 cm from the screen, room
lights dim if possible (the screen should be a noticeable light source).

WARNING: this flashes full-screen colours (slower than 3 changes per second).
Do not use it with anyone who has photosensitive epilepsy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lumen import hud  # noqa: E402
from lumen.camera import describe, open_camera  # noqa: E402
from lumen.face import FaceTracker  # noqa: E402
from lumen.light import Capture, analyze, make_challenge, verdict  # noqa: E402

LABELS = ["real", "print", "screen", "mask", "deepfake"]
CW, CH = 1280, 720          # canvas size (the window is shown full-screen)
PRE, POST = 1.0, 0.6        # seconds of black before / after the code


def paste_center(canvas, img, top):
    h, w = img.shape[:2]
    x = (CW - w) // 2
    canvas[top:top + h, x:x + w] = img


def align_phase(cap, tracker, win, args) -> bool:
    """Live preview until the user presses SPACE. Returns False to quit."""
    while True:
        ok, frame = cap.read()
        if not ok:
            return False
        view = cv.flip(frame, 1)
        faces = tracker.detect(view)
        for f in faces[:1]:
            hud.brackets(view, f.box)
        canvas = np.full((CH, CW, 3), (24, 20, 20), np.uint8)
        paste_center(canvas, view, 90)
        ready = len(faces) == 1
        msg = "SPACE to start the light challenge" if ready else "Show exactly one face to the camera"
        hud._text(canvas, "LUMEN // LIGHT CHALLENGE", (40, 50), 0.9, hud.GOLD, 2)
        hud._text(canvas, f"label: {args.label}   subject: {args.subject}", (40, 80), 0.5, hud.DIM)
        hud._text(canvas, msg, (40, CH - 80), 0.8, hud.GOLD if ready else hud.ALERT, 2)
        hud._text(canvas, "Screen brightness max, 30-45 cm away, hold still. Flashing colours ahead.",
                  (40, CH - 45), 0.55, hud.PEARL)
        cv.imshow(win, canvas)
        key = cv.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        if key == ord(" ") and ready:
            return True


def challenge_phase(cap, tracker, win, ch):
    capt, frames, times = Capture(), [], []
    canvas = np.zeros((CH, CW, 3), np.uint8)
    t0 = time.perf_counter() + PRE
    while True:
        t = time.perf_counter() - t0
        if t > ch.duration + POST:
            break
        canvas[:] = ch.bgr_at(t)
        cv.imshow(win, canvas)
        cv.waitKey(1)
        ok, frame = cap.read()
        if not ok:
            break
        tf = time.perf_counter() - t0
        faces = tracker.detect(frame)
        capt.add(tf, frame, faces[0] if faces else None)
        frames.append(frame)
        times.append(tf)
    return capt, frames, times


def result_screen(win, res) -> bool:
    """Show the result. Returns True to run again."""
    canvas = np.full((CH, CW, 3), (24, 20, 20), np.uint8)
    hud._text(canvas, "LUMEN // LIGHT CHALLENGE RESULT", (40, 50), 0.9, hud.GOLD, 2)
    label, reason = verdict(res)
    colour = {"live": hud.GOLD, "fake": hud.ALERT, "inconclusive": hud.ROSE}[label]
    hud._text(canvas, {"live": "LIVE FACE", "fake": "NOT A LIVE FACE", "inconclusive": "INCONCLUSIVE"}[label],
              (40, 120), 1.2, colour, 2)
    hud._text(canvas, reason, (42, 150), 0.6, colour)
    if res.ok:
        rows = [
            (f"code match (R2)      {res.r2:5.2f}", "how closely face brightness follows the random code"),
            (f"reflection strength  {res.gain * 100:5.1f} %", "brightness added by the screen"),
            (f"colour fidelity      {res.purity:5.2f}", "red flash -> red response"),
            (f"3D relief            {res.relief:5.3f}", "uneven reflection = real depth, even = flat"),
            (f"background strength  {res.bg_gain * 100:5.1f} %", "same measure, outside the face"),
            (f"face size            {res.face_px:5.0f} px", "needs 120 px or more to judge shape"),
            (f"movement             {res.motion * 100:5.1f} %", "of face width; must stay under 4.5"),
            (f"screen-camera delay  {res.lag * 1000:5.0f} ms", f"{res.frames_used} frames used"),
        ]
        for i, (a, b) in enumerate(rows):
            hud._text(canvas, a, (40, 210 + 44 * i), 0.7, hud.PEARL, 1)
            hud._text(canvas, b, (60, 230 + 44 * i), 0.45, hud.DIM)
        # Reflection map: where on the face the screen light landed.
        m = res.ratio_map.copy()
        lo, hi = np.percentile(m[res.skin > 0], [2, 98])
        m = np.clip((m - lo) / (hi - lo + 1e-9), 0, 1)
        heat = cv.applyColorMap((m * 255).astype(np.uint8), cv.COLORMAP_MAGMA)
        heat[res.skin == 0] = (30, 26, 26)
        heat = cv.resize(heat, (384, 384), interpolation=cv.INTER_CUBIC)
        canvas[180:564, CW - 384 - 60:CW - 60] = heat
        hud._text(canvas, "REFLECTION MAP", (CW - 384 - 60, 170), 0.5, hud.DIM)
        hud._text(canvas, "provisional thresholds from a small pilot set", (40, CH - 80), 0.5, hud.DIM)
    hud._text(canvas, "SPACE = run again     q = quit", (40, CH - 45), 0.6, hud.GOLD)
    cv.imshow(win, canvas)
    while True:
        key = cv.waitKey(30) & 0xFF
        if key == ord(" "):
            return True
        if key in (ord("q"), 27) or cv.getWindowProperty(win, cv.WND_PROP_VISIBLE) < 1:
            return False


def save(frames, times, ch, res, cam_mode, args) -> Path:
    out_dir = ROOT / "data" / "raw" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{args.subject}_{time.strftime('%Y%m%d_%H%M%S')}_light"
    h, w = frames[0].shape[:2]
    writer = cv.VideoWriter(str(base.with_suffix(".avi")), cv.VideoWriter_fourcc(*"FFV1"), 30, (w, h))
    codec = "FFV1"
    if not writer.isOpened():
        writer = cv.VideoWriter(str(base.with_suffix(".avi")), cv.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
        codec = "MJPG"
    for f in frames:
        writer.write(f)
    writer.release()
    meta = {
        "kind": "light_challenge", "label": args.label, "subject": args.subject, "note": args.note,
        "codec": codec, "camera": cam_mode, "exposure_locked": not args.auto_exposure,
        "seed": ch.seed, "step_seconds": ch.step_seconds, "codes_rgb": ch.codes.tolist(),
        "times": [round(t, 4) for t in times],
        "features": res.features() if res.ok else None, "verdict": verdict(res)[0],
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    base.with_suffix(".json").write_text(json.dumps(meta))
    return base.with_suffix(".avi")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="real", choices=LABELS)
    ap.add_argument("--subject", default="p00")
    ap.add_argument("--note", default="")
    ap.add_argument("--steps", type=int, default=14)
    ap.add_argument("--step-seconds", type=float, default=0.4)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--auto-exposure", action="store_true", help="leave camera auto-exposure on")
    ap.add_argument("--no-save", action="store_true", help="do not save the clip")
    args = ap.parse_args()
    if args.step_seconds < 0.34:
        sys.exit("step-seconds must be at least 0.34 (photosensitivity safety limit).")

    tracker = FaceTracker()
    cap = open_camera(args.camera, lock_exposure=not args.auto_exposure)
    cam_mode = describe(cap)
    print("Camera:", cam_mode)
    win = "LUMEN light challenge"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.setWindowProperty(win, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    while align_phase(cap, tracker, win, args):
        ch = make_challenge(args.steps, args.step_seconds)
        capt, frames, times = challenge_phase(cap, tracker, win, ch)
        res = analyze(capt, ch)
        print("Verdict:", *verdict(res), "|", res.features() if res.ok else "")
        if not args.no_save and frames:
            print("Saved", save(frames, times, ch, res, cam_mode, args))
        if not result_screen(win, res):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
