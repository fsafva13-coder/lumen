"""Milestone 4: the head-turn challenge.

LUMEN asks for a head turn in a random direction and checks that the nose
moves like a 3D nose (it cannot on a flat photo or screen).

Run:   python scripts/04_head_turn.py --label real --subject p01
Keys:  SPACE = start / run again     q / Esc = quit

For print / screen clips: hold the photo up, then swivel it left or right
as if it were turning its head.
"""
from __future__ import annotations

import argparse
import json
import random
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
from lumen.motion import analyze, verdict  # noqa: E402

LABELS = ["real", "print", "screen", "mask", "deepfake"]
STRAIGHT, TURN, BACK = 1.2, 3.0, 1.0   # seconds per phase


def banner(view, text, color):
    h, w = view.shape[:2]
    roi = view[h - 70:h, :]
    cv.addWeighted(np.zeros_like(roi), 0.6, roi, 0.4, 0, dst=roi)
    hud._text(view, text, (20, h - 26), 0.9, color, 2)


def wait_for_start(cap, tracker, win, args) -> bool:
    while True:
        ok, frame = cap.read()
        if not ok:
            return False
        view = cv.flip(frame, 1)
        faces = tracker.detect(view)
        for f in faces[:1]:
            hud.brackets(view, f.box)
        ready = len(faces) == 1
        hud.panel(view, [(f"HEAD-TURN CHALLENGE  {args.label} / {args.subject}", hud.PEARL)], width=420)
        banner(view, "SPACE to start" if ready else "Show exactly one face", hud.GOLD if ready else hud.ALERT)
        cv.imshow(win, view)
        key = cv.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv.getWindowProperty(win, cv.WND_PROP_VISIBLE) < 1:
            return False
        if key == ord(" ") and ready:
            return True


def run_challenge(cap, tracker, win, direction):
    times, lms, frames = [], [], []
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.perf_counter() - t0
        if t > STRAIGHT + TURN + BACK:
            break
        faces = tracker.detect(frame)
        frames.append(frame)
        if faces:
            times.append(t)
            lms.append(faces[0].raw_landmarks)
        view = cv.flip(frame, 1)
        if t < STRAIGHT:
            banner(view, "Look straight at the camera", hud.PEARL)
        elif t < STRAIGHT + TURN:
            arrow = "<<<  " if direction == "left" else ""
            tail = "  >>>" if direction == "right" else ""
            banner(view, f"{arrow}Slowly turn your head to YOUR {direction.upper()}{tail}", hud.GOLD)
        else:
            banner(view, "Back to centre", hud.PEARL)
        cv.imshow(win, view)
        cv.waitKey(1)
    return times, lms, frames


def show_result(win, frame, res) -> bool:
    view = cv.flip(frame, 1)
    label, reason = verdict(res)
    colour = {"live": hud.GOLD, "fake": hud.ALERT, "inconclusive": hud.ROSE}[label]
    lines = [({"live": "LIVE FACE", "fake": "NOT A LIVE FACE", "inconclusive": "INCONCLUSIVE"}[label], colour),
             (reason, colour)]
    if res.ok:
        lines += [(f"nose parallax   {res.parallax:5.3f}  (3D if 0.10+)", hud.PEARL),
                  (f"nose shift      {res.nose_shift:+5.3f}", hud.PEARL),
                  (f"face narrowing  {(1 - res.foreshorten) * 100:4.1f} %", hud.PEARL),
                  (f"eye distance    {res.face_px:4.0f} px", hud.PEARL)]
    hud.panel(view, lines, width=560)
    banner(view, "SPACE = run again     q = quit", hud.GOLD)
    cv.imshow(win, view)
    while True:
        key = cv.waitKey(30) & 0xFF
        if key == ord(" "):
            return True
        if key in (ord("q"), 27) or cv.getWindowProperty(win, cv.WND_PROP_VISIBLE) < 1:
            return False


def save(frames, times, lms, direction, res, cam_mode, args) -> Path:
    out_dir = ROOT / "data" / "raw" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{args.subject}_{time.strftime('%Y%m%d_%H%M%S')}_turn"
    h, w = frames[0].shape[:2]
    writer = cv.VideoWriter(str(base.with_suffix(".avi")), cv.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    meta = {
        "kind": "head_turn", "label": args.label, "subject": args.subject, "note": args.note,
        "camera": cam_mode, "direction_asked": direction,
        "times": [round(t, 4) for t in times],
        "landmarks": [np.round(l, 2).tolist() for l in lms],
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
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    tracker = FaceTracker()
    cap = open_camera(args.camera)
    cam_mode = describe(cap)
    print("Camera:", cam_mode)
    win = "LUMEN head turn"
    cv.namedWindow(win, cv.WINDOW_NORMAL)

    while wait_for_start(cap, tracker, win, args):
        direction = random.choice(["left", "right"])
        times, lms, frames = run_challenge(cap, tracker, win, direction)
        if not frames:
            break
        res = analyze(times, lms, ref_seconds=STRAIGHT - 0.3)
        print(f"Asked: {direction} | Verdict:", *verdict(res), "|", res.features() if res.ok else "")
        if not args.no_save:
            print("Saved", save(frames, times, lms, direction, res, cam_mode, args))
        if not show_result(win, frames[-1], res):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
