"""Milestone 1: live face lock + skin-signal acquisition with the LUMEN HUD.

Run:   python scripts/01_face_scan.py
Keys:  q / Esc = quit   s = save snapshot   r = reset signal
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2 as cv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lumen import hud  # noqa: E402
from lumen.camera import open_camera  # noqa: E402
from lumen.face import FaceTracker, mean_color  # noqa: E402
from lumen.signals import SkinSignal  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    tracker = FaceTracker()
    signal = SkinSignal(seconds=10)
    cap = open_camera(args.camera, args.width, args.height)
    snap_dir = ROOT / "data" / "snapshots"
    win = "LUMEN"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    t0, last, loop_fps = time.perf_counter(), time.perf_counter(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame not received; stopping.")
            break
        frame = cv.flip(frame, 1)  # mirror view
        now = time.perf_counter()
        t = now - t0
        loop_fps = 0.9 * loop_fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now

        faces = tracker.detect(frame)
        view = frame.copy()

        if not faces:
            status, color = "SEARCHING FOR SUBJECT", hud.DIM
            signal.reset()
        else:
            face = faces[0]
            rois = face.skin_rois(frame.shape)
            # Sample colour from the clean frame, not the drawn-on view
            signal.push(t, mean_color(frame, rois["forehead"]))
            hud.scanline(view, face.box, t)
            hud.brackets(view, face.box)
            hud.landmarks(view, face.landmarks)
            hud.roi_boxes(view, rois)
            if len(faces) > 1:
                status, color = f"MULTIPLE SUBJECTS ({len(faces)})", hud.ALERT
                for other in faces[1:]:
                    hud.brackets(view, other.box, hud.ALERT, thick=1)
            else:
                status, color = "SUBJECT LOCKED", hud.GOLD

        lines = [(status, color), (f"camera {loop_fps:4.1f} fps", hud.PEARL)]
        if faces:
            lines.append((f"face confidence {faces[0].score:.2f}", hud.PEARL))
        lines.append((f"signal {signal.duration:4.1f}s / {signal.seconds:.0f}s", hud.PEARL))
        hud.panel(view, lines)

        H, W = view.shape[:2]
        hud.waveform(view, signal.green_trace(), (14, H - 104, W - 28, 90))

        cv.imshow(win, view)
        key = cv.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            snap_dir.mkdir(parents=True, exist_ok=True)
            path = snap_dir / f"snap_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv.imwrite(str(path), view)
            print("Saved", path)
        if key == ord("r"):
            signal.reset()
        if cv.getWindowProperty(win, cv.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
