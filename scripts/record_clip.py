"""Record labelled clips for the LUMEN evaluation set.

Examples:
  python scripts/record_clip.py --label real   --subject safva
  python scripts/record_clip.py --label print  --subject safva --note "A4 colour print"
  python scripts/record_clip.py --label screen --subject safva --note "phone replay"

Only record people who have agreed to be recorded. Clips stay in data/raw
(git-ignored) and are never uploaded.
Keys: Space = start recording, q / Esc = quit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2 as cv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lumen import hud  # noqa: E402
from lumen.camera import describe, open_camera  # noqa: E402

LABELS = ["real", "print", "screen", "mask", "deepfake"]


def make_writer(base: Path, fps: float, size: tuple[int, int]):
    """Lossless FFV1 keeps the faint pulse signal intact; fall back to MJPG."""
    for fourcc, ext in (("FFV1", ".avi"), ("MJPG", ".avi")):
        path = base.with_suffix(ext)
        w = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            return w, path, fourcc
    sys.exit("No working video codec found (tried FFV1, MJPG).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=LABELS)
    ap.add_argument("--subject", required=True, help="short id, e.g. safva or p01")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--lock-exposure", action="store_true", help="freeze camera exposure/white balance")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--note", default="", help="lighting, device, glasses, etc.")
    args = ap.parse_args()

    cap = open_camera(args.camera, lock_exposure=args.lock_exposure)
    camera_mode = describe(cap)
    print("Camera:", camera_mode)

    out_dir = ROOT / "data" / "raw" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    win = "LUMEN recorder"
    cv.namedWindow(win, cv.WINDOW_NORMAL)

    recording, writer, start, frames = False, None, 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        view = cv.flip(frame, 1)
        if recording:
            writer.write(frame)  # save the unmirrored, unmodified frame
            frames += 1
            left = args.seconds - (time.perf_counter() - start)
            hud.panel(view, [(f"REC  {args.label.upper()}  {left:4.1f}s", hud.ALERT)])
            if left <= 0:
                break
        else:
            hud.panel(view, [(f"label: {args.label}  subject: {args.subject}", hud.PEARL),
                             ("SPACE to record, q to quit", hud.GOLD)])
        cv.imshow(win, view)
        key = cv.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" ") and not recording:
            fps = cap.get(cv.CAP_PROP_FPS) or 30
            h, w = frame.shape[:2]
            stamp = time.strftime("%Y%m%d_%H%M%S")
            writer, path, codec = make_writer(out_dir / f"{args.subject}_{stamp}", fps, (w, h))
            recording, start = True, time.perf_counter()

    cap.release()
    cv.destroyAllWindows()
    if writer is not None:
        writer.release()
        elapsed = time.perf_counter() - start
        meta = {
            "file": path.name, "label": args.label, "subject": args.subject,
            "codec": codec, "frames": frames, "seconds": round(elapsed, 2),
            "measured_fps": round(frames / max(elapsed, 1e-6), 2),
            "camera": camera_mode, "exposure_locked": args.lock_exposure,
            "note": args.note, "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(f"Saved {path} ({frames} frames, {meta['measured_fps']} fps, {codec})")


if __name__ == "__main__":
    main()
