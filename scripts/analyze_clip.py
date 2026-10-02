"""Run the pulse scan on recorded clips and print BPM / SNR per clip.

  python scripts/analyze_clip.py data/raw/real/safva_20261001_190000.avi
  python scripts/analyze_clip.py data/raw            # every clip, all labels
  python scripts/analyze_clip.py data/raw --csv results/pulse.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lumen.face import FaceTracker, SkinSampler  # noqa: E402
from lumen.pulse import estimate  # noqa: E402


def analyze(path: Path, tracker: FaceTracker) -> dict:
    cap = cv.VideoCapture(str(path))
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    meta = path.with_suffix(".json")
    if meta.exists():  # the true capture rate, not the nominal one in the file header
        fps = json.loads(meta.read_text()).get("measured_fps") or fps
    ts, rgb, i, missed = [], [], 0, 0
    sampler = SkinSampler()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = tracker.detect(frame)
        color = sampler.sample(frame, faces[0])[0] if faces else None
        if color is not None:
            ts.append(i / fps)
            rgb.append(color[::-1])
        else:
            missed += 1
        i += 1
    cap.release()
    r = estimate(np.array(ts), np.array(rgb)) if len(ts) > 1 else None
    return {
        "clip": path.name, "label": path.parent.name, "frames": i, "no_face": missed,
        "bpm": round(r.bpm, 1) if r and r.bpm else "", "snr_db": round(r.snr_db, 2) if r and r.snr_db is not None else "",
        "quality": r.quality if r else "no_face",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()
    clips = (sorted(c for c in args.path.rglob("*.avi") if not c.stem.endswith(("_light", "_turn")))
             if args.path.is_dir() else [args.path])
    if not clips:
        sys.exit("No .avi clips found.")
    tracker = FaceTracker()
    rows = []
    print(f"{'label':8} {'quality':9} {'bpm':>6} {'snr_db':>7}  clip")
    for c in clips:
        r = analyze(c, tracker)
        rows.append(r)
        print(f"{r['label']:8} {r['quality']:9} {str(r['bpm']):>6} {str(r['snr_db']):>7}  {r['clip']}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("Wrote", args.csv)


if __name__ == "__main__":
    main()
