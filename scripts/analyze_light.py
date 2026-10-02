"""Re-run the light-challenge analysis on saved clips and print a table.

  python scripts/analyze_light.py data/raw
  python scripts/analyze_light.py data/raw --csv results/light.csv
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
from lumen.face import FaceTracker  # noqa: E402
from lumen.light import Challenge, analyze, capture_from_frames, verdict  # noqa: E402


def load(meta_path: Path):
    meta = json.loads(meta_path.read_text())
    cap = cv.VideoCapture(str(meta_path.with_suffix(".avi")))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    ch = Challenge(meta["seed"], meta["step_seconds"], np.array(meta["codes_rgb"]))
    n = min(len(frames), len(meta["times"]))
    return meta, frames[:n], meta["times"][:n], ch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()
    metas = sorted(args.path.rglob("*_light.json")) if args.path.is_dir() else [args.path.with_suffix(".json")]
    if not metas:
        sys.exit("No *_light.json clips found. Record some with scripts/03_light_challenge.py")
    rows = []
    print(f"{'label':7} {'subject':7} {'verdict':12} {'r2':>5} {'gain%':>6} {'relief':>6} {'bg%':>6} {'face':>5} {'move':>5}  note")
    for m in metas:
        meta, frames, times, ch = load(m)
        res = analyze(capture_from_frames(frames, times, FaceTracker()), ch)
        row = {"label": meta["label"], "subject": meta["subject"], "clip": m.stem, "note": meta.get("note", ""),
               "ok": res.ok, "verdict": verdict(res)[0], **(res.features() if res.ok else {})}
        rows.append(row)
        if res.ok:
            print(f"{row['label']:7} {row['subject']:7} {row['verdict']:12} {res.r2:5.2f} {res.gain*100:6.1f} "
                  f"{res.relief:6.3f} {res.bg_gain*100:6.1f} {res.face_px:5.0f} {res.motion:5.3f}  {row['note']}")
        else:
            print(f"{row['label']:7} {row['subject']:7} inconclusive -- {res.reason}  {row['note']}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print("Wrote", args.csv)


if __name__ == "__main__":
    main()
