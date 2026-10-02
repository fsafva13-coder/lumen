"""Re-run the head-turn analysis on saved clips and print a table.

  python scripts/analyze_turn.py data/raw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lumen.motion import analyze, verdict  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    args = ap.parse_args()
    metas = sorted(args.path.rglob("*_turn.json"))
    if not metas:
        sys.exit("No *_turn.json clips found. Record some with scripts/04_head_turn.py")
    print(f"{'label':7} {'subject':7} {'verdict':12} {'parallax':>8} {'shift':>6} {'narrow%':>7} {'eyes':>5} {'asked':>6}  note")
    for m in metas:
        meta = json.loads(m.read_text())
        res = analyze(meta["times"], np.array(meta["landmarks"]), ref_seconds=0.9)
        v = verdict(res)[0]
        if res.ok:
            print(f"{meta['label']:7} {meta['subject']:7} {v:12} {res.parallax:8.3f} {res.nose_shift:+6.2f} "
                  f"{(1 - res.foreshorten) * 100:7.1f} {res.face_px:5.0f} {meta['direction_asked']:>6}  {meta.get('note', '')}")
        else:
            print(f"{meta['label']:7} {meta['subject']:7} {v:12} -- {res.reason}  {meta.get('note', '')}")


if __name__ == "__main__":
    main()
