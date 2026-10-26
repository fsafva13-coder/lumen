"""Score labelled verification sessions (data/sessions/*.json).

  python scripts/score_sessions.py              # all labelled sessions
  python scripts/score_sessions.py --aws-only   # only sessions analysed on AWS
  python scripts/score_sessions.py --md results/sessions.md

Metrics follow ISO/IEC 30107-3 presentation-attack-detection terms:
  APCER  attacks wrongly accepted as live   (lower is better)
  BPCER  real people not accepted           (lower is better)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ATTACKS = {"print", "screen", "mask", "deepfake"}


def load(aws_only: bool) -> list[dict]:
    out = []
    for f in sorted((ROOT / "data" / "sessions").glob("session_*.json")):
        s = json.loads(f.read_text())
        if not s.get("label"):
            continue
        if aws_only and s.get("where") != "aws":
            continue
        s["_file"] = f.name
        out.append(s)
    return out


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.0f}% ({a}/{b})" if b else "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aws-only", action="store_true")
    ap.add_argument("--md", type=Path, help="also write a Markdown table here")
    args = ap.parse_args()
    sessions = load(args.aws_only)
    if not sessions:
        sys.exit("No labelled sessions found. Run 05_verify.py with --label first.")

    by_label = defaultdict(list)
    for s in sessions:
        by_label[s["label"]].append(s)

    rows = []
    for label, ss in sorted(by_label.items()):
        dec = Counter(s["decision"] for s in ss)
        n = len(ss)
        secs = sorted(s["seconds"] for s in ss)
        steps = sum(len(s["steps"]) for s in ss) / n
        caught_at = Counter()
        for s in ss:
            if s["decision"] != "accept" and s["steps"]:
                last_bad = next((st["challenge"] for st in s["steps"] if st["verdict"] != "live"), "-")
                caught_at[last_bad] += 1
        rows.append({
            "label": label, "n": n, "accept": dec["accept"], "reject": dec["reject"], "refer": dec["refer"],
            "median_s": secs[len(secs) // 2], "steps": steps, "caught_at": dict(caught_at),
        })

    lines = ["| Presented | Sessions | Verified | Rejected | Referred to human | Median time | Avg. challenges |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['n']} | {r['accept']} | {r['reject']} | {r['refer']} | "
                     f"{r['median_s']:.0f} s | {r['steps']:.1f} |")
    real = by_label.get("real", [])
    attacks = [s for l, ss in by_label.items() if l in ATTACKS for s in ss]
    a_acc = sum(s["decision"] == "accept" for s in attacks)
    r_not = sum(s["decision"] != "accept" for s in real)
    summary = [
        "",
        f"APCER (attacks accepted as live): {pct(a_acc, len(attacks))}",
        f"BPCER (real people not verified): {pct(r_not, len(real))}",
        f"Sessions analysed on AWS: {sum(s.get('where') == 'aws' for s in sessions)} of {len(sessions)}",
    ]
    for r in rows:
        if r["caught_at"]:
            summary.append(f"  {r['label']}: not verified at -> " +
                           ", ".join(f"{k} {v}" for k, v in sorted(r["caught_at"].items())))
    text = "\n".join(lines + summary)
    print(text)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(text + "\n")
        print("\nWrote", args.md)


if __name__ == "__main__":
    main()
