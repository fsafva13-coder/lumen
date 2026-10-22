"""Milestone 5: a full verification session.

Runs the challenges in the order a policy chooses and gives one decision:
VERIFIED, NOT A LIVE PERSON, or NOT VERIFIED (refer to a human).
No video is saved; only the measurements and the decision trace are logged.

Run:   python scripts/05_verify.py
       python scripts/05_verify.py --label print --subject ad01   (for evaluation runs)
Keys:  SPACE = start / run again     q / Esc = quit

WARNING: flashes full-screen colours. Not for anyone with photosensitive epilepsy.
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
from lumen import challenges as C  # noqa: E402
from lumen import light, motion  # noqa: E402
from lumen.camera import describe, open_camera  # noqa: E402
from lumen.face import FaceTracker  # noqa: E402
from lumen.policy import DECISION_TEXT, fixed_policy  # noqa: E402
from lumen.remote import Remote  # noqa: E402
from lumen.voice import Voice  # noqa: E402


def run_session(cap, tracker, win, voice, policy) -> dict:
    history, start = [], time.perf_counter()
    while True:
        action = policy(history)
        if action in DECISION_TEXT:
            break
        t0 = time.perf_counter()
        if action == "light":
            res = C.light_challenge(cap, tracker, win, voice)
            label, reason = light.verdict(res)
            step = {"challenge": "light"}
        else:
            direction = random.choice(["left", "right"])
            res = C.head_turn(cap, tracker, win, voice, direction)
            label, reason = motion.verdict(res, direction)
            step = {"challenge": "turn", "asked": direction}
        step.update(verdict=label, reason=reason, features=res.features() if res.ok else None,
                    seconds=round(time.perf_counter() - t0, 2))
        history.append(step)
        print(f"  {step['challenge']:5} -> {label:12} {reason}")
        if label != "live":   # tell the person why before the next step
            voice.say(reason + ".")
            C.message(win, step["challenge"].upper(), reason, hud.ROSE, seconds=2.2)
    return {"decision": action, "steps": history, "seconds": round(time.perf_counter() - start, 2)}


def run_session_cloud(cap, tracker, win, voice, remote: Remote) -> dict:
    """Same session, but OpenCV analysis and the policy run on AWS Lambda.
    Only face patches (48x48) and landmark points leave the laptop."""
    sid = f"s{int(time.time())}"
    start = time.perf_counter()
    resp = remote.start(sid)
    history, action = resp["history"], resp["next"]
    while action not in DECISION_TEXT:
        t0 = time.perf_counter()
        if action == "light":
            capt, ch = C.light_capture(cap, tracker, win, voice)
            C.message(win, "LIGHT", "Analysing on AWS...", hud.PEARL, seconds=0.1)
            resp = remote.light(sid, history, capt, ch)
        else:
            direction = random.choice(["left", "right"])
            times, lms = C.head_turn_capture(cap, tracker, win, voice, direction)
            C.message(win, "HEAD TURN", "Analysing on AWS...", hud.PEARL, seconds=0.1)
            resp = remote.turn(sid, history, times, lms, direction, C.STRAIGHT - 0.3)
        history, action, step = resp["history"], resp["next"], resp["step"]
        step["seconds"] = round(time.perf_counter() - t0, 2)
        print(f"  {step['challenge']:5} -> {step['verdict']:12} {step['reason']}  [AWS]")
        if step["verdict"] != "live":
            voice.say(step["reason"] + ".")
            C.message(win, step["challenge"].upper(), step["reason"], hud.ROSE, seconds=2.2)
    return {"decision": action, "steps": history, "seconds": round(time.perf_counter() - start, 2),
            "session_id": sid}


def show_decision(win, session) -> bool:
    title, _ = DECISION_TEXT[session["decision"]]
    colour = {"accept": hud.GOLD, "reject": hud.ALERT, "refer": hud.ROSE}[session["decision"]]
    canvas = np.full((C.CH, C.CW, 3), C.BG, np.uint8)
    hud._text(canvas, "LUMEN // DECISION", (40, 50), 0.9, hud.GOLD, 2)
    hud._text(canvas, title, (40, 140), 1.6, colour, 3)
    hud._text(canvas, f"decided in {session['seconds']:.0f} s over {len(session['steps'])} challenge(s)",
              (42, 180), 0.6, hud.DIM)
    for i, s in enumerate(session["steps"]):
        c = {"live": hud.GOLD, "fake": hud.ALERT, "inconclusive": hud.ROSE}[s["verdict"]]
        name = "LIGHT" if s["challenge"] == "light" else f"HEAD TURN ({s['asked']})"
        hud._text(canvas, f"{i + 1}. {name}", (60, 250 + 70 * i), 0.8, hud.PEARL, 2)
        hud._text(canvas, f"{s['verdict'].upper()} - {s['reason']}", (90, 280 + 70 * i), 0.6, c)
    hud._text(canvas, "SPACE = run again     q = quit", (40, C.CH - 45), 0.6, hud.GOLD)
    cv.imshow(win, canvas)
    while True:
        key = cv.waitKey(30) & 0xFF
        if key == ord(" "):
            return True
        if key in (ord("q"), 27) or cv.getWindowProperty(win, cv.WND_PROP_VISIBLE) < 1:
            return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="", help="ground truth for evaluation: real / print / screen")
    ap.add_argument("--subject", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--mute", action="store_true")
    ap.add_argument("--cloud", default="", help="AWS Lambda function URL: run the analysis on AWS")
    ap.add_argument("--key", default="", help="access key for the cloud function")
    args = ap.parse_args()

    tracker, voice = FaceTracker(), Voice(enabled=not args.mute)
    cap = open_camera(args.camera)  # auto exposure; locked only during the light flashes
    cam_mode = describe(cap)
    print("Camera:", cam_mode)
    win = "LUMEN"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.setWindowProperty(win, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    log_dir = ROOT / "data" / "sessions"

    hint = "Screen brightness max, 30-45 cm away. Flashing colours, then a head turn."
    while C.wait_ready(cap, tracker, win, "VERIFICATION", hint):
        print("Session:")
        if args.cloud:
            session = run_session_cloud(cap, tracker, win, voice, Remote(args.cloud, args.key))
        else:
            session = run_session(cap, tracker, win, voice, fixed_policy)
        session.update(policy="fixed", where="aws" if args.cloud else "local", label=args.label, subject=args.subject, note=args.note,
                       camera=cam_mode, recorded_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        title, spoken = DECISION_TEXT[session["decision"]]
        print(f"  DECISION: {title} ({session['seconds']} s)")
        voice.say(spoken)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"session_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(session, indent=2))
        if not show_decision(win, session):
            break

    voice.close()
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
