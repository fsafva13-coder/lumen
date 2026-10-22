"""LUMEN cloud analysis: AWS Lambda entry point.

The laptop runs the camera and the screen flashes. It sends this function
only what the analysis needs - 48x48 face patches for the light challenge,
5 landmark points per frame for the head turn - never full video. Here,
OpenCV 5 does the analysis, the policy decides the next step, and every
decision is written to CloudWatch as one structured log line.
Nothing is stored: the patches exist only for the length of the request.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time

import numpy as np

from lumen import light, motion
from lumen.policy import fixed_policy

API_KEY = os.environ.get("LUMEN_KEY", "")


def _respond(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": {"content-type": "application/json"}, "body": json.dumps(body)}


def _light(req: dict):
    ch = light.Challenge(int(req["seed"]), float(req["step_seconds"]), np.array(req["codes_rgb"]))
    cap = light.Capture()
    cap.times = list(req["times"])
    cap.patches = list(np.load(io.BytesIO(base64.b64decode(req["patches_npy"]))).astype(np.float32))
    cap.background = [np.array(b) for b in req["background"]]
    cap.face_widths = list(req["face_widths"])
    cap.centres = [tuple(c) for c in req["centres"]]
    cap.frames_total = int(req["frames_total"])
    res = light.analyze(cap, ch)
    return res, light.verdict(res)


def _turn(req: dict):
    res = motion.analyze(req["times"], np.array(req["landmarks"]), ref_seconds=float(req.get("ref_seconds", 1.7)))
    return res, motion.verdict(res, req.get("direction"))


def lambda_handler(event, context):
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    if API_KEY and headers.get("x-lumen-key") != API_KEY:
        return _respond(401, {"error": "unauthorised"})
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode()
    req = json.loads(raw)
    t0 = time.perf_counter()

    step = None
    if req.get("challenge") == "light":
        res, (label, reason) = _light(req)
        step = {"challenge": "light"}
    elif req.get("challenge") == "turn":
        res, (label, reason) = _turn(req)
        step = {"challenge": "turn", "asked": req.get("direction")}
    history = list(req.get("history", []))
    if step is not None:
        step.update(verdict=label, reason=reason, features=res.features() if res.ok else None)
        history.append(step)
    action = fixed_policy(history)

    print(json.dumps({"lumen": "decision", "session": req.get("session"), "step": step,
                      "next": action, "steps_so_far": len(history),
                      "analysis_ms": round((time.perf_counter() - t0) * 1000, 1)}))
    return _respond(200, {"step": step, "next": action, "history": history})
