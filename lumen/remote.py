"""Client for the LUMEN analysis function on AWS Lambda (standard library only)."""
from __future__ import annotations

import base64
import io
import json
import sys
import urllib.error
import urllib.request

import numpy as np

from .light import Capture, Challenge


class Remote:
    def __init__(self, url: str, key: str = "", timeout: float = 30.0):
        self.url, self.key, self.timeout = url, key, timeout

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"content-type": "application/json", "x-lumen-key": self.key})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 401:
                sys.exit("AWS refused the key. Copy it again exactly (in CloudShell: cat ~/lumen_key.txt) "
                         "and put it in quotes: --key \"...\"")
            sys.exit(f"AWS returned an error {e.code}: {e.read()[:300]!r}")
        except urllib.error.URLError as e:
            sys.exit(f"Could not reach AWS ({e.reason}). Check the internet connection and the --cloud URL.")

    def start(self, session: str) -> dict:
        return self._post({"session": session, "history": []})

    def light(self, session: str, history: list, cap: Capture, ch: Challenge) -> dict:
        buf = io.BytesIO()
        np.save(buf, np.clip(np.stack(cap.patches), 0, 255).astype(np.uint8) if cap.patches
                else np.zeros((0, 48, 48, 3), np.uint8))
        return self._post({
            "session": session, "history": history, "challenge": "light",
            "seed": ch.seed, "step_seconds": ch.step_seconds, "codes_rgb": ch.codes.tolist(),
            "times": cap.times, "patches_npy": base64.b64encode(buf.getvalue()).decode(),
            "background": [np.asarray(b).tolist() for b in cap.background],
            "face_widths": cap.face_widths, "centres": [list(c) for c in cap.centres],
            "frames_total": cap.frames_total,
        })

    def turn(self, session: str, history: list, times, landmarks, direction: str, ref_seconds: float) -> dict:
        return self._post({
            "session": session, "history": history, "challenge": "turn", "direction": direction,
            "times": list(times), "landmarks": [np.asarray(l).tolist() for l in landmarks],
            "ref_seconds": ref_seconds,
        })
