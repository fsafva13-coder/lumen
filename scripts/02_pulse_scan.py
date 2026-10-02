"""Milestone 2: live pulse (rPPG) liveness scan.

Run:   python scripts/02_pulse_scan.py
Keys:  q / Esc = quit   s = save snapshot   r = reset

Tips for a clean signal: sit still, face a window or lamp (light on your face,
not behind you), and give it ~8 seconds. Not a medical device.
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
from lumen.camera import describe, open_camera  # noqa: E402
from lumen.face import FaceTracker, SkinSampler  # noqa: E402
from lumen.pulse import MIN_SECONDS, estimate  # noqa: E402
from lumen.signals import SkinSignal  # noqa: E402

VERDICT = {
    "acquiring": ("ACQUIRING PULSE...", hud.DIM),
    "none": ("NO PULSE SIGNAL", hud.ALERT),
    "weak": ("WEAK PULSE SIGNAL", hud.ROSE),
    "strong": ("PULSE DETECTED", hud.GOLD),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--window", type=float, default=20.0, help="seconds of signal analysed")
    ap.add_argument("--lock-exposure", action="store_true", help="freeze camera exposure/white balance")
    args = ap.parse_args()

    tracker = FaceTracker()
    sampler = SkinSampler()
    signal = SkinSignal(seconds=args.window)
    cap = open_camera(args.camera, lock_exposure=args.lock_exposure)
    print("Camera:", describe(cap))
    win = "LUMEN pulse"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    t0 = time.perf_counter()
    result, last_est, bpm_smooth = None, 0.0, None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv.flip(frame, 1)
        t = time.perf_counter() - t0
        faces = tracker.detect(frame)
        view = frame.copy()

        if len(faces) != 1:
            signal.reset()
            sampler.reset()
            result, bpm_smooth = None, None
            msg = "SEARCHING FOR SUBJECT" if not faces else f"MULTIPLE SUBJECTS ({len(faces)})"
            hud.panel(view, [(msg, hud.ALERT if faces else hud.DIM)])
        else:
            face = faces[0]
            color_bgr, skin_frac, box = sampler.sample(frame, face)
            if color_bgr is not None:
                signal.push(t, color_bgr)

            if t - last_est > 0.5:  # re-estimate twice per second
                result = estimate(*signal.arrays())
                last_est = t
                if result.bpm is not None and result.quality != "none":
                    bpm_smooth = result.bpm if bpm_smooth is None else 0.7 * bpm_smooth + 0.3 * result.bpm

            hud.scanline(view, face.box, t)
            hud.brackets(view, face.box)
            hud.roi_boxes(view, {f"skin {skin_frac*100:.0f}%": box})

            quality = result.quality if result else "acquiring"
            text, color = VERDICT[quality]
            snr = f"{result.snr_db:+.1f} dB" if result and result.snr_db is not None else "--"
            progress = min(signal.duration / MIN_SECONDS, 1.0)
            hud.panel(view, [
                (text, color),
                (f"signal quality (SNR) {snr}", hud.PEARL),
                (f"buffer {signal.duration:4.1f}s  |  {signal.fps:4.1f} fps", hud.PEARL),
                ("" if progress >= 1 else f"warming up {progress*100:3.0f}%", hud.DIM),
            ])
            hud.quality_bar(view, result.snr_db if result else None, (24, 132, 280, 10))
            shown = f"{bpm_smooth:.0f}" if bpm_smooth and quality != "none" else "--"
            hud.readout(view, "PULSE", shown, "bpm", color)

        H, W = view.shape[:2]
        wave = result.waveform[-int(6 * 30):] if result is not None and len(result.waveform) else []
        hud.waveform(view, wave, (14, H - 104, W - 28, 90), label="PULSE WAVE (POS, 0.7-3 Hz)")
        hud._text(view, "liveness signal - not a medical device", (W - 300, H - 88), 0.38, hud.DIM)

        cv.imshow(win, view)
        key = cv.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv.getWindowProperty(win, cv.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("r"):
            signal.reset()
            result, bpm_smooth = None, None
        if key == ord("s"):
            d = ROOT / "data" / "snapshots"
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"pulse_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv.imwrite(str(p), view)
            print("Saved", p)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
