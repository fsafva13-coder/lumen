# LUMEN

**An agentic proof-of-life vision system that interrogates a face with light, pulse, and motion to tell a living person from a fake.**

![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8?logo=opencv&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-Lambda%20%7C%20Bedrock%20%7C%20S3-FF9900?logo=amazonaws&logoColor=white)
![Status](https://img.shields.io/badge/status-in%20development-c9a96e)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## Project Overview

Face verification for online exams, banking, and remote check-ins is increasingly beaten by printed photos, screen replays, and deepfakes. LUMEN combines three independent liveness signals — **screen-light reflection**, **remote pulse (rPPG)**, and **challenge-driven head motion** — and an AI agent that picks the next challenge based on what OpenCV 5 has observed so far. The goal is anti-spoofing that is harder to fool than any single check, while reaching a decision faster than running every test.

## Features

- **Face lock** — OpenCV 5 YuNet detection with 5-point landmarks and skin ROIs *(done)*
- **Skin-signal acquisition** — live forehead colour trace for pulse extraction *(done)*
- **Pulse scan** — blood-flow signal from facial video (POS rPPG) with SNR-based quality *(done)*
- **Light challenge** — random screen colour sequences; checks that a real 3D face reflects them *(planned)*
- **Screen-replay detection** — FFT moiré analysis *(planned)*
- **Motion challenge** — head-pose verification with `solvePnP` *(planned)*
- **Adaptive agent** — Amazon Bedrock selects the next challenge, escalates to a human when uncertain *(planned)*
- **Privacy by design** — video deleted after scoring; only scores and decision traces are kept *(planned)*

## Tech Stack

| Layer | Tools |
|---|---|
| Vision | OpenCV 5 (YuNet, DNN, `solvePnP`, DFT), NumPy, SciPy |
| Agent | Amazon Bedrock (tool use) |
| Cloud | AWS Lambda (container), API Gateway, S3, DynamoDB, CloudWatch |
| Frontend | HTML/JS webcam client with HUD overlay |

## Screenshots / Demo

| Face lock HUD | Light challenge | Agent decision trace |
|---|---|---|
| `[screenshot: docs/face_lock.png]` | `[screenshot: docs/light_challenge.png]` | `[screenshot: docs/agent_trace.png]` |

## Installation

```bash
git clone https://github.com/fsafva13-coder/lumen.git
cd lumen
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py
```

## Usage

```bash
# Live face lock + skin signal HUD   (q = quit, s = snapshot, r = reset signal)
python scripts/01_face_scan.py

# Live pulse liveness scan (BPM + signal quality)
python scripts/02_pulse_scan.py

# Record labelled clips for evaluation (consenting subjects only)
python scripts/record_clip.py --label real  --subject p01
python scripts/record_clip.py --label print --subject p01 --note "A4 colour print"

# Batch-analyse recorded clips
python scripts/analyze_clip.py data/raw --csv results/pulse.csv
```

## Project Structure

```
lumen/
├── lumen/                 # core library
│   ├── face.py            # YuNet face tracker + skin ROIs
│   ├── signals.py         # rolling skin-colour buffer
│   ├── pulse.py           # POS rPPG + SNR
│   ├── camera.py          # webcam helper
│   └── hud.py             # scanning overlay
├── scripts/
│   ├── 01_face_scan.py    # milestone 1: face lock
│   ├── 02_pulse_scan.py   # milestone 2: pulse liveness
│   ├── record_clip.py     # evaluation-set recorder
│   ├── analyze_clip.py    # batch pulse analysis
│   └── download_models.py
├── models/                # YuNet ONNX (OpenCV Zoo)
├── data/raw/              # local clips (git-ignored)
└── docs/                  # report, diagrams, screenshots
```

## Results / Metrics

Evaluated with ISO/IEC 30107-3 presentation-attack metrics.

| Configuration | APCER ↓ | BPCER ↓ | Avg. time to decision |
|---|---|---|---|
| Pulse only | `TBD` | `TBD` | `TBD` |
| Light only | `TBD` | `TBD` | `TBD` |
| All checks (fixed) | `TBD` | `TBD` | `TBD` |
| **LUMEN agent (adaptive)** | `TBD` | `TBD` | `TBD` |

## Challenges & Learnings

- `[e.g. pulse signal quality under low light and webcam auto-exposure]`
- `[e.g. performance across skin tones and with glasses]`

## Future Improvements

- Mobile client with front-flash illumination
- Evaluation on public anti-spoofing benchmarks
- On-device inference for full offline operation

## Demo / Live Link

- Live endpoint: `[coming soon]`
- Video (≤ 5 min): `[coming soon]`

## License

MIT — see `LICENSE`. YuNet model © OpenCV Zoo contributors, MIT licence.
