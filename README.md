# H.A.N.D.S
H.A.N.D.S. (Hand-Activated Navigation &amp; Display System) — a real-time, local-first gesture control skill for E.D.I.T.H. Tracks your hand via webcam using MediaPipe and translates pinches, swipes, and grabs into Windows actions: window control, volume, scrolling, and clicks. No cloud, fully open source.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Project layout

```
src/hands/
  camera.py              webcam capture
  gesture_recognizer.py  MediaPipe hand landmarks -> gestures
  actions.py             gestures -> Windows actions
  config.py              tunable thresholds and camera settings
  main.py                wiring + CLI entry point
```
