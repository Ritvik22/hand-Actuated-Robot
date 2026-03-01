# Hand-Actuated Robot (Computer Vision + 3D Simulation + Imitation Learning)

This project lets a user control a simulated robot hand in real-time using their own hand tracked from a webcam, then train the robot on recorded 10-second episodes so it can imitate learned actions.

## Features

- **Computer vision hand tracking** (MediaPipe + OpenCV)
- **Finger joint angle extraction** from landmarks
- **3D robot simulation** in PyBullet with interactive objects
- **Real-time teleoperation** by moving your hand in camera view
- **Episode recording** (default 10 seconds)
- **Imitation learning** from recorded episodes
- **Model-driven replay** of learned actions
- **Deterministic reset** so each run starts from same setup

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Live control (hand -> robot)

```bash
python -m robot_hand_control.app live
```

### 2) Record an episode (10s default)

```bash
python -m robot_hand_control.app record --name pick_like --duration 10
```

### 3) Train imitation model

```bash
python -m robot_hand_control.app train
```

### 4) Run learned behavior

```bash
python -m robot_hand_control.app imitate
```

## Controls

- `r`: reset environment to the same initial setup
- `q`: quit

## Notes

- If `mediapipe` or camera access is unavailable, the app can run with synthetic landmarks (`--synthetic`) for debugging.
- Model artifacts and episodes are stored under `artifacts/`.
