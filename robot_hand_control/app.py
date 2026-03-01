from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .config import AppConfig
from .hand_tracker import HandTracker
from .imitation import Episode, EpisodeStore, ImitationPolicy
from .simulator import RobotHandSimulator


def _state_to_features(translation: np.ndarray, finger_angles: np.ndarray) -> np.ndarray:
    return np.concatenate([translation, finger_angles], axis=0).astype(np.float32)


def run_live(config: AppConfig, synthetic: bool = False) -> None:
    tracker = HandTracker(config.camera_index, config.width, config.height, synthetic=synthetic)
    sim = RobotHandSimulator(config.physics_timestep, gui=True)

    try:
        while True:
            state, frame = tracker.read()
            if state is not None:
                sim.apply_action(state.translation, state.finger_angles)
            sim.step()

            if frame is not None:
                import cv2

                cv2.imshow("Hand Tracking", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    sim.reset()
            elif synthetic:
                time.sleep(1.0 / config.control_rate_hz)
    finally:
        tracker.close()
        sim.close()


def run_record(config: AppConfig, name: str, duration: float, synthetic: bool = False) -> Path:
    tracker = HandTracker(config.camera_index, config.width, config.height, synthetic=synthetic)
    sim = RobotHandSimulator(config.physics_timestep, gui=True)
    store = EpisodeStore(config.episode_dir)

    observations = []
    actions = []

    start = time.time()
    try:
        while time.time() - start < duration:
            state, frame = tracker.read()
            if state is None:
                sim.step()
                continue

            obs = _state_to_features(state.translation, state.finger_angles)
            act = np.concatenate([state.translation, state.finger_angles], axis=0).astype(np.float32)

            observations.append(obs)
            actions.append(act)

            sim.apply_action(state.translation, state.finger_angles)
            sim.step()

            if frame is not None:
                import cv2

                cv2.putText(frame, f"Recording: {time.time() - start:0.1f}s/{duration}s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Hand Tracking", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    sim.reset()
            elif synthetic:
                time.sleep(1.0 / config.control_rate_hz)
    finally:
        tracker.close()
        sim.close()

    ep = Episode(np.vstack(observations), np.vstack(actions))
    return store.save_episode(name, ep)


def run_train(config: AppConfig) -> None:
    store = EpisodeStore(config.episode_dir)
    data = store.load_all()

    policy = ImitationPolicy()
    policy.train(data)
    policy.save(config.model_path)

    print(f"Trained model on {len(data.observations)} samples -> {config.model_path}")


def run_imitate(config: AppConfig, synthetic: bool = False) -> None:
    tracker = HandTracker(config.camera_index, config.width, config.height, synthetic=synthetic)
    sim = RobotHandSimulator(config.physics_timestep, gui=True)
    policy = ImitationPolicy()
    policy.load(config.model_path)

    try:
        while True:
            state, frame = tracker.read()
            if state is not None:
                obs = _state_to_features(state.translation, state.finger_angles)
                pred = policy.predict(obs)
                translation = pred[:3]
                finger = pred[3:]
                sim.apply_action(translation, finger)
            sim.step()

            if frame is not None:
                import cv2

                cv2.putText(frame, "Mode: Imitation", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
                cv2.imshow("Hand Tracking", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    sim.reset()
            elif synthetic:
                time.sleep(1.0 / config.control_rate_hz)
    finally:
        tracker.close()
        sim.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hand-actuated robot simulator")
    sub = parser.add_subparsers(dest="mode", required=True)

    live = sub.add_parser("live")
    live.add_argument("--synthetic", action="store_true")

    rec = sub.add_parser("record")
    rec.add_argument("--name", required=True)
    rec.add_argument("--duration", type=float, default=10.0)
    rec.add_argument("--synthetic", action="store_true")

    tr = sub.add_parser("train")

    imi = sub.add_parser("imitate")
    imi.add_argument("--synthetic", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig()

    if args.mode == "live":
        run_live(config, synthetic=args.synthetic)
    elif args.mode == "record":
        path = run_record(config, args.name, args.duration, synthetic=args.synthetic)
        print(f"Saved episode to {path}")
    elif args.mode == "train":
        run_train(config)
    elif args.mode == "imitate":
        run_imitate(config, synthetic=args.synthetic)


if __name__ == "__main__":
    main()
