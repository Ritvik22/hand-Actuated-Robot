from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .geometry import angle_three_points, hand_translation


@dataclass
class HandState:
    landmarks: np.ndarray
    translation: np.ndarray
    finger_angles: np.ndarray


FINGERS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


class HandTracker:
    def __init__(self, camera_index: int = 0, width: int = 960, height: int = 720, synthetic: bool = False):
        self.synthetic = synthetic
        self.cap = None
        self.hands = None
        self.cv2 = None
        self.mp = None
        self.width = width
        self.height = height

        if not synthetic:
            import cv2
            import mediapipe as mp

            self.cv2 = cv2
            self.mp = mp
            self.cap = cv2.VideoCapture(camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        if self.hands is not None:
            self.hands.close()

    def _synthetic_landmarks(self) -> np.ndarray:
        t = time.time()
        base = np.zeros((21, 3), dtype=np.float32)
        for i in range(21):
            base[i] = np.array([0.5 + 0.05 * np.sin(t + i * 0.25), 0.5 + 0.04 * np.cos(t + i * 0.2), 0.02 * np.sin(t * 0.8 + i)])
        return base

    def read(self) -> tuple[HandState | None, np.ndarray | None]:
        if self.synthetic:
            landmarks = self._synthetic_landmarks()
            return self._to_state(landmarks), None

        ok, frame = self.cap.read()
        if not ok:
            return None, None

        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None, frame

        hand_lm = result.multi_hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float32)

        self.mp.solutions.drawing_utils.draw_landmarks(
            frame,
            hand_lm,
            self.mp.solutions.hands.HAND_CONNECTIONS,
        )

        return self._to_state(landmarks), frame

    def _to_state(self, landmarks: np.ndarray) -> HandState:
        angles = []
        for _, (mcp, pip, dip, tip) in FINGERS.items():
            pip_angle = angle_three_points(landmarks[mcp], landmarks[pip], landmarks[dip])
            dip_angle = angle_three_points(landmarks[pip], landmarks[dip], landmarks[tip])
            angles.extend([pip_angle, dip_angle])

        translation = hand_translation(landmarks)
        return HandState(landmarks=landmarks, translation=translation, finger_angles=np.array(angles, dtype=np.float32))
