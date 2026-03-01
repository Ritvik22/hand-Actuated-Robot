from __future__ import annotations

import numpy as np


def _safe_unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros_like(v)
    return v / n


def angle_three_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC in radians from 3D points."""
    ba = _safe_unit(a - b)
    bc = _safe_unit(c - b)
    dot = np.clip(np.dot(ba, bc), -1.0, 1.0)
    return float(np.arccos(dot))


def hand_translation(landmarks: np.ndarray) -> np.ndarray:
    """Estimate global hand translation from wrist and palm center."""
    wrist = landmarks[0]
    mcp_ids = [5, 9, 13, 17]
    palm_center = landmarks[mcp_ids].mean(axis=0)
    return (wrist + palm_center) / 2.0
