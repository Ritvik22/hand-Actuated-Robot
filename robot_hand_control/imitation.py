from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Episode:
    observations: np.ndarray
    actions: np.ndarray


class EpisodeStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_episode(self, name: str, episode: Episode) -> Path:
        path = self.root / f"{name}.npz"
        np.savez_compressed(path, observations=episode.observations, actions=episode.actions)
        return path

    def load_all(self) -> Episode:
        obs = []
        acts = []
        for f in sorted(self.root.glob("*.npz")):
            d = np.load(f)
            obs.append(d["observations"])
            acts.append(d["actions"])
        if not obs:
            raise FileNotFoundError("No episodes found. Record at least one episode first.")
        return Episode(observations=np.vstack(obs), actions=np.vstack(acts))


class ImitationPolicy:
    def __init__(self):
        base = MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu", max_iter=600, random_state=42)
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", MultiOutputRegressor(base)),
        ])

    def train(self, episode: Episode) -> None:
        self.model.fit(episode.observations, episode.actions)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        return self.model.predict(obs.reshape(1, -1))[0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)
