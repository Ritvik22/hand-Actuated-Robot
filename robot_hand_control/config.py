from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    camera_index: int = 0
    width: int = 960
    height: int = 720
    physics_timestep: float = 1.0 / 120.0
    control_rate_hz: float = 30.0
    record_duration_s: float = 10.0
    episode_dir: Path = field(default_factory=lambda: Path("artifacts/episodes"))
    model_path: Path = field(default_factory=lambda: Path("artifacts/imitation_model.joblib"))
