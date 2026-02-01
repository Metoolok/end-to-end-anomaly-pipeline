from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    """
    Centralized configuration management for the anomaly detection pipeline.
    Encapsulates all hyperparameters and directory paths.
    """
    BASE_DIR: Path = Path("artifacts")
    MODEL_PATH: Path = BASE_DIR / "iso_forest.pkl"
    SCALER_PATH: Path = BASE_DIR / "scaler.pkl"

    STL_PERIOD: int = 24
    CONTAMINATION: float = 0.05
    RANDOM_SEED: int = 42
    PSI_THRESHOLD: float = 0.2

    def __post_init__(self):
        self.BASE_DIR.mkdir(parents=True, exist_ok=True)