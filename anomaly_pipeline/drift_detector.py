import numpy as np


class DriftDetector:
    """
    Implements statistical checks to identify distribution shifts (Concept Drift).
    """

    @staticmethod
    def calculate_psi(baseline: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
        hist_base, bins = np.histogram(baseline, bins=buckets)
        hist_curr, _ = np.histogram(current, bins=bins)

        base_per = np.clip(hist_base / len(baseline), 0.001, 1)
        curr_per = np.clip(hist_curr / len(current), 0.001, 1)

        return np.sum((base_per - curr_per) * np.log(base_per / curr_per))