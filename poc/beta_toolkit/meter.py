"""β-meter : compute 1/f^β spectral slope. Anomaly detection via β shift."""
from __future__ import annotations
import numpy as np


def compute_beta(signal: np.ndarray) -> float:
    """Compute spectral slope β tel que power ~ 1/f^β.

    Args:
        signal: 1D ou 2D array. Si 2D, shape (T, D) → mean power across D channels.

    Returns:
        β value (slope du log power vs log freq, signe positif convention).
        β=0 = white noise, β=1 = pink noise, β=2 = brown/Brownian, β=3+ = très smooth.
    """
    signal = np.asarray(signal)
    if signal.ndim == 1:
        signal = signal[:, None]
    if signal.shape[0] < 16:
        return 0.0
    fft = np.fft.rfft(signal, axis=0)
    power = (np.abs(fft) ** 2).mean(axis=1)
    freqs = np.arange(1, len(power))
    log_f = np.log(freqs + 1e-9)
    log_p = np.log(power[1:] + 1e-12)
    slope, _ = np.polyfit(log_f, log_p, 1)
    return float(-slope)


def sliding_beta(signal: np.ndarray, window: int = 64, stride: int = 8) -> np.ndarray:
    """Compute β over sliding windows. Returns array of β per window."""
    signal = np.asarray(signal)
    if signal.ndim > 1:
        # Use first channel only for sliding scan
        signal = signal[:, 0] if signal.shape[1] > 0 else signal.ravel()
    betas = []
    for i in range(0, len(signal) - window, stride):
        betas.append(compute_beta(signal[i:i + window]))
    return np.array(betas)


class BetaAnomalyDetector:
    """Zero-shot anomaly detector via β shift detection.

    Fit baseline β stats on normal data → score new data deviation.

    Usage:
        det = BetaAnomalyDetector(window=64)
        det.fit(normal_signal)
        scores = det.score(test_signal)
        anomaly_mask = scores > threshold
    """

    def __init__(self, window: int = 64, stride: int = 8):
        self.window = window
        self.stride = stride
        self.baseline_mean: float | None = None
        self.baseline_std: float | None = None

    def fit(self, normal_signal: np.ndarray) -> None:
        """Compute baseline β distribution on normal data."""
        betas = sliding_beta(normal_signal, self.window, self.stride)
        self.baseline_mean = float(betas.mean())
        self.baseline_std = float(betas.std() + 1e-9)

    def score(self, signal: np.ndarray) -> np.ndarray:
        """Anomaly score = |β(window) - baseline_β| / baseline_std (z-score)."""
        if self.baseline_mean is None:
            raise ValueError("Call fit() first")
        betas = sliding_beta(signal, self.window, self.stride)
        return np.abs(betas - self.baseline_mean) / self.baseline_std

    def predict(self, signal: np.ndarray, threshold: float = 2.0) -> np.ndarray:
        """Binary anomaly mask (1=anomaly) based on z-score threshold."""
        return (self.score(signal) > threshold).astype(int)
