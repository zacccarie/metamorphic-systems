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


def multi_feature_betas(signal: np.ndarray, window: int = 64, stride: int = 8) -> np.ndarray:
    """Compute β across 4 transforms : raw, |signal|, diff, signal².

    Each transform captures different anomaly aspect :
    - raw : spectral regime change, distribution shift
    - |signal| : amplitude shifts, variance clustering
    - diff : trend / drift detection
    - signal² : second-order moments, variance shifts

    Returns: shape (n_windows, 4) of β values per transform.
    """
    signal = np.asarray(signal).astype('float32')
    if signal.ndim > 1:
        signal = signal[:, 0]
    transforms = [
        signal,
        np.abs(signal),
        np.diff(signal, prepend=signal[0]),
        signal ** 2,
    ]
    betas = [sliding_beta(t, window, stride) for t in transforms]
    n = min(len(b) for b in betas)
    return np.stack([b[:n] for b in betas], axis=1)  # (N, 4)


class BetaMultiFeatureDetector:
    """β-detector enhanced with multi-transform features.

    Computes β across [raw, |signal|, diff, signal²] transforms.
    Anomaly score = max |z-score| across 4 features (worst-case detection).

    Catches amplitude/variance/trend anomalies that pure β-meter misses.
    """

    def __init__(self, window: int = 64, stride: int = 8):
        self.window = window
        self.stride = stride
        self.baseline_means: np.ndarray | None = None
        self.baseline_stds: np.ndarray | None = None

    def fit(self, normal_signal: np.ndarray) -> None:
        betas_4d = multi_feature_betas(normal_signal, self.window, self.stride)
        self.baseline_means = betas_4d.mean(axis=0)
        self.baseline_stds = betas_4d.std(axis=0) + 1e-9

    def score(self, signal: np.ndarray) -> np.ndarray:
        if self.baseline_means is None:
            raise ValueError("Call fit() first")
        betas_4d = multi_feature_betas(signal, self.window, self.stride)
        z = np.abs(betas_4d - self.baseline_means) / self.baseline_stds  # (N, 4)
        return z.max(axis=1)  # worst-case per window

    def feature_z(self, signal: np.ndarray) -> np.ndarray:
        """Return full 4D z-score matrix for ensemble use."""
        if self.baseline_means is None:
            raise ValueError("Call fit() first")
        betas_4d = multi_feature_betas(signal, self.window, self.stride)
        return np.abs(betas_4d - self.baseline_means) / self.baseline_stds  # (N, 4)


class BetaEnsembleDetector:
    """Hybrid : β multi-feature + Isolation Forest stack.

    β captures regime/dist shifts.
    IsoForest captures moment shifts (amplitude/variance).
    Ensemble = max(β z-score normalized, IsoForest score normalized).

    Hypothesis : ensemble dominates each individual on aggregate AUROC.
    """

    def __init__(self, window: int = 64, stride: int = 8, n_estimators: int = 100):
        from sklearn.ensemble import IsolationForest
        self.window = window
        self.stride = stride
        self.beta_det = BetaMultiFeatureDetector(window, stride)
        self.iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=n_estimators)
        self.iso_baseline_mean: float | None = None
        self.iso_baseline_std: float | None = None

    def _windowize(self, signal: np.ndarray) -> np.ndarray:
        sig = np.asarray(signal).astype('float32')
        if sig.ndim > 1:
            sig = sig[:, 0]
        return np.stack([sig[i:i + self.window]
                          for i in range(0, len(sig) - self.window, self.stride)])

    def fit(self, normal_signal: np.ndarray) -> None:
        self.beta_det.fit(normal_signal)
        X = self._windowize(normal_signal)
        self.iso.fit(X)
        # Calibrate IsoForest scores on normal train to compute z-score
        iso_scores = -self.iso.score_samples(X)
        self.iso_baseline_mean = float(iso_scores.mean())
        self.iso_baseline_std = float(iso_scores.std() + 1e-9)

    def score(self, signal: np.ndarray) -> np.ndarray:
        beta_z = self.beta_det.score(signal)  # (N_beta,)
        X = self._windowize(signal)
        iso_raw = -self.iso.score_samples(X)
        iso_z = (iso_raw - self.iso_baseline_mean) / self.iso_baseline_std

        n = min(len(beta_z), len(iso_z))
        return np.maximum(beta_z[:n], iso_z[:n])
