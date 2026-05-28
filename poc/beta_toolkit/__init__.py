"""β-toolkit : spectral 1/f^β anomaly detection + ODE-based synthetic time-series.

Findings derived from metamorphic-systems research project.

Modules :
- meter : β computation, sliding-window anomaly detector
- synthesizer : ODE Euler generator producing β-controlled signals
"""
from .meter import compute_beta, sliding_beta, BetaAnomalyDetector
from .synthesizer import ODESynthesizer, train_synthesizer

__version__ = "0.1.0"
__all__ = [
    "compute_beta",
    "sliding_beta",
    "BetaAnomalyDetector",
    "ODESynthesizer",
    "train_synthesizer",
]
