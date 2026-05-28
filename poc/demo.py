"""Demo β-toolkit : anomaly detection + ODE synthesizer."""
from __future__ import annotations
import numpy as np
import sys
sys.path.insert(0, '.')
from beta_toolkit import compute_beta, BetaAnomalyDetector, ODESynthesizer, train_synthesizer


def gen_colored_noise(n: int, beta: float, seed: int = 0) -> np.ndarray:
    """Generate 1/f^beta colored noise length n."""
    rng = np.random.default_rng(seed)
    white = rng.normal(0, 1, n)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1e-6
    scale = freqs ** (-beta / 2)
    fft = fft * scale
    sig = np.fft.irfft(fft, n=n)
    return ((sig - sig.mean()) / (sig.std() + 1e-9)).astype('float32')


def demo_anomaly_detection():
    print("\n=== Demo 1 : β-Anomaly Detection ===")
    rng = np.random.default_rng(42)

    # Setup : normal regime β=0.5, anomaly bursts β=1.8
    normal_sig = gen_colored_noise(4000, beta=0.5, seed=0)

    # Test signal : normal + 2 anomaly bursts
    test_chunks = []
    labels = []
    for i in range(10):
        if i in [3, 7]:  # anomaly chunks
            test_chunks.append(gen_colored_noise(200, beta=1.8, seed=100 + i))
            labels.append(1)
        else:
            test_chunks.append(gen_colored_noise(200, beta=0.5, seed=200 + i))
            labels.append(0)
    test_sig = np.concatenate(test_chunks)

    # Fit detector on normal, score test
    det = BetaAnomalyDetector(window=64, stride=8)
    det.fit(normal_sig)
    scores = det.score(test_sig)

    # Compute per-chunk mean score
    print(f"  Baseline β : {det.baseline_mean:.3f} ± {det.baseline_std:.3f}")
    print(f"  Detection result :")
    chunk_scores = []
    n_per_chunk = (len(test_sig) - 64) // 8 // 10  # approx windows per 200-sample chunk
    for i in range(10):
        s = i * n_per_chunk
        e = s + n_per_chunk
        mean_s = scores[s:e].mean()
        lab = "ANOM" if labels[i] else "norm"
        marker = "★" if labels[i] else " "
        print(f"    chunk {i} ({lab}) {marker}: mean z-score = {mean_s:.2f}")


def demo_synthesizer():
    print("\n=== Demo 2 : ODE β-Synthesizer ===")

    # Generate Lorenz training data
    n = 5000
    traj = np.zeros((n, 3), dtype=np.float64)
    traj[0] = [1.0, 1.0, 1.0]
    for i in range(1, n):
        x, y, z = traj[i - 1]
        traj[i, 0] = x + 0.005 * 10 * (y - x)
        traj[i, 1] = y + 0.005 * (x * (28 - z) - y)
        traj[i, 2] = z + 0.005 * (x * y - 8 / 3 * z)

    print(f"  Training ODE synthesizer (CPU, ~5s)...")
    model = train_synthesizer(traj, hidden=32, n_steps=16, dt=0.1, n_epochs=8, batch_size=64,
                                device="cpu", seed=0)

    print(f"  Generating 1000-step sequence via autoregressive rollout...")
    generated = model.generate(n_samples=1000, seed=42, device="cpu")
    print(f"  Generated shape : {generated.shape}")
    print(f"  β of generated  : {compute_beta(generated):.3f}  (expected ≈ 2.0)")
    print(f"  Max value       : {np.abs(generated).max():.2f}  (stable if < 100)")


if __name__ == "__main__":
    demo_anomaly_detection()
    demo_synthesizer()
    print("\n=== Done ===")
