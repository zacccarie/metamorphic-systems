"""Visualization β-meter en action : figures publication/portfolio.

Génère figures PNG démontrant :
1. β-anomaly detection sliding window sur signal mixte
2. ODE synthesizer trajectory β=2 stable
3. β taxonomy archs (data du paper)
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')
from beta_toolkit import compute_beta, sliding_beta, BetaAnomalyDetector, train_synthesizer

# Aesthetics : black bg, emissive colors
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#000000',
    'axes.facecolor': '#000000',
    'savefig.facecolor': '#000000',
    'axes.edgecolor': '#444444',
    'axes.labelcolor': '#cccccc',
    'xtick.color': '#999999',
    'ytick.color': '#999999',
    'text.color': '#ffffff',
    'font.family': 'monospace',
    'font.size': 10,
})

# Color palette (emissive)
COLORS = {
    'normal': '#00ff9f',     # mint green
    'anomaly': '#ff3366',    # red pink
    'beta_line': '#ffaa00',  # amber
    'signal': '#88ccff',     # light blue
    'highlight': '#ff00ff',  # magenta
    'fit': '#ffffff',
}


def gen_colored_noise(n: int, beta: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.normal(0, 1, n)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1e-6
    scale = freqs ** (-beta / 2)
    fft = fft * scale
    sig = np.fft.irfft(fft, n=n)
    return ((sig - sig.mean()) / (sig.std() + 1e-9)).astype('float32')


def fig1_anomaly_detection():
    """β-anomaly detection sliding window."""
    rng = np.random.default_rng(42)
    normal = gen_colored_noise(4000, beta=0.5, seed=0)

    chunks = []
    labels = []
    for i in range(10):
        if i in [3, 7]:
            chunks.append(gen_colored_noise(200, beta=1.8, seed=100 + i))
            labels.append(1)
        else:
            chunks.append(gen_colored_noise(200, beta=0.5, seed=200 + i))
            labels.append(0)
    test = np.concatenate(chunks)

    det = BetaAnomalyDetector(window=64, stride=8)
    det.fit(normal)
    scores = det.score(test)
    betas = sliding_beta(test, 64, 8)

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=False)

    # Top : raw signal with anomaly shading
    axs[0].plot(test, color=COLORS['signal'], lw=0.5, alpha=0.8)
    for i, lab in enumerate(labels):
        if lab:
            axs[0].axvspan(i * 200, (i + 1) * 200, color=COLORS['anomaly'], alpha=0.15)
    axs[0].set_title('Test signal : normal regime β=0.5 + anomaly bursts β=1.8',
                      color=COLORS['fit'], pad=10)
    axs[0].set_ylabel('amplitude')
    axs[0].grid(True, alpha=0.1)

    # Middle : β sliding
    win_ts = np.arange(len(betas)) * 8 + 32
    axs[1].plot(win_ts, betas, color=COLORS['beta_line'], lw=1.5)
    axs[1].axhline(det.baseline_mean, color=COLORS['normal'], ls='--', alpha=0.6,
                   label=f'baseline β = {det.baseline_mean:.2f}')
    axs[1].fill_between(win_ts, det.baseline_mean - 2*det.baseline_std,
                          det.baseline_mean + 2*det.baseline_std, color=COLORS['normal'], alpha=0.1)
    for i, lab in enumerate(labels):
        if lab:
            axs[1].axvspan(i * 200, (i + 1) * 200, color=COLORS['anomaly'], alpha=0.1)
    axs[1].set_ylabel('β (sliding 64-win)')
    axs[1].set_title('Spectral slope β over time', color=COLORS['fit'], pad=10)
    axs[1].legend(loc='upper right', facecolor='#111111', edgecolor='#333333')
    axs[1].grid(True, alpha=0.1)

    # Bottom : z-score anomaly score with threshold
    axs[2].plot(win_ts, scores, color=COLORS['highlight'], lw=1.5)
    axs[2].axhline(2.0, color=COLORS['anomaly'], ls='--', alpha=0.6, label='threshold z=2')
    for i, lab in enumerate(labels):
        if lab:
            axs[2].axvspan(i * 200, (i + 1) * 200, color=COLORS['anomaly'], alpha=0.1)
    axs[2].set_xlabel('time (sample index)')
    axs[2].set_ylabel('|β - baseline| / σ_baseline')
    axs[2].set_title('Anomaly score (z-score)', color=COLORS['fit'], pad=10)
    axs[2].legend(loc='upper right', facecolor='#111111', edgecolor='#333333')
    axs[2].grid(True, alpha=0.1)

    plt.tight_layout()
    plt.savefig('fig1_anomaly_detection.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  fig1_anomaly_detection.png saved")


def fig2_ode_synthesizer():
    """ODE synthesizer output β trajectory."""
    n = 5000
    traj = np.zeros((n, 3), dtype=np.float64)
    traj[0] = [1.0, 1.0, 1.0]
    for i in range(1, n):
        x, y, z = traj[i - 1]
        traj[i, 0] = x + 0.005 * 10 * (y - x)
        traj[i, 1] = y + 0.005 * (x * (28 - z) - y)
        traj[i, 2] = z + 0.005 * (x * y - 8 / 3 * z)

    print("  Training ODE synthesizer...")
    model = train_synthesizer(traj, hidden=32, n_steps=16, dt=0.1, n_epochs=8, batch_size=64,
                                device="cpu", seed=0)
    generated = model.generate(n_samples=1000, seed=42, device="cpu")

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

    # Top-left : training data X
    axs[0, 0].plot(traj[:1000, 0], color=COLORS['signal'], lw=0.5)
    axs[0, 0].set_title(f'Lorenz training data (X dim)  β={compute_beta(traj[:1000, 0]):.2f}',
                         color=COLORS['fit'], pad=10)
    axs[0, 0].set_ylabel('amplitude')
    axs[0, 0].grid(True, alpha=0.1)

    # Top-right : generated X
    axs[0, 1].plot(generated[:, 0], color=COLORS['highlight'], lw=0.5)
    axs[0, 1].set_title(f'ODE autoregressive rollout (X dim)  β={compute_beta(generated[:, 0]):.2f}',
                         color=COLORS['fit'], pad=10)
    axs[0, 1].set_ylabel('amplitude')
    axs[0, 1].grid(True, alpha=0.1)

    # Bottom-left : training power spectrum
    def fft_loglog(sig):
        fft = np.fft.rfft(sig)
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(sig))
        return freqs[1:], power[1:]

    f1, p1 = fft_loglog(traj[:1000, 0])
    axs[1, 0].loglog(f1, p1, color=COLORS['signal'], lw=0.7)
    slope = -compute_beta(traj[:1000, 0])
    fit_x = np.logspace(np.log10(f1[1]), np.log10(f1[-1]), 100)
    fit_y = np.exp(slope * np.log(fit_x) + np.log(p1.mean() * 10))
    axs[1, 0].loglog(fit_x, fit_y, color=COLORS['fit'], ls='--', alpha=0.7, label=f'β={-slope:.2f}')
    axs[1, 0].set_xlabel('frequency')
    axs[1, 0].set_ylabel('power')
    axs[1, 0].set_title('Lorenz spectrum (1/f^β)', color=COLORS['fit'], pad=10)
    axs[1, 0].legend(facecolor='#111111', edgecolor='#333333')
    axs[1, 0].grid(True, alpha=0.1)

    # Bottom-right : generated power spectrum
    f2, p2 = fft_loglog(generated[:, 0])
    axs[1, 1].loglog(f2, p2, color=COLORS['highlight'], lw=0.7)
    slope2 = -compute_beta(generated[:, 0])
    fit_y2 = np.exp(slope2 * np.log(fit_x) + np.log(p2.mean() * 10))
    axs[1, 1].loglog(fit_x, fit_y2, color=COLORS['fit'], ls='--', alpha=0.7, label=f'β={-slope2:.2f}')
    axs[1, 1].set_xlabel('frequency')
    axs[1, 1].set_ylabel('power')
    axs[1, 1].set_title('Generated spectrum (1/f^β)', color=COLORS['fit'], pad=10)
    axs[1, 1].legend(facecolor='#111111', edgecolor='#333333')
    axs[1, 1].grid(True, alpha=0.1)

    plt.tight_layout()
    plt.savefig('fig2_ode_synthesizer.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  fig2_ode_synthesizer.png saved")


def fig3_taxonomy_beta():
    """β taxonomy across architectures (data from cloud runs)."""
    archs = ['MLP', 'ResNet', 'RNN\n(tanh)', 'Attention', 'GRU', 'LSTM', 'ODE Euler']
    betas = [0.05, 0.02, 0.21, 0.50, 1.08, 1.17, 1.57]
    err = [0.01, 0.01, 0.05, 0.10, 0.08, 0.06, 0.006]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors_per = ['#444444', '#444444', '#888888', '#bbbb55', COLORS['normal'],
                   COLORS['beta_line'], COLORS['highlight']]
    bars = ax.bar(archs, betas, yerr=err, color=colors_per,
                   edgecolor='#ffffff', linewidth=0.5, capsize=4)

    # Annotate β values
    for bar, b, name in zip(bars, betas, archs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'{b:.2f}', ha='center', color=COLORS['fit'], fontsize=11, weight='bold')

    ax.axhline(1.0, color='#666666', ls=':', alpha=0.6, label='pink noise β=1')
    ax.axhline(2.0, color='#666666', ls=':', alpha=0.6, label='brown noise β=2')

    ax.set_ylabel('β generated under Gaussian input', fontsize=11)
    ax.set_title('Architecture β taxonomy — spectral signature of integration depth\n(cloud Modal A10, n=4-8 seeds per arch)',
                  color=COLORS['fit'], pad=15, fontsize=12)
    ax.set_ylim(0, 1.9)
    ax.legend(loc='upper left', facecolor='#111111', edgecolor='#333333')
    ax.grid(True, alpha=0.1, axis='y')

    plt.tight_layout()
    plt.savefig('fig3_taxonomy_beta.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  fig3_taxonomy_beta.png saved")


if __name__ == "__main__":
    print("Generating publication figures...")
    fig1_anomaly_detection()
    fig2_ode_synthesizer()
    fig3_taxonomy_beta()
    print("Done. 3 figures saved in current directory.")
