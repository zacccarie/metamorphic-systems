"""Robustness study : β-detector across multiple anomaly types.

Test β-anomaly detector on 7 distinct anomaly categories :
1. Spike (point anomaly)
2. Amplitude scaling (variance change)
3. Frequency shift (sine inject)
4. Trend (drift)
5. Regime change (β shift)
6. Variance shift (heteroscedastic)
7. Distribution shift (gaussian → uniform)

Output : AUROC per type + figure showing detection.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0, '.')
from beta_toolkit import BetaAnomalyDetector, sliding_beta

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
    'font.size': 9,
})

COLORS = {
    'normal': '#00ff9f',
    'anomaly': '#ff3366',
    'beta_line': '#ffaa00',
    'signal': '#88ccff',
    'highlight': '#ff00ff',
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


def make_test_signal(anomaly_type: str, normal_beta: float = 1.0, n_chunks: int = 10,
                       chunk_size: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Build test signal with anomaly bursts at chunks 3, 7.

    Returns: (signal, labels_per_sample)
    """
    rng = np.random.default_rng(seed)
    chunks = []
    labels_per_sample = []
    for i in range(n_chunks):
        base = gen_colored_noise(chunk_size, normal_beta, seed=seed + i)
        is_anom = i in [3, 7]
        if is_anom:
            if anomaly_type == 'spike':
                base[chunk_size // 2] = 10.0  # injection point
            elif anomaly_type == 'amplitude_scale':
                base = base * 3.0  # 3x amplitude
            elif anomaly_type == 'frequency_inject':
                t = np.arange(chunk_size) / chunk_size
                base = base + 2.0 * np.sin(2 * np.pi * 20 * t).astype('float32')
            elif anomaly_type == 'trend':
                t = np.linspace(0, 5, chunk_size).astype('float32')
                base = base + t
            elif anomaly_type == 'regime_change':
                base = gen_colored_noise(chunk_size, 2.5, seed=seed + i)
            elif anomaly_type == 'variance_shift':
                # Heteroscedastic : double variance in second half
                base[chunk_size // 2:] = base[chunk_size // 2:] * 2.5
            elif anomaly_type == 'distribution_shift':
                base = rng.uniform(-2, 2, chunk_size).astype('float32')  # uniform not gaussian
        chunks.append(base)
        labels_per_sample.extend([1 if is_anom else 0] * chunk_size)
    return np.concatenate(chunks), np.array(labels_per_sample)


def evaluate(anomaly_type: str, n_seeds: int = 5) -> dict:
    """Evaluate β-detector AUROC across n_seeds."""
    aurocs = []
    for seed in range(n_seeds):
        normal_sig = gen_colored_noise(4000, beta=1.0, seed=seed * 100)
        test_sig, labels_per_sample = make_test_signal(anomaly_type, seed=seed)

        det = BetaAnomalyDetector(window=64, stride=8)
        det.fit(normal_sig)
        scores = det.score(test_sig)

        # Aggregate per-window labels
        window_labels = []
        for i in range(0, len(test_sig) - 64, 8):
            mean_lab = labels_per_sample[i:i + 64].mean()
            window_labels.append(int(mean_lab > 0.5))
        window_labels = np.array(window_labels)

        if window_labels.sum() == 0 or window_labels.sum() == len(window_labels):
            continue  # avoid degenerate cases
        a = roc_auc_score(window_labels, scores)
        aurocs.append(max(a, 1 - a))

    return {
        'anomaly_type': anomaly_type,
        'auroc_mean': float(np.mean(aurocs)),
        'auroc_std': float(np.std(aurocs)),
        'n_seeds': len(aurocs),
    }


def fig_robustness():
    """Generate 7-panel figure showing β-detector across anomaly types."""
    types = ['spike', 'amplitude_scale', 'frequency_inject', 'trend',
              'regime_change', 'variance_shift', 'distribution_shift']
    type_labels = ['Point spike', 'Amplitude ×3', 'Frequency injection',
                    'Linear trend', 'β regime change', 'Variance shift',
                    'Distribution (G→U)']

    fig, axs = plt.subplots(7, 1, figsize=(14, 16), sharex=False)

    results = []
    for ax, atype, label in zip(axs, types, type_labels):
        # Single example
        normal_sig = gen_colored_noise(4000, beta=1.0, seed=0)
        test_sig, labels_per_sample = make_test_signal(atype, seed=42)
        det = BetaAnomalyDetector(window=64, stride=8)
        det.fit(normal_sig)
        scores = det.score(test_sig)
        win_ts = np.arange(len(scores)) * 8 + 32

        # AUROC averaged across seeds
        res = evaluate(atype, n_seeds=5)
        results.append(res)

        # Plot signal + score
        ax2 = ax.twinx()
        ax.plot(test_sig, color=COLORS['signal'], lw=0.4, alpha=0.7, label='signal')
        ax2.plot(win_ts, scores, color=COLORS['highlight'], lw=1.2, label='β z-score')
        ax2.axhline(2.0, color=COLORS['anomaly'], ls='--', alpha=0.5)
        for i in range(10):
            if labels_per_sample[i * 200] == 1:
                ax.axvspan(i * 200, (i + 1) * 200, color=COLORS['anomaly'], alpha=0.1)

        ax.set_ylabel('signal', color=COLORS['signal'])
        ax2.set_ylabel('z-score', color=COLORS['highlight'])
        ax.set_title(f'{label}  ·  AUROC = {res["auroc_mean"]:.3f} ± {res["auroc_std"]:.3f}',
                      color='#ffffff', loc='left', pad=4)
        ax.grid(True, alpha=0.1)
        ax.set_xlim(0, 2000)

    axs[-1].set_xlabel('time (sample index)')
    plt.tight_layout()
    plt.savefig('fig4_anomaly_types_robustness.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("  fig4_anomaly_types_robustness.png saved")

    return results


def fig_summary_bars(results):
    """Bar chart summarizing AUROC across anomaly types."""
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [r['anomaly_type'].replace('_', '\n') for r in results]
    means = [r['auroc_mean'] for r in results]
    stds = [r['auroc_std'] for r in results]

    # Color : green if AUROC>0.8, yellow 0.6-0.8, red <0.6
    cs = []
    for m in means:
        if m > 0.8:
            cs.append(COLORS['normal'])
        elif m > 0.6:
            cs.append(COLORS['beta_line'])
        else:
            cs.append(COLORS['anomaly'])

    bars = ax.bar(labels, means, yerr=stds, color=cs, edgecolor='#ffffff', linewidth=0.5, capsize=4)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{m:.2f}',
                 ha='center', color='#ffffff', fontsize=10, weight='bold')
    ax.axhline(0.5, color='#666666', ls=':', alpha=0.5, label='random AUROC=0.5')
    ax.axhline(0.8, color='#888888', ls=':', alpha=0.5, label='good detector ≥0.8')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('AUROC direction-agnostic')
    ax.set_title('β-detector across anomaly types — strengths & limits',
                  color='#ffffff', pad=10, fontsize=12)
    ax.legend(loc='lower right', facecolor='#111111', edgecolor='#333333')
    ax.grid(True, alpha=0.1, axis='y')
    plt.tight_layout()
    plt.savefig('fig5_anomaly_types_summary.png', dpi=140, bbox_inches='tight')
    plt.close()
    print("  fig5_anomaly_types_summary.png saved")


if __name__ == "__main__":
    print("Evaluating β-detector across anomaly types...")
    results = fig_robustness()
    print("\n=== Results AUROC ===")
    for r in results:
        print(f"  {r['anomaly_type']:<22} AUROC = {r['auroc_mean']:.3f} ± {r['auroc_std']:.3f}")
    fig_summary_bars(results)
    print("\nDone.")
