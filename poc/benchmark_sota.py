"""Benchmark β-meter vs SOTA baselines on 7 anomaly types.

Baselines :
- Isolation Forest (sklearn) - tree-based unsupervised
- One-Class SVM (sklearn) - kernel-based unsupervised
- LSTM Autoencoder (PyTorch) - reconstruction-based supervised(normal)
- β-meter (ours) - spectral unsupervised zero-train

Metrics : AUROC, training time, inference time per window, n_params
"""
from __future__ import annotations
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from beta_toolkit import BetaAnomalyDetector, sliding_beta
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
                      chunk_size: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    chunks = []
    labels_per_sample = []
    for i in range(n_chunks):
        base = gen_colored_noise(chunk_size, normal_beta, seed=seed + i)
        is_anom = i in [3, 7]
        if is_anom:
            if anomaly_type == 'spike':
                base[chunk_size // 2] = 10.0
            elif anomaly_type == 'amplitude_scale':
                base = base * 3.0
            elif anomaly_type == 'frequency_inject':
                t = np.arange(chunk_size) / chunk_size
                base = base + 2.0 * np.sin(2 * np.pi * 20 * t).astype('float32')
            elif anomaly_type == 'trend':
                t = np.linspace(0, 5, chunk_size).astype('float32')
                base = base + t
            elif anomaly_type == 'regime_change':
                base = gen_colored_noise(chunk_size, 2.5, seed=seed + i)
            elif anomaly_type == 'variance_shift':
                base[chunk_size // 2:] = base[chunk_size // 2:] * 2.5
            elif anomaly_type == 'distribution_shift':
                base = rng.uniform(-2, 2, chunk_size).astype('float32')
        chunks.append(base)
        labels_per_sample.extend([1 if is_anom else 0] * chunk_size)
    return np.concatenate(chunks), np.array(labels_per_sample)


# === LSTM Autoencoder ===
class LSTMAutoencoder(nn.Module):
    def __init__(self, window: int = 64, hidden: int = 32):
        super().__init__()
        self.window = window
        self.encoder = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=2, batch_first=True)
        self.decoder = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=2, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        # x : (B, window, 1)
        _, (h, c) = self.encoder(x)
        # Decode same length
        zero_input = torch.zeros_like(x)
        dec_out, _ = self.decoder(zero_input, (h, c))
        return self.out(dec_out).squeeze(-1)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        recon = self.forward(x)
        return ((recon - x.squeeze(-1)) ** 2).mean(dim=-1)


def train_lstm_ae(normal_signal: np.ndarray, window: int = 64, hidden: int = 32,
                   epochs: int = 30, batch_size: int = 32, lr: float = 1e-3,
                   device: str = "cpu") -> LSTMAutoencoder:
    device_obj = torch.device(device)
    n = len(normal_signal)
    X = np.stack([normal_signal[i:i + window] for i in range(0, n - window, 8)])
    Xt = torch.from_numpy(X[:, :, None]).to(device_obj)

    model = LSTMAutoencoder(window=window, hidden=hidden).to(device_obj)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(Xt.shape[0], device=device_obj)
        for i in range(0, Xt.shape[0], batch_size):
            j = idx[i:i + batch_size]
            opt.zero_grad()
            recon = model(Xt[j])
            loss = F.mse_loss(recon, Xt[j].squeeze(-1))
            loss.backward()
            opt.step()
    return model


def windowize(sig, win, stride=8):
    return np.stack([sig[i:i + win] for i in range(0, len(sig) - win, stride)])


def benchmark_method(method: str, normal_train: np.ndarray, test_sig: np.ndarray,
                      labels_per_sample: np.ndarray, window: int = 64, stride: int = 8) -> dict:
    """Train method on normal, score test, return metrics."""
    # Window labels (majority)
    window_labels = []
    for i in range(0, len(test_sig) - window, stride):
        window_labels.append(int(labels_per_sample[i:i + window].mean() > 0.5))
    window_labels = np.array(window_labels)
    if window_labels.sum() == 0 or window_labels.sum() == len(window_labels):
        return None

    t_train_start = time.perf_counter()

    if method == 'beta_meter':
        det = BetaAnomalyDetector(window=window, stride=stride)
        det.fit(normal_train)
        t_train = time.perf_counter() - t_train_start

        t_score_start = time.perf_counter()
        scores = det.score(test_sig)
        t_inf = (time.perf_counter() - t_score_start) / len(scores)
        n_params = 2  # mean + std

    elif method == 'isolation_forest':
        # Windowize normal data
        X_normal = windowize(normal_train, window, stride)
        clf = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
        clf.fit(X_normal)
        t_train = time.perf_counter() - t_train_start

        X_test = windowize(test_sig, window, stride)
        t_score_start = time.perf_counter()
        scores = -clf.score_samples(X_test)
        t_inf = (time.perf_counter() - t_score_start) / len(scores)
        n_params = 100 * 50  # ~100 trees, ~50 nodes each (approx)

    elif method == 'one_class_svm':
        X_normal = windowize(normal_train, window, stride)
        clf = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
        if X_normal.shape[0] > 2000:
            X_normal = X_normal[:2000]
        clf.fit(X_normal)
        t_train = time.perf_counter() - t_train_start

        X_test = windowize(test_sig, window, stride)
        t_score_start = time.perf_counter()
        scores = -clf.score_samples(X_test)
        t_inf = (time.perf_counter() - t_score_start) / len(scores)
        n_params = clf.support_vectors_.shape[0] * window

    elif method == 'lstm_ae':
        torch.manual_seed(42)
        model = train_lstm_ae(normal_train, window=window, epochs=20)
        t_train = time.perf_counter() - t_train_start

        X_test = windowize(test_sig, window, stride)
        Xt = torch.from_numpy(X_test[:, :, None]).float()
        t_score_start = time.perf_counter()
        with torch.no_grad():
            scores = model.reconstruction_error(Xt).numpy()
        t_inf = (time.perf_counter() - t_score_start) / len(scores)
        n_params = sum(p.numel() for p in model.parameters())

    else:
        return None

    # Direction-agnostic AUROC
    auroc_raw = roc_auc_score(window_labels, scores)
    auroc = max(auroc_raw, 1 - auroc_raw)

    return {
        'method': method,
        'auroc': auroc,
        'train_time_s': t_train,
        'inf_time_per_win_us': t_inf * 1e6,
        'n_params': n_params,
    }


def run_benchmark():
    types = ['spike', 'amplitude_scale', 'frequency_inject', 'trend',
              'regime_change', 'variance_shift', 'distribution_shift']
    methods = ['beta_meter', 'isolation_forest', 'one_class_svm', 'lstm_ae']
    n_seeds = 3

    results = {m: {t: [] for t in types} for m in methods}
    timing = {m: {'train': [], 'inf': [], 'params': 0} for m in methods}

    for atype in types:
        for seed in range(n_seeds):
            normal_train = gen_colored_noise(4000, beta=1.0, seed=seed * 100)
            test_sig, labels_per_sample = make_test_signal(atype, seed=seed)
            for method in methods:
                r = benchmark_method(method, normal_train, test_sig, labels_per_sample)
                if r is not None:
                    results[method][atype].append(r['auroc'])
                    timing[method]['train'].append(r['train_time_s'])
                    timing[method]['inf'].append(r['inf_time_per_win_us'])
                    timing[method]['params'] = r['n_params']

    return results, timing


def print_results(results, timing):
    types = list(results['beta_meter'].keys())
    methods = list(results.keys())
    print(f"\n=== AUROC per method × anomaly type ===\n")
    header = f"  {'anomaly type':<22}"
    for m in methods:
        header += f"{m:<22}"
    print(header)
    for atype in types:
        line = f"  {atype:<22}"
        for m in methods:
            vals = results[m][atype]
            if vals:
                mean = np.mean(vals); std = np.std(vals)
                line += f"{mean:.3f}±{std:.3f}    "
            else:
                line += "—                    "
        print(line)

    # Overall mean
    print(f"\n  {'MEAN ALL TYPES':<22}", end='')
    for m in methods:
        all_vals = [v for atype in types for v in results[m][atype]]
        if all_vals:
            print(f"{np.mean(all_vals):.3f}±{np.std(all_vals):.3f}    ", end='')
    print()

    print(f"\n=== Efficiency ===\n")
    print(f"  {'method':<22}{'train (s)':<14}{'inf (μs/win)':<16}{'params':<12}")
    for m in methods:
        tt = np.mean(timing[m]['train']) if timing[m]['train'] else 0
        it = np.mean(timing[m]['inf']) if timing[m]['inf'] else 0
        np_ = timing[m]['params']
        print(f"  {m:<22}{tt:<14.4f}{it:<16.2f}{np_:<12}")


def fig_benchmark(results, timing):
    types = list(results['beta_meter'].keys())
    methods = list(results.keys())
    colors = {'beta_meter': '#ff00ff', 'isolation_forest': '#00ff9f',
               'one_class_svm': '#ffaa00', 'lstm_ae': '#88ccff'}
    short = {'beta_meter': 'β-meter', 'isolation_forest': 'IsoForest',
              'one_class_svm': 'OC-SVM', 'lstm_ae': 'LSTM-AE'}

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})

    # Left : AUROC per type per method (grouped bars)
    n_types = len(types)
    n_methods = len(methods)
    x = np.arange(n_types)
    width = 0.2
    for i, m in enumerate(methods):
        means = [np.mean(results[m][t]) if results[m][t] else 0 for t in types]
        stds = [np.std(results[m][t]) if results[m][t] else 0 for t in types]
        axs[0].bar(x + i * width - 1.5 * width, means, width, yerr=stds, label=short[m],
                    color=colors[m], edgecolor='#ffffff', linewidth=0.3, capsize=2)
    axs[0].axhline(0.5, color='#666666', ls=':', alpha=0.6, label='random')
    axs[0].axhline(0.8, color='#888888', ls=':', alpha=0.6)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([t.replace('_', '\n') for t in types], fontsize=9)
    axs[0].set_ylabel('AUROC')
    axs[0].set_title('β-meter vs SOTA baselines · 7 anomaly types · 3 seeds',
                      color='#ffffff', pad=10, fontsize=12)
    axs[0].legend(loc='lower right', facecolor='#111111', edgecolor='#333333', ncol=2)
    axs[0].set_ylim(0, 1.1)
    axs[0].grid(True, alpha=0.1, axis='y')

    # Right : efficiency scatter (train time vs mean AUROC, marker size = params)
    ax = axs[1]
    for m in methods:
        all_vals = [v for atype in types for v in results[m][atype]]
        if not all_vals: continue
        mean_auroc = np.mean(all_vals)
        train_t = np.mean(timing[m]['train']) if timing[m]['train'] else 1e-6
        n_p = timing[m]['params']
        size = 40 + np.log10(max(n_p, 1)) * 80
        ax.scatter(train_t, mean_auroc, s=size, color=colors[m], edgecolor='#ffffff',
                    linewidth=1, alpha=0.8)
        ax.annotate(short[m], (train_t, mean_auroc), color='#ffffff', fontsize=10,
                     xytext=(8, 8), textcoords='offset points')
    ax.set_xscale('log')
    ax.set_xlabel('train time (s, log)')
    ax.set_ylabel('mean AUROC (all types)')
    ax.set_title('Efficiency Pareto', color='#ffffff', pad=10, fontsize=12)
    ax.grid(True, alpha=0.1)

    plt.tight_layout()
    plt.savefig('fig7_benchmark_sota.png', dpi=140, bbox_inches='tight')
    plt.close()
    print("  fig7_benchmark_sota.png saved")


if __name__ == "__main__":
    print("Running benchmark β-meter vs SOTA baselines...")
    results, timing = run_benchmark()
    print_results(results, timing)
    fig_benchmark(results, timing)
