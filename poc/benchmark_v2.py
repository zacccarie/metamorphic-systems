"""Benchmark v2 : β-meter naïf vs β multi-feature vs β ensemble vs SOTA.

Tests A + C improvements :
- A : β multi-feature (raw + |signal| + diff + signal²)
- C : β ensemble (β multi-feature + IsoForest)

Target : AUROC moyen 0.85+ (competitive avec SOTA OC-SVM 0.877)
"""
from __future__ import annotations
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from beta_toolkit import (
    BetaAnomalyDetector, BetaMultiFeatureDetector, BetaEnsembleDetector,
    sliding_beta,
)
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
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


def windowize(sig, win, stride=8):
    return np.stack([sig[i:i + win] for i in range(0, len(sig) - win, stride)])


def auroc_da(scores, labels):
    a = roc_auc_score(labels, scores)
    return max(a, 1 - a)


def benchmark_method(method: str, normal_train: np.ndarray, test_sig: np.ndarray,
                      labels_per_sample: np.ndarray, window: int = 64, stride: int = 8) -> dict:
    window_labels = []
    for i in range(0, len(test_sig) - window, stride):
        window_labels.append(int(labels_per_sample[i:i + window].mean() > 0.5))
    window_labels = np.array(window_labels)
    if window_labels.sum() == 0 or window_labels.sum() == len(window_labels):
        return None

    t0 = time.perf_counter()

    if method == 'beta_meter':
        det = BetaAnomalyDetector(window=window, stride=stride)
        det.fit(normal_train)
        train_t = time.perf_counter() - t0
        scores = det.score(test_sig)
        n_params = 2

    elif method == 'beta_multi':
        det = BetaMultiFeatureDetector(window=window, stride=stride)
        det.fit(normal_train)
        train_t = time.perf_counter() - t0
        scores = det.score(test_sig)
        n_params = 8  # 4 means + 4 stds

    elif method == 'beta_ensemble':
        det = BetaEnsembleDetector(window=window, stride=stride, n_estimators=100)
        det.fit(normal_train)
        train_t = time.perf_counter() - t0
        scores = det.score(test_sig)
        n_params = 8 + 5000  # β stats + IsoForest

    elif method == 'isolation_forest':
        X_normal = windowize(normal_train, window, stride)
        clf = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
        clf.fit(X_normal)
        train_t = time.perf_counter() - t0
        X_test = windowize(test_sig, window, stride)
        scores = -clf.score_samples(X_test)
        n_params = 5000

    elif method == 'one_class_svm':
        X_normal = windowize(normal_train, window, stride)
        clf = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
        if X_normal.shape[0] > 2000:
            X_normal = X_normal[:2000]
        clf.fit(X_normal)
        train_t = time.perf_counter() - t0
        X_test = windowize(test_sig, window, stride)
        scores = -clf.score_samples(X_test)
        n_params = clf.support_vectors_.shape[0] * window

    else:
        return None

    n = min(len(scores), len(window_labels))
    return {
        'method': method,
        'auroc': auroc_da(scores[:n], window_labels[:n]),
        'train_time': train_t,
        'n_params': n_params,
    }


def run_benchmark():
    types = ['spike', 'amplitude_scale', 'frequency_inject', 'trend',
              'regime_change', 'variance_shift', 'distribution_shift']
    methods = ['beta_meter', 'beta_multi', 'beta_ensemble', 'isolation_forest', 'one_class_svm']
    n_seeds = 3

    results = {m: {t: [] for t in types} for m in methods}
    timing = {m: {'train': [], 'params': 0} for m in methods}

    for atype in types:
        for seed in range(n_seeds):
            normal_train = gen_colored_noise(4000, beta=1.0, seed=seed * 100)
            test_sig, labels_per_sample = make_test_signal(atype, seed=seed)
            for method in methods:
                r = benchmark_method(method, normal_train, test_sig, labels_per_sample)
                if r is not None:
                    results[method][atype].append(r['auroc'])
                    timing[method]['train'].append(r['train_time'])
                    timing[method]['params'] = r['n_params']

    return results, timing


def print_results(results, timing):
    types = list(results['beta_meter'].keys())
    methods = list(results.keys())
    print(f"\n=== AUROC per method × anomaly type (3 seeds) ===\n")
    header = f"  {'anomaly type':<22}"
    for m in methods:
        short = m.replace('beta_', 'β_').replace('isolation_forest', 'IsoForest').replace('one_class_svm', 'OC-SVM')
        header += f"{short:<18}"
    print(header)
    for atype in types:
        line = f"  {atype:<22}"
        for m in methods:
            vals = results[m][atype]
            if vals:
                line += f"{np.mean(vals):.3f}±{np.std(vals):.3f}    "
            else:
                line += "—" + " " * 17
        print(line)

    print(f"\n  {'MEAN ALL TYPES':<22}", end='')
    for m in methods:
        all_vals = [v for atype in types for v in results[m][atype]]
        if all_vals:
            print(f"{np.mean(all_vals):.3f}±{np.std(all_vals):.3f}    ", end='')
    print()

    print(f"\n=== Efficiency ===\n")
    print(f"  {'method':<22}{'train (s)':<14}{'params':<12}")
    for m in methods:
        tt = np.mean(timing[m]['train']) if timing[m]['train'] else 0
        print(f"  {m:<22}{tt:<14.4f}{timing[m]['params']:<12}")


def fig_v2_results(results, timing):
    types = list(results['beta_meter'].keys())
    methods = list(results.keys())
    colors = {
        'beta_meter': '#ff66ff',
        'beta_multi': '#ff00ff',
        'beta_ensemble': '#ff8800',
        'isolation_forest': '#00ff9f',
        'one_class_svm': '#ffaa00',
    }
    short = {
        'beta_meter': 'β v1',
        'beta_multi': 'β multi',
        'beta_ensemble': 'β ensemble',
        'isolation_forest': 'IsoForest',
        'one_class_svm': 'OC-SVM',
    }

    fig, axs = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [2.2, 1]})

    n_types = len(types)
    n_methods = len(methods)
    x = np.arange(n_types)
    width = 0.16
    for i, m in enumerate(methods):
        means = [np.mean(results[m][t]) if results[m][t] else 0 for t in types]
        stds = [np.std(results[m][t]) if results[m][t] else 0 for t in types]
        axs[0].bar(x + i * width - 2 * width, means, width, yerr=stds, label=short[m],
                    color=colors[m], edgecolor='#ffffff', linewidth=0.3, capsize=2)
    axs[0].axhline(0.5, color='#666666', ls=':', alpha=0.6)
    axs[0].axhline(0.8, color='#888888', ls=':', alpha=0.6)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([t.replace('_', '\n') for t in types], fontsize=9)
    axs[0].set_ylabel('AUROC')
    axs[0].set_title('β v1 vs β multi-feature vs β ensemble vs SOTA · 7 anomaly types',
                      color='#ffffff', pad=10, fontsize=12)
    axs[0].legend(loc='lower right', facecolor='#111111', edgecolor='#333333', ncol=3, fontsize=10)
    axs[0].set_ylim(0, 1.1)
    axs[0].grid(True, alpha=0.1, axis='y')

    ax = axs[1]
    for m in methods:
        all_vals = [v for atype in types for v in results[m][atype]]
        if not all_vals: continue
        mean_auroc = np.mean(all_vals)
        train_t = max(np.mean(timing[m]['train']) if timing[m]['train'] else 1e-5, 1e-5)
        n_p = timing[m]['params']
        size = 50 + np.log10(max(n_p, 1)) * 80
        ax.scatter(train_t, mean_auroc, s=size, color=colors[m], edgecolor='#ffffff',
                    linewidth=1.2, alpha=0.85)
        ax.annotate(short[m], (train_t, mean_auroc), color='#ffffff', fontsize=10,
                     xytext=(10, 8), textcoords='offset points')
    ax.set_xscale('log')
    ax.set_xlabel('train time (s, log)')
    ax.set_ylabel('mean AUROC')
    ax.set_title('Pareto efficiency', color='#ffffff', pad=10, fontsize=12)
    ax.grid(True, alpha=0.1)
    ax.set_ylim(0.6, 1.0)

    plt.tight_layout()
    plt.savefig('fig8_benchmark_v2.png', dpi=140, bbox_inches='tight')
    plt.close()
    print("  fig8_benchmark_v2.png saved")


if __name__ == "__main__":
    print("Benchmark v2 : β multi-feature + ensemble vs SOTA...")
    results, timing = run_benchmark()
    print_results(results, timing)
    fig_v2_results(results, timing)
