"""Benchmark β ensemble OPTIMIZED config vs v2 default vs SOTA.

Config v3 (optimized via ablations) :
- window = 128 (vs 64)
- n_estimators = 50 (vs 100)
- transforms = ['raw'] only (vs 4 transforms)

Hypothesis : ≥0.90 AUROC moyen, beats v2 + SOTA OC-SVM 0.877.
"""
from __future__ import annotations
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from beta_toolkit import sliding_beta
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score


def gen_colored_noise(n, beta, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, n)
    fft = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1e-6
    sig = np.fft.irfft(fft * (freqs ** (-beta / 2)), n=n)
    return ((sig - sig.mean()) / (sig.std() + 1e-9)).astype('float32')


def make_test_signal(atype, seed=42, n_chunks=10, chunk_size=200, normal_beta=1.0):
    rng = np.random.default_rng(seed)
    chunks, labels = [], []
    for i in range(n_chunks):
        base = gen_colored_noise(chunk_size, normal_beta, seed=seed + i)
        is_anom = i in [3, 7]
        if is_anom:
            if atype == 'spike':
                base[chunk_size // 2] = 10.0
            elif atype == 'amplitude_scale':
                base = base * 3.0
            elif atype == 'frequency_inject':
                t = np.arange(chunk_size) / chunk_size
                base = base + 2.0 * np.sin(2 * np.pi * 20 * t).astype('float32')
            elif atype == 'trend':
                t = np.linspace(0, 5, chunk_size).astype('float32')
                base = base + t
            elif atype == 'regime_change':
                base = gen_colored_noise(chunk_size, 2.5, seed=seed + i)
            elif atype == 'variance_shift':
                base[chunk_size // 2:] = base[chunk_size // 2:] * 2.5
            elif atype == 'distribution_shift':
                base = rng.uniform(-2, 2, chunk_size).astype('float32')
        chunks.append(base)
        labels.extend([1 if is_anom else 0] * chunk_size)
    return np.concatenate(chunks), np.array(labels)


def windowize(sig, win, stride=8):
    return np.stack([sig[i:i + win] for i in range(0, len(sig) - win, stride)])


def auroc_da(scores, labels):
    a = roc_auc_score(labels, scores)
    return max(a, 1 - a)


def beta_ensemble_v3(normal_train, test_sig, window=128, stride=8, n_estimators=50):
    """OPTIMIZED config : window=128, n_est=50, only raw transform."""
    betas_train = sliding_beta(normal_train, window, stride)
    mean_b = betas_train.mean()
    std_b = betas_train.std() + 1e-9

    X_train = windowize(normal_train, window, stride)
    clf = IsolationForest(contamination=0.05, random_state=42, n_estimators=n_estimators)
    clf.fit(X_train)
    iso_train_scores = -clf.score_samples(X_train)
    iso_mean = iso_train_scores.mean()
    iso_std = iso_train_scores.std() + 1e-9

    betas_test = sliding_beta(test_sig, window, stride)
    z_beta = np.abs(betas_test - mean_b) / std_b
    X_test = windowize(test_sig, window, stride)
    iso_z = (-clf.score_samples(X_test) - iso_mean) / iso_std

    n = min(len(z_beta), len(iso_z))
    return np.maximum(z_beta[:n], iso_z[:n])


def benchmark_method(method, normal_train, test_sig, labels_per_sample, window=128, stride=8):
    window_labels = []
    for i in range(0, len(test_sig) - window, stride):
        window_labels.append(int(labels_per_sample[i:i + window].mean() > 0.5))
    window_labels = np.array(window_labels)
    if window_labels.sum() == 0 or window_labels.sum() == len(window_labels):
        return None
    t0 = time.perf_counter()

    if method == 'beta_v3':
        scores = beta_ensemble_v3(normal_train, test_sig, window=128, stride=8, n_estimators=50)
        train_t = time.perf_counter() - t0
        n_params = 2 + 50 * 50  # β stats + IsoForest 50 trees
    elif method == 'iso_forest_w128':
        X_normal = windowize(normal_train, window, stride)
        clf = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
        clf.fit(X_normal)
        train_t = time.perf_counter() - t0
        X_test = windowize(test_sig, window, stride)
        scores = -clf.score_samples(X_test)
        n_params = 100 * 50
    elif method == 'ocsvm_w128':
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
        'auroc': auroc_da(scores[:n], window_labels[:n]),
        'train_t': train_t,
        'n_params': n_params,
    }


def run():
    types = ['spike', 'amplitude_scale', 'frequency_inject', 'trend',
              'regime_change', 'variance_shift', 'distribution_shift']
    methods = ['beta_v3', 'iso_forest_w128', 'ocsvm_w128']
    results = {m: {t: [] for t in types} for m in methods}
    timing = {m: {'train': [], 'params': 0} for m in methods}

    for atype in types:
        for seed in range(5):  # 5 seeds for more robust CI
            normal_train = gen_colored_noise(4000, beta=1.0, seed=seed * 100)
            test_sig, labels_per_sample = make_test_signal(atype, seed=seed)
            for m in methods:
                r = benchmark_method(m, normal_train, test_sig, labels_per_sample)
                if r:
                    results[m][atype].append(r['auroc'])
                    timing[m]['train'].append(r['train_t'])
                    timing[m]['params'] = r['n_params']
    return results, timing


def print_results(results, timing):
    types = list(results['beta_v3'].keys())
    print(f"\n=== Benchmark v3 OPTIMIZED config (window=128, n_est=50, transforms=['raw'], 5 seeds) ===\n")
    print(f"  {'anomaly type':<22}{'β v3 (ours)':<22}{'IsoForest w128':<22}{'OC-SVM w128':<22}")
    for atype in types:
        line = f"  {atype:<22}"
        for m in ['beta_v3', 'iso_forest_w128', 'ocsvm_w128']:
            vals = results[m][atype]
            if vals:
                line += f"{np.mean(vals):.3f}±{np.std(vals):.3f}        "
            else:
                line += "—" + " " * 21
        print(line)
    print(f"\n  {'MEAN ALL':<22}", end='')
    for m in ['beta_v3', 'iso_forest_w128', 'ocsvm_w128']:
        all_v = [v for atype in types for v in results[m][atype]]
        if all_v:
            print(f"{np.mean(all_v):.3f}±{np.std(all_v):.3f}        ", end='')
    print()

    print(f"\n=== Efficiency ===\n")
    print(f"  {'method':<22}{'train (s)':<14}{'params':<12}")
    for m in ['beta_v3', 'iso_forest_w128', 'ocsvm_w128']:
        tt = np.mean(timing[m]['train']) if timing[m]['train'] else 0
        print(f"  {m:<22}{tt:<14.4f}{timing[m]['params']:<12}")


if __name__ == "__main__":
    print("Benchmark v3 OPTIMIZED (window=128, n_est=50, transforms=['raw']) vs SOTA window=128...")
    results, timing = run()
    print_results(results, timing)
