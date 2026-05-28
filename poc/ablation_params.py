"""Ablations β ensemble : window × n_estimators × transform count.

Identify Pareto optimum architecture pour β ensemble :
- Window size : [16, 32, 64, 128, 256]
- IsoForest n_estimators : [25, 50, 100, 200, 500]
- Transform ablation : enable/disable [raw, |s|, diff, s²] individuellement

Test influence params count + identify optimal config.
"""
from __future__ import annotations
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from beta_toolkit import sliding_beta
from sklearn.ensemble import IsolationForest
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


def beta_ensemble_score(normal_train, test_sig, window, stride, n_estimators, transforms):
    """β ensemble customizable : transforms subset + IsoForest n_estimators."""
    def get_betas(sig):
        sig = sig.astype('float32')
        results = []
        if 'raw' in transforms:
            results.append(sliding_beta(sig, window, stride))
        if 'abs' in transforms:
            results.append(sliding_beta(np.abs(sig), window, stride))
        if 'diff' in transforms:
            results.append(sliding_beta(np.diff(sig, prepend=sig[0]), window, stride))
        if 'sq' in transforms:
            results.append(sliding_beta(sig ** 2, window, stride))
        if not results:
            return None
        n = min(len(r) for r in results)
        return np.stack([r[:n] for r in results], axis=1)

    # Fit β baseline (skip si pas transforms)
    if transforms:
        betas_train = get_betas(normal_train)
        means = betas_train.mean(axis=0)
        stds = betas_train.std(axis=0) + 1e-9
    else:
        means = None
        stds = None

    # Fit IsoForest
    X_train = windowize(normal_train, window, stride)
    clf = IsolationForest(contamination=0.05, random_state=42, n_estimators=n_estimators)
    clf.fit(X_train)
    iso_train_scores = -clf.score_samples(X_train)
    iso_mean = iso_train_scores.mean()
    iso_std = iso_train_scores.std() + 1e-9

    # Score test
    X_test = windowize(test_sig, window, stride)
    iso_test = -clf.score_samples(X_test)
    iso_z = (iso_test - iso_mean) / iso_std

    if transforms:
        betas_test = get_betas(test_sig)
        z_beta = np.abs(betas_test - means) / stds
        beta_score = z_beta.max(axis=1)
        n = min(len(beta_score), len(iso_z))
        return np.maximum(beta_score[:n], iso_z[:n])
    else:
        return iso_z


def evaluate_config(window, stride, n_estimators, transforms, types, n_seeds=3):
    """Run ensemble config across types + seeds. Return mean AUROC, train time."""
    aurocs = []
    train_times = []
    for atype in types:
        for seed in range(n_seeds):
            normal_train = gen_colored_noise(4000, beta=1.0, seed=seed * 100)
            test_sig, labels_per_sample = make_test_signal(atype, seed=seed)
            window_labels = []
            for i in range(0, len(test_sig) - window, stride):
                window_labels.append(int(labels_per_sample[i:i + window].mean() > 0.5))
            window_labels = np.array(window_labels)
            if window_labels.sum() == 0 or window_labels.sum() == len(window_labels):
                continue
            t0 = time.perf_counter()
            scores = beta_ensemble_score(normal_train, test_sig, window, stride,
                                          n_estimators, transforms)
            train_t = time.perf_counter() - t0
            n = min(len(scores), len(window_labels))
            aurocs.append(auroc_da(scores[:n], window_labels[:n]))
            train_times.append(train_t)
    return {
        'auroc_mean': float(np.mean(aurocs)),
        'auroc_std': float(np.std(aurocs)),
        'train_time': float(np.mean(train_times)),
    }


def ablation_window_size():
    types = ['spike', 'amplitude_scale', 'frequency_inject', 'trend',
              'regime_change', 'variance_shift', 'distribution_shift']
    windows = [16, 32, 64, 128, 256]
    all_transforms = ['raw', 'abs', 'diff', 'sq']
    print("\n=== ABLATION 1 : window size (n_est=100, all transforms) ===\n")
    print(f"  {'window':<10}{'AUROC':<16}{'train (s)':<12}")
    results = []
    for w in windows:
        r = evaluate_config(window=w, stride=max(4, w // 16), n_estimators=100,
                              transforms=all_transforms, types=types, n_seeds=3)
        r['window'] = w
        results.append(r)
        print(f"  {w:<10}{r['auroc_mean']:.3f}±{r['auroc_std']:.3f}    {r['train_time']:.3f}")
    return results


def ablation_n_estimators():
    types = ['spike', 'amplitude_scale', 'frequency_inject', 'trend',
              'regime_change', 'variance_shift', 'distribution_shift']
    n_estimators_list = [25, 50, 100, 200, 500]
    all_transforms = ['raw', 'abs', 'diff', 'sq']
    print("\n=== ABLATION 2 : IsoForest n_estimators (window=64, all transforms) ===\n")
    print(f"  {'n_est':<10}{'AUROC':<16}{'train (s)':<12}")
    results = []
    for n_est in n_estimators_list:
        r = evaluate_config(window=64, stride=8, n_estimators=n_est,
                              transforms=all_transforms, types=types, n_seeds=3)
        r['n_estimators'] = n_est
        results.append(r)
        print(f"  {n_est:<10}{r['auroc_mean']:.3f}±{r['auroc_std']:.3f}    {r['train_time']:.3f}")
    return results


def ablation_transforms():
    types = ['spike', 'amplitude_scale', 'frequency_inject', 'trend',
              'regime_change', 'variance_shift', 'distribution_shift']
    # Per-transform : test each individually + cumulative add
    configs = [
        ('iso_only', []),
        ('raw', ['raw']),
        ('raw+abs', ['raw', 'abs']),
        ('raw+abs+diff', ['raw', 'abs', 'diff']),
        ('raw+abs+diff+sq (all)', ['raw', 'abs', 'diff', 'sq']),
        # Leave-one-out
        ('all-raw', ['abs', 'diff', 'sq']),
        ('all-abs', ['raw', 'diff', 'sq']),
        ('all-diff', ['raw', 'abs', 'sq']),
        ('all-sq', ['raw', 'abs', 'diff']),
    ]
    print("\n=== ABLATION 3 : transform set (window=64, n_est=100) ===\n")
    print(f"  {'config':<30}{'AUROC':<16}{'train (s)':<12}")
    results = []
    for name, transforms in configs:
        r = evaluate_config(window=64, stride=8, n_estimators=100,
                              transforms=transforms, types=types, n_seeds=3)
        r['config'] = name
        r['transforms'] = transforms
        results.append(r)
        print(f"  {name:<30}{r['auroc_mean']:.3f}±{r['auroc_std']:.3f}    {r['train_time']:.3f}")
    return results


def ablation_multi_window():
    """Test multi-window concat (16+64+256) ensemble."""
    types = ['spike', 'amplitude_scale', 'frequency_inject', 'trend',
              'regime_change', 'variance_shift', 'distribution_shift']
    # Multi-window : compute β for multiple window sizes, concat features
    def multi_window_score(normal_train, test_sig, windows_list, stride, n_estimators, transforms):
        all_beta_scores = []
        for w in windows_list:
            scores = beta_ensemble_score(normal_train, test_sig, w, stride, n_estimators, transforms)
            # Resample to common length (last window)
            target_n = (len(test_sig) - max(windows_list)) // stride
            if len(scores) > target_n:
                scores = scores[:target_n]
            elif len(scores) < target_n:
                scores = np.pad(scores, (0, target_n - len(scores)), constant_values=scores.mean())
            all_beta_scores.append(scores)
        all_beta_scores = np.stack(all_beta_scores)
        return all_beta_scores.max(axis=0)

    all_transforms = ['raw', 'abs', 'diff', 'sq']
    configs = [
        ('single w=32', [32]),
        ('single w=64 (baseline)', [64]),
        ('single w=128', [128]),
        ('multi [32, 64]', [32, 64]),
        ('multi [32, 64, 128]', [32, 64, 128]),
        ('multi [16, 32, 64, 128]', [16, 32, 64, 128]),
    ]
    print("\n=== ABLATION 4 : multi-window concat (n_est=100, all transforms) ===\n")
    print(f"  {'config':<30}{'AUROC':<16}{'train (s)':<12}")
    results = []
    for name, windows_list in configs:
        aurocs = []
        train_times = []
        for atype in types:
            for seed in range(3):
                normal_train = gen_colored_noise(4000, beta=1.0, seed=seed * 100)
                test_sig, labels_per_sample = make_test_signal(atype, seed=seed)
                window_labels = []
                max_w = max(windows_list)
                for i in range(0, len(test_sig) - max_w, 8):
                    window_labels.append(int(labels_per_sample[i:i + max_w].mean() > 0.5))
                window_labels = np.array(window_labels)
                if window_labels.sum() == 0 or window_labels.sum() == len(window_labels):
                    continue
                t0 = time.perf_counter()
                scores = multi_window_score(normal_train, test_sig, windows_list,
                                              stride=8, n_estimators=100,
                                              transforms=all_transforms)
                t = time.perf_counter() - t0
                n = min(len(scores), len(window_labels))
                aurocs.append(auroc_da(scores[:n], window_labels[:n]))
                train_times.append(t)
        m = float(np.mean(aurocs))
        s = float(np.std(aurocs))
        tt = float(np.mean(train_times))
        results.append({'config': name, 'auroc_mean': m, 'auroc_std': s, 'train_time': tt})
        print(f"  {name:<30}{m:.3f}±{s:.3f}    {tt:.3f}")
    return results


def fig_ablations(res_w, res_n, res_t, res_mw):
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    # Window ablation
    axs[0, 0].errorbar([r['window'] for r in res_w], [r['auroc_mean'] for r in res_w],
                        yerr=[r['auroc_std'] for r in res_w], color='#ff00ff', lw=2, capsize=4,
                        marker='o', markersize=8)
    axs[0, 0].set_xlabel('window size')
    axs[0, 0].set_ylabel('AUROC')
    axs[0, 0].set_title('Window size ablation', color='#ffffff', pad=10)
    axs[0, 0].set_xscale('log', base=2)
    axs[0, 0].grid(True, alpha=0.1)

    # n_estimators ablation
    axs[0, 1].errorbar([r['n_estimators'] for r in res_n], [r['auroc_mean'] for r in res_n],
                        yerr=[r['auroc_std'] for r in res_n], color='#00ff9f', lw=2, capsize=4,
                        marker='s', markersize=8)
    axs[0, 1].set_xlabel('IsoForest n_estimators')
    axs[0, 1].set_ylabel('AUROC')
    axs[0, 1].set_title('IsoForest size ablation', color='#ffffff', pad=10)
    axs[0, 1].set_xscale('log', base=2)
    axs[0, 1].grid(True, alpha=0.1)

    # Transform ablation
    labels_t = [r['config'].replace(' ', '\n', 1) for r in res_t]
    cs = ['#ff3366' if 'iso_only' in r['config'] else ('#00ff9f' if 'all' in r['config'] and '-' not in r['config'] else ('#ff00ff' if r['config'].startswith('all-') else '#ffaa00')) for r in res_t]
    axs[1, 0].bar(range(len(res_t)), [r['auroc_mean'] for r in res_t],
                   yerr=[r['auroc_std'] for r in res_t], color=cs, edgecolor='#ffffff', linewidth=0.5, capsize=3)
    axs[1, 0].set_xticks(range(len(res_t)))
    axs[1, 0].set_xticklabels(labels_t, rotation=30, ha='right', fontsize=8)
    axs[1, 0].set_ylabel('AUROC')
    axs[1, 0].set_title('Transform ablation (add + leave-one-out)', color='#ffffff', pad=10)
    axs[1, 0].set_ylim(0.7, 1.0)
    axs[1, 0].grid(True, alpha=0.1, axis='y')

    # Multi-window
    labels_mw = [r['config'].replace(' ', '\n', 1) for r in res_mw]
    axs[1, 1].bar(range(len(res_mw)), [r['auroc_mean'] for r in res_mw],
                   yerr=[r['auroc_std'] for r in res_mw], color='#88ccff', edgecolor='#ffffff', linewidth=0.5, capsize=3)
    axs[1, 1].set_xticks(range(len(res_mw)))
    axs[1, 1].set_xticklabels(labels_mw, rotation=30, ha='right', fontsize=8)
    axs[1, 1].set_ylabel('AUROC')
    axs[1, 1].set_title('Multi-window concat ablation', color='#ffffff', pad=10)
    axs[1, 1].set_ylim(0.7, 1.0)
    axs[1, 1].grid(True, alpha=0.1, axis='y')

    plt.tight_layout()
    plt.savefig('fig9_ablations.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("  fig9_ablations.png saved")


if __name__ == "__main__":
    print("β ensemble architecture ablations...")
    res_w = ablation_window_size()
    res_n = ablation_n_estimators()
    res_t = ablation_transforms()
    res_mw = ablation_multi_window()
    fig_ablations(res_w, res_n, res_t, res_mw)
