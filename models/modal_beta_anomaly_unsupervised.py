"""β-detector UNSUPERVISED : random-init LSTM captures regime shift.

Hypothèse killer : β shift activations émerge naturellement quand input change
de régime spectral, MÊME sans entraînement. LSTM random-init = filtre temporel
universel qui propage propriétés spectrales du signal.

Si confirmé → β-detector = anomaly detector zero-shot, sans label, sans training.
Avantage majeur sur MSE-detector qui requiert training.

Comparaisons :
  - β random-init LSTM : zero training
  - β trained LSTM     : reference (above PoC)
  - MSE trained LSTM   : standard baseline
  - β raw input        : skyline (sans NN, juste signal)
"""
from __future__ import annotations
import modal

app = modal.App("metamorphic-beta-unsupervised")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("numpy>=1.26", "scikit-learn>=1.4")
)


def gen_colored_noise(n: int, beta: float, rng) -> 'np.ndarray':
    import numpy as np
    white = rng.normal(0, 1, n)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1e-6
    scale = freqs ** (-beta / 2)
    fft = fft * scale
    sig = np.fft.irfft(fft, n=n)
    sig = (sig - sig.mean()) / (sig.std() + 1e-9)
    return sig.astype('float32')


@app.function(image=image, gpu="A10", timeout=900)
def run_unsupervised(seed: int = 0, beta_normal: float = 0.5, beta_anomaly: float = 1.0) -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    device = torch.device("cuda")
    rng = np.random.default_rng(seed)

    n_train = 8000
    window = 32
    train_signal = gen_colored_noise(n_train, beta_normal, rng)

    chunks = []
    labels_per_chunk = []
    for _ in range(20):
        kind = rng.integers(0, 2)
        beta = beta_normal if kind == 0 else beta_anomaly
        chunks.append(gen_colored_noise(200, beta, rng))
        labels_per_chunk.append(kind)
    test_signal = np.concatenate(chunks)
    test_labels_per_sample = np.repeat(labels_per_chunk, 200)

    def windowize(sig, win):
        X = np.stack([sig[i:i + win] for i in range(len(sig) - win - 1)])
        Y = sig[win:win + len(X)]
        return X.astype('float32'), Y.astype('float32')

    Xtr, Ytr = windowize(train_signal, window)
    Xte, Yte = windowize(test_signal, window)
    label_te = test_labels_per_sample[window:window + len(Xte)]

    Xtr_t = torch.from_numpy(Xtr[:, :, None]).to(device)
    Ytr_t = torch.from_numpy(Ytr).to(device)
    Xte_t = torch.from_numpy(Xte[:, :, None]).to(device)
    Yte_t = torch.from_numpy(Yte).to(device)

    class LSTMNet(nn.Module):
        def __init__(self, hidden=64):
            super().__init__()
            self.rnn = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=2, batch_first=True)
            self.head = nn.Linear(hidden, 1)
        def forward(self, x):
            h, _ = self.rnn(x)
            return self.head(h[:, -1]).squeeze(-1), h[:, -1, :]

    torch.manual_seed(seed)
    model_random = LSTMNet(hidden=64).to(device)
    torch.manual_seed(seed + 1000)
    model_trained = LSTMNet(hidden=64).to(device)
    opt = torch.optim.Adam(model_trained.parameters(), lr=1e-3)

    # Train second model on normal regime
    for ep in range(15):
        model_trained.train()
        idx = torch.randperm(Xtr_t.shape[0], device=device)
        for i in range(0, Xtr_t.shape[0], 64):
            j = idx[i:i + 64]
            opt.zero_grad()
            pred, _ = model_trained(Xtr_t[j])
            loss = F.mse_loss(pred, Ytr_t[j])
            loss.backward(); opt.step()

    # Collect activations both models + mse from trained
    def collect(model):
        model.eval()
        acts = []
        mses = []
        with torch.no_grad():
            for i in range(0, Xte_t.shape[0], 128):
                pred, h = model(Xte_t[i:i + 128])
                acts.append(h.cpu().numpy())
                mses.append(((pred - Yte_t[i:i + 128]) ** 2).cpu().numpy())
        return np.concatenate(acts), np.concatenate(mses)

    acts_random, _ = collect(model_random)
    acts_trained, mse_trained = collect(model_trained)

    eval_win = 64
    def compute_beta(h):
        if h.shape[0] < 16:
            return 0.0
        fft = np.fft.rfft(h, axis=0)
        power = (np.abs(fft) ** 2).mean(axis=1)
        freqs = np.arange(1, len(power))
        log_f = np.log(freqs + 1e-9)
        log_p = np.log(power[1:] + 1e-12)
        slope, _ = np.polyfit(log_f, log_p, 1)
        return float(-slope)

    def compute_beta_1d(sig):
        """β du signal raw (1D)."""
        if len(sig) < 16:
            return 0.0
        fft = np.fft.rfft(sig)
        power = np.abs(fft) ** 2
        freqs = np.arange(1, len(power))
        log_f = np.log(freqs + 1e-9)
        log_p = np.log(power[1:] + 1e-12)
        slope, _ = np.polyfit(log_f, log_p, 1)
        return float(-slope)

    beta_random_scores = []
    beta_trained_scores = []
    beta_raw_scores = []
    mse_scores = []
    labels_window = []

    raw_signal_aligned = test_signal[window:window + len(acts_random)]

    for i in range(0, len(acts_random) - eval_win, 8):
        b_r = compute_beta(acts_random[i:i + eval_win])
        b_t = compute_beta(acts_trained[i:i + eval_win])
        b_raw = compute_beta_1d(raw_signal_aligned[i:i + eval_win])
        m = float(mse_trained[i:i + eval_win].mean())
        lab = int(label_te[i:i + eval_win].mean() > 0.5)
        beta_random_scores.append(b_r)
        beta_trained_scores.append(b_t)
        beta_raw_scores.append(b_raw)
        mse_scores.append(m)
        labels_window.append(lab)

    beta_random = np.array(beta_random_scores)
    beta_trained = np.array(beta_trained_scores)
    beta_raw = np.array(beta_raw_scores)
    mse_arr = np.array(mse_scores)
    labels = np.array(labels_window)

    def auroc_da(scores, labels):
        a = roc_auc_score(labels, scores)
        return max(a, 1 - a)

    return {
        'seed': seed,
        'beta_normal': beta_normal,
        'beta_anomaly': beta_anomaly,
        'auroc_beta_random_init': auroc_da(beta_random, labels),
        'auroc_beta_trained': auroc_da(beta_trained, labels),
        'auroc_beta_raw_input': auroc_da(beta_raw, labels),
        'auroc_mse_trained': auroc_da(mse_arr, labels),
        'mean_beta_random_normal': float(beta_random[labels == 0].mean()),
        'mean_beta_random_anom': float(beta_random[labels == 1].mean()),
    }


@app.local_entrypoint()
def main():
    import statistics
    configs = [
        ('easy', 0.5, 2.0),
        ('medium', 0.5, 1.5),
        ('hard', 0.5, 1.0),
        ('very_hard', 0.5, 0.8),
    ]
    seeds = list(range(6))
    inputs = [(s, bn, ba) for (_, bn, ba) in configs for s in seeds]
    results = list(run_unsupervised.starmap(inputs))

    print(f"\n=== β-detector zero-shot comparison ===\n")
    hdr = f"  {'difficulty':<12} {'β raw':<14} {'β rand-LSTM':<14} {'β train-LSTM':<14} {'MSE train':<14}"
    print(hdr)
    for (name, bn, ba) in configs:
        runs = [r for r in results if abs(r['beta_normal'] - bn) < 1e-3 and abs(r['beta_anomaly'] - ba) < 1e-3]
        b_raw = [r['auroc_beta_raw_input'] for r in runs]
        b_rnd = [r['auroc_beta_random_init'] for r in runs]
        b_trn = [r['auroc_beta_trained'] for r in runs]
        m_trn = [r['auroc_mse_trained'] for r in runs]
        def m_s(x):
            return f"{statistics.mean(x):.3f}±{statistics.stdev(x):.3f}" if len(x) > 1 else f"{x[0]:.3f}"
        print(f"  {name:<12} {m_s(b_raw):<14} {m_s(b_rnd):<14} {m_s(b_trn):<14} {m_s(m_trn):<14}")

    print(f"\n=== Random-init LSTM β shift sanity ===")
    for (name, bn, ba) in configs:
        runs = [r for r in results if abs(r['beta_normal'] - bn) < 1e-3 and abs(r['beta_anomaly'] - ba) < 1e-3]
        bn_act = statistics.mean([r['mean_beta_random_normal'] for r in runs])
        ba_act = statistics.mean([r['mean_beta_random_anom'] for r in runs])
        print(f"  {name:<12} : β_rand normal={bn_act:.2f}  β_rand anomaly={ba_act:.2f}  shift={ba_act - bn_act:+.2f}")
