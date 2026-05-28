"""β-detector under NOISE : where LSTM matters.

Hypothèse : raw signal β fragile face au bruit. LSTM trained learns to denoise
+ extract β robust. β des activations LSTM = anomaly detector noise-robust.

Setup :
- Normal regime : β=0.5 + noise σ
- Anomaly : β=1.5 + noise σ
- Sweep σ ∈ [0, 0.3, 0.6, 1.0, 1.5]
- Compare raw β / random LSTM β / trained LSTM β / MSE detectors
- Expect : LSTM trained > raw β under high noise
"""
from __future__ import annotations
import modal

app = modal.App("metamorphic-beta-anomaly-noise")

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
def run_noisy(seed: int = 0, sigma: float = 0.5, beta_anomaly: float = 1.5) -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    device = torch.device("cuda")
    rng = np.random.default_rng(seed)
    beta_normal = 0.5
    n_train = 8000
    window = 32

    train_signal = gen_colored_noise(n_train, beta_normal, rng)
    train_signal = train_signal + rng.normal(0, sigma, n_train).astype('float32')

    chunks = []
    labels_per_chunk = []
    for _ in range(20):
        kind = rng.integers(0, 2)
        beta = beta_normal if kind == 0 else beta_anomaly
        sig = gen_colored_noise(200, beta, rng)
        sig = sig + rng.normal(0, sigma, 200).astype('float32')
        chunks.append(sig)
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
    model_trained = LSTMNet(hidden=64).to(device)
    opt = torch.optim.Adam(model_trained.parameters(), lr=1e-3)

    for ep in range(15):
        model_trained.train()
        idx = torch.randperm(Xtr_t.shape[0], device=device)
        for i in range(0, Xtr_t.shape[0], 64):
            j = idx[i:i + 64]
            opt.zero_grad()
            pred, _ = model_trained(Xtr_t[j])
            loss = F.mse_loss(pred, Ytr_t[j])
            loss.backward(); opt.step()

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
        if len(sig) < 16:
            return 0.0
        fft = np.fft.rfft(sig)
        power = np.abs(fft) ** 2
        freqs = np.arange(1, len(power))
        log_f = np.log(freqs + 1e-9)
        log_p = np.log(power[1:] + 1e-12)
        slope, _ = np.polyfit(log_f, log_p, 1)
        return float(-slope)

    beta_trained_scores = []
    beta_raw_scores = []
    mse_scores = []
    labels_window = []
    raw_signal_aligned = test_signal[window:window + len(acts_trained)]

    for i in range(0, len(acts_trained) - eval_win, 8):
        b_t = compute_beta(acts_trained[i:i + eval_win])
        b_raw = compute_beta_1d(raw_signal_aligned[i:i + eval_win])
        m = float(mse_trained[i:i + eval_win].mean())
        lab = int(label_te[i:i + eval_win].mean() > 0.5)
        beta_trained_scores.append(b_t)
        beta_raw_scores.append(b_raw)
        mse_scores.append(m)
        labels_window.append(lab)

    beta_trained = np.array(beta_trained_scores)
    beta_raw = np.array(beta_raw_scores)
    mse_arr = np.array(mse_scores)
    labels = np.array(labels_window)

    def auroc_da(scores, labels):
        a = roc_auc_score(labels, scores)
        return max(a, 1 - a)

    return {
        'seed': seed,
        'sigma': sigma,
        'beta_anomaly': beta_anomaly,
        'auroc_beta_raw': auroc_da(beta_raw, labels),
        'auroc_beta_trained': auroc_da(beta_trained, labels),
        'auroc_mse': auroc_da(mse_arr, labels),
    }


@app.local_entrypoint()
def main():
    import statistics
    sigmas = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0]
    beta_anom = 1.5
    seeds = list(range(5))
    inputs = [(s, sig, beta_anom) for sig in sigmas for s in seeds]
    results = list(run_noisy.starmap(inputs))

    print(f"\n=== β-detector noise robustness (β_anom={beta_anom}) ===\n")
    print(f"  {'σ noise':<10} {'β raw signal':<16} {'β LSTM trained':<18} {'MSE LSTM':<14}")
    for sig in sigmas:
        runs = [r for r in results if abs(r['sigma'] - sig) < 1e-3]
        br = [r['auroc_beta_raw'] for r in runs]
        bt = [r['auroc_beta_trained'] for r in runs]
        ms = [r['auroc_mse'] for r in runs]
        def m_s(x):
            return f"{statistics.mean(x):.3f}±{statistics.stdev(x):.3f}" if len(x) > 1 else f"{x[0]:.3f}"
        print(f"  {sig:<10} {m_s(br):<16} {m_s(bt):<18} {m_s(ms):<14}")
