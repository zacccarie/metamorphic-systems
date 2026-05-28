"""β-meter sur vraie dynamique : Lorenz attractor avec regime shift.

Setup :
- Normal regime : Lorenz ρ=28 (chaotic attractor stable)
- Anomaly : Lorenz ρ=350 (système hyper-chaotic, structure différente)
- Test β-detector raw + MSE-detector LSTM
- Real dynamical system → showcase of β-meter sur signal non-Gaussian
"""
from __future__ import annotations
import modal

app = modal.App("metamorphic-beta-anomaly-lorenz")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("numpy>=1.26", "scikit-learn>=1.4")
)


def gen_lorenz(n: int, rho: float, dt: float = 0.005, seed: int = 0) -> 'np.ndarray':
    """Generate Lorenz trajectory length n, parameter rho. Burn-in 1000 steps."""
    import numpy as np
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, 3).astype('float64')
    # Burn-in to attractor
    for _ in range(1000):
        dx = 10 * (x[1] - x[0])
        dy = x[0] * (rho - x[2]) - x[1]
        dz = x[0] * x[1] - 8/3 * x[2]
        x = x + dt * np.array([dx, dy, dz])
        if not np.all(np.isfinite(x)):
            x = rng.uniform(-5, 5, 3).astype('float64')
    traj = np.zeros((n, 3))
    for i in range(n):
        traj[i] = x
        dx = 10 * (x[1] - x[0])
        dy = x[0] * (rho - x[2]) - x[1]
        dz = x[0] * x[1] - 8/3 * x[2]
        x = x + dt * np.array([dx, dy, dz])
        if not np.all(np.isfinite(x)):
            x = rng.uniform(-5, 5, 3).astype('float64')
    sig = traj[:, 0]
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    sig = (sig - sig.mean()) / (sig.std() + 1e-9)
    return sig.astype('float32')


@app.function(image=image, gpu="A10", timeout=900)
def run_lorenz_anomaly(seed: int = 0, rho_anomaly: float = 350.0, chunk_size: int = 200) -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    device = torch.device("cuda")
    rng = np.random.default_rng(seed)
    rho_normal = 28.0
    window = 32

    train_signal = gen_lorenz(8000, rho_normal, seed=seed)

    chunks = []
    labels_per_chunk = []
    for _ in range(20):
        kind = rng.integers(0, 2)
        rho = rho_normal if kind == 0 else rho_anomaly
        sig = gen_lorenz(chunk_size, rho, seed=seed * 100 + int(_))
        chunks.append(sig)
        labels_per_chunk.append(kind)
    test_signal = np.concatenate(chunks)
    test_labels_per_sample = np.repeat(labels_per_chunk, chunk_size)

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
    model = LSTMNet(hidden=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(15):
        model.train()
        idx = torch.randperm(Xtr_t.shape[0], device=device)
        for i in range(0, Xtr_t.shape[0], 64):
            j = idx[i:i + 64]
            opt.zero_grad()
            pred, _ = model(Xtr_t[j])
            loss = F.mse_loss(pred, Ytr_t[j])
            loss.backward(); opt.step()

    model.eval()
    acts = []
    mses = []
    with torch.no_grad():
        for i in range(0, Xte_t.shape[0], 128):
            pred, h = model(Xte_t[i:i + 128])
            acts.append(h.cpu().numpy())
            mses.append(((pred - Yte_t[i:i + 128]) ** 2).cpu().numpy())
    acts = np.concatenate(acts)
    mse = np.concatenate(mses)

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

    beta_raw = []
    beta_lstm = []
    mse_scores = []
    labels_w = []
    raw_aligned = test_signal[window:window + len(acts)]
    for i in range(0, len(acts) - eval_win, 8):
        beta_raw.append(compute_beta_1d(raw_aligned[i:i + eval_win]))
        beta_lstm.append(compute_beta(acts[i:i + eval_win]))
        mse_scores.append(float(mse[i:i + eval_win].mean()))
        labels_w.append(int(label_te[i:i + eval_win].mean() > 0.5))

    beta_raw = np.array(beta_raw)
    beta_lstm = np.array(beta_lstm)
    mse_arr = np.array(mse_scores)
    labels = np.array(labels_w)

    def auroc_da(s, l):
        a = roc_auc_score(l, s)
        return max(a, 1 - a)

    return {
        'seed': seed,
        'rho_anomaly': rho_anomaly,
        'auroc_beta_raw': auroc_da(beta_raw, labels),
        'auroc_beta_lstm': auroc_da(beta_lstm, labels),
        'auroc_mse': auroc_da(mse_arr, labels),
        'mean_beta_raw_normal': float(beta_raw[labels == 0].mean()),
        'mean_beta_raw_anom': float(beta_raw[labels == 1].mean()),
    }


@app.local_entrypoint()
def main():
    import statistics
    rhos = [40.0, 60.0, 80.0, 100.0, 150.0]  # increasing deviation from ρ=28 (stable range)
    seeds = list(range(5))
    inputs = [(s, rho, 200) for rho in rhos for s in seeds]
    results = list(run_lorenz_anomaly.starmap(inputs))

    print(f"\n=== β-meter on Lorenz regime shift (normal ρ=28) ===\n")
    print(f"  {'ρ anom':<10} {'β_raw normal':<14} {'β_raw anom':<14} {'AUROC β':<14} {'AUROC β LSTM':<14} {'AUROC MSE':<14}")
    for rho in rhos:
        runs = [r for r in results if abs(r['rho_anomaly'] - rho) < 1e-3]
        bn_a = statistics.mean([r['mean_beta_raw_normal'] for r in runs])
        ba_a = statistics.mean([r['mean_beta_raw_anom'] for r in runs])
        b_raw = [r['auroc_beta_raw'] for r in runs]
        b_lstm = [r['auroc_beta_lstm'] for r in runs]
        ms = [r['auroc_mse'] for r in runs]
        def m_s(x):
            return f"{statistics.mean(x):.3f}±{statistics.stdev(x):.3f}" if len(x) > 1 else f"{x[0]:.3f}"
        print(f"  {rho:<10.1f} {bn_a:<14.3f} {ba_a:<14.3f} {m_s(b_raw):<14} {m_s(b_lstm):<14} {m_s(ms):<14}")
