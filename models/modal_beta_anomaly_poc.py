"""PoC anomaly detection via β-shift activations LSTM.

Hypothèse : LSTM trained sur normal regime apprend signature β cible.
Sous régime anomaly (β différent), activations LSTM shift β.
Sliding window β = anomaly score, comparable à reconstruction-MSE.

Setup :
- Signal normal : β≈0.5 (slight memory)
- Signal anomaly : β≈1.5 (strong memory)
- Train LSTM next-step sur normal only
- Test : mix normal + anomaly windows
- Compare AUROC : β-detector vs MSE-detector
"""
from __future__ import annotations
import modal

app = modal.App("metamorphic-beta-anomaly-poc")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("numpy>=1.26", "scikit-learn>=1.4")
)


def gen_colored_noise(n: int, beta: float, rng) -> 'np.ndarray':
    """Génère bruit 1/f^beta de longueur n via IFFT."""
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
def run_poc(seed: int = 0, beta_normal: float = 0.5, beta_anomaly: float = 1.5) -> dict:
    """Build dataset → train LSTM → eval β-detector vs MSE-detector → AUROC."""
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    device = torch.device("cuda")
    rng = np.random.default_rng(seed)

    n_train = 8000
    n_test_per_regime = 2000
    window = 32

    # Données train : seulement régime normal
    train_signal = gen_colored_noise(n_train, beta_normal, rng)

    # Données test : alternance normal/anomaly chunks 200 samples
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
            self.last_act = None
        def forward(self, x):
            h, _ = self.rnn(x)
            self.last_act = h[:, -1, :].detach()  # last timestep activations
            return self.head(h[:, -1]).squeeze(-1)

    torch.manual_seed(seed)
    model = LSTMNet(hidden=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train sur normal uniquement
    for ep in range(15):
        model.train()
        idx = torch.randperm(Xtr_t.shape[0], device=device)
        for i in range(0, Xtr_t.shape[0], 64):
            j = idx[i:i + 64]
            opt.zero_grad()
            pred = model(Xtr_t[j])
            loss = F.mse_loss(pred, Ytr_t[j])
            loss.backward(); opt.step()

    # Eval sur test mixte
    model.eval()
    activations_per_window = []
    mse_per_sample = []
    with torch.no_grad():
        for i in range(0, Xte_t.shape[0], 128):
            batch = Xte_t[i:i + 128]
            pred = model(batch)
            activations_per_window.append(model.last_act.cpu().numpy())
            mse_per_sample.append(((pred - Yte_t[i:i + 128]) ** 2).cpu().numpy())
    activations = np.concatenate(activations_per_window)  # (N, hidden)
    mse = np.concatenate(mse_per_sample)  # (N,)

    # Sliding window β-detector sur activations
    eval_win = 64  # 64 timesteps activations → compute β
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

    beta_scores = []
    mse_scores = []
    labels_window = []
    for i in range(0, len(activations) - eval_win, 8):  # stride 8
        b = compute_beta(activations[i:i + eval_win])
        m = float(mse[i:i + eval_win].mean())
        # Window label : majority of underlying samples
        lab = int(label_te[i:i + eval_win].mean() > 0.5)
        beta_scores.append(b)
        mse_scores.append(m)
        labels_window.append(lab)

    beta_scores = np.array(beta_scores)
    mse_scores = np.array(mse_scores)
    labels_window = np.array(labels_window)

    # AUROC direction-agnostic : detector valid si AUROC > 0.5 OU < 0.5
    # Use max(AUROC, 1-AUROC) = strength of separation
    auroc_beta_raw = roc_auc_score(labels_window, beta_scores)
    auroc_mse_raw = roc_auc_score(labels_window, mse_scores)
    auroc_beta = max(auroc_beta_raw, 1 - auroc_beta_raw)
    auroc_mse = max(auroc_mse_raw, 1 - auroc_mse_raw)

    # Mean β per regime (sanity)
    mean_beta_normal = float(beta_scores[labels_window == 0].mean())
    mean_beta_anom = float(beta_scores[labels_window == 1].mean())
    mean_mse_normal = float(mse_scores[labels_window == 0].mean())
    mean_mse_anom = float(mse_scores[labels_window == 1].mean())

    return {
        'seed': seed,
        'beta_normal_input': beta_normal,
        'beta_anomaly_input': beta_anomaly,
        'auroc_beta_detector': auroc_beta,
        'auroc_mse_detector': auroc_mse,
        'mean_beta_act_normal': mean_beta_normal,
        'mean_beta_act_anomaly': mean_beta_anom,
        'mean_mse_normal': mean_mse_normal,
        'mean_mse_anomaly': mean_mse_anom,
    }


@app.local_entrypoint()
def main():
    """Sweep beta_anomaly difficulty + seeds."""
    import statistics

    # Sweep difficulty : closer beta_anomaly → harder to detect
    configs = [
        ('easy', 0.5, 2.0),
        ('medium', 0.5, 1.5),
        ('hard', 0.5, 1.0),
        ('very_hard', 0.5, 0.8),
    ]
    seeds = list(range(6))

    inputs = [(s, bn, ba) for (_, bn, ba) in configs for s in seeds]
    results = list(run_poc.starmap(inputs))

    print(f"\n=== PoC β-anomaly detector vs MSE-detector ===\n")
    print(f"  {'difficulty':<12} {'β_norm':<8} {'β_anom':<8} {'AUROC β':<14} {'AUROC MSE':<14} {'verdict':<10}")
    for (name, bn, ba) in configs:
        runs = [r for r in results if abs(r['beta_normal_input'] - bn) < 1e-3 and abs(r['beta_anomaly_input'] - ba) < 1e-3]
        ab = [r['auroc_beta_detector'] for r in runs]
        am = [r['auroc_mse_detector'] for r in runs]
        mb = statistics.mean(ab); sb = statistics.stdev(ab) if len(ab) > 1 else 0
        mm = statistics.mean(am); sm = statistics.stdev(am) if len(am) > 1 else 0
        verdict = "β WIN" if mb > mm + 0.02 else "MSE WIN" if mm > mb + 0.02 else "TIE"
        print(f"  {name:<12} {bn:<8.2f} {ba:<8.2f} {mb:.3f}±{sb:.3f}  {mm:.3f}±{sm:.3f}  {verdict}")

    # Sanity : mean β activations per regime
    print(f"\n=== Sanity check : β activations shift ===")
    for (name, bn, ba) in configs:
        runs = [r for r in results if abs(r['beta_normal_input'] - bn) < 1e-3 and abs(r['beta_anomaly_input'] - ba) < 1e-3]
        bn_a = statistics.mean([r['mean_beta_act_normal'] for r in runs])
        ba_a = statistics.mean([r['mean_beta_act_anomaly'] for r in runs])
        print(f"  {name:<12} : β_act normal={bn_a:.2f}  β_act anomaly={ba_a:.2f}  shift={ba_a - bn_a:+.2f}")
