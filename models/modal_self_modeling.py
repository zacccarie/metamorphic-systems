"""Modal cloud sweep — Self-Modeling MLP α sweep.

Test si self-modeling head (prédiction propre activation) améliore task perf.
"""
from __future__ import annotations
import modal

app = modal.App("metamorphic-self-modeling")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("numpy>=1.26")
)


@app.function(image=image, gpu="A10", timeout=300)
def train_smmlp(alpha: float = 0.0, seed: int = 0) -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("cuda")

    def gen_chirp(fc, fm, sigma, sr=8000, duration=0.5, n_samples=32, seed=0):
        rng = np.random.default_rng(seed)
        n = int(sr * duration)
        t = np.linspace(0, duration, n, endpoint=False)
        f_t = fc * (1 + fm * t / duration)
        phase = 2 * np.pi * np.cumsum(f_t) / sr
        sig = np.sin(phase + rng.uniform(0, 2 * np.pi))
        sig = sig + rng.normal(0, sigma, n)
        idx = np.linspace(0, n - 1, n_samples).astype(int)
        return sig[idx].astype(np.float32)

    def make_dataset(n_per_class, seed):
        freqs = np.array([800 + 25 * i for i in range(8)])
        mods = np.array([0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 0.25, -0.25])
        X, y = [], []
        for c, (fc, fm) in enumerate(zip(freqs, mods)):
            for i in range(n_per_class):
                X.append(gen_chirp(fc, fm, 1.5, seed=seed * 10000 + c * 1000 + i))
                y.append(c)
        X = np.stack(X); y = np.array(y, dtype=np.int64)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(y))
        return torch.from_numpy(X[perm]), torch.from_numpy(y[perm])

    class SMMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, 8)
            self.self_model = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 32))

        def forward(self, x):
            h1 = F.gelu(self.fc1(x))
            h2 = F.gelu(self.fc2(h1))
            out = self.fc3(h2)
            h2_pred = self.self_model(h1)
            return out, h2_pred, h2

    torch.manual_seed(seed); np.random.seed(seed)
    xtr, ytr = make_dataset(400, 0)
    xva, yva = make_dataset(120, 999)
    xtr, ytr, xva, yva = xtr.to(device), ytr.to(device), xva.to(device), yva.to(device)
    model = SMMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = xtr.shape[0]
    history_acc = []
    history_self = []
    for ep in range(12):
        model.train()
        idx = torch.randperm(n, device=device)
        ep_self = 0
        for i in range(0, n, 32):
            j = idx[i:i + 32]
            opt.zero_grad()
            out, h2p, h2t = model(xtr[j])
            tl = F.cross_entropy(out, ytr[j])
            sl = F.mse_loss(h2p, h2t.detach())
            (tl + alpha * sl).backward()
            opt.step()
            ep_self += sl.item()
        history_self.append(ep_self / (n / 32))
        model.eval()
        with torch.no_grad():
            out, _, _ = model(xva)
            acc = (out.argmax(-1) == yva).float().mean().item()
        history_acc.append(acc)

    return {
        'seed': seed, 'alpha': alpha,
        'val_acc': history_acc[-1],
        'final_self_loss': history_self[-1],
        'init_self_loss': history_self[0],
        'self_decrease': history_self[0] - history_self[-1],
    }


@app.local_entrypoint()
def alpha_sweep():
    """Sweep α ∈ [0, 0.01, 0.1, 0.5, 1.0, 2.0] × 8 seeds parallel."""
    alphas = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0]
    seeds = list(range(8))
    inputs = [(a, s) for a in alphas for s in seeds]
    results = list(train_smmlp.starmap(inputs))

    import statistics
    print(f"\n=== Self-Modeling MLP α sweep ===")
    print(f"  {'α':<6} {'val_acc':<18} {'self_loss final':<20} {'self_decrease'}")
    for a in alphas:
        runs = [r for r in results if abs(r['alpha'] - a) < 1e-6]
        accs = [r['val_acc'] for r in runs]
        selfs = [r['final_self_loss'] for r in runs]
        decs = [r['self_decrease'] for r in runs]
        print(f"  {a:<6} {statistics.mean(accs):.4f}±{statistics.stdev(accs):.4f}    "
              f"{statistics.mean(selfs):.4f}±{statistics.stdev(selfs):.4f}    "
              f"{statistics.mean(decs):.4f}")
