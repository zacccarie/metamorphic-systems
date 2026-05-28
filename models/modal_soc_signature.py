"""Test SOC signature : 1/f^β converge à β≈2 pendant training ?

Hypothèse Bak 1987 : système auto-organisé critique → spectre 1/f^2.
Test : tracker β spectral des activations hidden d'un MLP pendant training.
Si β → 2 (ou plus précisément converge stable), signature SOC empirique.

Compare 3 conditions :
- depth=1 : peu d'expressivité
- depth=3 : medium
- depth=5 : deep enough pour structure
"""
from __future__ import annotations
import modal

app = modal.App("metamorphic-soc")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("numpy>=1.26", "scipy>=1.13")
)


@app.function(image=image, gpu="A10", timeout=600)
def train_track_beta(seed: int = 0, depth: int = 3) -> dict:
    """Train MLP, track β spectral slope every epoch."""
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("cuda")

    # Lorenz dataset (rich signal for spectral analysis)
    def lorenz(n=5000):
        xs = np.zeros((n, 3), dtype=np.float64)
        xs[0] = [1.0, 1.0, 1.0]
        for i in range(1, n):
            x, y, z = xs[i - 1]
            xs[i, 0] = x + 0.01 * 10 * (y - x)
            xs[i, 1] = y + 0.01 * (x * (28 - z) - y)
            xs[i, 2] = z + 0.01 * (x * y - 8 / 3 * z)
        return xs.astype(np.float32)

    traj = lorenz(5000)
    mu, std = traj.mean(0), traj.std(0) + 1e-6
    traj = (traj - mu) / std

    window = 4
    X = np.stack([traj[i: i + window].flatten() for i in range(len(traj) - window - 1)])
    Y = traj[window:window + len(X)]
    X = torch.from_numpy(X[:1000]).to(device); Y = torch.from_numpy(Y[:1000]).to(device)

    # MLP variable depth
    class MLP(nn.Module):
        def __init__(self, in_d, h, out_d, depth):
            super().__init__()
            layers = []
            d_in = in_d
            for _ in range(depth):
                layers.append(nn.Linear(d_in, h))
                layers.append(nn.GELU())
                d_in = h
            layers.append(nn.Linear(h, out_d))
            self.net = nn.Sequential(*layers)
            self.hooks = []
            self.activations = []

        def forward(self, x):
            self.activations.clear()
            h = x
            for layer in self.net:
                h = layer(h)
                if isinstance(layer, nn.GELU):
                    self.activations.append(h.detach())
            return h

    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(12, 64, 3, depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def compute_beta(activations):
        """Compute spectral slope β from time-series of activations.

        activations : list of tensors (B, H) — one per layer
        Return β fitted from log P(f) vs log f.
        """
        if not activations or activations[0].shape[0] < 16:
            return 0.0
        # Use first layer activations as time series (one sample per row)
        h = activations[0].cpu().numpy()  # (B, H)
        # FFT along batch dim (treat batch as time)
        fft = np.fft.rfft(h, axis=0)
        power = (np.abs(fft) ** 2).mean(axis=1)
        # Skip DC, fit log-log
        freqs = np.arange(1, len(power))
        log_f = np.log(freqs + 1e-9)
        log_p = np.log(power[1:] + 1e-12)
        slope, _ = np.polyfit(log_f, log_p, 1)
        return float(-slope)  # negative since P ~ 1/f^β means slope = -β in log-log

    betas = []
    losses = []
    n_epochs = 20
    for ep in range(n_epochs):
        model.train()
        idx = torch.randperm(X.shape[0], device=device)
        ep_loss = 0
        for i in range(0, X.shape[0], 32):
            j = idx[i:i + 32]
            opt.zero_grad()
            out = model(X[j])
            loss = F.mse_loss(out, Y[j])
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        # Compute β at this epoch
        model.eval()
        with torch.no_grad():
            _ = model(X[:128])
            beta = compute_beta(model.activations)
        betas.append(beta)
        losses.append(ep_loss / (X.shape[0] / 32))

    return {
        'seed': seed, 'depth': depth,
        'betas': betas,
        'losses': losses,
        'beta_init': betas[0],
        'beta_final': betas[-1],
        'beta_converged': float(np.mean(betas[-5:])),
        'loss_final': losses[-1],
    }


@app.local_entrypoint()
def soc_sweep():
    """Track β over training for depth ∈ [1, 3, 5] × 8 seeds."""
    depths = [1, 3, 5]
    seeds = list(range(8))
    inputs = [(s, d) for d in depths for s in seeds]
    results = list(train_track_beta.starmap(inputs))

    import statistics
    print(f"\n=== SOC signature : β convergence over training ===")
    for d in depths:
        runs = [r for r in results if r['depth'] == d]
        b_init = [r['beta_init'] for r in runs]
        b_conv = [r['beta_converged'] for r in runs]
        loss = [r['loss_final'] for r in runs]
        print(f"\n  depth={d}")
        print(f"    β init     : {statistics.mean(b_init):.3f} ± {statistics.stdev(b_init):.3f}")
        print(f"    β converged: {statistics.mean(b_conv):.3f} ± {statistics.stdev(b_conv):.3f}")
        print(f"    loss final : {statistics.mean(loss):.5f}")

    # Trajectory analysis : do betas converge ?
    print(f"\n=== β trajectory (mean per epoch) ===")
    for d in depths:
        runs = [r for r in results if r['depth'] == d]
        # average each epoch across seeds
        n_epochs = len(runs[0]['betas'])
        mean_traj = [
            statistics.mean([r['betas'][ep] for r in runs])
            for ep in range(n_epochs)
        ]
        print(f"  depth={d} : β over epochs = {[round(b, 2) for b in mean_traj[:10]]}...")


