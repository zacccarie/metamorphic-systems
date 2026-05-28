"""Test if growth events (Net2Net mutations) produce 1/f² spectral signature.

Reproduces architrace VG setup under Gaussian noise input. Si β émerge sous
growth + Gaussian → architrace claim survives. Si β reste 0 → claim invalide
définitivement.
"""
from __future__ import annotations
import modal

app = modal.App("metamorphic-growth-soc")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("numpy>=1.26")
)


@app.function(image=image, gpu="A10", timeout=600)
def train_with_growth(seed: int = 0, signal_type: str = "gaussian") -> dict:
    """Train MLP with periodic Net2Net growth events. Track β throughout."""
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("cuda")
    rng = np.random.default_rng(seed)

    # Generate signal
    n = 5000
    if signal_type == "gaussian":
        traj = rng.normal(0, 1, (n, 3)).astype(np.float32)
    else:  # lorenz
        traj = np.zeros((n, 3), dtype=np.float64)
        traj[0] = [1.0, 1.0, 1.0]
        for i in range(1, n):
            x, y, z = traj[i - 1]
            traj[i, 0] = x + 0.01 * 10 * (y - x)
            traj[i, 1] = y + 0.01 * (x * (28 - z) - y)
            traj[i, 2] = z + 0.01 * (x * y - 8 / 3 * z)
        traj = traj.astype(np.float32)

    mu, std = traj.mean(0), traj.std(0) + 1e-6
    traj = (traj - mu) / std
    window = 4
    X = np.stack([traj[i: i + window].flatten() for i in range(len(traj) - window - 1)])
    Y = traj[window:window + len(X)]
    X = torch.from_numpy(X[:800]).to(device); Y = torch.from_numpy(Y[:800]).to(device)

    class GrowableMLP(nn.Module):
        def __init__(self, in_d, h, out_d):
            super().__init__()
            self.fc1 = nn.Linear(in_d, h).to(device)
            self.fc2 = nn.Linear(h, h).to(device)
            self.fc3 = nn.Linear(h, out_d).to(device)
            self.last_act = None
        def forward(self, x):
            h = F.gelu(self.fc1(x))
            h = F.gelu(self.fc2(h))
            self.last_act = h.detach()
            return self.fc3(h)

    def grow(model, add=4):
        """Net2Net widening : grow fc2 + fc3 by `add` units."""
        old_h = model.fc2.weight.shape[0]
        in_d = model.fc2.weight.shape[1]
        new_h = old_h + add
        # New fc2 : in_d → new_h
        new_fc2 = nn.Linear(in_d, new_h).to(device)
        new_fc2.weight.data[:old_h] = model.fc2.weight.data
        new_fc2.bias.data[:old_h] = model.fc2.bias.data
        for k in range(add):
            src = np.random.randint(old_h)
            new_fc2.weight.data[old_h + k] = model.fc2.weight.data[src] + torch.randn_like(model.fc2.weight.data[src]) * 0.01
            new_fc2.bias.data[old_h + k] = model.fc2.bias.data[src]
        new_fc3 = nn.Linear(new_h, model.fc3.weight.shape[0]).to(device)
        new_fc3.weight.data[:, :old_h] = model.fc3.weight.data
        new_fc3.weight.data[:, old_h:] = 0
        new_fc3.bias.data = model.fc3.bias.data
        model.fc2 = new_fc2
        model.fc3 = new_fc3
        return new_h

    def compute_beta(h):
        h = h.cpu().numpy()
        if h.shape[0] < 16:
            return 0.0
        fft = np.fft.rfft(h, axis=0)
        power = (np.abs(fft) ** 2).mean(axis=1)
        freqs = np.arange(1, len(power))
        log_f = np.log(freqs + 1e-9)
        log_p = np.log(power[1:] + 1e-12)
        slope, _ = np.polyfit(log_f, log_p, 1)
        return float(-slope)

    torch.manual_seed(seed); np.random.seed(seed)
    model = GrowableMLP(12, 16, 3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    betas_per_step = []
    growth_events = []
    prev_loss = float('inf')
    plateau = 0

    for ep in range(20):
        model.train()
        idx = torch.randperm(X.shape[0], device=device)
        ep_loss = 0
        for i in range(0, X.shape[0], 32):
            j = idx[i:i + 32]
            opt.zero_grad()
            out = model(X[j])
            loss = F.mse_loss(out, Y[j])
            loss.backward(); opt.step()
            ep_loss += loss.item()
        avg_loss = ep_loss / (X.shape[0] / 32)
        # Detect plateau → grow
        if prev_loss - avg_loss < 0.001:
            plateau += 1
        else:
            plateau = 0
        if plateau >= 2 and model.fc2.weight.shape[0] < 80:
            new_h = grow(model, add=4)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            growth_events.append({'epoch': ep, 'new_hidden': new_h})
            plateau = 0
        prev_loss = avg_loss

        model.eval()
        with torch.no_grad():
            _ = model(X[:256])
            beta = compute_beta(model.last_act)
        betas_per_step.append(beta)

    return {
        'seed': seed,
        'signal_type': signal_type,
        'betas': betas_per_step,
        'beta_init': betas_per_step[0],
        'beta_final': betas_per_step[-1],
        'beta_mean_late': float(sum(betas_per_step[-5:]) / 5),
        'growth_events': len(growth_events),
        'final_hidden': model.fc2.weight.shape[0],
    }


@app.local_entrypoint()
def growth_soc_test():
    """Run growth + signal control."""
    signals = ['gaussian', 'lorenz']
    seeds = list(range(8))
    inputs = [(s, sig) for sig in signals for s in seeds]
    results = list(train_with_growth.starmap(inputs))

    import statistics
    print(f"\n=== Growth SOC test (architrace setup reproduit) ===")
    for sig in signals:
        runs = [r for r in results if r['signal_type'] == sig]
        b_init = [r['beta_init'] for r in runs]
        b_late = [r['beta_mean_late'] for r in runs]
        grow = [r['growth_events'] for r in runs]
        h_final = [r['final_hidden'] for r in runs]
        print(f"\n  signal = {sig}")
        print(f"    β init       : {statistics.mean(b_init):.3f} ± {statistics.stdev(b_init):.3f}")
        print(f"    β late mean  : {statistics.mean(b_late):.3f} ± {statistics.stdev(b_late):.3f}")
        print(f"    growth events: {statistics.mean(grow):.1f}")
        print(f"    final hidden : {statistics.mean(h_final):.1f}")
