"""Test if paired+magnitude bias (algebra ablation finding M4) extends to dynamics.

Complex echoua sur Lorenz (Real 4.4× mieux). Mais c'était Complex algebra spécifique.
Real bias = paired channels + magnitude activation. Tester ça sur Lorenz directement.

Hypothèses :
- A : paired+magnitude wins Lorenz too → bias domain-agnostic, vrai inductive bias.
- B : paired+magnitude loses Lorenz → bias task-dependent (audio phase only).
"""
from __future__ import annotations
import modal

app = modal.App("metamorphic-paired-lorenz")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("numpy>=1.26")
)


@app.function(image=image, gpu="A10", timeout=300)
def train_lorenz(seed: int = 0, mode: str = "paired_mag") -> dict:
    """Modes :
    - real_baseline : standard MLP (d=3, h=64) — original winner
    - complex : ComplexMLP (had failed)
    - paired_mag : 2 branch Linear with magnitude norm activation
    - paired_gelu : 2 branch Linear with separate GELU each (NO magnitude)
    """
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("cuda")

    def lorenz(n=5000, dt=0.01, sigma=10, rho=28, beta=8 / 3):
        xs = np.zeros((n, 3), dtype=np.float64)
        xs[0] = [1.0, 1.0, 1.0]
        for i in range(1, n):
            x, y, z = xs[i - 1]
            xs[i, 0] = x + dt * sigma * (y - x)
            xs[i, 1] = y + dt * (x * (rho - z) - y)
            xs[i, 2] = z + dt * (x * y - beta * z)
        return xs.astype(np.float32)

    def make_windows(traj, window=4):
        n = len(traj) - window - 1
        X = np.stack([traj[i: i + window].flatten() for i in range(n)])
        Y = traj[window:window + n]
        return X.astype(np.float32), Y.astype(np.float32)

    # Models
    class RealBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(12, 64), nn.GELU(),
                nn.Linear(64, 64), nn.GELU(),
                nn.Linear(64, 64), nn.GELU(),
                nn.Linear(64, 3),
            )
        def forward(self, x):
            return self.net(x)

    class ComplexMLP(nn.Module):
        def __init__(self):
            super().__init__()
            h = 32
            self.Wr = nn.ModuleList([nn.Linear(12, h), nn.Linear(h, h), nn.Linear(h, h), nn.Linear(h, 3)])
            self.Wi = nn.ModuleList([nn.Linear(12, h), nn.Linear(h, h), nn.Linear(h, h), nn.Linear(h, 3)])
            self.biases = nn.ParameterList([nn.Parameter(torch.zeros(h)) for _ in range(3)])
        def forward(self, x):
            r, i = x, torch.zeros_like(x)
            for idx in range(3):
                nr = self.Wr[idx](r) - self.Wi[idx](i)
                ni = self.Wr[idx](i) + self.Wi[idx](r)
                mag = torch.sqrt(nr * nr + ni * ni + 1e-12)
                b = self.biases[idx].view(*([1] * (mag.ndim - 1)), -1)
                scale = torch.relu(mag + b) / mag
                r, i = nr * scale, ni * scale
            r = self.Wr[3](r) - self.Wi[3](i)
            return r  # Real part for regression

    class PairedMag(nn.Module):
        """2 branch Linear with magnitude norm activation (no complex algebra)."""
        def __init__(self):
            super().__init__()
            h = 32
            self.b1 = nn.ModuleList([nn.Linear(12, h), nn.Linear(h, h), nn.Linear(h, h), nn.Linear(h, 3)])
            self.b2 = nn.ModuleList([nn.Linear(12, h), nn.Linear(h, h), nn.Linear(h, h), nn.Linear(h, 3)])
            self.biases = nn.ParameterList([nn.Parameter(torch.zeros(h)) for _ in range(3)])
        def forward(self, x):
            r, i = x, x  # Same input both branches
            for idx in range(3):
                nr = self.b1[idx](r)
                ni = self.b2[idx](i)
                mag = torch.sqrt(nr * nr + ni * ni + 1e-12)
                b = self.biases[idx].view(*([1] * (mag.ndim - 1)), -1)
                scale = torch.relu(mag + b) / mag
                r, i = nr * scale, ni * scale
            return self.b1[3](r)

    class PairedGELU(nn.Module):
        """2 branch Linear with separate GELU each (NO magnitude)."""
        def __init__(self):
            super().__init__()
            h = 32
            self.b1 = nn.ModuleList([nn.Linear(12, h), nn.Linear(h, h), nn.Linear(h, h), nn.Linear(h, 3)])
            self.b2 = nn.ModuleList([nn.Linear(12, h), nn.Linear(h, h), nn.Linear(h, h), nn.Linear(h, 3)])
        def forward(self, x):
            r, i = x, x
            for idx in range(3):
                r = F.gelu(self.b1[idx](r))
                i = F.gelu(self.b2[idx](i))
            return self.b1[3](r) + self.b2[3](i)

    # Setup data
    traj_tr = lorenz(5000, sigma=10)
    traj_va = lorenz(1500, sigma=10)
    mu = traj_tr.mean(0); std = traj_tr.std(0) + 1e-6
    traj_tr = (traj_tr - mu) / std
    traj_va = (traj_va - mu) / std
    x_tr, y_tr = make_windows(traj_tr)
    x_va, y_va = make_windows(traj_va)
    x_tr = torch.from_numpy(x_tr[:400]).to(device); y_tr = torch.from_numpy(y_tr[:400]).to(device)
    x_va = torch.from_numpy(x_va[:200]).to(device); y_va = torch.from_numpy(y_va[:200]).to(device)

    torch.manual_seed(seed); np.random.seed(seed)
    if mode == "real_baseline":
        model = RealBaseline().to(device)
    elif mode == "complex":
        model = ComplexMLP().to(device)
    elif mode == "paired_mag":
        model = PairedMag().to(device)
    elif mode == "paired_gelu":
        model = PairedGELU().to(device)
    else:
        raise ValueError(mode)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = x_tr.shape[0]
    for ep in range(12):
        model.train()
        idx = torch.randperm(n, device=device)
        for i in range(0, n, 32):
            j = idx[i:i + 32]
            opt.zero_grad()
            loss = F.mse_loss(model(x_tr[j]), y_tr[j])
            loss.backward(); opt.step()
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        mse = F.mse_loss(model(x_va), y_va).item()
    return {'seed': seed, 'mode': mode, 'val_mse': float(mse), 'n_params': n_params}


@app.local_entrypoint()
def sweep_lorenz():
    """4 modes × 10 seeds. Test paired+magnitude bias on Lorenz."""
    modes = ['real_baseline', 'complex', 'paired_mag', 'paired_gelu']
    seeds = list(range(10))
    inputs = [(s, m) for m in modes for s in seeds]
    results = list(train_lorenz.starmap(inputs))

    import statistics
    print(f"\n=== Lorenz : paired+magnitude generalization test ===")
    print(f"  {'mode':<18} {'val_mse':<20} {'params':<8}")
    for m in modes:
        runs = [r for r in results if r['mode'] == m]
        mses = [r['val_mse'] for r in runs]
        ps = runs[0]['n_params']
        print(f"  {m:<18} {statistics.mean(mses):.5f}±{statistics.stdev(mses):.5f}    {ps}")
