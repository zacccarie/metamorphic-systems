"""Push ODE Euler vers β=2 via integration depth.

Hypothesis : more integration steps + smaller dt = closer to true Brownian (β=2).
Si β converge vers 2 → architrace claim survives sur architecture spécifique.
"""
from __future__ import annotations
import modal

app = modal.App("metamorphic-ode-push")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("numpy>=1.26")
)


@app.function(image=image, gpu="A10", timeout=600)
def train_ode_beta(seed: int = 0, n_steps: int = 4, dt: float = 0.1) -> dict:
    """ODE Euler avec n_steps × dt total integration time.

    Plus de steps OR plus petit dt × plus longue trajectoire = plus de
    accumulation random walk = β → 2.
    """
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("cuda")
    rng = np.random.default_rng(seed)
    n = 5000
    traj = rng.normal(0, 1, (n, 3)).astype(np.float32)
    window = n_steps  # same as integration steps
    X = np.stack([traj[i: i + window].flatten() for i in range(len(traj) - window - 1)])
    Y = traj[window:window + len(X)]
    X = torch.from_numpy(X[:1000]).to(device); Y = torch.from_numpy(Y[:1000]).to(device)

    class ODE_Euler(nn.Module):
        def __init__(self, n_steps, dt):
            super().__init__()
            self.window = n_steps
            self.dt = dt
            self.f = nn.Sequential(nn.Linear(64 + 3, 64), nn.Tanh(), nn.Linear(64, 64))
            self.head = nn.Linear(64, 3)
            self.last_act = None
        def forward(self, x):
            B = x.shape[0]
            x = x.view(B, self.window, 3)
            h = torch.zeros(B, 64, device=x.device)
            all_h = []
            for t in range(self.window):
                inp = torch.cat([h, x[:, t]], dim=-1)
                h = h + self.dt * self.f(inp)
                all_h.append(h)
            all_h = torch.stack(all_h, dim=1)
            self.last_act = all_h.reshape(-1, 64).detach()
            return self.head(h)

    torch.manual_seed(seed); np.random.seed(seed)
    model = ODE_Euler(n_steps, dt).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

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

    for ep in range(8):  # fewer epochs (compute heavier with bigger window)
        model.train()
        idx = torch.randperm(X.shape[0], device=device)
        for i in range(0, X.shape[0], 32):
            j = idx[i:i + 32]
            opt.zero_grad()
            loss = F.mse_loss(model(X[j]), Y[j])
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        _ = model(X[:128])
        beta = compute_beta(model.last_act)
    return {'seed': seed, 'n_steps': n_steps, 'dt': dt, 'beta': beta}


@app.local_entrypoint()
def push_beta():
    """Vary integration depth × dt."""
    configs = [
        (4, 0.1), (8, 0.1), (16, 0.1), (32, 0.1),  # vary n_steps
        (8, 0.01), (8, 0.5), (8, 1.0),  # vary dt
    ]
    seeds = list(range(6))
    inputs = [(s, ns, dt) for (ns, dt) in configs for s in seeds]
    results = list(train_ode_beta.starmap(inputs))

    import statistics
    print(f"\n=== ODE β push test ===")
    print(f"  {'n_steps':<10} {'dt':<8} {'β':<22}")
    for ns, dt in configs:
        runs = [r for r in results if r['n_steps'] == ns and abs(r['dt'] - dt) < 1e-6]
        betas = [r['beta'] for r in runs]
        m = statistics.mean(betas)
        s = statistics.stdev(betas) if len(betas) > 1 else 0
        print(f"  {ns:<10} {dt:<8} {m:.3f} ± {s:.3f}")
