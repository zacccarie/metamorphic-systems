"""ODE β-controlled GENERATOR via autoregressive rollout.

Goal : générer séquences temporelles longues avec β contrôlable.
Train ODE Euler, then autoregressive rollout 1000 timesteps → measure β of generated.

Use case : data augmentation rare events, controlled β signal generation.
"""
from __future__ import annotations
import modal

app = modal.App("metamorphic-ode-autoregressive")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("numpy>=1.26")
)


@app.function(image=image, gpu="A10", timeout=900)
def gen_autoregressive(seed: int = 0, dt: float = 0.1, n_steps_train: int = 32, hidden: int = 64,
                        rollout_len: int = 1000) -> dict:
    """Train ODE Euler → autoregressive rollout → measure β of generated sequence."""
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("cuda")
    rng = np.random.default_rng(seed)

    # Train data : Lorenz attractor (natural β≈2 signal)
    n_data = 5000
    traj = np.zeros((n_data, 3), dtype=np.float64)
    traj[0] = rng.uniform(-5, 5, 3)
    for i in range(1, n_data):
        x, y, z = traj[i - 1]
        traj[i, 0] = x + 0.005 * 10 * (y - x)
        traj[i, 1] = y + 0.005 * (x * (28 - z) - y)
        traj[i, 2] = z + 0.005 * (x * y - 8 / 3 * z)
    traj = ((traj - traj.mean(0)) / (traj.std(0) + 1e-6)).astype('float32')

    window = 8
    X = np.stack([traj[i: i + window].flatten() for i in range(len(traj) - window - 1)])
    Y = traj[window:window + len(X)]
    X = torch.from_numpy(X[:2000]).to(device)
    Y = torch.from_numpy(Y[:2000]).to(device)

    class ODENet(nn.Module):
        def __init__(self, hidden=64, n_steps=32, dt=0.1):
            super().__init__()
            self.window = window
            self.hidden = hidden
            self.n_steps = n_steps
            self.dt = dt
            self.f = nn.Sequential(nn.Linear(hidden + 3, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
            self.head = nn.Linear(hidden, 3)

        def forward(self, x):
            B = x.shape[0]
            x = x.view(B, self.window, 3)
            h = torch.zeros(B, self.hidden, device=x.device)
            for t in range(self.n_steps):
                inp_idx = min(t, self.window - 1)
                inp = torch.cat([h, x[:, inp_idx]], dim=-1)
                h = h + self.dt * self.f(inp)
            return self.head(h)

    torch.manual_seed(seed); np.random.seed(seed)
    model = ODENet(hidden=hidden, n_steps=n_steps_train, dt=dt).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(15):
        model.train()
        idx = torch.randperm(X.shape[0], device=device)
        for i in range(0, X.shape[0], 32):
            j = idx[i:i + 32]
            opt.zero_grad()
            loss = F.mse_loss(model(X[j]), Y[j])
            loss.backward(); opt.step()

    # Autoregressive rollout from random init
    model.eval()
    with torch.no_grad():
        buf = torch.randn(1, window, 3, device=device)
        generated = []
        for _ in range(rollout_len):
            pred = model(buf.reshape(1, -1))  # (1, 3)
            generated.append(pred.squeeze(0).cpu().numpy())
            buf = torch.cat([buf[:, 1:], pred.unsqueeze(1)], dim=1)
        generated = np.array(generated)  # (rollout_len, 3)

    # β of generated sequence (along time axis)
    def compute_beta(sig):
        if sig.shape[0] < 16:
            return 0.0
        fft = np.fft.rfft(sig, axis=0)
        power = (np.abs(fft) ** 2).mean(axis=1) if sig.ndim > 1 else np.abs(fft) ** 2
        freqs = np.arange(1, len(power))
        log_f = np.log(freqs + 1e-9)
        log_p = np.log(power[1:] + 1e-12)
        slope, _ = np.polyfit(log_f, log_p, 1)
        return float(-slope)

    beta_gen = compute_beta(generated)

    # Check stability (didn't blow up)
    max_val = float(np.abs(generated).max())
    has_nan = bool(np.any(~np.isfinite(generated)))

    return {
        'seed': seed,
        'dt': dt,
        'n_steps': n_steps_train,
        'beta_generated': beta_gen,
        'max_val': max_val,
        'has_nan': has_nan,
    }


@app.local_entrypoint()
def main():
    import statistics
    configs = []
    for dt in [0.05, 0.1, 0.2, 0.5]:
        for n_steps in [8, 16, 32, 64]:
            configs.append((dt, n_steps))

    seeds = list(range(3))
    inputs = [(s, dt, ns, 64, 1000) for (dt, ns) in configs for s in seeds]
    results = list(gen_autoregressive.starmap(inputs))

    print(f"\n=== ODE autoregressive β output 1000-step rollout ===\n")
    print(f"  {'dt':<8} {'n_steps':<10} {'β generated':<20} {'max_val':<12} {'stable':<8}")
    for (dt, ns) in configs:
        runs = [r for r in results if abs(r['dt'] - dt) < 1e-3 and r['n_steps'] == ns]
        bgs = [r['beta_generated'] for r in runs]
        mvs = [r['max_val'] for r in runs]
        nans = [r['has_nan'] for r in runs]
        m = statistics.mean(bgs); s = statistics.stdev(bgs) if len(bgs) > 1 else 0
        mv = max(mvs)
        stable = "OK" if not any(nans) and mv < 1e6 else "BLOW"
        print(f"  {dt:<8.2f} {ns:<10d} {m:.3f}±{s:.3f}        {mv:<12.2g} {stable:<8}")
