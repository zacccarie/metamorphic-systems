"""ODE β-controlled synthesizer : generator paramétré dt/n_steps → β tunable.

Hypothèse : ODE Euler trained sur signal target produit activations β plafond
≈1.57. En tunant dt + n_steps (integration depth), on contrôle β output ∈ [0, 1.57].

Application PoC : data augmentation pour rare events.
- Train ODE generator avec specific dt/n_steps configs
- Generate samples avec β cible
- Validate β output matches target via FFT slope

Setup :
- Train ODE Euler sur Lorenz (target β≈2)
- Sweep dt ∈ [0.01, 0.05, 0.1, 0.5, 1.0] × n_steps ∈ [8, 16, 32, 64, 128]
- Measure β generated output per config
- Show β tunable parameter space
"""
from __future__ import annotations
import modal

app = modal.App("metamorphic-ode-synthesizer")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("numpy>=1.26")
)


@app.function(image=image, gpu="A10", timeout=900)
def train_and_generate(seed: int = 0, dt: float = 0.1, n_steps: int = 16, hidden: int = 64) -> dict:
    """Train ODE Euler + measure β des outputs générés."""
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("cuda")
    rng = np.random.default_rng(seed)
    n_data = 5000

    # Gaussian noise input (worst case for β generation, max challenge)
    traj = rng.normal(0, 1, (n_data, 3)).astype('float32')
    window = 8
    X = np.stack([traj[i: i + window].flatten() for i in range(len(traj) - window - 1)])
    Y = traj[window:window + len(X)]
    X = torch.from_numpy(X[:1000]).to(device)
    Y = torch.from_numpy(Y[:1000]).to(device)

    class ODEGenerator(nn.Module):
        def __init__(self, hidden=64, n_steps=16, dt=0.1):
            super().__init__()
            self.window = window
            self.hidden = hidden
            self.n_steps = n_steps
            self.dt = dt
            self.f = nn.Sequential(nn.Linear(hidden + 3, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
            self.proj_in = nn.Linear(3, hidden)
            self.head = nn.Linear(hidden, 3)
            self.last_act = None

        def forward(self, x):
            B = x.shape[0]
            x = x.view(B, self.window, 3)
            h = torch.zeros(B, self.hidden, device=x.device)
            all_h = []
            for t in range(self.n_steps):
                inp_idx = min(t, self.window - 1)
                inp = torch.cat([h, x[:, inp_idx]], dim=-1)
                h = h + self.dt * self.f(inp)
                all_h.append(h)
            all_h = torch.stack(all_h, dim=1)
            self.last_act = all_h.reshape(-1, self.hidden).detach()
            return self.head(h)

    torch.manual_seed(seed); np.random.seed(seed)
    model = ODEGenerator(hidden=hidden, n_steps=n_steps, dt=dt).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(10):
        model.train()
        idx = torch.randperm(X.shape[0], device=device)
        for i in range(0, X.shape[0], 32):
            j = idx[i:i + 32]
            opt.zero_grad()
            loss = F.mse_loss(model(X[j]), Y[j])
            loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        _ = model(X[:256])
        acts = model.last_act.cpu().numpy()

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

    beta_acts = compute_beta(acts)

    # Generate samples via rollout from random initial state
    with torch.no_grad():
        gen_batch = 64
        x_init = torch.randn(gen_batch, window, 3, device=device).reshape(gen_batch, -1)
        gen_out = model(x_init)
    beta_gen_out = compute_beta(gen_out.cpu().numpy().reshape(-1, 3))

    return {
        'seed': seed,
        'dt': dt,
        'n_steps': n_steps,
        'hidden': hidden,
        'beta_activations': beta_acts,
        'beta_generated_output': beta_gen_out,
    }


@app.local_entrypoint()
def main():
    import statistics
    configs = []
    for dt in [0.01, 0.05, 0.1, 0.3, 0.5]:
        for n_steps in [8, 16, 32, 64]:
            configs.append((dt, n_steps))

    seeds = list(range(3))
    inputs = [(s, dt, ns, 64) for (dt, ns) in configs for s in seeds]
    results = list(train_and_generate.starmap(inputs))

    print(f"\n=== β tunable via dt × n_steps ===\n")
    print(f"  {'dt':<8} {'n_steps':<10} {'β activations':<20}")
    for (dt, ns) in configs:
        runs = [r for r in results if abs(r['dt'] - dt) < 1e-3 and r['n_steps'] == ns]
        bas = [r['beta_activations'] for r in runs]
        m = statistics.mean(bas); s = statistics.stdev(bas) if len(bas) > 1 else 0
        print(f"  {dt:<8.2f} {ns:<10d} {m:.3f}±{s:.3f}")

    print(f"\n=== β output generated (after rollout from random) ===")
    print(f"  {'dt':<8} {'n_steps':<10} {'β gen output':<20}")
    for (dt, ns) in configs:
        runs = [r for r in results if abs(r['dt'] - dt) < 1e-3 and r['n_steps'] == ns]
        bgs = [r['beta_generated_output'] for r in runs]
        m = statistics.mean(bgs); s = statistics.stdev(bgs) if len(bgs) > 1 else 0
        print(f"  {dt:<8.2f} {ns:<10d} {m:.3f}±{s:.3f}")
