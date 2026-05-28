"""Ablations ODE synthesizer : hidden × n_steps × dt × nonlinearity.

Identify architecture optimum pour β-controllable ODE generator.

Targets :
1. β output stability (low variance)
2. β tunability range (max-min β achievable)
3. Training speed
4. Reconstruction error on real data
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.insert(0, '.')
from beta_toolkit import compute_beta, train_synthesizer
import matplotlib.pyplot as plt

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#000000',
    'axes.facecolor': '#000000',
    'savefig.facecolor': '#000000',
    'axes.edgecolor': '#444444',
    'axes.labelcolor': '#cccccc',
    'xtick.color': '#999999',
    'ytick.color': '#999999',
    'text.color': '#ffffff',
    'font.family': 'monospace',
    'font.size': 9,
})


def gen_lorenz(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    traj = np.zeros((n, 3), dtype=np.float64)
    traj[0] = rng.uniform(-5, 5, 3)
    for i in range(1, n):
        x, y, z = traj[i - 1]
        traj[i, 0] = x + 0.005 * 10 * (y - x)
        traj[i, 1] = y + 0.005 * (x * (28 - z) - y)
        traj[i, 2] = z + 0.005 * (x * y - 8 / 3 * z)
    return ((traj - traj.mean(0)) / (traj.std(0) + 1e-6)).astype('float32')


class ODEConfigurable(nn.Module):
    """Configurable ODE Euler integrator avec nonlinearity choice."""
    def __init__(self, dim_in=3, hidden=64, n_steps=32, dt=0.1, window=8, activation='tanh'):
        super().__init__()
        self.dim_in = dim_in
        self.hidden = hidden
        self.n_steps = n_steps
        self.dt = dt
        self.window = window
        act = {'tanh': nn.Tanh(), 'gelu': nn.GELU(), 'relu': nn.ReLU(), 'silu': nn.SiLU()}[activation]
        self.f = nn.Sequential(nn.Linear(hidden + dim_in, hidden), act, nn.Linear(hidden, hidden))
        self.head = nn.Linear(hidden, dim_in)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.window, self.dim_in)
        h = torch.zeros(B, self.hidden, device=x.device)
        for t in range(self.n_steps):
            inp_idx = min(t, self.window - 1)
            inp = torch.cat([h, x[:, inp_idx]], dim=-1)
            h = h + self.dt * self.f(inp)
        return self.head(h)

    @torch.no_grad()
    def generate(self, n_samples=1000, seed=42):
        torch.manual_seed(seed)
        device = next(self.parameters()).device
        buf = torch.randn(1, self.window, self.dim_in, device=device)
        out = []
        for _ in range(n_samples):
            pred = self.forward(buf.reshape(1, -1))
            out.append(pred.squeeze(0).cpu().numpy())
            buf = torch.cat([buf[:, 1:], pred.unsqueeze(1)], dim=1)
        return np.array(out)


def train_ode(data, hidden, n_steps, dt, window, activation, n_epochs=10, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if data.ndim == 1:
        data = data[:, None]
    T, D = data.shape
    X = np.stack([data[i:i + window].flatten() for i in range(T - window - 1)])
    Y = data[window:window + len(X)]
    Xt = torch.from_numpy(X[:2000].astype('float32'))
    Yt = torch.from_numpy(Y[:2000].astype('float32'))

    model = ODEConfigurable(dim_in=D, hidden=hidden, n_steps=n_steps, dt=dt,
                              window=window, activation=activation)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(n_epochs):
        idx = torch.randperm(Xt.shape[0])
        for i in range(0, Xt.shape[0], 32):
            j = idx[i:i + 32]
            opt.zero_grad()
            pred = model(Xt[j])
            loss = F.mse_loss(pred, Yt[j])
            loss.backward()
            opt.step()
    return model, float(loss.item())


def evaluate_ode(hidden, n_steps, dt, window, activation, n_seeds=3, n_samples_gen=1000):
    """Train + generate + measure β + reconstruction MSE."""
    data = gen_lorenz(5000, seed=0)
    betas_gen = []
    losses = []
    train_times = []
    stable = []
    for seed in range(n_seeds):
        t0 = time.perf_counter()
        model, loss = train_ode(data, hidden, n_steps, dt, window, activation, n_epochs=10, seed=seed)
        train_t = time.perf_counter() - t0
        gen = model.generate(n_samples=n_samples_gen, seed=seed + 100)
        if not np.all(np.isfinite(gen)) or np.abs(gen).max() > 1e4:
            stable.append(False)
            continue
        stable.append(True)
        betas_gen.append(compute_beta(gen))
        losses.append(loss)
        train_times.append(train_t)
    if not betas_gen:
        return None
    n_params = sum(p.numel() for p in model.parameters())
    return {
        'beta_mean': float(np.mean(betas_gen)),
        'beta_std': float(np.std(betas_gen)),
        'loss_mean': float(np.mean(losses)),
        'train_time': float(np.mean(train_times)),
        'n_params': n_params,
        'stable_rate': float(np.mean(stable)),
    }


def ablation_hidden():
    print("\n=== ODE ABLATION 1 : hidden dim ===\n")
    print(f"  {'hidden':<10}{'β output':<18}{'loss':<12}{'train (s)':<12}{'params':<12}")
    results = []
    for hidden in [16, 32, 64, 128, 256]:
        r = evaluate_ode(hidden=hidden, n_steps=32, dt=0.1, window=8, activation='tanh')
        if r:
            r['hidden'] = hidden
            results.append(r)
            print(f"  {hidden:<10}{r['beta_mean']:.3f}±{r['beta_std']:.3f}      {r['loss_mean']:.4f}      {r['train_time']:.2f}        {r['n_params']:<12}")
    return results


def ablation_n_steps():
    print("\n=== ODE ABLATION 2 : n_steps integration ===\n")
    print(f"  {'n_steps':<10}{'β output':<18}{'loss':<12}{'train (s)':<12}{'params':<12}")
    results = []
    for n_steps in [4, 8, 16, 32, 64, 128]:
        r = evaluate_ode(hidden=64, n_steps=n_steps, dt=0.1, window=8, activation='tanh')
        if r:
            r['n_steps'] = n_steps
            results.append(r)
            print(f"  {n_steps:<10}{r['beta_mean']:.3f}±{r['beta_std']:.3f}      {r['loss_mean']:.4f}      {r['train_time']:.2f}        {r['n_params']:<12}")
    return results


def ablation_activation():
    print("\n=== ODE ABLATION 3 : activation function ===\n")
    print(f"  {'activation':<12}{'β output':<18}{'loss':<12}{'train (s)':<12}")
    results = []
    for act in ['tanh', 'gelu', 'silu', 'relu']:
        r = evaluate_ode(hidden=64, n_steps=32, dt=0.1, window=8, activation=act)
        if r:
            r['activation'] = act
            results.append(r)
            print(f"  {act:<12}{r['beta_mean']:.3f}±{r['beta_std']:.3f}      {r['loss_mean']:.4f}      {r['train_time']:.2f}")
    return results


def fig_ode_ablations(res_h, res_n, res_a):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    # Hidden
    axs[0].errorbar([r['hidden'] for r in res_h], [r['beta_mean'] for r in res_h],
                     yerr=[r['beta_std'] for r in res_h], color='#ff00ff', lw=2, capsize=4,
                     marker='o', markersize=8)
    axs[0].axhline(2.0, color='#666666', ls=':', alpha=0.5, label='β=2 brown noise')
    axs[0].set_xlabel('hidden dim')
    axs[0].set_ylabel('β generated')
    axs[0].set_title('ODE hidden dim ablation', color='#ffffff', pad=10)
    axs[0].set_xscale('log', base=2)
    axs[0].legend(facecolor='#111111', edgecolor='#333333')
    axs[0].grid(True, alpha=0.1)

    # n_steps
    axs[1].errorbar([r['n_steps'] for r in res_n], [r['beta_mean'] for r in res_n],
                     yerr=[r['beta_std'] for r in res_n], color='#00ff9f', lw=2, capsize=4,
                     marker='s', markersize=8)
    axs[1].axhline(2.0, color='#666666', ls=':', alpha=0.5, label='β=2 brown noise')
    axs[1].set_xlabel('n_steps')
    axs[1].set_ylabel('β generated')
    axs[1].set_title('ODE n_steps ablation', color='#ffffff', pad=10)
    axs[1].set_xscale('log', base=2)
    axs[1].legend(facecolor='#111111', edgecolor='#333333')
    axs[1].grid(True, alpha=0.1)

    # Activation
    acts = [r['activation'] for r in res_a]
    means = [r['beta_mean'] for r in res_a]
    stds = [r['beta_std'] for r in res_a]
    cs = ['#ffaa00', '#ff00ff', '#00ff9f', '#88ccff']
    axs[2].bar(acts, means, yerr=stds, color=cs[:len(acts)], edgecolor='#ffffff', linewidth=0.5, capsize=4)
    axs[2].axhline(2.0, color='#666666', ls=':', alpha=0.5)
    axs[2].set_ylabel('β generated')
    axs[2].set_title('ODE activation ablation', color='#ffffff', pad=10)
    axs[2].grid(True, alpha=0.1, axis='y')

    plt.tight_layout()
    plt.savefig('fig10_ode_ablations.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("  fig10_ode_ablations.png saved")


if __name__ == "__main__":
    print("ODE synthesizer architecture ablations...")
    res_h = ablation_hidden()
    res_n = ablation_n_steps()
    res_a = ablation_activation()
    fig_ode_ablations(res_h, res_n, res_a)
