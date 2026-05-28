"""Test which architectures genuinely MODIFY input β spectrum.

Hypothèse : MLP = passive filter (confirmé). Mais RNN, ResNet, Attention
peuvent (théoriquement) générer corrélations long-terme et donc 1/f² même
sous Gaussian input.

Test sous Gaussian noise → mesurer β des activations à chaque architecture.
Si β > 0.5 significativement → architecture génère structure spontanément.
"""
from __future__ import annotations
import modal

app = modal.App("metamorphic-arch-beta")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("numpy>=1.26")
)


@app.function(image=image, gpu="A10", timeout=600)
def train_arch_beta(seed: int = 0, arch: str = "mlp") -> dict:
    """Train arch under Gaussian noise input, measure β activations.

    archs : mlp, rnn, resnet, attention
    """
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("cuda")
    rng = np.random.default_rng(seed)
    n = 5000
    # Gaussian random walk to make input slightly correlated (similar to short Lorenz window setup)
    traj = rng.normal(0, 1, (n, 3)).astype(np.float32)

    window = 4
    X = np.stack([traj[i: i + window].flatten() for i in range(len(traj) - window - 1)])
    Y = traj[window:window + len(X)]
    X = torch.from_numpy(X[:1000]).to(device); Y = torch.from_numpy(Y[:1000]).to(device)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(12, 64), nn.GELU(),
                nn.Linear(64, 64), nn.GELU(),
                nn.Linear(64, 64), nn.GELU(),
                nn.Linear(64, 3),
            )
            self.last_act = None
        def forward(self, x):
            for i, layer in enumerate(self.net):
                x = layer(x)
                if i == 4:  # save middle GELU output
                    self.last_act = x.detach()
            return x

    class RNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.window = 4
            self.rnn = nn.GRU(input_size=3, hidden_size=64, num_layers=2, batch_first=True)
            self.head = nn.Linear(64, 3)
            self.last_act = None
        def forward(self, x):
            B = x.shape[0]
            x = x.view(B, self.window, 3)
            h, _ = self.rnn(x)
            self.last_act = h.reshape(-1, h.shape[-1]).detach()  # all timesteps
            return self.head(h[:, -1])

    class ResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(12, 64)
            self.fc1 = nn.Linear(64, 64); self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 64); self.fc4 = nn.Linear(64, 64)
            self.head = nn.Linear(64, 3)
            self.last_act = None
        def forward(self, x):
            h = self.proj(x)
            h = h + F.gelu(self.fc2(F.gelu(self.fc1(h))))
            h = h + F.gelu(self.fc4(F.gelu(self.fc3(h))))
            self.last_act = h.detach()
            return self.head(h)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.window = 4
            self.tok_proj = nn.Linear(3, 32)
            layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64,
                                                 dropout=0.0, batch_first=True, activation='gelu')
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Linear(32, 3)
            self.last_act = None
        def forward(self, x):
            B = x.shape[0]
            tok = x.view(B, self.window, 3)
            tok = self.tok_proj(tok)
            z = self.encoder(tok)
            self.last_act = z.reshape(-1, z.shape[-1]).detach()
            return self.head(z.mean(dim=1))

    factory = {'mlp': MLP, 'rnn': RNN, 'resnet': ResNet, 'attention': Attention}[arch]
    torch.manual_seed(seed); np.random.seed(seed)
    model = factory().to(device)
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

    # Train brief
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
        beta = compute_beta(model.last_act)
    n_params = sum(p.numel() for p in model.parameters())
    return {'seed': seed, 'arch': arch, 'beta': beta, 'n_params': n_params}


@app.local_entrypoint()
def arch_sweep():
    """4 archs × 8 seeds parallel."""
    archs = ['mlp', 'rnn', 'resnet', 'attention']
    seeds = list(range(8))
    inputs = [(s, a) for a in archs for s in seeds]
    results = list(train_arch_beta.starmap(inputs))

    import statistics
    print(f"\n=== β under Gaussian noise per architecture ===")
    print(f"  {'arch':<14} {'β activations':<22} {'params':<10}")
    for a in archs:
        runs = [r for r in results if r['arch'] == a]
        betas = [r['beta'] for r in runs]
        ps = runs[0]['n_params']
        m = statistics.mean(betas)
        s = statistics.stdev(betas)
        emergence = "EMERGES" if m > 0.5 else "passive"
        print(f"  {a:<14} {m:.3f} ± {s:.3f}        {ps:<10}  [{emergence}]")
