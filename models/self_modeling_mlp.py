"""Self-Modeling MLP (SMMLP) — architecture qui prédit ses propres activations.

Concept :
- NN standard : task_out = forward(x)
- SMMLP : task_out, self_pred = forward(x)
  où self_pred = prédiction des activations de couche middle au prochain forward.

Loss = task_loss + α · self_modeling_loss
où self_modeling_loss = MSE(self_pred_t, actual_activations_{t+1} sur même input)

Hypothèse à TUER :
- α=0 : standard model
- α>0 : self-modeling améliore task performance (active inference style)
- Si α>0 dégrade : self-modeling pas utile per se sur cette task.

Test setup matched M4 (chirps n=400 σ=1.5, 8 classes).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def make_dataset(n_per_class, n_classes=8, noise=1.5, seed=0):
    freqs = np.array([800 + 25 * i for i in range(n_classes)])
    mods = np.array([0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 0.25, -0.25])
    X, y = [], []
    for c, (fc, fm) in enumerate(zip(freqs, mods)):
        for i in range(n_per_class):
            X.append(gen_chirp(fc, fm, noise, seed=seed * 10000 + c * 1000 + i))
            y.append(c)
    X = np.stack(X); y = np.array(y, dtype=np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    return torch.from_numpy(X[perm]), torch.from_numpy(y[perm])


class SelfModelingMLP(nn.Module):
    """MLP qui produit task_out + prédiction de propres activations layer 2.

    α = poids du self-modeling loss. α=0 → comme MLP standard.
    """
    def __init__(self, in_dim, hidden, out_dim, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)
        # Self-model head : prédit h_2 (output fc2) à partir de h_1 (output fc1)
        # Hypothèse : "what will my layer 2 produce ?"
        self.self_model = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x):
        h1 = F.gelu(self.fc1(x))
        h2 = F.gelu(self.fc2(h1))
        task_out = self.fc3(h2)
        # Self-prediction : prédire h2 depuis h1 (avant qu'il soit calculé)
        h2_predicted = self.self_model(h1)
        return task_out, h2_predicted, h2  # h2 = vraie valeur "future"


def train(model, xtr, ytr, xva, yva, epochs=12, batch_size=32, lr=1e-3, device='cpu'):
    model.to(device); xtr, ytr, xva, yva = xtr.to(device), ytr.to(device), xva.to(device), yva.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = xtr.shape[0]
    history = {'val_acc': [], 'task_loss': [], 'self_loss': []}
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(n, device=device)
        ep_task_loss = 0.0; ep_self_loss = 0.0
        for i in range(0, n, batch_size):
            j = idx[i:i + batch_size]
            opt.zero_grad()
            task_out, h2_pred, h2_true = model(xtr[j])
            task_loss = F.cross_entropy(task_out, ytr[j])
            self_loss = F.mse_loss(h2_pred, h2_true.detach())  # detach h2_true (no grad backprop on it)
            total_loss = task_loss + model.alpha * self_loss
            total_loss.backward()
            opt.step()
            ep_task_loss += task_loss.item() * j.shape[0]
            ep_self_loss += self_loss.item() * j.shape[0]
        history['task_loss'].append(ep_task_loss / n)
        history['self_loss'].append(ep_self_loss / n)
        model.eval()
        with torch.no_grad():
            task_out, _, _ = model(xva)
            acc = (task_out.argmax(-1) == yva).float().mean().item()
        history['val_acc'].append(acc)
    return history


def run(alpha: float, seed: int):
    device = torch.device('cpu')
    torch.manual_seed(seed); np.random.seed(seed)
    xtr, ytr = make_dataset(400, seed=0)
    xva, yva = make_dataset(120, seed=999)
    model = SelfModelingMLP(32, 32, 8, alpha=alpha)
    h = train(model, xtr, ytr, xva, yva, epochs=12, device=device)
    return {
        'seed': seed, 'alpha': alpha,
        'val_acc': h['val_acc'][-1],
        'final_task_loss': h['task_loss'][-1],
        'final_self_loss': h['self_loss'][-1],
    }


if __name__ == "__main__":
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    r = run(alpha, seed)
    print(json.dumps(r))
