"""β-regularization on sparse parity grokking — cross-task validation.

2e tâche (vs modular addition) : (n,k)-sparse parity. Label = XOR de k bits fixes
parmi n. Classique grokking (Nanda, Merrill 2023). MLP + high weight decay.

Test : loss = CE - λ·β_act accélère-t-il grokking AUSSI ici ?
Si OUI → β-reg robuste cross-task → vrai résultat.
Si NON → spécifique modular addition → nuancer.

β_act = power-law slope eigenspectrum hidden activations (différentiable svdvals).

Robust : volume + detach-safe + fetch stateless.
"""
from __future__ import annotations
import modal

app = modal.App("beta-sparse-parity")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("numpy>=1.26")
)

results_vol = modal.Volume.from_name("beta-parity-results", create_if_missing=True)


@app.function(image=image, gpu="A10", timeout=7200, volumes={"/results": results_vol})
def run_one(seed: int = 0, lam: float = 0.0, n_bits: int = 30, k: int = 3,
            n_train: int = 1000, weight_decay: float = 1e-2, steps: int = 20000) -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import json, os

    device = torch.device("cuda")
    torch.manual_seed(seed); np.random.seed(seed)

    # k fixed bits among n define the parity
    rng = np.random.default_rng(seed)
    parity_bits = rng.choice(n_bits, size=k, replace=False)

    def make(n_samples):
        X = rng.integers(0, 2, size=(n_samples, n_bits)).astype(np.float32)
        Y = (X[:, parity_bits].sum(axis=1) % 2).astype(np.int64)
        return X, Y

    Xtr_np, Ytr_np = make(n_train)
    Xte_np, Yte_np = make(2000)
    Xtr = torch.tensor(Xtr_np, device=device); Ytr = torch.tensor(Ytr_np, device=device)
    Xte = torch.tensor(Xte_np, device=device); Yte = torch.tensor(Yte_np, device=device)

    width = 256

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(n_bits, width)
            self.fc2 = nn.Linear(width, width)
            self.head = nn.Linear(width, 2)
        def forward(self, x, want_act=False):
            h1 = F.relu(self.fc1(x))
            h2 = F.relu(self.fc2(h1))
            logits = self.head(h2)
            return (logits, h2) if want_act else logits

    model = MLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)

    def beta_diff(act):
        act = act - act.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(act)
        eig = (s ** 2) + 1e-10
        k_ = eig.shape[0]
        ranks = torch.arange(1, k_ + 1, device=act.device, dtype=act.dtype)
        lr = torch.log(ranks); le = torch.log(eig)
        lrc = lr - lr.mean(); lec = le - le.mean()
        slope = (lrc * lec).sum() / (lrc * lrc).sum().clamp_min(1e-9)
        return -slope

    @torch.no_grad()
    def acc(Xs, Ys):
        model.eval(); a = (model(Xs).argmax(-1) == Ys).float().mean().item(); model.train(); return a

    @torch.no_grad()
    def beta_eval(Xs):
        model.eval(); _, h = model(Xs, want_act=True); b = beta_diff(h).item(); model.train(); return b

    traj = []
    for step in range(steps + 1):
        model.train(); opt.zero_grad()
        logits, h2 = model(Xtr, want_act=True)
        ce = F.cross_entropy(logits, Ytr)
        loss = ce - lam * beta_diff(h2) if lam > 0 else ce
        loss.backward(); opt.step()
        if step % 100 == 0:
            traj.append({'step': step, 'train_acc': acc(Xtr, Ytr),
                         'val_acc': acc(Xte, Yte), 'beta_act': beta_eval(Xte)})

    grok = next((t['step'] for t in traj if t['val_acc'] > 0.9), None)
    out = {'seed': seed, 'lam': lam, 'n_bits': n_bits, 'k': k, 'n_train': n_train,
           'weight_decay': weight_decay, 'grok_step': grok,
           'final_val': traj[-1]['val_acc'], 'trajectory': traj}
    os.makedirs("/results", exist_ok=True)
    with open(f"/results/parity_lam{lam}_seed{seed}.json", "w") as f:
        json.dump(out, f)
    results_vol.commit()
    print(f"[parity lam{lam} seed{seed}] grok={grok} final={out['final_val']:.3f}")
    return out


@app.local_entrypoint()
def run():
    lams = [0.0, 0.01]   # focus baseline vs sweet spot
    seeds = [0, 1, 2]
    for lam in lams:
        for s in seeds:
            run_one.spawn(s, lam, 30, 3, 1000, 1e-2, 20000)
    print(f"Spawned {len(lams)*len(seeds)} parity runs. Fetch: ::fetch")


@app.function(image=image, volumes={"/results": results_vol})
def fetch_all() -> list:
    import os, json
    out = []
    if os.path.isdir("/results"):
        for fn in sorted(os.listdir("/results")):
            if fn.endswith(".json"):
                with open(f"/results/{fn}") as f:
                    out.append(json.load(f))
    return out


@app.local_entrypoint()
def fetch():
    import statistics
    res = fetch_all.remote()
    if not res:
        print("No results yet."); return
    lams = sorted(set(r['lam'] for r in res))
    print(f"\n=== β-reg on SPARSE PARITY (n=30,k=3) — cross-task test ({len(res)} runs) ===\n")
    print(f"  {'λ':<10}{'grok_step':<24}{'final_val':<12}{'n':<4}")
    base = None
    for lam in lams:
        runs = [r for r in res if r['lam'] == lam]
        gs = [r['grok_step'] for r in runs if r['grok_step'] is not None]
        fv = [r['final_val'] for r in runs]
        if gs:
            m = statistics.mean(gs); s = statistics.stdev(gs) if len(gs) > 1 else 0
            if lam == 0.0: base = m
            gss = f"{m:.0f}±{s:.0f}"
        else:
            gss = "NO GROK (n=%d)" % len(runs)
        print(f"  {lam:<10}{gss:<24}{statistics.mean(fv):<12.3f}{len(runs):<4}")
    if base:
        print(f"\n=== Verdict cross-task (baseline grok={base:.0f}) ===")
        for lam in lams:
            if lam == 0.0: continue
            runs = [r for r in res if r['lam'] == lam]
            gs = [r['grok_step'] for r in runs if r['grok_step'] is not None]
            if gs:
                m = statistics.mean(gs); d = (m - base) / base * 100
                v = "β ACCELERATES ★ (cross-task CONFIRMED)" if d < -10 else ("slows" if d > 10 else "neutral")
                print(f"  λ={lam}: grok {m:.0f} ({d:+.0f}%) → {v}")
