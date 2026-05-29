"""β-accelerated grokking — CAUSAL test : β lever ou symptôme ?

Hypothèse : ajouter objectif β-maximization (push repr vers low-rank power-law)
fait grokker PLUS VITE → β CAUSE généralisation accélérée → β = LEVIER.

loss = CrossEntropy - λ · β_act    (encourage β haut = repr compressée)

β_act différentiable : singular values (torch.linalg.svdvals) → power-law slope
closed-form (cov/var en log-log). Backprop à travers.

Variants : λ ∈ {0 (baseline), 0.001, 0.01, 0.05} × seeds {0,1,2}
Métrique : grok_step (premier val_acc>0.9). Si chute avec λ>0 → β levier.

Résultats possibles :
- grok_step chute → β LEVIER, extraction accélérée (finding majeur)
- inchangé → β symptôme, monitor only (honnête)
- monte/casse → forcer β nuit (collapse)

Robust : volume + detach-safe + fetch stateless.
"""
from __future__ import annotations
import modal

app = modal.App("beta-accelerated-grokking")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("numpy>=1.26")
)

results_vol = modal.Volume.from_name("beta-accel-results", create_if_missing=True)


@app.function(image=image, gpu="A10", timeout=7200, volumes={"/results": results_vol})
def run_one(seed: int = 0, lam: float = 0.0, p: int = 97, weight_decay: float = 1.0,
            steps: int = 20000) -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import json, os

    device = torch.device("cuda")
    torch.manual_seed(seed); np.random.seed(seed)

    eq_token = p
    vocab = p + 1
    pairs = [(a, b) for a in range(p) for b in range(p)]
    X = np.array([[a, b, eq_token] for a, b in pairs], dtype=np.int64)
    Y = np.array([(a + b) % p for a, b in pairs], dtype=np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, Y = X[perm], Y[perm]
    n_train = int(0.5 * len(X))
    Xtr = torch.tensor(X[:n_train], device=device); Ytr = torch.tensor(Y[:n_train], device=device)
    Xte = torch.tensor(X[n_train:], device=device); Yte = torch.tensor(Y[n_train:], device=device)

    d_model, n_head, n_layer = 128, 4, 2

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                     nn.Linear(4 * d_model, d_model))
        def forward(self, x):
            a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
            x = x + a
            x = x + self.mlp(self.ln2(x))
            return x

    class GrokGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab, d_model)
            self.pos = nn.Embedding(3, d_model)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, p)
        def forward(self, idx, want_last_act=False):
            B, T = idx.shape
            x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
            for blk in self.blocks:
                x = blk(x)
            last_act = x[:, -1, :]  # eq-position, pre-final-LN
            logits = self.head(self.ln(x)[:, -1])
            return (logits, last_act) if want_last_act else logits

    model = GrokGPT().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay, betas=(0.9, 0.98))

    def beta_differentiable(act):
        """Power-law slope of eigenspectrum, DIFFERENTIABLE.

        act : (n, d). Centered → singular values → eig=s² → slope log(eig) vs log(rank).
        """
        act = act - act.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(act)         # differentiable
        eig = s ** 2
        eig = eig + 1e-10
        k = eig.shape[0]
        ranks = torch.arange(1, k + 1, device=act.device, dtype=act.dtype)
        log_r = torch.log(ranks)
        log_e = torch.log(eig)
        lr_c = log_r - log_r.mean()
        le_c = log_e - log_e.mean()
        slope = (lr_c * le_c).sum() / (lr_c * lr_c).sum().clamp_min(1e-9)
        return -slope  # positive = decaying

    @torch.no_grad()
    def acc(Xs, Ys):
        model.eval()
        a = (model(Xs).argmax(-1) == Ys).float().mean().item()
        model.train()
        return a

    @torch.no_grad()
    def beta_eval(Xs):
        model.eval()
        _, la = model(Xs, want_last_act=True)
        b = beta_differentiable(la).item()
        model.train()
        return b

    trajectory = []
    log_every = 100
    for step in range(steps + 1):
        model.train()
        opt.zero_grad()
        logits, last_act = model(Xtr, want_last_act=True)
        ce = F.cross_entropy(logits, Ytr)
        if lam > 0:
            b = beta_differentiable(last_act)
            loss = ce - lam * b      # maximize β
        else:
            loss = ce
        loss.backward()
        opt.step()
        if step % log_every == 0:
            trajectory.append({
                'step': step, 'ce': float(ce.item()),
                'train_acc': acc(Xtr, Ytr), 'val_acc': acc(Xte, Yte),
                'beta_act': beta_eval(Xte),
            })

    grok_step = next((t['step'] for t in trajectory if t['val_acc'] > 0.9), None)
    out = {'seed': seed, 'lam': lam, 'weight_decay': weight_decay, 'steps': steps,
           'grok_step': grok_step, 'final_val': trajectory[-1]['val_acc'],
           'trajectory': trajectory}
    os.makedirs("/results", exist_ok=True)
    with open(f"/results/accel_lam{lam}_seed{seed}.json", "w") as f:
        json.dump(out, f)
    results_vol.commit()
    print(f"[lam{lam} seed{seed}] grok_step={grok_step} final_val={out['final_val']:.3f}")
    return out


@app.local_entrypoint()
def run():
    """Spawn all λ × seed detached."""
    lams = [0.0, 0.001, 0.01, 0.05]
    seeds = [0, 1, 2]
    for lam in lams:
        for s in seeds:
            run_one.spawn(s, lam, 97, 1.0, 20000)
    print(f"Spawned {len(lams)*len(seeds)} runs. Fetch: modal run modal_beta_accelerated.py::fetch")


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
        print("No results yet.")
        return
    lams = sorted(set(r['lam'] for r in res))
    print(f"\n=== β-accelerated grokking : grok_step vs λ ({len(res)} runs) ===\n")
    print(f"  {'λ':<10}{'grok_step (mean±std)':<26}{'final_val':<14}{'n':<4}")
    baseline_grok = None
    for lam in lams:
        runs = [r for r in res if r['lam'] == lam]
        gsteps = [r['grok_step'] for r in runs if r['grok_step'] is not None]
        fvals = [r['final_val'] for r in runs]
        if gsteps:
            m = statistics.mean(gsteps); s = statistics.stdev(gsteps) if len(gsteps) > 1 else 0
            if lam == 0.0:
                baseline_grok = m
            gs_str = f"{m:.0f}±{s:.0f}"
        else:
            gs_str = "NO GROK"
        fv = statistics.mean(fvals)
        print(f"  {lam:<10}{gs_str:<26}{fv:<14.3f}{len(runs):<4}")

    if baseline_grok:
        print(f"\n=== Verdict (vs baseline grok_step={baseline_grok:.0f}) ===")
        for lam in lams:
            if lam == 0.0:
                continue
            runs = [r for r in res if r['lam'] == lam]
            gsteps = [r['grok_step'] for r in runs if r['grok_step'] is not None]
            if gsteps:
                m = statistics.mean(gsteps)
                delta = (m - baseline_grok) / baseline_grok * 100
                verdict = "β ACCELERATES ★" if delta < -10 else ("slows" if delta > 10 else "neutral")
                print(f"  λ={lam}: grok {m:.0f} ({delta:+.0f}%) → {verdict}")
