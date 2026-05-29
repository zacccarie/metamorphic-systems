"""β signature of grokking transition.

Hypothèse jamais testée : la pente power-law du spectre eigenvalues des activations
(et poids) a une SIGNATURE au moment du grokking (mémorisation → généralisation).

Connecte :
- Grokking (Power 2022) : généralisation soudaine tardive
- Heavy-tailed spectra ↔ généralisation (Martin & Mahoney 2021)
- Neural scaling from power-law spectra (Maloney 2022)
- Notre β observable (spectral slope)

Setup grokking classique :
- Tâche : (a + b) mod p, p=97
- Tiny transformer 2 layers d=128
- AdamW high weight_decay (critique pour grokking)
- Train ~30k steps, val groks tard
- Track CHAQUE checkpoint : train_acc, val_acc, β_activations, β_weights

Si β shift au moment grokking → finding nouveau.

Robust pattern : results → Volume, --detach safe, fetch stateless.

Run :
  modal run --detach modal_grokking_beta.py::run
  modal run modal_grokking_beta.py::fetch
"""
from __future__ import annotations
import modal

app = modal.App("grokking-beta")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("numpy>=1.26")
)

results_vol = modal.Volume.from_name("grokking-beta-results", create_if_missing=True)


def _eigenspectrum_beta(mat) -> float:
    """Power-law slope β of eigenvalue spectrum of mat^T mat (covariance).

    mat : (n_samples, d). Returns β such that eigenvalue_k ~ k^{-β}.
    Higher β = more concentrated/structured spectrum.
    """
    import numpy as np
    mat = mat - mat.mean(axis=0, keepdims=True)
    # Covariance eigenvalues via SVD (singular values squared)
    try:
        s = np.linalg.svd(mat, compute_uv=False)
    except Exception:
        return 0.0
    eig = (s ** 2)
    eig = eig[eig > 1e-12]
    if len(eig) < 8:
        return 0.0
    ranks = np.arange(1, len(eig) + 1)
    log_r = np.log(ranks)
    log_e = np.log(eig + 1e-12)
    slope, _ = np.polyfit(log_r, log_e, 1)
    return float(-slope)  # positive = decaying spectrum


@app.function(image=image, gpu="A10", timeout=7200, volumes={"/results": results_vol})
def run_grokking(seed: int = 0, p: int = 97, weight_decay: float = 1.0,
                 train_frac: float = 0.5, steps: int = 30000) -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import json, os

    device = torch.device("cuda")
    torch.manual_seed(seed); np.random.seed(seed)

    # Dataset : all (a,b) pairs, label (a+b) mod p
    # Sequence : [a, b, eq_token] → predict (a+b)%p
    eq_token = p
    vocab = p + 1
    pairs = [(a, b) for a in range(p) for b in range(p)]
    X = np.array([[a, b, eq_token] for a, b in pairs], dtype=np.int64)
    Y = np.array([(a + b) % p for a, b in pairs], dtype=np.int64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, Y = X[perm], Y[perm]
    n_train = int(train_frac * len(X))
    Xtr = torch.tensor(X[:n_train], device=device)
    Ytr = torch.tensor(Y[:n_train], device=device)
    Xte = torch.tensor(X[n_train:], device=device)
    Yte = torch.tensor(Y[n_train:], device=device)

    d_model, n_head, n_layer, ctx = 128, 4, 2, 3

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
            self.pos = nn.Embedding(ctx, d_model)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, p)
        def forward(self, idx, return_acts=False):
            B, T = idx.shape
            x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
            acts = []
            for blk in self.blocks:
                x = blk(x)
                if return_acts:
                    acts.append(x.detach())
            x = self.ln(x)
            logits = self.head(x[:, -1])  # predict from eq position
            return (logits, acts) if return_acts else logits

    model = GrokGPT().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay, betas=(0.9, 0.98))

    def accuracy(Xs, Ys):
        model.eval()
        with torch.no_grad():
            logits = model(Xs)
            acc = (logits.argmax(-1) == Ys).float().mean().item()
        model.train()
        return acc

    trajectory = []
    log_every = 200

    for step in range(steps + 1):
        model.train()
        opt.zero_grad()
        logits = model(Xtr)
        loss = F.cross_entropy(logits, Ytr)
        loss.backward()
        opt.step()

        if step % log_every == 0:
            tr_acc = accuracy(Xtr, Ytr)
            te_acc = accuracy(Xte, Yte)
            # β eigenspectrum of activations (last layer, eq position) on test set
            model.eval()
            with torch.no_grad():
                _, acts = model(Xte, return_acts=True)
                # last layer, eq-position activations : (n_test, d_model)
                act_mat = acts[-1][:, -1, :].cpu().numpy()
            model.train()
            beta_act = _eigenspectrum_beta(act_mat)
            # β of weight matrix spectrum (head + mlp of last block)
            W = model.blocks[-1].mlp[0].weight.detach().cpu().numpy()
            beta_w = _eigenspectrum_beta(W)
            trajectory.append({
                'step': step, 'train_loss': float(loss.item()),
                'train_acc': tr_acc, 'val_acc': te_acc,
                'beta_act': beta_act, 'beta_weight': beta_w,
            })

    # Detect grokking step : first step where val_acc > 0.9
    grok_step = None
    for t in trajectory:
        if t['val_acc'] > 0.9:
            grok_step = t['step']
            break

    out = {
        'seed': seed, 'p': p, 'weight_decay': weight_decay,
        'train_frac': train_frac, 'steps': steps,
        'grok_step': grok_step,
        'trajectory': trajectory,
    }
    os.makedirs("/results", exist_ok=True)
    with open(f"/results/grok_seed{seed}_wd{weight_decay}.json", "w") as f:
        json.dump(out, f)
    results_vol.commit()
    print(f"[seed{seed} wd{weight_decay}] grok_step={grok_step} final_val={trajectory[-1]['val_acc']:.3f}")
    return out


@app.function(image=image, volumes={"/results": results_vol})
def run_all():
    """All seeds × wd in one container (survives detach)."""
    for wd in [1.0]:
        for seed in [0, 1, 2]:
            run_grokking.local(seed=seed, weight_decay=wd, steps=30000)
    print("All grokking runs done.")


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
def run():
    """Spawn single-container all-runs (detach-safe)."""
    # Direct call inside one A10 container; launched via --detach
    import json
    for seed in [0, 1, 2]:
        run_grokking.spawn(seed, 97, 1.0, 0.5, 30000)
    print("Spawned 3 grokking runs (seed 0,1,2). Fetch: modal run modal_grokking_beta.py::fetch")


@app.local_entrypoint()
def dump_json():
    """Print full results as single JSON line for local viz parsing."""
    import json
    results = fetch_all.remote()
    print("JSON_DUMP:" + json.dumps(results))


@app.local_entrypoint()
def fetch():
    results = fetch_all.remote()
    if not results:
        print("No results yet.")
        return
    print(f"\n=== Grokking β signature ({len(results)} runs) ===\n")
    for r in results:
        traj = r['trajectory']
        gs = r['grok_step']
        print(f"\nseed{r['seed']} wd{r['weight_decay']} : grok_step={gs}")
        if gs is None:
            print("  NO GROKKING (val never >0.9)")
            # still show β trend
        # Show β around transition
        print(f"  {'step':<8}{'train_acc':<11}{'val_acc':<11}{'β_act':<10}{'β_weight':<10}")
        # Sample key points
        n = len(traj)
        idxs = sorted(set([0, n//8, n//4, n//2, 3*n//4, n-1]))
        # Add grok-step neighbors
        if gs is not None:
            for i, t in enumerate(traj):
                if t['step'] == gs:
                    idxs.extend([max(0, i-1), i, min(n-1, i+1)])
        for i in sorted(set(idxs)):
            t = traj[i]
            mark = "← GROK" if (gs and t['step'] == gs) else ""
            print(f"  {t['step']:<8}{t['train_acc']:<11.3f}{t['val_acc']:<11.3f}{t['beta_act']:<10.3f}{t['beta_weight']:<10.3f}{mark}")

    # Aggregate β_act delta around grok
    print(f"\n=== β_act shift at grokking (mean across grokked seeds) ===")
    deltas = []
    for r in results:
        traj = r['trajectory']; gs = r['grok_step']
        if gs is None:
            continue
        # β_act before (memorization, val<0.5) vs after (generalized, val>0.9)
        before = [t['beta_act'] for t in traj if t['val_acc'] < 0.5]
        after = [t['beta_act'] for t in traj if t['val_acc'] > 0.9]
        if before and after:
            import statistics
            d = statistics.mean(after) - statistics.mean(before)
            deltas.append(d)
            print(f"  seed{r['seed']} : β_act memorize={statistics.mean(before):.3f} → grok={statistics.mean(after):.3f}  Δ={d:+.3f}")
    if deltas:
        import statistics
        print(f"\n  Mean Δβ_act = {statistics.mean(deltas):+.3f} ± {statistics.stdev(deltas) if len(deltas)>1 else 0:.3f}")
        print(f"  → {'β SIGNATURE at grokking' if abs(statistics.mean(deltas)) > 0.1 else 'no clear β signature'}")
