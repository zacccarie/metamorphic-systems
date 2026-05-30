"""β-reg on permutation composition grokking — 3rd task.

S_5 group : compose two permutations of 5 elements. 120 perms.
Input [idx_a, idx_b, eq] → output idx(a∘b). Group-theoretic structure.
Distinct de modular-add (arithmetic) + sparse-parity (boolean).

Test : loss = CE - λ·β_act accélère grokking ? 3e confirmation cross-task.
Robust : volume + detach.
"""
from __future__ import annotations
import modal

app = modal.App("beta-permutation")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("numpy>=1.26")
)
results_vol = modal.Volume.from_name("beta-perm-results", create_if_missing=True)


@app.function(image=image, gpu="A10", timeout=7200, volumes={"/results": results_vol})
def run_one(seed: int = 0, lam: float = 0.0, weight_decay: float = 1.0, steps: int = 25000) -> dict:
    import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json, os, itertools
    device = torch.device("cuda")
    torch.manual_seed(seed); np.random.seed(seed)

    perms = list(itertools.permutations(range(5)))   # 120
    P = len(perms)
    idx_of = {p: i for i, p in enumerate(perms)}
    def compose(a, b):  # a∘b : apply b then a
        return tuple(a[b[i]] for i in range(5))
    comp_table = np.array([[idx_of[compose(perms[a], perms[b])] for b in range(P)] for a in range(P)])

    eq = P; vocab = P + 1
    X = np.array([[a, b, eq] for a in range(P) for b in range(P)], dtype=np.int64)
    Y = np.array([comp_table[a, b] for a in range(P) for b in range(P)], dtype=np.int64)
    rng = np.random.default_rng(seed); perm = rng.permutation(len(X))
    X, Y = X[perm], Y[perm]
    ntr = int(0.5 * len(X))
    Xtr = torch.tensor(X[:ntr], device=device); Ytr = torch.tensor(Y[:ntr], device=device)
    Xte = torch.tensor(X[ntr:], device=device); Yte = torch.tensor(Y[ntr:], device=device)

    d, nh, nl = 128, 4, 2
    class Block(nn.Module):
        def __init__(s):
            super().__init__()
            s.ln1 = nn.LayerNorm(d); s.attn = nn.MultiheadAttention(d, nh, batch_first=True)
            s.ln2 = nn.LayerNorm(d); s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        def forward(s, x):
            a, _ = s.attn(s.ln1(x), s.ln1(x), s.ln1(x), need_weights=False); x = x + a
            return x + s.mlp(s.ln2(x))
    class GPT(nn.Module):
        def __init__(s):
            super().__init__()
            s.tok = nn.Embedding(vocab, d); s.pos = nn.Embedding(3, d)
            s.blocks = nn.ModuleList([Block() for _ in range(nl)])
            s.ln = nn.LayerNorm(d); s.head = nn.Linear(d, P)
        def forward(s, idx, want_act=False):
            x = s.tok(idx) + s.pos(torch.arange(idx.shape[1], device=idx.device))
            for b in s.blocks: x = b(x)
            la = x[:, -1, :]
            return (s.head(s.ln(x)[:, -1]), la) if want_act else s.head(s.ln(x)[:, -1])

    model = GPT().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay, betas=(0.9, 0.98))

    def beta_diff(act):
        act = act - act.mean(0, keepdim=True)
        s = torch.linalg.svdvals(act); eig = (s**2) + 1e-10
        k = eig.shape[0]; r = torch.arange(1, k+1, device=act.device, dtype=act.dtype)
        lr = torch.log(r); le = torch.log(eig); lrc = lr-lr.mean(); lec = le-le.mean()
        return -((lrc*lec).sum() / (lrc*lrc).sum().clamp_min(1e-9))

    @torch.no_grad()
    def acc(Xs, Ys):
        model.eval(); a = (model(Xs).argmax(-1) == Ys).float().mean().item(); model.train(); return a
    @torch.no_grad()
    def be(Xs):
        model.eval(); _, h = model(Xs, want_act=True); b = beta_diff(h).item(); model.train(); return b

    traj = []
    for step in range(steps+1):
        model.train(); opt.zero_grad()
        lo, la = model(Xtr, want_act=True); ce = F.cross_entropy(lo, Ytr)
        loss = ce - lam*beta_diff(la) if lam > 0 else ce
        loss.backward(); opt.step()
        if step % 150 == 0:
            traj.append({'step': step, 'train_acc': acc(Xtr, Ytr), 'val_acc': acc(Xte, Yte), 'beta_act': be(Xte)})
    grok = next((t['step'] for t in traj if t['val_acc'] > 0.9), None)
    out = {'seed': seed, 'lam': lam, 'grok_step': grok, 'final_val': traj[-1]['val_acc'], 'trajectory': traj}
    os.makedirs("/results", exist_ok=True)
    with open(f"/results/perm_lam{lam}_seed{seed}.json", "w") as f: json.dump(out, f)
    results_vol.commit()
    print(f"[perm lam{lam} seed{seed}] grok={grok} final={out['final_val']:.3f}")
    return out


@app.local_entrypoint()
def run():
    for lam in [0.0, 0.01]:
        for s in [0, 1, 2]:
            run_one.spawn(s, lam, 1.0, 25000)
    print("Spawned 6 permutation runs. Fetch: ::fetch")


@app.function(image=image, volumes={"/results": results_vol})
def fetch_all() -> list:
    import os, json
    out = []
    if os.path.isdir("/results"):
        for fn in sorted(os.listdir("/results")):
            if fn.endswith(".json"):
                with open(f"/results/{fn}") as f: out.append(json.load(f))
    return out


@app.local_entrypoint()
def fetch():
    import statistics
    res = fetch_all.remote()
    if not res: print("No results yet."); return
    lams = sorted(set(r['lam'] for r in res))
    print(f"\n=== β-reg PERMUTATION S_5 — 3rd task ({len(res)} runs) ===\n")
    print(f"  {'λ':<10}{'grok_step':<24}{'final_val':<12}{'n':<4}")
    base = None
    for lam in lams:
        runs = [r for r in res if r['lam'] == lam]
        gs = [r['grok_step'] for r in runs if r['grok_step'] is not None]
        fv = [r['final_val'] for r in runs]
        if gs:
            m = statistics.mean(gs); s = statistics.stdev(gs) if len(gs)>1 else 0
            if lam == 0.0: base = m
            gss = f"{m:.0f}±{s:.0f}"
        else: gss = f"NO GROK (n={len(runs)})"
        print(f"  {lam:<10}{gss:<24}{statistics.mean(fv):<12.3f}{len(runs):<4}")
    if base:
        print(f"\n=== Verdict (baseline grok={base:.0f}) ===")
        for lam in lams:
            if lam == 0.0: continue
            runs = [r for r in res if r['lam'] == lam]; gs = [r['grok_step'] for r in runs if r['grok_step'] is not None]
            if gs:
                m = statistics.mean(gs); d = (m-base)/base*100
                v = "β ACCELERATES ★ (3rd task CONFIRMED)" if d < -10 else ("slows" if d > 10 else "neutral")
                print(f"  λ={lam}: grok {m:.0f} ({d:+.0f}%) → {v}")
