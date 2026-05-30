"""L4 strengthening : 10 seeds + cheap β estimator.

3 conditions × 10 seeds sur modular addition (benchmark grokking canonique) :
- baseline   : λ=0
- beta_full  : λ=0.01, β via svdvals complet (O(d³))
- beta_cheap : λ=0.01, β via torch.svd_lowrank q=32 (randomized, cheap, differentiable)

Buts :
1. CI serré sur l'effet -24% (10 seeds vs 3)
2. Valider que cheap β récupère l'accélération à coût réduit (overhead train_time)

Robust : volume + detach single-container + fetch stateless.

Run :
  modal run --detach modal_beta_seeds.py::run_all
  modal run modal_beta_seeds.py::fetch
"""
from __future__ import annotations
import modal

app = modal.App("beta-seeds-l4")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("numpy>=1.26")
)
results_vol = modal.Volume.from_name("beta-seeds-results", create_if_missing=True)


@app.function(image=image, gpu="A10", timeout=14400, volumes={"/results": results_vol})
def run_all(steps: int = 6000):
    """30 runs (3 conditions × 10 seeds) séquentiels dans 1 container. Survit detach."""
    conditions = ["baseline", "beta_full", "beta_cheap"]
    for cond in conditions:
        for seed in range(10):
            _train(cond, seed, steps)
    print("All 30 runs done.")


def _train(cond: str, seed: int, steps: int) -> dict:
    import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json, os, time
    device = torch.device("cuda")
    torch.manual_seed(seed); np.random.seed(seed)
    p = 97; eq = p; vocab = p + 1
    lam = 0.0 if cond == "baseline" else 0.01

    X = np.array([[a, b, eq] for a in range(p) for b in range(p)], dtype=np.int64)
    Y = np.array([(a + b) % p for a in range(p) for b in range(p)], dtype=np.int64)
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
            s.ln = nn.LayerNorm(d); s.head = nn.Linear(d, p)
        def forward(s, idx, want_act=False):
            x = s.tok(idx) + s.pos(torch.arange(idx.shape[1], device=idx.device))
            for b in s.blocks: x = b(x)
            la = x[:, -1, :]
            return (s.head(s.ln(x)[:, -1]), la) if want_act else s.head(s.ln(x)[:, -1])

    model = GPT().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))

    def beta_full(act):
        act = act - act.mean(0, keepdim=True)
        s = torch.linalg.svdvals(act); eig = (s**2) + 1e-10
        k = eig.shape[0]; r = torch.arange(1, k+1, device=act.device, dtype=act.dtype)
        lr = torch.log(r); le = torch.log(eig); lrc = lr-lr.mean(); lec = le-le.mean()
        return -((lrc*lec).sum() / (lrc*lrc).sum().clamp_min(1e-9))

    def beta_cheap(act, q=32):
        act = act - act.mean(0, keepdim=True)
        # randomized low-rank SVD : top-q singular values, differentiable, O(n d q)
        U, S, V = torch.svd_lowrank(act, q=q)
        eig = (S**2) + 1e-10
        r = torch.arange(1, eig.shape[0]+1, device=act.device, dtype=act.dtype)
        lr = torch.log(r); le = torch.log(eig); lrc = lr-lr.mean(); lec = le-le.mean()
        return -((lrc*lec).sum() / (lrc*lrc).sum().clamp_min(1e-9))

    beta_fn = beta_cheap if cond == "beta_cheap" else beta_full

    @torch.no_grad()
    def acc(Xs, Ys):
        model.eval(); a = (model(Xs).argmax(-1) == Ys).float().mean().item(); model.train(); return a

    traj = []; t0 = time.time()
    for step in range(steps + 1):
        model.train(); opt.zero_grad()
        lo, la = model(Xtr, want_act=True); ce = F.cross_entropy(lo, Ytr)
        loss = ce - lam * beta_fn(la) if lam > 0 else ce
        loss.backward(); opt.step()
        if step % 100 == 0:
            traj.append((step, acc(Xtr, Ytr), acc(Xte, Yte)))
    train_time = time.time() - t0
    grok = next((s for (s, tr, va) in traj if va > 0.9), None)
    final_val = traj[-1][2]

    out = {'cond': cond, 'seed': seed, 'lam': lam, 'grok_step': grok,
           'final_val': final_val, 'train_time': train_time}
    os.makedirs("/results", exist_ok=True)
    with open(f"/results/{cond}_seed{seed}.json", "w") as f:
        json.dump(out, f)
    results_vol.commit()
    print(f"[{cond} seed{seed}] grok={grok} final={final_val:.3f} t={train_time:.0f}s")
    return out


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
def run_all_local():
    run_all.spawn(6000)
    print("Spawned run_all (30 runs). Fetch: modal run modal_beta_seeds.py::fetch")


@app.local_entrypoint()
def fetch():
    import statistics
    res = fetch_all.remote()
    if not res:
        print("No results yet."); return
    conds = ["baseline", "beta_full", "beta_cheap"]
    print(f"\n=== L4 : 10 seeds × 3 conditions (modular addition) — {len(res)} runs ===\n")
    print(f"  {'cond':<14}{'grok_step':<20}{'final_val':<16}{'train(s)':<12}{'n':<4}")
    base = None
    for c in conds:
        runs = [r for r in res if r['cond'] == c]
        if not runs: continue
        gs = [r['grok_step'] for r in runs if r['grok_step'] is not None]
        fv = [r['final_val'] for r in runs]
        tt = [r['train_time'] for r in runs]
        if gs:
            m = statistics.mean(gs); s = statistics.stdev(gs) if len(gs) > 1 else 0
            if c == "baseline": base = m
            gss = f"{m:.0f}±{s:.0f}"
        else:
            gss = f"NO GROK (n={len(runs)})"
        print(f"  {c:<14}{gss:<20}{statistics.mean(fv):.3f}±{statistics.stdev(fv) if len(fv)>1 else 0:.3f}   {statistics.mean(tt):<12.0f}{len(runs):<4}")
    if base:
        print(f"\n=== Verdict (baseline grok={base:.0f}) ===")
        for c in ["beta_full", "beta_cheap"]:
            runs = [r for r in res if r['cond'] == c]
            gs = [r['grok_step'] for r in runs if r['grok_step'] is not None]
            if gs:
                m = statistics.mean(gs); d = (m - base) / base * 100
                print(f"  {c}: grok {m:.0f} ({d:+.0f}%)")
        # overhead cheap vs full
        tf = [r['train_time'] for r in res if r['cond'] == 'beta_full']
        tc = [r['train_time'] for r in res if r['cond'] == 'beta_cheap']
        tb = [r['train_time'] for r in res if r['cond'] == 'baseline']
        if tf and tc and tb:
            import statistics as st
            print(f"\n  Overhead vs baseline : full β = {st.mean(tf)/st.mean(tb):.2f}×, cheap β = {st.mean(tc)/st.mean(tb):.2f}×")
