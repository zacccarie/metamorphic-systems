"""β-reg sur SCAN — compositional generalization (Lake & Baroni 2018).

Le pont toy→tangible. SCAN = commandes → séquences d'actions. Split "length"
(train séquences courtes, test longues) = généralisation compositionnelle, profil
grokking (modèles mémorisent train, échouent à généraliser).

Test : loss = CE − λ·β sur activations decoder accélère/améliore-t-il la
généralisation compositionnelle (exact-match test) ?

Claim : β-reg améliore l'accuracy exact-match sur le split compositionnel.
Kill-switch : si baseline et β-reg flooren­t (0%) ou égaux → niche étroite / null.

Robust : volume + detach + fetch stateless.
Entrypoints : smoke (foreground sanity), run_all (detached full), fetch.
"""
from __future__ import annotations
import modal

app = modal.App("scan-beta")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("numpy>=1.26", "requests>=2.31")
)
results_vol = modal.Volume.from_name("scan-beta-results", create_if_missing=True)

SCAN_BASE = "https://raw.githubusercontent.com/brendenlake/SCAN/master"
SPLITS = {
    "length": ("length_split/tasks_train_length.txt", "length_split/tasks_test_length.txt"),
    "simple": ("simple_split/tasks_train_simple.txt", "simple_split/tasks_test_simple.txt"),
}


def _load_split(split):
    import requests, os
    out = {}
    for name, rel in zip(["train", "test"], SPLITS[split]):
        path = f"/tmp/scan_{split}_{name}.txt"
        if not os.path.exists(path):
            txt = requests.get(f"{SCAN_BASE}/{rel}").text
            with open(path, "w") as f:
                f.write(txt)
        else:
            txt = open(path).read()
        pairs = []
        for line in txt.strip().split("\n"):
            # format: IN: <cmd> OUT: <actions>
            if "IN:" in line and "OUT:" in line:
                inp = line.split("IN:")[1].split("OUT:")[0].strip().split()
                outp = line.split("OUT:")[1].strip().split()
                pairs.append((inp, outp))
        out[name] = pairs
    return out


def _build_vocab(pairs_all):
    words = set()
    for inp, outp in pairs_all:
        words.update(inp); words.update(outp)
    # specials : PAD=0, BOS=1, SEP=2, EOS=3
    itos = ["<pad>", "<bos>", "<sep>", "<eos>"] + sorted(words)
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def _train_eval(split, seed, lam, max_len, steps, device_str="cuda"):
    import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json, os, time
    device = torch.device(device_str)
    torch.manual_seed(seed); np.random.seed(seed)

    data = _load_split(split)
    stoi, itos = _build_vocab(data["train"] + data["test"])
    V = len(itos); PAD, BOS, SEP, EOS = 0, 1, 2, 3

    def encode(inp, outp):
        ids = [BOS] + [stoi[w] for w in inp] + [SEP] + [stoi[w] for w in outp] + [EOS]
        return ids

    def make_set(pairs):
        seqs, sep_pos = [], []
        for inp, outp in pairs:
            ids = encode(inp, outp)
            if len(ids) > max_len:
                continue
            seqs.append(ids); sep_pos.append(1 + len(inp) + 1)  # index juste après SEP
        return seqs, sep_pos

    tr_seqs, tr_sep = make_set(data["train"])
    te_seqs, te_sep = make_set(data["test"])

    def batch(seqs, seps, idxs):
        B = len(idxs)
        X = np.full((B, max_len), PAD, dtype=np.int64)
        tgt = np.full((B, max_len), PAD, dtype=np.int64)  # PAD = ignore
        for bi, i in enumerate(idxs):
            s = seqs[i]; L = len(s)
            X[bi, :L] = s
            # next-token targets, mais loss SEULEMENT sur région OUT (après SEP)
            for t in range(L - 1):
                if t >= seps[i] - 1:  # cible = token OUT ou EOS
                    tgt[bi, t] = s[t + 1]
        return torch.tensor(X, device=device), torch.tensor(tgt, device=device)

    d, nh, nl = 128, 4, 4
    class Block(nn.Module):
        def __init__(s):
            super().__init__()
            s.ln1 = nn.LayerNorm(d); s.attn = nn.MultiheadAttention(d, nh, batch_first=True)
            s.ln2 = nn.LayerNorm(d); s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        def forward(s, x, cmask):
            a, _ = s.attn(s.ln1(x), s.ln1(x), s.ln1(x), attn_mask=cmask, need_weights=False)
            x = x + a
            return x + s.mlp(s.ln2(x))
    class GPT(nn.Module):
        def __init__(s):
            super().__init__()
            s.tok = nn.Embedding(V, d, padding_idx=PAD); s.pos = nn.Embedding(max_len, d)
            s.blocks = nn.ModuleList([Block() for _ in range(nl)])
            s.ln = nn.LayerNorm(d); s.head = nn.Linear(d, V)
        def forward(s, idx, want_act=False):
            T = idx.shape[1]
            x = s.tok(idx) + s.pos(torch.arange(T, device=idx.device))
            cmask = torch.triu(torch.full((T, T), float('-inf'), device=idx.device), 1)
            for b in s.blocks: x = b(x, cmask)
            h = s.ln(x)
            return (s.head(h), h) if want_act else s.head(h)

    model = GPT().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    def beta(act_flat):
        # act_flat : (M, d) activations OUT-region, M tokens
        a = act_flat - act_flat.mean(0, keepdim=True)
        s = torch.linalg.svdvals(a); eig = (s**2) + 1e-10
        r = torch.arange(1, eig.shape[0]+1, device=a.device, dtype=a.dtype)
        lr = torch.log(r); le = torch.log(eig); lrc = lr-lr.mean(); lec = le-le.mean()
        return -((lrc*lec).sum() / (lrc*lrc).sum().clamp_min(1e-9))

    @torch.no_grad()
    def exact_match(seqs, seps, n_eval=300):
        model.eval()
        idxs = np.random.choice(len(seqs), min(n_eval, len(seqs)), replace=False)
        correct = 0
        for i in idxs:
            s = seqs[i]; sp = seps[i]
            prefix = s[:sp]  # [BOS] IN [SEP]
            gold = s[sp:]    # OUT ... EOS
            cur = list(prefix)
            for _ in range(max_len - len(prefix)):
                x = torch.tensor([cur], device=device)
                logits = model(x)
                nxt = int(logits[0, -1].argmax())
                cur.append(nxt)
                if nxt == EOS:
                    break
            gen = cur[sp:]
            if gen == gold:
                correct += 1
        model.train()
        return correct / len(idxs)

    bs = 256; t0 = time.time()
    for step in range(steps):
        model.train()
        idxs = np.random.randint(0, len(tr_seqs), bs)
        X, tgt = batch(tr_seqs, tr_sep, idxs)
        logits, h = model(X, want_act=True)
        ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), ignore_index=PAD)
        if lam > 0:
            mask = (tgt.reshape(-1) != PAD)
            act = h.reshape(-1, d)[mask]
            if act.shape[0] > d:
                ce = ce - lam * beta(act)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    train_time = time.time() - t0
    tr_acc = exact_match(tr_seqs, tr_sep)
    te_acc = exact_match(te_seqs, te_sep)
    out = {'split': split, 'seed': seed, 'lam': lam, 'train_acc': tr_acc,
           'test_acc': te_acc, 'train_time': train_time, 'vocab': V,
           'n_train': len(tr_seqs), 'n_test': len(te_seqs), 'max_len': max_len}
    os.makedirs("/results", exist_ok=True)
    with open(f"/results/{split}_lam{lam}_seed{seed}.json", "w") as f:
        json.dump(out, f)
    results_vol.commit()
    print(f"[{split} lam{lam} seed{seed}] train_acc={tr_acc:.3f} test_acc={te_acc:.3f}")
    return out


@app.function(image=image, gpu="A10", timeout=900, volumes={"/results": results_vol})
def smoke():
    """Sanity rapide : length split, 1 seed, peu de steps. Vérifie pipeline + pas de crash."""
    print("SMOKE: simple split (doit avoir test_acc élevé)...")
    r1 = _train_eval("simple", seed=0, lam=0.0, max_len=64, steps=800)
    print("SMOKE: length split baseline (compositionnel, dur)...")
    r2 = _train_eval("length", seed=0, lam=0.0, max_len=64, steps=800)
    print(f"\nSMOKE OK. simple test_acc={r1['test_acc']:.3f} (attendu haut), length test_acc={r2['test_acc']:.3f} (attendu bas)")
    print(f"vocab={r1['vocab']} n_train_simple={r1['n_train']} n_train_length={r2['n_train']}")


@app.function(image=image, gpu="A10", timeout=14400, volumes={"/results": results_vol})
def run_all(steps: int = 3000):
    """Full : length split, baseline vs β-reg, 3 seeds. Détaché."""
    for lam in [0.0, 0.01]:
        for seed in [0, 1, 2]:
            _train_eval("length", seed, lam, max_len=64, steps=steps)
    print("All SCAN runs done.")


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
def smoke_run():
    smoke.remote()


@app.local_entrypoint()
def launch():
    run_all.spawn(3000)
    print("Spawned SCAN run_all (length split, baseline+β-reg, 3 seeds). Fetch: ::fetch")


@app.local_entrypoint()
def fetch():
    import statistics
    res = fetch_all.remote()
    if not res:
        print("No results yet."); return
    # filtrer length split runs (full)
    length = [r for r in res if r['split'] == 'length']
    print(f"\n=== β-reg sur SCAN length split (compositional) — {len(length)} runs ===\n")
    print(f"  {'λ':<8}{'train_acc':<18}{'test_acc (compositional ↑)':<28}{'n':<4}")
    for lam in sorted(set(r['lam'] for r in length)):
        runs = [r for r in length if r['lam'] == lam]
        tr = [r['train_acc'] for r in runs]; te = [r['test_acc'] for r in runs]
        def ms(x): return f"{statistics.mean(x):.3f}±{statistics.stdev(x) if len(x)>1 else 0:.3f}"
        print(f"  {lam:<8}{ms(tr):<18}{ms(te):<28}{len(runs):<4}")
    base = [r['test_acc'] for r in length if r['lam'] == 0.0]
    breg = [r['test_acc'] for r in length if r['lam'] == 0.01]
    if base and breg:
        b0 = statistics.mean(base); b1 = statistics.mean(breg)
        print(f"\n  Compositional test_acc : baseline {b0:.3f} → β-reg {b1:.3f}  (Δ {b1-b0:+.3f})")
        if b1 > b0 + 0.03:
            print("  → ★ β-reg AMÉLIORE la généralisation compositionnelle. Pont toy→tangible.")
        elif abs(b1 - b0) <= 0.03:
            print("  → neutre. Pas d'effet sur ce benchmark.")
        else:
            print("  → β-reg NUIT ici.")
