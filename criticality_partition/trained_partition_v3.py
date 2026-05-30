"""P2 v3 : criticalité = opérateur de partition ? Métrique généralisation REDESIGNÉE.

Fix du v2 : la généralisation few-shot (refit w_ctx, tout gelé) échouait partout
(loss 7-16, bruit). v3 remplace par LINEAR PROBE TRANSFER, standard et propre :

- Entraîner réseau à rayon spectral épinglé g sur contextes train
- Geler. Extraire états cachés finaux sur contextes HELD-OUT
- Ridge regression closed-form : hidden → target, sur split held-out
- Mesurer R² test sur held-out → qualité de la représentation transférée

Si la représentation (partition apprise) à la criticalité supporte mieux les contextes
nouveaux → probe R² pique à g≈1.

Garde bimodality (v2 : pic g=1.05, marchait).

Claim : probe R² ET bimodality piquent à criticalité.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def bimodality(x):
    x = np.asarray(x, dtype=np.float64); n = len(x)
    if n < 4: return 0.0
    m, s = x.mean(), x.std() + 1e-12
    skew = ((x - m) ** 3).mean() / s ** 3
    kurt = ((x - m) ** 4).mean() / s ** 4
    denom = kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((skew ** 2 + 1) / max(denom, 1e-9))


def make_task(K, n_seq, T=20, d=4, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1, (K, d))
    data = {}
    for k in range(K):
        x = rng.normal(0, 1, (n_seq, T, d)).astype('float32')
        shared = np.tanh(2 * x[:, :, 0].mean(axis=1))     # invariant (mémoire)
        variant = 0.5 * (x[:, -1, :] @ U[k])               # variant (contexte)
        y = (shared + variant).astype('float32')
        data[k] = (x, y)
    return data, U


class PinnedRNN(nn.Module):
    def __init__(self, d, N, K, g):
        super().__init__()
        self.N = N; self.g = g
        self.W_in = nn.Linear(d, N)
        self.W_rec = nn.Linear(N, N, bias=False)
        self.ctx_emb = nn.Embedding(K, N)
        self.w_shared = nn.Linear(N, 1)
        self.w_ctx = nn.Embedding(K, N)
        with torch.no_grad(): self._pin()
    def _pin(self):
        W = self.W_rec.weight.data.cpu().numpy()
        r = np.max(np.abs(np.linalg.eigvals(W)))
        if r > 1e-6: self.W_rec.weight.data *= (self.g / r)
    def forward(self, x, ctx, return_h=False):
        B, T, d = x.shape
        h = torch.tanh(self.ctx_emb(ctx))
        for t in range(T):
            h = torch.tanh(self.W_rec(h) + self.W_in(x[:, t]))
        y = self.w_shared(h).squeeze(-1) + (h * self.w_ctx(ctx)).sum(-1)
        return (y, h) if return_h else y


def ridge_probe(H_tr, y_tr, H_te, y_te, alpha=1.0):
    """Ridge closed-form. Retourne R² test."""
    H_tr = np.asarray(H_tr); H_te = np.asarray(H_te)
    n, d = H_tr.shape
    A = H_tr.T @ H_tr + alpha * np.eye(d)
    w = np.linalg.solve(A, H_tr.T @ y_tr)
    pred = H_te @ w
    ss_res = ((y_te - pred) ** 2).sum()
    ss_tot = ((y_te - y_te.mean()) ** 2).sum() + 1e-9
    return 1.0 - ss_res / ss_tot


def train_eval(g, seed, K=8, K_test=2, N=96, epochs=60):
    torch.manual_seed(seed); np.random.seed(seed)
    data, U = make_task(K, n_seq=500, seed=seed)
    train_ctx = list(range(K - K_test)); test_ctx = list(range(K - K_test, K))

    model = PinnedRNN(d=4, N=N, K=K, g=g)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    def batch(ctxs, ns=128):
        xs, ys, cs = [], [], []
        for k in ctxs:
            x, y = data[k]; idx = np.random.randint(0, x.shape[0], ns)
            xs.append(torch.tensor(x[idx])); ys.append(torch.tensor(y[idx]))
            cs.append(torch.full((ns,), k, dtype=torch.long))
        return torch.cat(xs), torch.cat(ys), torch.cat(cs)

    for ep in range(epochs):
        model.train()
        x, y, c = batch(train_ctx)
        opt.zero_grad(); loss = F.mse_loss(model(x, c), y); loss.backward(); opt.step()
        with torch.no_grad(): model._pin()

    model.eval()
    # GÉNÉRALISATION v3 : linear probe transfer sur held-out
    with torch.no_grad():
        # held-out : extraire hidden + target, split train/test du probe
        Hs, Ys = [], []
        for k in test_ctx:
            x, y = data[k]
            xt = torch.tensor(x); ct = torch.full((x.shape[0],), k, dtype=torch.long)
            _, h = model(xt, ct, return_h=True)
            Hs.append(h.cpu().numpy()); Ys.append(y)
        H = np.concatenate(Hs); Yv = np.concatenate(Ys)
        n = len(H); idx = np.random.permutation(n); sp = int(0.7 * n)
        probe_r2 = ridge_probe(H[idx[:sp]], Yv[idx[:sp]], H[idx[sp:]], Yv[idx[sp:]])

        # PARTITION : sélectivité (var shared vs var contexte)
        x0, _, _ = batch([train_ctx[0]], ns=400)
        _, h_sh = model(x0, torch.full((x0.shape[0],), train_ctx[0], dtype=torch.long), return_h=True)
        var_shared = h_sh.var(0).cpu().numpy()
        hs = []
        for k in train_ctx:
            ck = torch.full((x0.shape[0],), k, dtype=torch.long)
            _, hk = model(x0, ck, return_h=True); hs.append(hk.cpu().numpy())
        var_ctx = np.stack(hs).var(0).mean(0)
        sel = (var_shared - var_ctx) / (var_shared + var_ctx + 1e-9)
        bim = bimodality(sel)

    return {'g': g, 'seed': seed, 'probe_r2': float(probe_r2), 'bimodality': bim}


def main():
    gains = [0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.6]
    seeds = [0, 1, 2, 3, 4]
    print("P2 v3 : criticalité=partition ? (linear probe transfer + bimodality)\n")
    print(f"  {'g':<7}{'probe_R² (↑)':<20}{'bimodality (↑)':<20}")
    res = {}
    for g in gains:
        runs = [train_eval(g, s) for s in seeds]
        pr = np.mean([r['probe_r2'] for r in runs]); prs = np.std([r['probe_r2'] for r in runs])
        bm = np.mean([r['bimodality'] for r in runs]); bms = np.std([r['bimodality'] for r in runs])
        res[g] = (pr, prs, bm, bms)
        bf = " ★BIMODAL" if bm > 0.555 else ""
        print(f"  {g:<7}{pr:.3f}±{prs:.3f}      {bm:.3f}±{bms:.3f}{bf}")

    best_probe = max(res, key=lambda g: res[g][0])
    best_bim = max(res, key=lambda g: res[g][2])
    print(f"\n  Meilleur probe R² : g={best_probe}")
    print(f"  Meilleure partition (bimodality) : g={best_bim}")
    print(f"\n  Prédiction : les deux près de g≈1 (criticalité).")
    p_ok = abs(best_probe - 1.0) <= 0.15
    b_ok = abs(best_bim - 1.0) <= 0.15
    if p_ok and b_ok:
        print("  → ★ CLAIM CONFIRMÉ : probe ET partition piquent à la criticalité.")
    elif p_ok or b_ok:
        print(f"  → PARTIEL : {'probe' if p_ok else 'partition'} pique au critique.")
    else:
        print("  → CLAIM RÉFUTÉ : ni l'un ni l'autre.")


if __name__ == "__main__":
    main()
