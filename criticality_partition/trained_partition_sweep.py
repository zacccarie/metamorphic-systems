"""P2 corrigé : criticalité = opérateur de partition ? (système ENTRAÎNÉ)

Fix du design bâclé précédent : on entraîne K réseaux RNN, chacun avec rayon
spectral récurrent ÉPINGLÉ à un gain g (le vrai bouton edge-of-chaos). La partition
invariant/variant émerge alors de l'APPRENTISSAGE (pas d'un reservoir random).

Tâche multi-contexte :
- shared : y dépend d'une intégration temporelle d'une dim (MÉMOIRE, invariant tous contextes)
- variant : readout linéaire spécifique-contexte du dernier pas

Claim pré-enregistré : généralisation held-out ET propreté de la partition piquent
au gain critique (g ≈ 1, bord du chaos).

Mesures par g :
- generalization : loss sur contextes held-out (on ne refit QUE le readout-contexte,
  récurrent + shared gelés → teste si la computation invariante transfère)
- bimodality : sélectivité des unités (var expliquée shared vs contexte), bimodal = partition propre

Kill-switch : si ni généralisation ni bimodalité ne piquent près de g=1 → thèse morte.

Vanilla RNN, numpy+torch, local.
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
    """K contextes. y_k = tanh(2*mean_t(x[:,0])) [shared] + 0.5*(u_k . x[-1]) [variant]."""
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1, (K, d))  # readout variant par contexte
    data = {}
    for k in range(K):
        x = rng.normal(0, 1, (n_seq, T, d)).astype('float32')
        shared = np.tanh(2 * x[:, :, 0].mean(axis=1))
        variant = 0.5 * (x[:, -1, :] @ U[k])
        y = (shared + variant).astype('float32')
        data[k] = (x, y)
    return data, U


class PinnedRNN(nn.Module):
    """Vanilla RNN, rayon spectral récurrent ré-épinglé à g après chaque step."""
    def __init__(self, d, N, K, g):
        super().__init__()
        self.N = N; self.g = g
        self.W_in = nn.Linear(d, N)
        self.W_rec = nn.Linear(N, N, bias=False)
        self.ctx_emb = nn.Embedding(K, N)        # contexte injecté à l'état initial
        self.w_shared = nn.Linear(N, 1)           # readout partagé (invariant)
        self.w_ctx = nn.Embedding(K, N)           # readout variant par contexte
        with torch.no_grad():
            self._pin()

    def _pin(self):
        W = self.W_rec.weight.data.cpu().numpy()
        r = np.max(np.abs(np.linalg.eigvals(W)))
        if r > 1e-6:
            self.W_rec.weight.data *= (self.g / r)

    def forward(self, x, ctx, return_h=False):
        B, T, d = x.shape
        h = torch.tanh(self.ctx_emb(ctx))
        for t in range(T):
            h = torch.tanh(self.W_rec(h) + self.W_in(x[:, t]))
        y_shared = self.w_shared(h).squeeze(-1)
        y_ctx = (h * self.w_ctx(ctx)).sum(-1)
        y = y_shared + y_ctx
        return (y, h) if return_h else y


def train_eval(g, seed, K=8, K_test=2, N=96, epochs=60):
    torch.manual_seed(seed); np.random.seed(seed)
    data, U = make_task(K, n_seq=400, seed=seed)
    train_ctx = list(range(K - K_test)); test_ctx = list(range(K - K_test, K))

    model = PinnedRNN(d=4, N=N, K=K, g=g)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    def batch(ctxs, ns=128):
        xs, ys, cs = [], [], []
        for k in ctxs:
            x, y = data[k]
            idx = np.random.randint(0, x.shape[0], ns)
            xs.append(torch.tensor(x[idx])); ys.append(torch.tensor(y[idx]))
            cs.append(torch.full((ns,), k, dtype=torch.long))
        return torch.cat(xs), torch.cat(ys), torch.cat(cs)

    for ep in range(epochs):
        model.train()
        x, y, c = batch(train_ctx)
        opt.zero_grad()
        loss = F.mse_loss(model(x, y if False else c), y)  # guard: pass ctx
        loss.backward(); opt.step()
        with torch.no_grad():
            model._pin()  # ré-épingle rayon spectral à g

    # Généralisation held-out : gèle tout sauf readout-contexte, refit few-shot
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.w_ctx.weight.requires_grad_(True)
    opt2 = torch.optim.Adam([model.w_ctx.weight], lr=3e-3)
    for ep in range(40):
        x, y, c = batch(test_ctx)
        opt2.zero_grad()
        loss = F.mse_loss(model(x, c), y)
        loss.backward(); opt2.step()
    with torch.no_grad():
        x, y, c = batch(test_ctx, ns=400)
        gen_loss = F.mse_loss(model(x, c), y).item()

    # Partition : sélectivité des unités (var shared vs var contexte)
    with torch.no_grad():
        # var due au shared : fixer contexte, varier seq
        x, y, c = batch([train_ctx[0]] * 1, ns=400)
        _, h_shared = model(x, c, return_h=True)
        var_shared = h_shared.var(0).cpu().numpy()
        # var due au contexte : même seq stats, varier contexte
        x2, _, _ = batch([train_ctx[0]], ns=400)
        hs = []
        for k in train_ctx:
            ck = torch.full((x2.shape[0],), k, dtype=torch.long)
            _, hk = model(x2, ck, return_h=True)
            hs.append(hk.cpu().numpy())
        hs = np.stack(hs)  # (Kt, B, N)
        var_ctx = hs.var(0).mean(0)  # variance across contextes, moyennée
    sel = (var_shared - var_ctx) / (var_shared + var_ctx + 1e-9)
    bim = bimodality(sel)

    # Criticalité empirique : rayon spectral effectif (épinglé = g, sanity)
    Wr = model.W_rec.weight.data.cpu().numpy()
    eff_radius = float(np.max(np.abs(np.linalg.eigvals(Wr))))

    return {'g': g, 'seed': seed, 'gen_loss': gen_loss, 'bimodality': bim,
            'eff_radius': eff_radius}


def main():
    gains = [0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.6]
    seeds = [0, 1, 2]
    print("P2 corrigé : criticalité = opérateur de partition ? (réseaux entraînés)\n")
    print(f"  {'g':<7}{'gen_loss (↓)':<20}{'bimodality (↑)':<20}{'eff_radius':<12}")
    res = {}
    for g in gains:
        runs = [train_eval(g, s) for s in seeds]
        gl = np.mean([r['gen_loss'] for r in runs]); gls = np.std([r['gen_loss'] for r in runs])
        bm = np.mean([r['bimodality'] for r in runs]); bms = np.std([r['bimodality'] for r in runs])
        er = np.mean([r['eff_radius'] for r in runs])
        res[g] = (gl, gls, bm, bms, er)
        bf = " ★BIMODAL" if bm > 0.555 else ""
        print(f"  {g:<7}{gl:.4f}±{gls:.4f}    {bm:.3f}±{bms:.3f}      {er:.3f}{bf}")

    best_gen = min(res, key=lambda g: res[g][0])
    best_bim = max(res, key=lambda g: res[g][2])
    print(f"\n  Meilleure généralisation (gen_loss min) : g={best_gen}")
    print(f"  Meilleure partition (bimodality max)     : g={best_bim}")
    print(f"\n  Prédiction : les deux près de g≈1 (criticalité).")
    gen_ok = abs(best_gen - 1.0) <= 0.15
    bim_ok = abs(best_bim - 1.0) <= 0.15
    if gen_ok and bim_ok:
        print("  → ★ CLAIM CONFIRMÉ : généralisation ET partition piquent à la criticalité.")
    elif gen_ok or bim_ok:
        print(f"  → PARTIEL : {'généralisation' if gen_ok else 'partition'} pique au critique, pas l'autre.")
    else:
        print("  → CLAIM RÉFUTÉ : ni l'un ni l'autre ne pique à g≈1.")


if __name__ == "__main__":
    main()
