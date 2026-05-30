"""Criticalité = opérateur de partition ? Test discriminant.

Claim pré-enregistré : dans un reservoir (echo state network) piloté par une famille
de contextes (signal PARTAGÉ + signal SPÉCIFIQUE-contexte), la séparation des unités en
codeurs-invariants (répondent au partagé) vs codeurs-variants (répondent au contexte)
est MAXIMALE à la criticalité (gain spectral g ≈ 1).

Prédiction AVANT run : bimodalité de la sélectivité pique près de g=1.
- g < 1 (ordre) : dynamique s'éteint, codage faible, pas de séparation
- g ≈ 1 (criticalité) : séparation propre invariant/variant ★
- g > 1 (chaos) : tout se mélange, attribution impossible

Métrique : coefficient de bimodalité (Sarle) de la distribution de sélectivité.
  b = (skew² + 1) / kurtosis  ;  b > 0.555 = bimodal.
Kill-switch : si b ne pique pas près de g=1 → thèse morte.

Reservoir computing, numpy pur, local, secondes.
"""
from __future__ import annotations
import numpy as np


def bimodality_coefficient(x: np.ndarray) -> float:
    """Coefficient de bimodalité de Sarle. >0.555 = bimodal, ~0.33 = unimodal normal."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 4:
        return 0.0
    m = x.mean()
    s = x.std() + 1e-12
    skew = ((x - m) ** 3).mean() / s ** 3
    kurt = ((x - m) ** 4).mean() / s ** 4  # kurtosis non-excess
    # correction d'échantillon
    denom = kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if abs(denom) < 1e-9:
        return 0.0
    return float((skew ** 2 + 1) / denom)


def run_one(g: float, seed: int, N: int = 400, K: int = 6, T: int = 1500,
            n_in: int = 6, washout: int = 200) -> dict:
    """Un reservoir au gain g. Mesure séparation invariant/variant + criticalité empirique."""
    rng = np.random.default_rng(seed)

    # Reservoir : W recurrent random, rayon spectral réglé à g
    W = rng.normal(0, 1, (N, N))
    eig = np.linalg.eigvals(W)
    W = W / (np.max(np.abs(eig)) + 1e-12) * g   # rayon spectral = g
    W_in_shared = rng.normal(0, 1, (N, n_in)) * 0.5
    W_in_ctx = rng.normal(0, 1, (N, n_in)) * 0.5

    # Signal PARTAGÉ : même séquence pour tous les contextes (invariant du cadre)
    shared = rng.normal(0, 1, (T, n_in))
    # Signaux SPÉCIFIQUES-contexte : un par contexte (variant)
    ctx_signals = [rng.normal(0, 1, (T, n_in)) for _ in range(K)]

    def drive(inp_seq):
        h = np.zeros(N)
        H = np.zeros((T, N))
        for t in range(T):
            h = np.tanh(W @ h + inp_seq[t])
            H[t] = h
        return H[washout:]

    # Pilotage : pour chaque contexte, input = partagé + spécifique
    # On veut décomposer la variance de chaque unité : part due au partagé vs au contexte
    # Drive partagé seul (contexte off)
    H_shared = drive(shared @ W_in_shared.T)
    var_shared = H_shared.var(axis=0)  # variance par unité sous drive partagé

    # Drive contexte seul (partagé off), moyenné sur contextes
    var_ctx_per = []
    for k in range(K):
        H_k = drive(ctx_signals[k] @ W_in_ctx.T)
        var_ctx_per.append(H_k.var(axis=0))
    var_ctx = np.mean(var_ctx_per, axis=0)

    # Sélectivité par unité : +1 = pur invariant (partagé), -1 = pur variant (contexte)
    selectivity = (var_shared - var_ctx) / (var_shared + var_ctx + 1e-9)

    # Métrique principale : bimodalité (= partition propre en 2 populations)
    bimod = bimodality_coefficient(selectivity)

    # Criticalité empirique : sensibilité aux perturbations (proxy Lyapunov)
    # Deux trajectoires partagées proches → divergence
    h1 = np.zeros(N); h2 = np.zeros(N); h2[0] += 1e-3
    div = []
    inp = shared @ W_in_shared.T
    for t in range(min(T, 600)):
        h1 = np.tanh(W @ h1 + inp[t])
        h2 = np.tanh(W @ h2 + inp[t])
        div.append(np.linalg.norm(h1 - h2))
    div = np.array(div) + 1e-12
    # taux de divergence moyen (log) sur la fenêtre post-washout
    lyap_proxy = float(np.mean(np.diff(np.log(div[washout:washout + 200]))))

    return {
        'g': g, 'seed': seed,
        'bimodality': bimod,
        'lyap_proxy': lyap_proxy,
        'mean_selectivity': float(selectivity.mean()),
        'frac_invariant': float((selectivity > 0.3).mean()),
        'frac_variant': float((selectivity < -0.3).mean()),
    }


def main():
    gains = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
    seeds = list(range(5))
    print("Criticalité = opérateur de partition ? Sweep gain spectral.\n")
    print(f"  {'g':<6}{'bimodality':<18}{'lyap_proxy':<16}{'%invar':<10}{'%var':<10}")
    results = {}
    for g in gains:
        runs = [run_one(g, s) for s in seeds]
        bm = np.mean([r['bimodality'] for r in runs])
        bs = np.std([r['bimodality'] for r in runs])
        ly = np.mean([r['lyap_proxy'] for r in runs])
        fi = np.mean([r['frac_invariant'] for r in runs])
        fv = np.mean([r['frac_variant'] for r in runs])
        results[g] = (bm, bs, ly, fi, fv)
        crit = " ← λ≈0 (critique)" if abs(ly) < 0.02 else ""
        bimod_flag = " ★BIMODAL" if bm > 0.555 else ""
        print(f"  {g:<6}{bm:.3f}±{bs:.3f}      {ly:+.4f}        {fi:<10.2f}{fv:<10.2f}{crit}{bimod_flag}")

    # Verdict : bimodalité pique-t-elle près de la criticalité ?
    best_g = max(results, key=lambda g: results[g][0])
    print(f"\n  Bimodalité MAX à g={best_g} (b={results[best_g][0]:.3f})")
    # Criticalité empirique = g où lyap_proxy traverse 0
    crit_g = min(results, key=lambda g: abs(results[g][2]))
    print(f"  Criticalité empirique (λ≈0) à g={crit_g}")
    gap = abs(best_g - crit_g)
    print(f"\n  Écart pic-bimodalité ↔ criticalité : {gap:.2f}")
    if gap <= 0.2:
        print("  → ★ CLAIM CONFIRMÉ : séparation pique À la criticalité. Partition = opérateur critique.")
    elif gap <= 0.4:
        print("  → PARTIEL : séparation près de la criticalité, pas exactement.")
    else:
        print("  → CLAIM RÉFUTÉ : séparation ne pique pas à la criticalité. Thèse morte.")


if __name__ == "__main__":
    main()
