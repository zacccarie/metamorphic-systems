# Phase 1 : ODE-Transformer hybrid scaling experiment

## Hypothèse

Concept fort source : **β taxonomy architectures** (ODE Euler = β plateau 1.57 sous noise).

Hypothèse Phase 1 : remplacer FFN par ODE Euler block dans transformer améliore :
- Long-context coherence
- Val loss à params matched
- β des activations vers structure pink-brown noise (β=1-2 = mémoire structurée)

Cf finding metamorphic-systems : architectures recurrentes intégratives = source vraie d'auto-organisation spectrale.

## Setup

- Dataset : TinyShakespeare (1M chars, char-level)
- Model : 6 layers, d_model=384, n_head=6 (~25M params)
- 4 variants × 3 seeds × 3000 steps
- GPU : Modal A10

## Variants comparées

| Variant | FFN replacements | ODE positions |
|---|---|---|
| vanilla | 0 (standard transformer) | — |
| ode_hybrid_25 | 1 mid layer | L3 |
| ode_hybrid_50 | 3 layers | L1, L3, L4 |
| ode_full | All 6 layers | L0-L5 |

ODE block = 4 internal Euler steps avec hidden_mult=2 (vs FFN mult=4) pour matched params.

## Métriques

- **Final val_loss** (target : ode_hybrid bat vanilla)
- **β activations per layer** (target : ODE positions montrent β proche 1.57)
- **Train time** (target : pas trop overhead)
- **Stability** (3 seeds, std reasonable)

## Décision Phase 2

Si ode_hybrid_25 ou _50 bat vanilla → continuer Phase 2 (1.3B-3B scale).
Si tous variants ≥ vanilla → continuer (encouraging).
Si vanilla domine clairement → option 1 NOT viable, pivot vers Option 3 (β-adaptive compute).

## Files

- `modal_ode_transformer.py` : Modal cloud experiment
- `results.json` : results after run
- `analysis.md` : findings post-run
