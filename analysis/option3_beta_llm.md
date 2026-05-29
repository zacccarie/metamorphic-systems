# Option 3 — β-meter adaptive compute LLM : findings

Date : 2026-05-29
Setup : Modal A10 (TinyLlama) + A100-80GB (Qwen 7B)
Scripts : `option3_llm/modal_beta_llama.py`, `modal_beta_qwen7b.py`

## Hypothèse

β des activations residual stream LLM = signal difficulty token → trigger adaptive compute.
Si β stable = confident → fast path. Si β shift = struggle → more thinking.

## Tests

### TinyLlama 1.1B (10 easy + 10 hard)

| Régime | β_norm | std |
|---|---|---|
| EASY | 0.264 | 0.251 |
| HARD | 0.061 | 0.125 |
| Δ | -0.203 | — |
| Separability | **0.54** | weak |

Direction : HARD → β plus BAS.

### Qwen2.5 7B (30 easy + 30 medium + 30 hard, 29 layers)

| Layer | EASY | MEDIUM | HARD | Δ(H-E) | sep |
|---|---|---|---|---|---|
| L16 | 0.36 | 0.48 | 0.55 | +0.19 | 0.29 |
| L20 | 0.40 | 0.59 | 0.56 | +0.16 | 0.25 |
| best L19 | — | — | — | — | 0.32 |
| norm L27 | — | — | — | — | **0.63** |

Direction : HARD → β plus HAUT (inverse TinyLlama).

## Findings critiques

### 1. Scale n'aide pas
TinyLlama sep=0.54 → Qwen 7B sep=0.63. Gain marginal. Pas le saut espéré (target ≥1.5).

### 2. Direction INCONSISTANTE cross-model
- TinyLlama : HARD → β bas
- Qwen 7B : HARD → β haut

Signal pas robuste, dépend architecture/training. Pas universal.

### 3. Confirme β taxonomy (cohérence théorique)
β residual stream culmine ~0.5 mid-layers (L16-24). Matche finding **Attention = β 0.5** de 700 runs cloud A10. Théorie cohérente interne.

### 4. Qwen 7B qualité génération
Easy + hard tous deux cohérents. (TinyLlama sortait vide sur easy → artefact chat template).

## Verdict

**β-meter adaptive compute LLM : PoC FAIBLE, pas production-ready.**

- Separability 0.18-0.63 << 1.5 requis pour gating fiable
- Direction model-dependent
- 7B scale ne résout pas problème

**Falsification honnête** : β activations LLM ≠ signal difficulty propre. Hypothèse trop optimiste.

## Ce qui reste valide

- β taxonomy confirmée : attention layers β~0.5 mid-depth (cohérent finding cloud)
- β computable cheap sur LLM activations (FFT, µs)
- Pas un detector difficulty mais MESURE structure spectrale réelle

## Alternatives non testées (si continuer)

1. **β des attention weights** (pas residual stream) : peut-être plus discriminant
2. **β multi-scale temporel** : combiner short + long window
3. **β + entropy logits** : ensemble avec uncertainty classique
4. **Supervised probe sur β features** : si labels difficulty disponibles
5. **β des gradients** (training-time) vs activations (inference)

## Comparaison vs SOTA adaptive compute

| Méthode | Signal | Robuste | Production |
|---|---|---|---|
| Speculative decoding (Medusa, Eagle) | draft model accept | ✓ | ✓ déployé |
| Early exit (CALM) | confidence threshold | ✓ | ✓ |
| Entropy-based | logit entropy | ✓ | ✓ |
| **β-meter (ours)** | spectral slope activations | ✗ weak | ✗ |

β-meter PAS compétitif vs méthodes établies pour adaptive compute LLM.

## Décision

Option 3 abandonné comme adaptive-compute tool. β reste outil de **caractérisation** (taxonomy confirmée), pas de **décision** runtime.

Pivot recommandé : Option 1 (ODE-Transformer hybrid) reste meilleur pari scaling.
