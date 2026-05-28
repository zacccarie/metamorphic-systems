# Falsifications scientifiques

Documentation des claims réfutés ou nuancés par expériences cloud.
Falsifier > confirmer (per PROTOCOLE.md).

## F1 : Architrace "1/f² universel = signature SOC" — INVALIDÉ

### Claim original (architrace S4)
> "Slope spectrale 1/f² (β ≈ -1.85 ± 0.13) sur croissance VG. Universel inter-seed. Signature SOC."

### Tests faits (cloud Modal A10, 100+ runs total)

1. **Signal control** (4 signaux × 6 seeds × MLP depth=3)

| Signal input | β input | β activations init | β activations final |
|---|---|---|---|
| Lorenz | 2.10 | 2.19 | 2.21 |
| **Gaussian** | **0.006** | **0.04** | **0.03** |
| Sine | 2.43 | 2.75 | 2.72 |
| Pink (1/f) | 1.78 | 1.83 | 1.84 |

→ **β activations ≈ β input**. MLP est passive filter.

2. **Depth emergence** (Gaussian × 5 depths × 6 seeds)

| Depth | β last layer |
|---|---|
| 1-8 | 0.02 to 0.08 (jamais > 0.1) |

→ **β stays ~0** quelle que soit profondeur. Pas d'émergence par depth.

3. **Growth events** (Net2Net + Gaussian × 8 seeds)

| Signal | β late | growth events |
|---|---|---|
| Gaussian | **0.034** | 0.2 |
| Lorenz | 2.33 | 0.6 |

→ **β stays ~0 sous growth + Gaussian**. Pas d'émergence par mutations.

### Verdict

**Architrace claim "1/f² universel" est probablement un artefact des inputs naturels** (audio/video/Lorenz = signaux ayant nativement β ≈ 2).

Le système n'auto-organise PAS spectre 1/f². Il transmet le spectre du signal input.

### Implication

- Architrace paper section §4 doit être **drastiquement nuancé**
- Le claim "signature SOC" sur croissance VG → **invalidé**
- Tests futurs : générer inputs avec différents β controlés, mesurer si l'architecture **modifie** ce β
- Vraie auto-organisation = β output significantly different from β input

## F2 : "Naive self-modeling helps task performance" — RÉFUTÉ

### Claim potentiel (active inference inspiration)
> "MLP qui prédit ses propres activations devrait améliorer task performance"

### Test (cloud A10, 6 α × 8 seeds = 48 runs)

α sweep ∈ [0, 0.01, 0.1, 0.5, 1.0, 2.0]

| α | val_acc | self_loss final |
|---|---|---|
| 0.0 | 0.557 | 1.94 |
| 0.01 - 2.0 | 0.557 - 0.559 | 0.044 |

→ val_acc **invariant**. Self-modeling head apprend (loss 1.94 → 0.044) mais **n'aide pas task**.

### Verdict

Naive self-modeling de **propres activations depuis même input** = trop facile (déterministe). Self-prediction utile **uniquement si** :
- Non-trivial (noise injection, état t+1 vs t)
- Free energy minimization style (prédiction incertaine)

## F3 : "ComplexNet algebra essential" — RÉFUTÉ (M4 algebra ablation)

### Claim original (M4 paper section antérieure)
> "Complex multiplication W_r x − W_i x_im encodes phase composition multiplicatively"

### Test (cloud A10, 3 algebras × 12 seeds = 36 runs)

| Algebra mode | val_acc chirps |
|---|---|
| complex_full | 0.6495 |
| **two_branch_add** | **0.6494** |
| two_branch_subtract | 0.4290 |

→ two_branch_add (mêmes 2 matrices, signs différents) ≡ complex_full. Specific complex multiplication NOT load-bearing.

### Verdict

ComplexMLP edge = **paired channels + magnitude activation**. Pas l'algèbre complexe spécifique.

## F4 : "ODE rollout natively good" (M6) — RÉFUTÉ

### Claim potentiel
> "LiquidNet ODE structure should handle autoregressive rollout well"

### Test (10 seeds Lorenz rollout)

| Model | single_mse | rollout_mse |
|---|---|---|
| LSTM | 0.00064 | 0.0250 |
| LiquidNet | 0.00014 | **0.0676** |
| MLP | 0.00004 | **0.0010** |

→ LiquidNet rollout **67× pire** que MLP simple.

### Verdict

ODE continuous dynamics compounds errors → rollout autoregressive harmful. Plain MLP step-discrete stable.

## F5 : "Anti-Bitter Lesson at high noise" — NUANCÉ

### Claim observé (local 2D scaling)
> "À σ=3, Δ Complex-Real croît avec n_per_class (anti-Bitter)"

### Test (cloud A10, n_per_class extended [1500, 2500, 5000, 8000])

| n | Δ |
|---|---|
| 1500 | +0.053 |
| 2500 | +0.042 |
| 5000 | +0.026 |
| 8000 | +0.015 |

→ Trend visible n≤1500 = **catch-up delay**, pas anti-Bitter Lesson persistant.

### Verdict

Bitter Lesson **universel à enough data**. Pas de regime où inductive bias bat scaling.

## F6 : "Complex algebra universal" — RÉFUTÉ (cross-domain)

| Domain | Complex verdict |
|---|---|
| Audio chirps | WIN (Δ+0.235) |
| ESC50 animals | WIN faible (Δ+0.085) |
| UrbanSound8K | INDIST (Δ+0.016) |
| CIFAR-10 vision | INDIST mais param-eff |
| Lorenz dynamics | LOSE 4× |
| Spectrogram STFT | LOSE 2× (Real Re+Im concat wins) |
| paired_mag Lorenz | LOSE worst (0.027 vs Real 0.002) |

→ Niche **strict** : magnitude pooling + phase-coded temporal signals + low-data.

## F7 : "MSE aux loss can replace InfoNCE on LLM-on-world-task" (Q5) — RÉFUTÉ

### Test (cloud H100 Pythia-410m LoRA, α sweep + 4 seeds)

| Condition | val_mse | Δ vs baseline |
|---|---|---|
| baseline | 0.165 | 0 |
| InfoNCE α=1 | 0.133 | -0.033 (mieux) |
| MSE α ∈ [0.1, 0.5, 1, 2] | 0.21-0.34 | tous pires |

→ MSE wrong type aux loss. InfoNCE contrastive nécessaire pour ce regime.

## Synthèse falsifications

| # | Claim | Verdict | Test count |
|---|---|---|---|
| F1 | architrace 1/f² SOC universel | INVALIDÉ | 5 tests convergents cloud |
| F2 | naive self-modeling helps | RÉFUTÉ | 48 runs cloud |
| F3 | complex multiplication essential | RÉFUTÉ | 36 runs cloud |
| F4 | ODE good rollout | RÉFUTÉ | 30 runs local |
| F5 | anti-Bitter at high noise | NUANCÉ | 32 runs cloud |
| F6 | Complex algebra universal | RÉFUTÉ | 100+ runs cross-domain |
| F7 | MSE = InfoNCE aux loss | RÉFUTÉ | 29 runs cloud H100 |

**Patterns émergeant** :
1. Beaucoup de "signatures" architecturales = artefacts d'inputs
2. Bitter Lesson universel : pas de regime où inductive bias bat scaling indéfiniment
3. Active inference naive ne suffit pas
4. Niches d'architectures spécifiques = strict + interprétation rétrécie

C'est de la **vraie science honnête**. La rigueur sauve le programme de claims excessifs.
