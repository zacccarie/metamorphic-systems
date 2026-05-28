# PoC β-anomaly detection — findings

Date : 2026-05-28
Setup : Modal A10 cloud, 4-6 seeds × configs
Repo : `models/modal_beta_anomaly_poc.py`, `modal_beta_anomaly_unsupervised.py`, `modal_beta_anomaly_noise.py`

## Question initiale

Comment trouver application concrète au finding "Taxonomy β architectures" (MLP=0, RNN=0.21, Attention=0.5, GRU=1.08, LSTM=1.17, ODE=1.57) ?

## Hypothèse PoC

LSTM trained sur signal β=β_normal → activations LSTM acquièrent β cible. Sous régime anomaly β différent, β activations shift → anomaly detector.

## Setup expérimental

Signal synthétique 1/f^β colored noise :
- Normal regime : β=0.5 (signal quasi-blanc faible mémoire)
- Anomaly regime : β ∈ {0.8, 1.0, 1.5, 2.0} (signal smoother, forte mémoire)
- Train LSTM next-step prediction sur 8000 samples normal
- Test : signal mixte 4000 samples (20 chunks aléatoires normal/anomaly)
- Sliding window AUROC : β-detector vs MSE-detector

## Findings

### 1. β-detector match MSE-detector sur synthétique

| Difficulté | β_anom | β LSTM trained | MSE LSTM | Verdict |
|---|---|---|---|---|
| easy | 2.0 | 0.987 | 0.992 | TIE |
| medium | 1.5 | 0.980 | 0.985 | TIE |
| hard | 1.0 | 0.895 | 0.908 | TIE |
| very_hard | 0.8 | 0.788 | 0.798 | TIE |

**Verdict** : β-detector AUROC ≈ MSE-detector AUROC à 1-2% près. Pas d'avantage clair via β des activations.

### 2. Random-init LSTM β-detector PERD vs trained

| Difficulté | β rand-LSTM | β train-LSTM |
|---|---|---|
| easy | 0.603 | 0.987 |
| medium | 0.686 | 0.980 |
| hard | 0.707 | 0.895 |
| very_hard | 0.660 | 0.788 |

Random-init LSTM = tanh kill memory, β activations saturé. Training nécessaire pour β-detector.

### 3. Raw signal β-meter MATCH trained LSTM (zero-shot)

| Difficulté | β raw signal | β train-LSTM |
|---|---|---|
| easy | 0.986 | 0.987 |
| medium | 0.979 | 0.980 |
| hard | 0.887 | 0.895 |
| very_hard | 0.776 | 0.788 |

**Killer finding** : β du signal raw (FFT + log-slope) = aussi bon que β LSTM trained. Pas de NN nécessaire.

### 4. Noise robustness — LSTM ne denoise pas

| σ noise | β raw | β LSTM trained | MSE LSTM |
|---|---|---|---|
| 0.0 | 0.975 | 0.975 | 0.975 |
| 0.3 | 0.934 | 0.929 | 0.960 |
| 0.6 | 0.835 | 0.816 | 0.904 |
| 1.0 | 0.723 | 0.688 | 0.751 |
| 1.5 | 0.639 | 0.604 | 0.613 |
| 2.0 | 0.590 | 0.574 | 0.592 |

LSTM β-detector ne bat jamais raw β sous bruit. MSE-detector slight edge moderate noise. Tous échouent σ≥1.5.

## Conclusion honnête

**Hypothèse PoC initiale (β LSTM activations > raw signal) : RÉFUTÉE.**

LSTM trained propage β du signal aux activations linéairement. Pas de denoising, pas d'extraction supplémentaire. β des activations ≈ β du signal pour LSTM trained sur ce task synthétique.

## 4 voies actionnables β taxonomy (révisées)

### A. β-meter raw signal — SIMPLE & ACTIONNABLE ✓ (prouvé)
- Library Python : `beta_meter(signal) → float`
- AUROC 0.78-0.99 zero-shot, no training, no labels
- Application : monitoring time-series, OOD detection unsupervised
- USP : simplicité extreme + science derrière (1/f^β SOC literature Bak 1987)
- **Path** : open-source npm/pip package "spectral-anomaly", demo viz Three.js

### B. β-aware architecture selector — PLAUSIBLE (non testé)
- Tool : input signal → measure β → recommend arch
  - β ≈ 0 (white noise) → MLP suffit
  - β ≈ 0.5-1 (weak memory) → Attention/GRU
  - β > 1 (strong memory) → LSTM/ODE
- **Path** : Jupyter notebook + benchmark suite + paper "β as arch selection prior"

### C. Deep model drift monitoring — THEORETICAL (à valider)
- Hypothèse : monitor β des activations layer-par-layer sur model deployed
- Si β shift par rapport baseline → data drift OU model degradation
- USP : detector internal, pas need ground truth
- **Path** : test sur model deployed (e.g. Pythia-410m sur stream texte) + measure β layer activations

### D. ODE Euler generator β=1.57 — DATA AUG ✓ (testable)
- ODE Euler trained generates samples avec β contrôlé
- dt, n_steps controls β output ∈ [0, 1.57]
- Application : data augmentation pour rare events, balanced training
- **Path** : `ode_synthesizer.py` + benchmark vs uniform noise augmentation

## Recommandation immédiate

**Path A + Path D** = combo most defensible :
- A : β-meter zero-shot detector → open source library "spectral-anomaly-detector"
- D : ODE synthesizer β-controlled → data augmentation tool

Combiné : "β-toolkit" for time-series practitioners.

## Limitations

- Tests synthétiques 1/f^β colored noise uniquement
- Real-world signals (ECG, financial, IoT) non testés
- Aucune comparison vs deep anomaly detectors SOTA (Anomaly Transformer, TFD)
- β robustness face à mode mixed signals non characterized

## Prochains tests recommandés

1. UCR Anomaly Detection benchmark (real-world)
2. β-detector vs Anomaly Transformer (SOTA fair comparison)
3. β-meter sur ECG MIT-BIH arrhythmia (medical applicable)
4. β-meter sur cryptocurrency volatility regime change
