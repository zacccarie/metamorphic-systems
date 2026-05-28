# Metamorphic Systems

## Vision

Programme de recherche sur **architectures neuronales auto-modifiantes** traitées comme **systèmes dynamiques observables**.

Le réseau n'est plus un pipeline d'optimisation statique. Il devient :
- un **organisme** qui modifie sa propre structure
- un **observateur** qui mesure son propre état
- un **prédicteur** qui anticipe sa propre évolution
- un **système physique** caractérisé par des signatures spectrales (1/f^β), Lyapunov, attracteurs

## Question centrale

> Une architecture neuronale qui s'observe et se modifie peut-elle développer des propriétés émergentes utiles (performance, robustesse, proto-conscience computationnelle) que les architectures fixes ne peuvent atteindre ?

## Boucle conceptuelle

```
x_{t+1} = F(x_t, φ(x_t))

où :
- x_t   = état du système (poids + topologie + activations)
- F     = règle de transition (forward + mutation policy)
- φ     = auto-observation (mesures internes : spectral slope, Lyapunov, φ_id)
```

L'architecture observe `x_t` via `φ`, et utilise cette observation pour décider sa propre transition `x_{t+1}`.

## Concepts pivots (voir GLOSSARY.md)

- **Systèmes dynamiques** : `x_{t+1} = F(x_t)`. NN = système dynamique sur l'espace des poids.
- **Spectral slope 1/f²** : signature universelle de systèmes auto-organisés critiques (Bak 1987).
- **Exposant de Lyapunov λ** : mesure de la divergence de trajectoires proches. λ>0 chaos, λ<0 stable.
- **Self-organized criticality (SOC)** : système qui converge à la frontière chaos/ordre.
- **Active inference** (Friston) : agents qui minimisent free energy en prédisant.
- **Strange loop** (Hofstadter) : système qui se référence à travers les niveaux d'abstraction.
- **φ_id / IIT** (Tononi) : information intégrée, candidate-mesure de conscience.

## Inventaire des projets existants

Voir `inventory/` pour fiches détaillées :

| Projet | Rôle | Statut |
|---|---|---|
| `adaptive-nn` | Précurseur. VerifyGrowNet (verify-then-keep growth) | Superseded par architrace |
| `dynamoscope` | Lunette latente. Lyapunov + SINDy + DMD + φ_id | Paper 401 KiB arxiv-ready |
| `architrace` | Fusion adaptive-nn × dynamoscope. 27 sprints, paper draft | 11 findings solides + 6 Q ouvertes |
| `metamorph` | Programme M1-M7 systématique | M4 caractérisé 16 axes nuit 2026-05-28 |
| `latent-flow` | À auditer | Inventaire à faire |
| `ark` | Node-based architecture designer Python (successeur ai-arch-builder) | Tool, pas modèle |

## Modèles candidats actifs

Voir `CHARTERS/` pour cahier des charges complet.

| ID | Nom | Hypothèse à tuer | Statut |
|---|---|---|---|
| M1 | CriticalGrowNet | Architecture peut s'auto-régler vers β=-2 via mutation policy | Saturated synthetic, needs harder task |
| M2 | MetaController GNN | Méta-réseau apprend politique de mutation | Non démarré |
| M3 | RetroProbe | Géométrie hidden state prédit Δpp transfer | Plateau n=8 |
| M4 | ComplexMLP | Phase-coded signals → complex algebra inductive bias | Caractérisé 16 axes. Real bias = paired channels + magnitude activation |
| M5 | InnerDialogueNet | Self-modeling produit dialogue intérieur émergent | Design exigé |
| M6 | LiquidODE | ODE dynamics handles autoregressive rollout | Réfuté (rollout 67× pire MLP) |
| M7 | TBD | À définir | - |
| M8+ | Slots libres | Nouvelles idées | - |

## Méthodologie

Voir `PROTOCOLE.md` pour détails. Pipeline standard 5 étages :

0. **Sanity** — compile + forward + overfit single batch
1. **Baseline** — comparaison matched-params (paramètres + FLOPs + compute)
2. **Dynamiques** — β spectral, λ Lyapunov, intrinsic dim, φ_id, RQA
3. **Falsification dédiée** — test conçu pour TUER l'hypothèse
4. **Comparaison strong baselines** — Transformer, S4, Mamba, Neural ODE, ConvNeXt, wav2vec
5. **Démo / portfolio**

Strict standards (per scientific-critical-thinking) :
- Pre-register hypothesis + analysis plan
- 10+ seeds minimum, report std + CI + p-values
- Matched params + FLOPs + scheduler + augment + dataset
- Test scaling laws (perf vs data/params/noise/horizon)
- OOD test
- Ablations identifiant composante load-bearing
- Real strong baselines (pas juste petit MLP)

## Stack

- **PyTorch** + **Modal** (cloud A10/H100 parallel)
- **autoresearch** loop pattern (edit → run → grep → keep/revert via git)
- **dynamoscope** API (HTTP) pour analyses dynamiques
- **ARK** pour visualisation future (Phase 2)

## Convention répertoire

```
metamorphic-systems/
├── README.md                  vision + ToC
├── GLOSSARY.md                définitions terminologie
├── PROTOCOLE.md               méthodologie scientifique
├── CHARTERS/                  cahier des charges par modèle (Mn.md)
├── inventory/                 fiches résumées des repos existants
├── models/                    implementations propres
├── benchmarks/                bancs d'essai par modèle
├── analysis/                  scaling laws, ablations, viz
├── paper/                     papers + meta-paper
└── memory/                    notes intermédiaires
```

## Status snapshot (2026-05-28)

### Session cloud nuit 2026-05-28

~700+ Modal A10 runs. Findings principaux :

**Falsifications (7)** — voir `analysis/falsifications.md` :
- F1 : architrace "1/f² universel SOC" INVALIDÉ (MLP passive filter)
- F2 : ComplexNet "complex algebra essential" RÉFUTÉ (paired channels suffit)
- F3 : Self-modeling naive utile RÉFUTÉ (val_acc invariant)
- F4 : ODE rollout good RÉFUTÉ (67× pire MLP)
- F5 : Anti-Bitter Lesson NUANCÉ (catch-up delay)
- F6 : Complex algebra universal RÉFUTÉ (niche M4 strict)
- F7 : MSE = InfoNCE aux loss RÉFUTÉ

**Finding positif majeur** : β taxonomy architectures (voir `analysis/falsifications.md`)
- MLP=0.05, ResNet=0.02 (passive)
- RNN=0.21 (tanh saturé)
- Attention=0.50 (partial mixing)
- GRU=1.08, LSTM=1.17 (gated memory → pink)
- **ODE Euler=1.57 ± 0.006** (integration scale)

### PoC livrés (`poc/`)

| PoC | Status | AUROC/result |
|---|---|---|
| β-meter raw signal | ✓ Validé | AUROC 0.78-0.99 (synth), 0.98 (dist shift) |
| ODE β-controlled gen | ✓ Validé | β≈2.0 stable autoregressive rollout |
| β-toolkit Python pkg | ✓ Functional | demo.py runs CPU 10s |
| Web demo live | ✓ Functional | port 4633, canvas-based interactive |
| Niche characterization | ✓ 7 anomaly types | excels regime+dist shift, blind to moments |

### Figures publication

- `poc/fig1_anomaly_detection.png` : sliding β + z-score on synthetic
- `poc/fig2_ode_synthesizer.png` : training vs generated spectra
- `poc/fig3_taxonomy_beta.png` : architecture bar chart β taxonomy
- `poc/fig4_anomaly_types_robustness.png` : 7-panel detection by type
- `poc/fig5_anomaly_types_summary.png` : AUROC niche summary
