# Protocole scientifique

## Principes (per scientific-critical-thinking standards)

1. **Pre-register** : hypothèse + design + analyse PLAN avant lancement.
2. **Strong baselines** : pas juste petit MLP. Transformer, S4, Mamba, Neural ODE, ConvNeXt, wav2vec selon domaine.
3. **Matched setup** : params + FLOPs + batch + scheduler + augment + dataset.
4. **n_seeds ≥ 10** : moyenne + std + CI + test permutation/bootstrap.
5. **Falsification dédiée** : test conçu pour TUER l'hypothèse (pas juste confirmer).
6. **Scaling laws** : perf vs data / params / noise / horizon. Pas juste 1 point.
7. **OOD test** : tester hors distribution train.
8. **Ablations** : identifier composante load-bearing.
9. **Reproducibility** : code + seeds + configs + datasets versionnés.

## Pipeline standard 5 stages

### Stage 0 — Sanity
- Compile model
- Forward pass shape check
- Overfit single batch (loss → 0 sur 1 batch = model fonctionnel)
- Gradient flow check (no NaN, no zero grad)

**Pass criterion** : compile + overfit OK.

### Stage 1 — Baseline matched
- Construire baseline matched-params (RealMLP / LSTM / Transformer selon domain)
- Ajouter strong baselines additionnels
- Run on default task, n_seeds=10
- Report : means, stds, params, FLOPs

**Pass criterion** : test_metric ± 2% baseline OR delta significatif (p<0.05).

### Stage 2 — Dynamiques (auto-observation)
Mesurer signatures internes :
- **β spectral** (slope 1/f^β sur activations latentes)
- **λ Lyapunov** (divergence trajectoires nearby init)
- **eff_rank** (Effective rank des hidden activations)
- **intrinsic_dim** (TwoNN, MLE)
- **φ_id** (information intégrée, via dynamoscope HTTP API)
- **RQA** : DET, LAM (récurrence)

**Goal** : caractériser le **régime dynamique** de l'architecture.

### Stage 3 — Falsification dédiée
Test conçu spécifiquement pour TUER l'hypothèse principale :

Exemples :
- "ComplexNet edge = inductive bias" → falsifié par Real MLP avec FFT phase features (counterfactual)
- "Algebra complexe = essential" → falsifié par two_branch_add ≡ complex_full
- "ODE rollout = good" → falsifié par LiquidNet rollout 67× pire que MLP
- "Anti-Bitter Lesson at high noise" → falsifié par n=8000 Δ chute

**Pass criterion** : si hypothèse survit la falsification, finding robuste. Sinon réfutation propre = aussi valuable.

### Stage 4 — Strong baselines + scaling
- Compare to Transformer / S4 / Mamba / Neural ODE / wav2vec / ConvNeXt selon domain
- Vary data scale (10, 100, 1000, 10000)
- Vary noise levels
- Vary task complexity (n_classes, horizon)
- Identify niche : où le modèle gagne vs où il perd

**Pass criterion** : map clean du regime utile + caveats explicites.

### Stage 5 — Démo / portfolio
- Visualisation Three.js / matplotlib des dynamiques
- Paper section (per format `paper/`)
- Reproductibilité : commit + tag + script run.sh

## Standards par stage

| Stage | n_seeds min | Wall time max | Cloud OK ? |
|---|---|---|---|
| 0 Sanity | 1 | 5 min | Non (debug local) |
| 1 Baseline | 10 | 30 min/seed | Oui (parallel) |
| 2 Dynamiques | 10 | 1h/seed | Oui |
| 3 Falsification | 10 | 1h/seed | Oui |
| 4 Strong baselines + scaling | 10 par cell | 6h total | **Modal parallel** |
| 5 Démo | 3 (figs reproductibles) | 1h | Local |

## Test types frameworks (per scientific-critical-thinking)

### Hypothesis testing
- **Pre-registered hypotheses** (write before run)
- **Distinguish primary vs secondary outcomes**
- **Multiple comparison correction** (Bonferroni, FDR)
- **Effect sizes** rapportés en plus de p-values
- **Confidence intervals** (bootstrap si non-normal)

### Risk of bias

Cochrane-style checklist par expé :
- Randomization (seed range diverse)
- Allocation concealment (preregister)
- Blinding (impossible pour design, mais : analyses pré-spécifiées)
- Selective reporting (rapport tous outcomes plannés)
- Attrition (n_crash documenté)

### GRADE

Qualité d'une preuve :
- **High** : 10+ seeds, replicated cross-dataset, ablation confirme, strong baselines comparés
- **Moderate** : 10 seeds, 1 dataset, baseline raisonnable
- **Low** : 3-5 seeds, single dataset, weak baseline
- **Very low** : 1 seed, anecdotal

## Falsification logbook

Toujours documenter dans `analysis/falsifications.md` :
- Hypothèse initiale
- Test de falsification désigné
- Résultat (validation ou falsification)
- Interprétation honest

Self-falsifications attendues. Exemples du programme :
- M4 "complex algebra essential" → FALSIFIÉ (algebra ablation, paired channels suffisent)
- M4 "anti-Bitter Lesson high noise" → FALSIFIÉ (n=8000 Δ chute)
- M6 "ODE good rollout" → FALSIFIÉ (LiquidNet 67× pire MLP)
- Q5 architrace "MSE aux loss universel" → FALSIFIÉ (MSE < InfoNCE sur world-task)

## Vulgarisation obligatoire

Chaque finding doit avoir :
- **Niveau technique** : pour reviewer
- **Niveau vulgarisé** : pour Zac, bac/licence
- **Niveau visuel** : figure ou diagram explicatif

## Liens repos

- `dynamoscope` HTTP API pour analyses dynamiques (Lyapunov, SINDy, DMD, φ_id)
- `adaptive-nn` pour code VG VerifyGrowNet
- `architrace` pour findings architrace S1-S20+
- `ai-arch-builder` / `ark` pour visualisation (Phase 2, non bloquant)
