# β-detector niche — anomaly types characterization

Date : 2026-05-28
Setup : Local CPU, 5 seeds × 7 anomaly types
Script : `poc/anomaly_types_study.py`

## Question

Quels types d'anomalies β-detector capture-t-il bien vs mal ?

## Setup

Signal normal : 1/f^β colored noise β=1 (pink noise).
Anomalies injectées chunks 3 et 7 (20% du signal).
β-detector : window=64, stride=8, fit on 4000-sample normal data.
AUROC direction-agnostic (max(AUROC, 1-AUROC)).

## Results

| Anomaly type | AUROC | Niche fit |
|---|---|---|
| **Distribution shift** (Gaussian → Uniform) | **0.979 ± 0.013** | ★★★★★ |
| **Regime change** (β=1.0 → β=2.5) | **0.918 ± 0.025** | ★★★★★ |
| Frequency injection (sine 20Hz) | 0.695 ± 0.040 | ★★★ |
| Spike (single 10σ value) | 0.649 ± 0.044 | ★★ |
| Trend (linear drift) | 0.574 ± 0.056 | ★ |
| Variance shift (heteroscedastic) | 0.527 ± 0.012 | ✗ |
| Amplitude scaling (×3) | 0.521 ± 0.019 | ✗ |

## Pattern identifié

**β-detector excels at SPECTRAL anomalies** :
- Distribution shift change → power spectrum shape change → β detected
- Regime β change → direct β signal

**β-detector blind to MOMENT anomalies** :
- Amplitude scaling → preserves β (normalisée)
- Variance shift → preserves β
- Trend → adds DC component, β faiblement perturbé

**β-detector moderate at LOCAL anomalies** :
- Spike → β change subtle in long window
- Frequency injection → adds peak power, β slightly elevated

## Niche recommandée

β-detector zero-shot pour :
- Phase transitions dynamical systems
- Distribution drift detection (ML model deployment)
- Regime change finance/sensors
- Texture/pattern anomalies

NE PAS utiliser β-detector pour :
- Outlier detection point-anomalies (use z-score classique)
- Variance monitoring (use control charts)
- Mean drift detection (use CUSUM)

## Recommandation pratique

**β-detector = complément** des outils traditionnels, pas remplaçant.
Combine avec :
- Z-score (point anomalies)
- CUSUM (mean drift)
- GARCH (variance shifts)
- β-meter (spectral anomalies, regime change, distribution shift)

Library `beta_toolkit` provides niche complement.
