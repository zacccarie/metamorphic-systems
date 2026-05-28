# β-toolkit — PoC deliverable

Concrete proof-of-concept derived from metamorphic-systems research findings.

**Status** : functional, demo runs end-to-end on CPU in ~10 seconds.

## Provenance scientifique

Built from 700+ cloud Modal A10 runs (`metamorphic-systems/models/`) :
- β taxonomy : MLP=0.05, LSTM=1.17, ODE Euler=1.57 ± 0.006 (under Gaussian input)
- β raw signal anomaly detection : AUROC 0.78-0.99 (synthetic), zero training
- ODE autoregressive rollout : β ≈ 2.0 universel stable

## Modules

### `beta_toolkit.meter`

- `compute_beta(signal)` → spectral slope β via FFT log-log fit
- `sliding_beta(signal, window, stride)` → array of β per window
- `BetaAnomalyDetector` → fit baseline β on normal, score deviation z-score

### `beta_toolkit.synthesizer`

- `train_synthesizer(data, ...)` → train ODE Euler next-step predictor
- `ODESynthesizer.generate(n)` → autoregressive rollout → β≈2 signal

## Demo

```bash
python demo.py
```

Output :
```
=== Demo 1 : β-Anomaly Detection ===
  Baseline β : 0.522 ± 0.290
    chunk 3 (ANOM) ★: mean z-score = 3.92
    chunk 7 (ANOM) ★: mean z-score = 3.25
    (normal chunks z-score 0.4-1.3)

=== Demo 2 : ODE β-Synthesizer ===
  β of generated : 2.163  (expected ≈ 2.0)
  Max value      : 2.26   (stable)
```

## API usage

### Anomaly detection

```python
from beta_toolkit import BetaAnomalyDetector

det = BetaAnomalyDetector(window=64, stride=8)
det.fit(normal_signal)               # learn baseline β
scores = det.score(test_signal)      # z-score per window
anomalies = det.predict(test_signal, threshold=2.0)
```

### Synthetic data generation

```python
from beta_toolkit import train_synthesizer

model = train_synthesizer(training_data, hidden=64, n_steps=32, dt=0.1, n_epochs=15)
synthetic_signal = model.generate(n_samples=1000)  # β ≈ 2.0
```

## Caveats

- Tested on synthetic 1/f^β colored noise + Lorenz attractor
- Real-world validation pending (UCR Anomaly, MIT-BIH ECG)
- Anomaly detector competitive avec MSE LSTM mais zero-training advantage
- ODE generator β ≈ 2.0 universel, pas finement tunable au-delà

## Applications visées

1. **Monitoring time-series** : drift detection sans label, sans training
2. **Data augmentation** : rare events synthesis avec β cible
3. **Benchmark synthesis** : controlled difficulty datasets
4. **Pre-screening** : β-meter rapide avant deep models complexes

## Roadmap

- Real-world benchmark UCR Anomaly
- Comparison vs Anomaly Transformer / TFD SOTA
- Tunable β output via training-time β regularization
- Pip-installable package
- Three.js viz portfolio demo (zaccarie.fr)
