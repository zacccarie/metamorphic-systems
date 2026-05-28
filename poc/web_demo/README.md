# β-meter live web demo

Single-file HTML + vanilla JS + Canvas2D. No build chain.

## Run

Open `index.html` directement dans navigateur, OR :

```bash
python3 -m http.server 4633 --directory poc/web_demo
# → http://127.0.0.1:4633
```

## Features

- Live signal generation (colored noise, β controllable)
- Sliding window β computation (FFT-based, 64 samples)
- Auto-fit baseline β over first 200 windows
- Z-score anomaly detection vs baseline
- 4 anomaly injection modes :
  - Spike (point anomaly)
  - β regime change (shift to β=2.5)
  - Distribution shift (Gaussian → Uniform)
  - Sine inject (frequency anomaly)
- Refit / Pause controls

## Aesthetics

Black background + emissive palette :
- Signal : light blue (#88ccff)
- β line : amber (#ffaa00)
- Z-score : magenta (#ff00ff)
- Normal : mint green (#00ff9f)
- Anomaly : red pink (#ff3366)
- Highlight : magenta

Font : Menlo / monospace.

## Tested

- Regime change anomaly → z-score 1.6+ (CAUTION)
- Distribution shift anomaly → z-score 2.0+ (ANOMALY detected, red status)

Portfolio-grade visualization of metamorphic-systems β taxonomy research.
