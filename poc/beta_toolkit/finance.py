"""β-trader : adapt β-toolkit pour finance regime detection + trading signals.

Principe :
- β des log-returns capture la mémoire long-terme du marché
- β shift = changement de régime (calm→volatile, trend→mean-revert)
- Combiner β multi-échelle (5min, 1h, 1d) + signaux techniques classiques

Use cases :
1. Risk management : réduit leverage quand β shift
2. Regime-aware trading : stratégie différente par β bucket
3. Backtest synthétique : ODE generator pour rare events
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .meter import compute_beta, sliding_beta, BetaAnomalyDetector


class BetaRegimeDetector:
    """Multi-scale β regime detector for financial time-series.

    Computes β across multiple horizons → fingerprint regime.
    Distinguishes : trending (β>2), mean-revert (β<1), random walk (β≈2), noise (β≈0).
    """

    def __init__(self, windows: tuple[int, ...] = (32, 64, 128), strides: tuple[int, ...] = (4, 8, 16)):
        self.windows = windows
        self.strides = strides

    def fit_baseline(self, returns: np.ndarray) -> dict:
        """Compute baseline β stats across multiple windows on training returns."""
        stats = {}
        for w, s in zip(self.windows, self.strides):
            betas = sliding_beta(returns, window=w, stride=s)
            stats[w] = {
                'mean': float(betas.mean()),
                'std': float(betas.std() + 1e-9),
            }
        self.baseline = stats
        return stats

    def score(self, returns: np.ndarray) -> dict:
        """Return per-window β + z-score per scale."""
        out = {}
        for w, s in zip(self.windows, self.strides):
            betas = sliding_beta(returns, window=w, stride=s)
            z = (betas - self.baseline[w]['mean']) / self.baseline[w]['std']
            out[w] = {'beta': betas, 'z': z}
        return out

    def classify_regime(self, beta: float) -> str:
        """Map β value to regime label."""
        if beta < 0.3: return 'noise'
        if beta < 1.0: return 'mean_revert'
        if beta < 1.5: return 'weak_trend'
        if beta < 2.3: return 'random_walk'  # market efficient
        if beta < 3.5: return 'strong_trend'
        return 'oversmooth'


class BetaTrader:
    """β-aware risk-managed trading signal.

    Strategy : measure 3 spectral signals simultaneously :
    1. β_prices : trend strength (random walk β≈2, oversmooth β>2.5)
    2. β_|returns| : volatility clustering (high β = persistent vol = GARCH-like)
    3. β_returns : returns autocorrelation (typically β≈0 EMH)

    Position logic :
    - Long bias by default in trending market (β_prices > 2.3 = oversmooth uptrend)
    - Reduce size when β_|r| spikes (vol clustering = upcoming volatile period)
    - Flat when β_prices in random_walk regime (β_p ∈ [1.5, 2.3])

    NOT financial advice. Educational PoC only.
    """

    def __init__(self, window: int = 64, stride: int = 4):
        self.window = window
        self.stride = stride

    def fit(self, train_prices: np.ndarray, train_returns: np.ndarray) -> None:
        """Compute baseline β stats for prices, |returns|, returns."""
        betas_p = sliding_beta(train_prices, self.window, self.stride)
        betas_ar = sliding_beta(np.abs(train_returns), self.window, self.stride)
        betas_r = sliding_beta(train_returns, self.window, self.stride)
        self.baseline_price_beta = float(betas_p.mean())
        self.baseline_price_std = float(betas_p.std() + 1e-9)
        self.baseline_absret_beta = float(betas_ar.mean())
        self.baseline_absret_std = float(betas_ar.std() + 1e-9)
        self.baseline_ret_beta = float(betas_r.mean())
        self.baseline_ret_std = float(betas_r.std() + 1e-9)

    def generate_signals(self, prices: np.ndarray, returns: np.ndarray) -> pd.DataFrame:
        """Compute multi-spectral signals + position + size."""
        betas_p = sliding_beta(prices[1:], self.window, self.stride)
        betas_ar = sliding_beta(np.abs(returns), self.window, self.stride)
        betas_r = sliding_beta(returns, self.window, self.stride)

        n = min(len(betas_p), len(betas_ar), len(betas_r))
        betas_p = betas_p[:n]
        betas_ar = betas_ar[:n]
        betas_r = betas_r[:n]

        signals = []
        for bp, bar, br in zip(betas_p, betas_ar, betas_r):
            # Position : long when prices show strong trend (β > 2.3)
            if bp > 2.3:
                pos = 1  # trend
            elif bp < 1.5:
                pos = -1  # anti-trend
            else:
                pos = 1  # default long bias (default to market exposure)

            # Size : reduce when volatility clustering high (β_|r| z-score)
            z_absret = abs(bar - self.baseline_absret_beta) / self.baseline_absret_std
            size = max(0.2, 1 - z_absret / 4.0) if z_absret > 1.5 else 1.0

            regime = self._classify(bp)
            signals.append({
                'beta_price': bp, 'beta_absret': bar, 'beta_ret': br,
                'z_absret': z_absret, 'regime': regime, 'pos': pos, 'size': size,
            })

        df = pd.DataFrame(signals)
        time_idx = np.arange(n) * self.stride + self.window
        df['time_idx'] = time_idx
        return df

    def _classify(self, beta_price: float) -> str:
        if beta_price < 1.0: return 'mean_revert'
        if beta_price < 1.5: return 'weak_trend'
        if beta_price < 2.3: return 'random_walk'
        if beta_price < 3.5: return 'strong_trend'
        return 'oversmooth'


def backtest(prices: np.ndarray, signals: pd.DataFrame, fee: float = 0.001) -> dict:
    """Simple backtest : apply signals, track equity curve."""
    equity = 1.0
    position = 0
    size = 0
    history = [equity]
    n_trades = 0

    for i in range(len(signals) - 1):
        t = int(signals.iloc[i]['time_idx'])
        t_next = int(signals.iloc[i + 1]['time_idx'])
        if t_next >= len(prices):
            break
        target_pos = signals.iloc[i]['pos']
        target_size = signals.iloc[i]['size']

        # Trade if change
        if target_pos != position or abs(target_size - size) > 0.1:
            n_trades += 1
            equity *= (1 - fee)

        position = target_pos
        size = target_size

        if t_next < len(prices):
            ret = (prices[t_next] - prices[t]) / prices[t]
            equity *= 1 + position * size * ret
            history.append(equity)

    final = equity
    bh = prices[-1] / prices[0]
    sharpe = np.mean(np.diff(history) / (np.array(history[:-1]) + 1e-9)) / (np.std(np.diff(history) / (np.array(history[:-1]) + 1e-9)) + 1e-9) * np.sqrt(252)

    return {
        'final_equity': float(final),
        'buy_hold_final': float(bh),
        'n_trades': n_trades,
        'sharpe_approx': float(sharpe),
        'equity_curve': history,
    }
