"""β-trader demo : real market data (BTC + SPY) → regime detection + backtest.

Pulls historical data via yfinance, applies β-regime detector, backtests.
Honest comparison vs buy-hold.
"""
from __future__ import annotations
import numpy as np
import yfinance as yf
import sys
sys.path.insert(0, '.')
from beta_toolkit import BetaRegimeDetector, BetaTrader, backtest, compute_beta
import matplotlib.pyplot as plt

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#000000',
    'axes.facecolor': '#000000',
    'savefig.facecolor': '#000000',
    'axes.edgecolor': '#444444',
    'axes.labelcolor': '#cccccc',
    'xtick.color': '#999999',
    'ytick.color': '#999999',
    'text.color': '#ffffff',
    'font.family': 'monospace',
    'font.size': 9,
})


def fetch(ticker: str, period: str = "2y", interval: str = "1d") -> np.ndarray:
    """Fetch ticker prices via yfinance."""
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    return df['Close'].values.astype('float32')


def analyze_ticker(ticker: str, period: str = "5y") -> dict:
    """Full β-regime analysis : fetch → compute → backtest → return results."""
    prices = fetch(ticker, period=period)
    log_returns = np.diff(np.log(prices)).astype('float32')

    # Train on first 70% (no future leakage)
    split = int(0.7 * len(log_returns))
    train_returns = log_returns[:split]
    train_prices = prices[:split + 1]
    test_returns = log_returns[split:]
    test_prices = prices[split:]

    overall_beta_price = compute_beta(prices)
    overall_beta_absret = compute_beta(np.abs(log_returns))
    overall_beta_ret = compute_beta(log_returns)

    trader = BetaTrader(window=64, stride=4)
    trader.fit(train_prices, train_returns)
    signals_test = trader.generate_signals(test_prices, test_returns)

    bt = backtest(test_prices, signals_test, fee=0.001)

    multi_beta = {
        'prices': overall_beta_price,
        '|returns|': overall_beta_absret,
        'returns': overall_beta_ret,
    }

    return {
        'ticker': ticker,
        'n_prices': len(prices),
        'n_returns': len(log_returns),
        'overall_beta': overall_beta_price,
        'multi_scale_beta': multi_beta,
        'final_strategy': bt['final_equity'],
        'final_buy_hold': bt['buy_hold_final'],
        'n_trades': bt['n_trades'],
        'sharpe': bt['sharpe_approx'],
        'equity_curve': bt['equity_curve'],
        'signals_summary': signals_test['regime'].value_counts().to_dict(),
        'prices_test': test_prices,
    }


def fig_finance_results(results: list[dict]):
    """Plot equity curves vs buy-hold + regime distribution."""
    n = len(results)
    fig, axs = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axs = axs.reshape(2, 1)

    for col, r in enumerate(results):
        eq = np.array(r['equity_curve'])
        n_steps = len(eq)
        prices = r['prices_test'][:n_steps]
        bh_curve = prices / prices[0]

        # Top : equity curves
        ax = axs[0, col]
        ax.plot(eq, color='#00ff9f', lw=1.5, label=f'β-trader (final {r["final_strategy"]:.2f})')
        ax.plot(bh_curve, color='#ff00ff', lw=1.5, label=f'buy-hold (final {r["final_buy_hold"]:.2f})')
        ax.set_title(f'{r["ticker"]}  ·  β overall = {r["overall_beta"]:.2f}',
                      color='#ffffff', pad=10)
        ax.set_ylabel('equity (normalized)')
        ax.legend(loc='upper left', facecolor='#111111', edgecolor='#333333', fontsize=9)
        ax.grid(True, alpha=0.1)

        # Bottom : regime distribution + multi-scale β
        ax2 = axs[1, col]
        regimes = r['signals_summary']
        labels = list(regimes.keys())
        counts = list(regimes.values())
        cs = ['#00ff9f' if 'trend' in l else '#ffaa00' if 'walk' in l else '#ff3366' if 'revert' in l else '#888888' for l in labels]
        ax2.bar(labels, counts, color=cs, edgecolor='#ffffff', linewidth=0.5)
        ax2.set_title(f'Regime distribution test set  ·  β scales : ' +
                      ' / '.join(f'{w}→{b:.2f}' for w, b in r['multi_scale_beta'].items()),
                       color='#ffffff', pad=10)
        ax2.set_xticklabels(labels, rotation=20, ha='right')
        ax2.grid(True, alpha=0.1, axis='y')

    plt.tight_layout()
    plt.savefig('fig6_finance_results.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("  fig6_finance_results.png saved")


if __name__ == "__main__":
    print("Fetching market data + running β-trader backtest...")
    # Intraday (where vol clustering more visible)
    intraday = [('BTC-USD', '60d', '1h'), ('SPY', '60d', '1h'), ('QQQ', '60d', '1h')]
    daily = [('BTC-USD', '5y', '1d'), ('SPY', '5y', '1d'), ('QQQ', '5y', '1d'), ('TSLA', '5y', '1d')]
    results = []
    for t, period, interval in intraday + daily:
        try:
            print(f"\n  {t} ({period}, {interval}) :")
            prices = fetch(t, period=period, interval=interval)
            log_returns = np.diff(np.log(prices)).astype('float32')
            split = int(0.7 * len(log_returns))
            train_returns = log_returns[:split]
            train_prices = prices[:split + 1]
            test_returns = log_returns[split:]
            test_prices = prices[split:]
            trader = BetaTrader(window=64, stride=4)
            trader.fit(train_prices, train_returns)
            signals_test = trader.generate_signals(test_prices, test_returns)
            bt = backtest(test_prices, signals_test, fee=0.001)
            multi_beta = {
                'prices': compute_beta(prices),
                '|returns|': compute_beta(np.abs(log_returns)),
                'returns': compute_beta(log_returns),
            }
            r = {
                'ticker': f'{t} {interval}',
                'overall_beta': multi_beta['prices'],
                'multi_scale_beta': multi_beta,
                'final_strategy': bt['final_equity'],
                'final_buy_hold': bt['buy_hold_final'],
                'n_trades': bt['n_trades'],
                'sharpe': bt['sharpe_approx'],
                'equity_curve': bt['equity_curve'],
                'signals_summary': signals_test['regime'].value_counts().to_dict(),
                'prices_test': test_prices,
            }
            results.append(r)
            print(f"    β prices : {multi_beta['prices']:.3f}")
            print(f"    β |returns| : {multi_beta['|returns|']:.3f}")
            print(f"    β returns : {multi_beta['returns']:.3f}")
            print(f"    trader/B&H : {bt['final_equity']:.3f} / {bt['buy_hold_final']:.3f}")
            print(f"    n_trades : {bt['n_trades']}  sharpe : {bt['sharpe_approx']:.2f}")
            print(f"    regimes : {signals_test['regime'].value_counts().to_dict()}")
        except Exception as e:
            print(f"    ERROR : {e}")

    if results:
        fig_finance_results(results)
        print("\n=== Summary ===")
        for r in results:
            verdict = "WIN" if r['final_strategy'] > r['final_buy_hold'] else "LOSE"
            print(f"  {r['ticker']:<10} β={r['overall_beta']:.2f} | trader {r['final_strategy']:.2f} vs B&H {r['final_buy_hold']:.2f} → {verdict}")
