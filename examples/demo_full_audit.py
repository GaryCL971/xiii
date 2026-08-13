"""
Complete XIII demo — all checks available in v0.1 (8/9).

Case: an FTMO "candidate" strategy with equity, trades, and sources.
XIII launches the full audit and flags all problems.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import xiii

ANN = 252
rng = np.random.default_rng(42)
HERE = Path(__file__).parent


def build_realistic_equity(n_days=4 * 252, sharpe_base=1.2, drawdown_base=-0.08):
    """Realistic equity curve: robust edge + volatility + 1 bad year."""
    idx = pd.bdate_range("2022-01-01", periods=n_days)

    # 3 years OK, 1 year bad (simulated crisis)
    parts = [
        rng.normal(0.00060, 0.010, 252),      # yr 1: OK
        rng.normal(0.00060, 0.010, 252),      # yr 2: OK
        rng.normal(0.00060, 0.010, 252),      # yr 3: OK
        rng.normal(-0.00020, 0.015, n_days - 3*252),  # yr 4: bad (negative return + high vol)
    ]
    returns = np.concatenate(parts)
    equity = 100_000 * (1 + returns).cumprod()

    return pd.Series(equity, index=idx)


def build_trades_from_equity(equity, n_trades_target=50):
    """Simulates a trade sequence consistent with equity."""
    n_days = len(equity)
    # Same seeded generator as the equity curve: the demo output must be reproducible,
    # otherwise the screenshots in the README document a run nobody can obtain again.
    trade_days = np.sort(rng.choice(n_days - 1, size=n_trades_target, replace=False))

    trades = []
    for i, day_idx in enumerate(trade_days):
        entry_idx = day_idx
        exit_idx = min(day_idx + int(rng.integers(1, 20)), n_days - 1)

        entry_price = equity.iloc[entry_idx]
        exit_price = equity.iloc[exit_idx]
        pnl = exit_price - entry_price

        trades.append({
            "datetime": equity.index[entry_idx],
            "symbol": "GBPUSD",
            "type": "BUY",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_usd": pnl,
            "bars_held": exit_idx - entry_idx,
        })

    return pd.DataFrame(trades)


print("\n" + "="*70)
print("  XIII — Complete Audit v0.1 (8/9 checks)")
print("="*70)

# Build test data
equity = build_realistic_equity()
trades = build_trades_from_equity(equity)

print(f"\n  Data: {len(equity)} days of equity, {len(trades)} trades")
print(f"  Source: {HERE / 'strategie_fautive.py'}")

# Full audit
rep = xiii.audit(
    equity=equity,
    trades=trades,
    source_file=HERE / "strategie_fautive.py",
    broker=xiii.FTMO,
    deployed_sizing=1.08,
    sl_pips=15,
    tp_pips=30,
)

rep.print()

print(f"\n  Summary:")
print(f"    Checks executed: {len(rep.verdicts)}/{rep.checks_total}")
print(f"    FAIL: {sum(1 for v in rep.verdicts if v.status == 'FAIL')}")
print(f"    WARN: {sum(1 for v in rep.verdicts if v.status == 'WARN')}")
print(f"    PASS: {sum(1 for v in rep.verdicts if v.status == 'PASS')}")
print(f"    SKIP: {sum(1 for v in rep.verdicts if v.status == 'SKIP')}")
print(f"\n  Verdict: {'OK (stage v0.1)' if rep.passed else 'BLOCK (stage v0.1)'}")
