"""
xiii.metrics — performance primitives, EXTRACTED as-is from the lab.

Source: research/sizing_fix.py. Logic unchanged: these are exactly the
calculations that uncovered the mirage (Sharpe claimed 2.53 -> honest 1.22).
Pure functions, no dependencies other than numpy/pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252  # trading days per year


def max_drawdown(returns: pd.Series) -> float:
    """Maximum (negative) drawdown from a series of periodic returns."""
    eq = (1 + returns).cumprod()
    return float((eq / eq.cummax() - 1).min())


def annualized_return(returns: pd.Series, ann: int = ANN) -> float:
    """CAGR from a series of periodic returns."""
    n = len(returns)
    if n == 0:
        return float("nan")
    eq = (1 + returns).cumprod()
    n_years = n / ann
    return float(eq.iloc[-1] ** (1 / n_years) - 1)


def sharpe(returns: pd.Series, ann: int = ANN) -> float:
    """Annualized Sharpe (rf=0). 0.0 if volatility is zero."""
    sd = returns.std()
    return float(returns.mean() / sd * np.sqrt(ann)) if sd > 0 else 0.0


def k_for_dd(
    returns: pd.Series,
    target_dd: float = -0.09,
    lo: float = 0.05,
    hi: float = 5.0,
    iters: int = 60,
) -> float:
    """Sizing multiplier k such that maxDD(k * returns) = target_dd.

    Binary search: with compounding, maxDD is not linear in k,
    so we solve numerically. This is the core of check C1 (honest sizing):
    we compare the k that respects the broker limit against actual deployed sizing.

    Extracted from research/sizing_fix.py (k_for_dd function, identical logic).
    """
    for _ in range(iters):
        mid = (lo + hi) / 2
        if max_drawdown(mid * returns) < target_dd:  # too much DD -> reduce k
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2
