"""
xiii.metrics — primitives de performance, EXTRAITES telles quelles du labo.

Source : research/sizing_fix.py. Logique inchangée : ce sont exactement les
calculs qui ont servi à débusquer le mirage (Sharpe annoncé 2.53 -> honnête 1.22).
Fonctions pures, sans dépendance autre que numpy/pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252  # jours de trading par an


def max_drawdown(returns: pd.Series) -> float:
    """Drawdown maximal (négatif) d'une série de rendements périodiques."""
    eq = (1 + returns).cumprod()
    return float((eq / eq.cummax() - 1).min())


def annualized_return(returns: pd.Series, ann: int = ANN) -> float:
    """CAGR à partir d'une série de rendements périodiques."""
    n = len(returns)
    if n == 0:
        return float("nan")
    eq = (1 + returns).cumprod()
    n_years = n / ann
    return float(eq.iloc[-1] ** (1 / n_years) - 1)


def sharpe(returns: pd.Series, ann: int = ANN) -> float:
    """Sharpe annualisé (rf=0). 0.0 si volatilité nulle."""
    sd = returns.std()
    return float(returns.mean() / sd * np.sqrt(ann)) if sd > 0 else 0.0


def k_for_dd(
    returns: pd.Series,
    target_dd: float = -0.09,
    lo: float = 0.05,
    hi: float = 5.0,
    iters: int = 60,
) -> float:
    """Multiplicateur de sizing k tel que maxDD(k * returns) = target_dd.

    Recherche dichotomique : avec le compounding, maxDD n'est pas linéaire en k,
    on résout donc numériquement. C'est le cœur du check C1 (sizing honnête) :
    on compare le k qui respecte la limite broker au sizing réellement déployé.

    Extrait de research/sizing_fix.py (fonction k_for_dd, logique identique).
    """
    for _ in range(iters):
        mid = (lo + hi) / 2
        if max_drawdown(mid * returns) < target_dd:  # trop de DD -> réduire k
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2
