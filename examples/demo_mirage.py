"""
Démo XIII — audit complet v0.1.dev (4 checks actifs) :
  CAS 1 : backtest 'vendable' + code source fautif  -> le carnage
  CAS 2 : edge stable, sans code source             -> PASS + SKIP signalés

Lancer depuis le dossier xiii/ :  python examples/demo_mirage.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import xiii

ANN = 252
rng = np.random.default_rng(13)
HERE = Path(__file__).parent


def equity_from(returns: np.ndarray) -> pd.Series:
    idx = pd.bdate_range("2010-01-04", periods=len(returns))
    return pd.Series((1 + returns).cumprod() * 100_000, index=idx)


# 1) MIRAGE : 14 ans corrects (Sharpe ~1) + 2 ans dopés (Sharpe ~2.5),
#    produit par un script truffé des 3 fautes classiques.
mirage = np.concatenate([
    rng.normal(0.00065, 0.0102, 14 * ANN),
    rng.normal(0.00130, 0.0082, 2 * ANN),
])
print("\n#### CAS 1 — backtest 'vendable' + source fautive ####")
rep = xiii.audit(
    equity=equity_from(mirage),
    source_file=HERE / "strategie_fautive.py",
)
rep.print()
print("passed =", rep.passed)

# 2) HONNÊTE : edge stable sur 16 ans
honest = rng.normal(0.00060, 0.0100, 16 * ANN)
print("\n#### CAS 2 — backtest honnête (edge stable) ####")
rep2 = xiii.audit(equity=equity_from(honest))
rep2.print()
print("passed =", rep2.passed)
