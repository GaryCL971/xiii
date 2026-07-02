"""
XIII demo — complete audit v0.1.dev (4 active checks):
  CASE 1: 'sellable' backtest + faulty source code  -> carnage
  CASE 2: stable edge, no source code               -> PASS + SKIP flagged

Run from xiii/ folder: python examples/demo_mirage.py
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


# 1) MIRAGE: 14 good years (Sharpe ~1) + 2 doped years (Sharpe ~2.5),
#    produced by a script riddled with 3 classic faults.
mirage = np.concatenate([
    rng.normal(0.00065, 0.0102, 14 * ANN),
    rng.normal(0.00130, 0.0082, 2 * ANN),
])
print("\n#### CASE 1 — 'sellable' backtest + faulty source ####")
rep = xiii.audit(
    equity=equity_from(mirage),
    source_file=HERE / "strategie_fautive.py",
)
rep.print()
print("passed =", rep.passed)

# 2) HONEST: stable edge over 16 years
honest = rng.normal(0.00060, 0.0100, 16 * ANN)
print("\n#### CASE 2 — honest backtest (stable edge) ####")
rep2 = xiii.audit(equity=equity_from(honest))
rep2.print()
print("passed =", rep2.passed)
