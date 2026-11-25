"""Informative replay of intraday session boundaries shifted by ±15 minutes."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ..report import CheckResult


def time_shift_sensitivity(
    replay: Callable[[int], pd.Series], *, periods_per_year: int = 252,
) -> dict:
    """Replay whole strategy windows at -15, 0 and +15 minutes.

    The callback shifts the session, then recalculates net returns on the same
    time grid, with zero on inactive periods. Compare on common finite support.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    offsets = (-15, 0, 15)
    series = [replay(offset) for offset in offsets]
    if any(not isinstance(s, pd.Series) or not s.index.is_unique or not s.index.equals(series[0].index)
           for s in series):
        raise ValueError("session replays require aligned return Series with unique indices")
    r = np.column_stack([s.to_numpy(dtype=float) for s in series])
    r = r[np.isfinite(r).all(axis=1)]
    if r.shape[0] < 2:
        raise ValueError("at least two common finite observations are required")
    spread = r.std(axis=0, ddof=1)
    expectancy = r.mean(axis=0)
    sharpe = [float(m / sd * np.sqrt(periods_per_year)) if sd > 0 else None
              for m, sd in zip(expectancy, spread)]

    def degradation(reference, value):
        if reference is None or value is None or reference == 0:
            return None
        return float((reference - value) / abs(reference))

    return {"n": r.shape[0], "periods_per_year": periods_per_year, "offsets": {
        str(offset): {"expectancy": float(expectancy[i]), "sharpe": sharpe[i],
                      "expectancy_degradation": degradation(expectancy[1], expectancy[i]),
                      "sharpe_degradation": degradation(sharpe[1], sharpe[i])}
        for i, offset in enumerate(offsets)
    }}


def time_shift_check(replay, *, periods_per_year: int = 252) -> CheckResult:
    check_id = "D3_time_shift_sensitivity"
    try:
        evidence = time_shift_sensitivity(replay, periods_per_year=periods_per_year)
    except (ValueError, TypeError) as exc:
        return CheckResult(check_id, "D", "WARN", "Session sensitivity unavailable", str(exc))
    # Informative warning only; 25% is a display convention, not a gate.
    flagged = any(v is not None and v >= 0.25
                  for key in ("-15", "15")
                  for v in (evidence["offsets"][key]["expectancy_degradation"],
                            evidence["offsets"][key]["sharpe_degradation"]))
    return CheckResult(check_id, "D", "WARN" if flagged else "PASS",
                       "Session boundary sensitivity" if flagged else "Stable under ±15-minute shifts",
                       "Informative comparison; diagnostic warning at 25% relative degradation.", evidence)
