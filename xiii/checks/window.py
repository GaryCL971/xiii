"""
xiii.checks.window — section B of protocol (window honesty).

B1_window_sensitivity : THE check that catches the mirage.
Recalculates Sharpe over full history vs recent sub-windows.
If a short favorable window inflates Sharpe, the "sellable" number
is a window artifact — exactly the flaw that sold a headline Sharpe of 2.53
which did not survive the full history.

Reference logic: 13th Man audit from 2026-06-21
("2.3 years 2023-2025 inflated Sharpe by +49% vs 2021-2025").

Input: a series of ~daily returns (pd.Series). B1 reasons in observation count
(≈252/yr); support for irregular series comes later.
"""
from __future__ import annotations

import pandas as pd

from ..metrics import ANN, sharpe
from ..report import CheckResult

_ID = "B1_window_sensitivity"


def _yr(y: int) -> str:
    return "1 yr" if y == 1 else f"{y} yrs"


def b1_window_sensitivity(
    returns: pd.Series | None,
    windows_years: tuple[int, ...] = (1, 2, 3, 5),
    warn_pct: float = 20.0,
    fail_pct: float = 50.0,
    min_obs: int = 252,
) -> CheckResult:
    """Detects a Sharpe inflated by a recent favorable sub-window.

    warn_pct / fail_pct: thresholds for Sharpe inflation (%) of a sub-window
    vs full history. Defaults: +50% inflation -> FAIL, +20% -> WARN.
    """
    if returns is None or len(returns.dropna()) < min_obs:
        n = 0 if returns is None else len(returns.dropna())
        return CheckResult(
            _ID, "B", "SKIP",
            "Insufficient history to test window sensitivity",
            f"Requires >= {min_obs} dated points (~1 yr daily); received {n}.",
        )

    r = returns.dropna()
    n = len(r)
    sh_full = sharpe(r)
    years_full = round(n / ANN, 1)

    windows = {"full": {"years": years_full, "sharpe": round(sh_full, 3)}}
    worst_infl = 0.0          # max inflation (%) of a sub-window
    worst_y: int | None = None
    worst_regime = False      # True if full<=0 but short window>0 (regime edge)

    for y in windows_years:
        w = int(y * ANN)
        if w >= n or w < min_obs:
            continue
        sh_w = sharpe(r.iloc[-w:])
        windows[f"last_{y}y"] = {"years": y, "sharpe": round(sh_w, 3)}
        if sh_full > 0:
            infl = (sh_w / sh_full - 1.0) * 100.0
        else:
            # full history unprofitable: any positive window is a regime edge
            infl = float("inf") if sh_w > 0 else 0.0
        if infl > worst_infl:
            worst_infl = infl
            worst_y = y
            worst_regime = sh_full <= 0 and sh_w > 0

    evidence = {
        "sharpe_full": round(sh_full, 3),
        "years_full": years_full,
        "windows": windows,
        "worst_window_years": worst_y,
        "inflation_pct": None if worst_y is None else (
            "inf" if worst_infl == float("inf") else round(worst_infl, 1)
        ),
        "thresholds": {"warn_pct": warn_pct, "fail_pct": fail_pct},
    }

    if worst_y is None:
        return CheckResult(
            _ID, "B", "PASS",
            "Stable window: no short sub-window more flattering",
            f"Full history Sharpe {sh_full:.2f} over {years_full} yrs; "
            f"tested sub-windows do not exceed it.",
            evidence,
        )

    if worst_regime:
        return CheckResult(
            _ID, "B", "FAIL",
            f"Mirage: strategy only 'works' on recent period ({_yr(worst_y)})",
            f"Full history Sharpe {sh_full:.2f} (<= 0), but positive on "
            f"{_yr(worst_y)} window. This is a REGIME edge, not robust: "
            f"it disappears outside its favorable window.",
            evidence,
        )

    sh_short = windows[f"last_{worst_y}y"]["sharpe"]
    if worst_infl >= fail_pct:
        status, verdict = "FAIL", "mirage"
    elif worst_infl >= warn_pct:
        status, verdict = "WARN", "watch carefully"
    else:
        return CheckResult(
            _ID, "B", "PASS",
            f"Honest window: max inflation +{worst_infl:.0f}% (< {warn_pct:.0f}%)",
            f"Full Sharpe {sh_full:.2f}; best sub-window ({_yr(worst_y)}) "
            f"{sh_short:.2f}. Difference in noise.",
            evidence,
        )

    return CheckResult(
        _ID, "B", status,
        f"Sharpe inflated +{worst_infl:.0f}% by {_yr(worst_y)} window ({verdict})",
        f"Full history ({years_full} yrs): Sharpe {sh_full:.2f}. "
        f"{_yr(worst_y)} window: Sharpe {sh_short:.2f}. "
        f"The 'sellable' number comes from the favorable window. "
        f"Size and decide on {sh_full:.2f}, not {sh_short:.2f}.",
        evidence,
    )
