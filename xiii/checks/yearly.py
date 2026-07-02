"""
xiii.checks.yearly — section B of protocol (year-by-year honesty).

B3_yearly_breakdown: detects annual breaches (negative years, drawdown breaches)
that reveal a fragile or regime-dependent edge.

Logic: if a strategy has 16 years of data but 2 years are catastrophic,
the average Sharpe masks the true risk.
"""
from __future__ import annotations

import pandas as pd

from ..metrics import ANN, max_drawdown, sharpe, annualized_return
from ..report import CheckResult

_ID = "B3_yearly_breakdown"


def b3_yearly_breakdown(
    returns: pd.Series | None,
    min_obs_per_year: int = 50,
) -> CheckResult:
    """Year-by-year breakdown. Flags each negative year / each breach.

    Detects: a strategy 'profitable on average' but broken in some years.
    Typical of mirage: good on 2023-2025, catastrophic on 2015.
    """
    _id = "B3_yearly_breakdown"
    if returns is None or len(returns.dropna()) < min_obs_per_year:
        n = 0 if returns is None else len(returns.dropna())
        return CheckResult(
            _id, "B", "SKIP",
            "Insufficient history for year-by-year breakdown",
            f"Requires >= {min_obs_per_year} points (~3 months daily); received {n}.",
        )

    r = returns.dropna()
    if not isinstance(r.index, pd.DatetimeIndex):
        return CheckResult(
            _id, "B", "SKIP",
            "Index not dated",
            "Need a datetime index to group by year. "
            "Pass `equity` instead of `returns` if index is derived from a curve.",
        )

    yearly = {}
    bad_years = []  # negative years or breaches
    for year, group in r.groupby(r.index.year):
        if len(group) < min_obs_per_year:
            continue
        sh = sharpe(group)
        ann_ret = annualized_return(group)
        dd = max_drawdown(group)
        yearly[year] = {
            "trades": len(group),
            "sharpe": round(sh, 2),
            "ret_ann": round(ann_ret * 100, 1),
            "maxdd": round(dd * 100, 1),
        }
        if ann_ret < 0 or dd < -0.10:
            bad_years.append(
                (year, ann_ret, dd,
                 "negative return" if ann_ret < 0 else "DD < -10%")
            )

    evidence = {"yearly": yearly, "bad_years_count": len(bad_years)}

    if not yearly:
        return CheckResult(
            _id, "B", "SKIP",
            "No complete year",
            "Each year must have >= {min_obs_per_year} points.",
        )

    if bad_years:
        summary = []
        for year, ann_ret, dd, reason in bad_years[:2]:
            summary.append(f"{year}: {reason} (return {ann_ret*100:+.1f}%, DD {dd*100:.1f}%)")
        detail = ", ".join(summary)
        if len(bad_years) > 2:
            detail += f", +{len(bad_years)-2} others"

        status = "FAIL" if len(bad_years) >= 2 else "WARN"
        return CheckResult(
            _id, "B", status,
            f"{len(bad_years)} problematic year(s): {detail}",
            f"A 'robust' strategy should not have years of total loss. "
            f"2+ bad years → fragile edge or regime-dependent. "
            f"Investigate: is this a market crisis (2008, 2020) or a real signal problem?",
            evidence,
        )

    return CheckResult(
        _id, "B", "PASS",
        f"Year-by-year breakdown: all positive, maxDD acceptable",
        f"{len(yearly)} years analyzed, all with Sharpe >= 0 and DD >= -10%. "
        f"Check detail to identify weak years (< Sharpe 1.0).",
        evidence,
    )
