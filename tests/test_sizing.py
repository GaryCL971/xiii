"""C1 sign semantics.

Drawdowns and broker limits are both negative fractions, so "worse" means "more
negative" and the comparison runs opposite to the intuition built on positive
percentages. C1 shipped inverted: it passed a -46% drawdown against a -10% limit
as "35.9 pp margin", and failed a safe -5% curve. These tests pin the direction.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from xiii.brokers import FTMO, VANTAGE
from xiii.checks.sizing import c1_sizing_vs_maxdd


def curve(max_dd: float, n: int = 600) -> pd.Series:
    """Equity that slides straight to `max_dd` (negative fraction), then flat."""
    half = n // 2
    path = np.concatenate([np.linspace(1.0, 1.0 + max_dd, half), np.full(n - half, 1.0 + max_dd)])
    return pd.Series(path * 100_000, index=pd.bdate_range("2015-01-01", periods=n))


@pytest.mark.parametrize("max_dd", [-0.11, -0.15, -0.46])
def test_drawdown_deeper_than_limit_fails(max_dd):
    assert c1_sizing_vs_maxdd(curve(max_dd), FTMO, 1.0).status == "FAIL"


@pytest.mark.parametrize("max_dd", [-0.01, -0.05, -0.07])
def test_drawdown_well_inside_limit_passes(max_dd):
    assert c1_sizing_vs_maxdd(curve(max_dd), FTMO, 1.0).status == "PASS"


@pytest.mark.parametrize("max_dd", [-0.085, -0.095])
def test_thin_headroom_warns(max_dd):
    """Inside the limit, but under 2 pp of headroom."""
    assert c1_sizing_vs_maxdd(curve(max_dd), FTMO, 1.0).status == "WARN"


def test_sizing_multiplier_can_push_a_safe_curve_into_breach():
    """The Exp.75 failure mode: the edge was fine, the deployed sizing was not."""
    assert c1_sizing_vs_maxdd(curve(-0.095), FTMO, 1.0).status == "WARN"
    assert c1_sizing_vs_maxdd(curve(-0.095), FTMO, 1.08).status == "FAIL"


def test_breach_is_reported_as_breach_not_as_headroom():
    """The original bug called a -46% drawdown '35.9 pp margin'. Pin the wording too."""
    result = c1_sizing_vs_maxdd(curve(-0.46), FTMO, 1.0)
    assert result.status == "FAIL"
    assert "margin" not in result.headline.lower()

    breach_pp = float(re.search(r"BREACH BY (-?[\d.]+) pp", result.detail).group(1))
    assert breach_pp > 0, "a breach must be reported as a positive overshoot"


def test_evidence_reports_the_sized_drawdown():
    result = c1_sizing_vs_maxdd(curve(-0.05), FTMO, 2.0)
    assert result.evidence["dd_base_pct"] == pytest.approx(-5.0, abs=0.1)
    assert result.evidence["dd_after_sizing_pct"] == pytest.approx(-10.0, abs=0.1)


def test_broker_without_risk_rules_skips():
    assert c1_sizing_vs_maxdd(curve(-0.46), VANTAGE, 1.0).status == "SKIP"


def test_missing_inputs_skip():
    assert c1_sizing_vs_maxdd(None, FTMO, 1.0).status == "SKIP"
    assert c1_sizing_vs_maxdd(curve(-0.05), None, 1.0).status == "SKIP"


def test_short_history_skips():
    assert c1_sizing_vs_maxdd(curve(-0.46, n=100), FTMO, 1.0).status == "SKIP"
