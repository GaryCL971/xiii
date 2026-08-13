"""C1 sign AND magnitude.

Two separate bugs have shipped in this check, both silent (a wrong verdict with
no error), which is why every fix here is pinned by a test:

1. Sign — drawdowns and broker limits are both negative fractions, so "worse"
   means "more negative" and the comparison runs opposite to the intuition built
   on positive percentages. C1 shipped inverted: it passed a -46% drawdown
   against a -10% limit as "35.9 pp margin", and failed a safe -5% curve.

2. Magnitude — drawdown does not scale linearly under compounding, so
   `dd_base * deployed_sizing` is only an approximation, and a bad one for
   deployed_sizing far from 1.0: it UNDERSTATES the true compounded drawdown for
   k < 1 (see test_scaling_down_a_deep_drawdown_can_silently_pass below — the
   exact failure mode: de-sizing to fit a limit, the normal prop-firm move,
   landed on the lenient side and could report a PASS on a real breach).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from xiii.brokers import FTMO, VANTAGE
from xiii.checks.sizing import c1_sizing_vs_maxdd
from xiii.metrics import max_drawdown


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
    """dd_after_sizing_pct is the TRUE compounded drawdown, not dd_base * k.

    At k=2.0 those two differ by 0.3 pp even on this gentle curve — small here,
    but the same formula that produces this gap is what silently passed a real
    breach in test_scaling_down_a_deep_drawdown_can_silently_pass below.
    """
    equity = curve(-0.05)
    result = c1_sizing_vs_maxdd(equity, FTMO, 2.0)
    true_dd = max_drawdown(equity.pct_change().dropna() * 2.0)

    assert result.evidence["dd_base_pct"] == pytest.approx(-5.0, abs=0.1)
    assert result.evidence["dd_after_sizing_pct"] == pytest.approx(true_dd * 100, abs=0.05)
    assert result.evidence["dd_after_sizing_pct"] != pytest.approx(-10.0, abs=0.1), (
        "this is the naive dd_base * 2.0 estimate — asserting it here would silently "
        "restore the magnitude bug this test exists to catch"
    )


def test_scaling_down_a_deep_drawdown_can_silently_pass():
    """The exact false-PASS shape found in review: de-size to fit a limit (the
    normal prop-firm move on a strategy with real crisis exposure), and the
    linear model UNDERSTATES the true compounded drawdown enough to flip the
    verdict. At base maxDD -80%, sized to 0.08x:
      linear model : -6.4%  (margin +3.6 pp -> PASS)
      true (fixed)  : -12.0% (margin -2.0 pp -> FAIL, a real breach)
    """
    result = c1_sizing_vs_maxdd(curve(-0.80), FTMO, 0.08)
    assert result.status == "FAIL"
    assert result.evidence["dd_after_sizing_pct"] < -10.0


def test_max_safe_sizing_is_reported_on_breach():
    result = c1_sizing_vs_maxdd(curve(-0.80), FTMO, 0.08)
    assert result.status == "FAIL"
    k_max = result.evidence["max_safe_sizing"]
    assert k_max is not None
    # k_for_dd's binary search returns a midpoint sitting almost exactly ON the
    # target, so its drawdown lands within a hair of the limit on either side —
    # asserting a strict ">= limit" here would hit the same floating-point coin
    # flip the production code's saturation guard was written to avoid.
    dd_at_k_max = max_drawdown(curve(-0.80).pct_change().dropna() * k_max)
    assert dd_at_k_max == pytest.approx(FTMO.max_total_drawdown, abs=0.005)
    assert "Max sizing that fits" in result.detail


def test_max_safe_sizing_is_none_when_no_sizing_fits():
    """k_for_dd searches within [0.05, 5.0]. If the curve still breaches even at
    that floor, no realistic sizing fits — reporting the search's boundary as if
    it were a real answer would be the same class of silent wrong-number bug
    this whole file exists to catch, just relocated to a new field.
    """
    result = c1_sizing_vs_maxdd(curve(-0.97), FTMO, 0.05)
    assert result.status == "FAIL"
    assert result.evidence["max_safe_sizing"] is None
    assert "No realistic sizing" in result.detail


def test_broker_without_risk_rules_skips():
    assert c1_sizing_vs_maxdd(curve(-0.46), VANTAGE, 1.0).status == "SKIP"


def test_missing_inputs_skip():
    assert c1_sizing_vs_maxdd(None, FTMO, 1.0).status == "SKIP"
    assert c1_sizing_vs_maxdd(curve(-0.05), None, 1.0).status == "SKIP"


def test_short_history_skips():
    assert c1_sizing_vs_maxdd(curve(-0.46, n=100), FTMO, 1.0).status == "SKIP"
