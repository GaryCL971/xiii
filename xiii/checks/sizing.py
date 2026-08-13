"""
xiii.checks.sizing — section C of protocol (sizing honesty).

C1_sizing_vs_maxdd: the CRITICAL check of the Thirteenth Man.

Real flaw: "my backtest maxDD = -2.6%, so 1.08x sizing stays inside FTMO's
-10% limit". Reason: -2.6% was measured on a short favorable window. Sized
on the honest full-history drawdown instead, the same 1.08x crossed the limit.

C1 takes the deployed sizing and validates that it respects broker limits.
"""
from __future__ import annotations

import pandas as pd

from ..brokers import BrokerConfig
from ..metrics import max_drawdown, k_for_dd
from ..report import CheckResult

_ID = "C1_sizing_vs_maxdd"


def c1_sizing_vs_maxdd(
    equity: pd.Series | None,
    broker: BrokerConfig | None,
    deployed_sizing: float = 1.0,
) -> CheckResult:
    """Validates that deployed sizing respects broker drawdown limits.

    Inputs:
      - equity: equity curve (returns derived)
      - broker: BrokerConfig (FTMO, Vantage, etc.)
      - deployed_sizing: multiplier currently deployed (e.g., 1.08×)

    Logic:
      1. Reconstructs returns from equity
      2. Recomputes maxDD on the SIZED returns (deployed_sizing * returns) —
         drawdown does not scale linearly under compounding, so `dd_base *
         deployed_sizing` is an approximation that gets worse the further
         deployed_sizing sits from 1.0 (see metrics.k_for_dd's docstring)
      3. Validates vs broker limits (FTMO: -10%, Vantage: n/a)
      4. On a breach, reports the largest sizing that would fit (metrics.k_for_dd)
    """
    _id = "C1_sizing_vs_maxdd"

    if equity is None or broker is None:
        return CheckResult(
            _id, "C", "SKIP",
            "Missing inputs",
            "Pass equity (equity curve) + broker (FTMO|VANTAGE). "
            "C1 validates that sizing respects drawdown limits.",
        )

    if not broker.has_risk_rules:
        return CheckResult(
            _id, "C", "SKIP",
            f"Broker '{broker.name}' has no risk rules",
            "Vantage does not impose DD limit (free real account). "
            "FTMO enforces -10% → C1 applies.",
        )

    # Reconstruct returns
    equity_clean = equity.dropna().astype(float)
    if len(equity_clean) < 252:
        return CheckResult(
            _id, "C", "SKIP",
            "Insufficient equity history",
            f"Requires >= 1 yr (~252 days); received {len(equity_clean)}.",
        )

    returns = equity_clean.pct_change().dropna()

    # Measure real maxDD, then re-derive it on the SIZED returns — not by scaling
    # dd_base linearly, which diverges sharply from the true compounded value the
    # further deployed_sizing sits from 1.0 (metrics.k_for_dd's docstring).
    dd_base = max_drawdown(returns)
    dd_sized = max_drawdown(deployed_sizing * returns)

    limit = broker.max_total_drawdown
    # Both are negative fractions (e.g. -0.46 vs -0.10). A curve breaches when its
    # drawdown is DEEPER than the limit, i.e. more negative — so the comparison runs
    # the opposite way to the intuition built on positive percentages.
    margin = dd_sized - limit  # > 0 = headroom still available, < 0 = depth of the breach

    # Advisory only, never the verdict above: the largest sizing that keeps maxDD
    # at the limit. k_for_dd's binary search returns a midpoint sitting almost
    # exactly ON the target, so checking k_max itself for safety is a coin flip
    # on floating-point noise — checking it directly here misreported every case,
    # safe and not, as invalid. What actually signals "no sizing fits" is the
    # search saturating at its own floor (0.05x): if even that still breaches,
    # there is no realistic sizing that fits, and reporting the search's boundary
    # as if it were a real answer would be the same kind of silent wrong-number
    # bug this fix exists to remove.
    k_max = k_for_dd(returns, target_dd=limit)
    k_max_valid = max_drawdown(0.05 * returns) >= limit
    evidence = {
        "dd_base_pct": round(dd_base * 100, 1),
        "deployed_sizing": deployed_sizing,
        "dd_after_sizing_pct": round(dd_sized * 100, 1),
        "broker_max_dd_limit_pct": broker.max_total_drawdown * 100,
        "max_safe_sizing": round(k_max, 3) if k_max_valid else None,
    }

    if margin < 0:
        safe_hint = (
            f" Max sizing that fits: ×{k_max:.2f}."
            if k_max_valid else
            " No realistic sizing keeps this within the limit — the edge itself "
            "is too fragile, not just the sizing."
        )
        return CheckResult(
            _id, "C", "FAIL",
            f"Sizing would exceed {broker.name} limit",
            f"Base maxDD: {dd_base*100:.1f}%. "
            f"After sizing ×{deployed_sizing}: {dd_sized*100:.1f}%. "
            f"{broker.name} limit: {limit*100:.1f}%. "
            f"→ BREACH BY {-margin*100:.1f} pp.{safe_hint}",
            evidence,
        )

    if margin < 0.02:  # less than 2 pp of headroom
        return CheckResult(
            _id, "C", "WARN",
            f"Tight DD margin: {margin*100:.1f} pp under limit",
            f"A maxDD slip of +{margin*100:.1f} pp → breach. "
            f"Little buffer. {margin*100:.1f} pp away from cliff.",
            evidence,
        )

    return CheckResult(
        _id, "C", "PASS",
        f"Sizing compliant: {margin*100:.1f} pp margin vs {broker.name} limit",
        f"Base maxDD {dd_base*100:.1f}%, "
        f"after sizing ×{deployed_sizing} → {dd_sized*100:.1f}%. "
        f"Limit: {limit*100:.1f}%. Safe.",
        evidence,
    )
