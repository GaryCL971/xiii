"""
xiii.checks.sizing — section C of protocol (sizing honesty).

C1_sizing_vs_maxdd: the CRITICAL check of the Thirteenth Man.

Real flaw: "my backtest maxDD = -2.6%, at 1.08x I go below -10% FTMO".
Reason: backtest measured on a short window; real maxDD over 16 yrs = -10.9%.
At 1.08×, you hit -11.8% → breach.

C1 takes the deployed sizing and validates that it respects broker limits.
"""
from __future__ import annotations

import numpy as np

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
      2. Calculates real maxDD of the curve
      3. Applies sizing: maxDD * deployed_sizing
      4. Validates vs broker limits (FTMO: -10%, Vantage: n/a)
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

    # Measure real maxDD
    dd_base = max_drawdown(returns)
    dd_sized = dd_base * deployed_sizing

    evidence = {
        "dd_base_pct": round(dd_base * 100, 1),
        "deployed_sizing": deployed_sizing,
        "dd_after_sizing_pct": round(dd_sized * 100, 1),
        "broker_max_dd_limit_pct": broker.max_total_drawdown * 100,
    }

    limit = broker.max_total_drawdown
    margin = dd_sized - limit  # margin (negative = safe, positive = breach)

    if dd_sized >= limit:
        return CheckResult(
            _id, "C", "FAIL",
            f"Sizing would exceed {broker.name} limit",
            f"Base maxDD: {dd_base*100:.1f}%. "
            f"After sizing ×{deployed_sizing}: {dd_sized*100:.1f}%. "
            f"{broker.name} limit: {limit*100:.1f}%. "
            f"→ BREACH BY {margin*100:.1f} pp. Reduce sizing or edge.",
            evidence,
        )

    if margin > -0.02:  # less than 2 pp margin
        return CheckResult(
            _id, "C", "WARN",
            f"Tight DD margin: {-margin*100:.1f} pp under limit",
            f"A maxDD slip of +{-margin*100:.1f} pp → breach. "
            f"Little buffer. {-margin*100:.1f} pp away from cliff.",
            evidence,
        )

    return CheckResult(
        _id, "C", "PASS",
        f"Sizing compliant: {-margin*100:.1f} pp margin vs {broker.name} limit",
        f"Base maxDD {dd_base*100:.1f}%, "
        f"after sizing ×{deployed_sizing} → {dd_sized*100:.1f}%. "
        f"Limit: {limit*100:.1f}%. Safe.",
        evidence,
    )
