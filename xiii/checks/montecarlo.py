"""
xiii.checks.montecarlo — section C of protocol (Monte Carlo broker constraints).

C2_montecarlo_constraints: bootstrap of real trade sequence to validate
probabilities under broker constraints (FTMO: pass challenge, respect maxDD,
max daily loss, etc.).

Logic: a backtest with 50 trades / 4 months may pass on average, but
a low-magnitude draw (50 permutations) can blow limits.
Monte Carlo simulates 1000 possible trajectories with same P&L sequence, reveals
worst case.

NB: v0.1 = light version (no autocorrelation-aware order adjustment).
Full = pattern-aware shuffling for crisis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..brokers import BrokerConfig
from ..report import CheckResult

_ID = "C2_montecarlo_constraints"


def c2_montecarlo_constraints(
    trades: pd.DataFrame | None,
    broker: BrokerConfig | None,
    deployed_sizing: float = 1.0,
    n_sims: int = 1000,
    seed: int = 13,
) -> CheckResult:
    """Bootstrap Monte Carlo of broker constraints.

    Inputs:
      - trades: DataFrame with columns [datetime, pnl_usd] (at least)
      - broker: BrokerConfig
      - deployed_sizing: multiplier (e.g., 1.08)
      - n_sims: number of trajectories to simulate (default 1000)

    Logic:
      1. Load real P&L sequence
      2. Re-sample (permute) N times
      3. For each trajectory, check:
         - maxDD (FTMO: <= -10%)
         - daily maxDD (FTMO: <= -5%)
         - cumulative profit >= target (FTMO: +10%)
      4. Report P(pass challenge), P(breach DD), P(breach daily)
    """
    _id = "C2_montecarlo_constraints"

    if trades is None or len(trades) < 10 or broker is None:
        n = 0 if trades is None else len(trades)
        return CheckResult(
            _id, "C", "SKIP",
            "Insufficient trade data",
            f"Requires >= 10 trades; received {n}. "
            "C2 simulates 1000 re-orderings to test robustness vs broker limits.",
        )

    if not broker.has_risk_rules:
        return CheckResult(
            _id, "C", "SKIP",
            f"Broker '{broker.name}' has no risk rules",
            "Vantage has no FTMO limits. C2 applies to FTMO.",
        )

    # Validate that trades has a pnl column or similar
    pnl_col = None
    for col in ["pnl_usd", "pnl", "profit"]:
        if col in trades.columns:
            pnl_col = col
            break
    if pnl_col is None:
        return CheckResult(
            _id, "C", "SKIP",
            "P&L column not found",
            "Trades must contain pnl_usd, pnl, or profit.",
        )

    pnl = trades[pnl_col].values * deployed_sizing
    n_trades = len(pnl)

    # FTMO parameters
    max_total_dd = broker.max_total_drawdown or -0.10
    max_daily_dd = broker.max_daily_loss or -0.05
    profit_target = broker.profit_target or 0.10
    initial_capital = 100_000  # assumption

    rng = np.random.default_rng(seed)
    results = {
        "pass_challenge": 0,
        "breach_total_dd": 0,
        "breach_daily_dd": 0,
        "breach_profit": 0,
    }

    for _ in range(n_sims):
        # Random permutation of P&L sequence
        shuffled = rng.permutation(pnl)

        # Cumulative trajectory
        equity = initial_capital + np.cumsum(shuffled)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / initial_capital
        total_dd = dd.min()

        # Daily max loss (simplified simulation: worst trade in one day)
        daily_dd = pnl.min() / initial_capital

        # Final profit
        final_pnl = shuffled.sum()

        # Verify constraints
        pass_total_dd = total_dd >= max_total_dd
        pass_daily_dd = daily_dd >= max_daily_dd
        pass_profit = final_pnl >= profit_target * initial_capital

        if pass_total_dd and pass_daily_dd and pass_profit:
            results["pass_challenge"] += 1
        if not pass_total_dd:
            results["breach_total_dd"] += 1
        if not pass_daily_dd:
            results["breach_daily_dd"] += 1
        if not pass_profit:
            results["breach_profit"] += 1

    # Probabilities
    p_pass = results["pass_challenge"] / n_sims
    p_breach_total_dd = results["breach_total_dd"] / n_sims
    p_breach_daily_dd = results["breach_daily_dd"] / n_sims
    p_breach_profit = results["breach_profit"] / n_sims

    evidence = {
        "n_trades": n_trades,
        "n_sims": n_sims,
        "p_pass_challenge": round(p_pass, 3),
        "p_breach_total_dd": round(p_breach_total_dd, 3),
        "p_breach_daily_dd": round(p_breach_daily_dd, 3),
        "p_breach_profit": round(p_breach_profit, 3),
    }

    if p_pass < 0.5:
        return CheckResult(
            _id, "C", "FAIL",
            f"Monte Carlo: P(pass challenge) = {p_pass:.1%} (< 50%)",
            f"Over 1000 re-orderings, only {results['pass_challenge']} pass FTMO. "
            f"Your trade sequence is too order-sensitive. "
            f"Risk: if market randomization favors bad sequence, breach likely.",
            evidence,
        )

    if p_breach_total_dd > 0.2:
        return CheckResult(
            _id, "C", "WARN",
            f"Monte Carlo: P(breach maxDD) = {p_breach_total_dd:.1%}",
            f"{int(results['breach_total_dd'])} / {n_sims} runs exceed -10%. "
            f"Significant risk: sequence matters.",
            evidence,
        )

    return CheckResult(
        _id, "C", "PASS",
        f"Monte Carlo robust: P(pass challenge) = {p_pass:.1%}",
        f"{results['pass_challenge']} / {n_sims} trajectories pass FTMO. "
        f"Your trade sequence is robust to re-orderings.",
        evidence,
    )
