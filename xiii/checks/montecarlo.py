"""C2 Monte Carlo on daily returns, with circular blocks and compound equity."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._risk_paths import BLOCK_DAYS, BREACH_DD, N_PATHS, SEED, block_bootstrap, breach_paths, courbe_equite
from ..brokers import BrokerConfig
from ..report import CheckResult

_ID = "C2_montecarlo_constraints"


def _trade_returns(trades: pd.DataFrame, initial_capital: float) -> np.ndarray:
    pnl_col = next((col for col in ("pnl_usd", "pnl", "profit") if col in trades), None)
    if pnl_col is None:
        raise ValueError("Trades must contain pnl_usd, pnl, or profit.")
    if "datetime" not in trades:
        raise ValueError("A datetime column is required to group trade P&L by day.")
    dates = pd.to_datetime(trades["datetime"], errors="coerce")
    pnl = pd.to_numeric(trades[pnl_col], errors="coerce")
    if dates.isna().any() or not np.isfinite(pnl.to_numpy(dtype=float)).all():
        raise ValueError("Trade dates and P&L must be valid.")
    daily = pnl.groupby(dates.dt.normalize()).sum()
    # A trade ledger has no rows for inactive sessions. Retain weekdays with
    # zero P&L and any weekend sessions actually present in the input.
    sessions = pd.bdate_range(daily.index.min(), daily.index.max()).union(daily.index)
    return daily.reindex(sessions, fill_value=0.0).to_numpy() / initial_capital


def c2_montecarlo_constraints(
    trades: pd.DataFrame | None,
    broker: BrokerConfig | None,
    deployed_sizing: float = 1.0,
    n_sims: int = N_PATHS,
    seed: int | np.random.Generator = SEED,
    *,
    daily_returns: pd.Series | np.ndarray | None = None,
    initial_capital: float = 100_000.0,
) -> CheckResult:
    """Estimate C2 with daily observations at the supplied strategy sizing.

    Daily net returns, when supplied, take precedence over realized trade P&L.
    The trade fallback groups by datetime and divides by reference capital.
    Apply deployed_sizing once to the resulting fractional returns. Daily
    loss uses close-to-close equity versus initial capital; intraday equity
    and the absorbing two-phase Challenge require a different simulation.
    """
    if broker is None or (daily_returns is None and (trades is None or trades.empty)):
        return CheckResult(_ID, "C", "SKIP", "Insufficient daily data",
                           "Requires a broker and daily returns or dated trades.")
    if not broker.has_risk_rules:
        return CheckResult(_ID, "C", "SKIP", f"Broker '{broker.name}' has no risk rules",
                           "C2 applies to brokers with drawdown rules.")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be finite and positive.")
    if not np.isfinite(deployed_sizing) or deployed_sizing < 0:
        raise ValueError("deployed_sizing must be finite and nonnegative.")
    if n_sims < 1:
        raise ValueError("n_sims must be positive.")

    try:
        if daily_returns is None:
            returns = _trade_returns(trades, initial_capital)
            source = "trade_pnl_by_day"
        else:
            source = "daily_returns"
            if isinstance(daily_returns, pd.Series) and isinstance(daily_returns.index, pd.DatetimeIndex):
                if daily_returns.index.hasnans or daily_returns.index.normalize().has_duplicates:
                    raise ValueError("Daily returns must have at most one observation per calendar day.")
                daily_returns = daily_returns.sort_index()
            returns = np.asarray(daily_returns, dtype=float)
        paths = block_bootstrap(deployed_sizing * returns, n_chemins=n_sims, graine=seed)
    except (TypeError, ValueError, AttributeError) as exc:
        return CheckResult(_ID, "C", "SKIP", "Daily Monte Carlo unavailable", str(exc))

    equity = courbe_equite(paths)
    total_limit = broker.max_total_drawdown or BREACH_DD
    total = breach_paths(equity, total_limit)
    daily = np.diff(equity, axis=1, prepend=1.0).min(axis=1) < (broker.max_daily_loss or -0.05)
    profit = equity[:, -1] - 1.0 < (broker.profit_target or 0.10)
    passed = ~(total | daily | profit)
    p_pass = float(passed.mean())
    p_total = float(total.mean())
    evidence = {
        "n_trades": len(trades) if trades is not None else 0,
        "n_days": paths.shape[1],
        "n_sims": n_sims,
        "block_days": BLOCK_DAYS,
        "returns_source": source,
        "p_pass_challenge": round(p_pass, 3),
        "p_breach_total_dd": round(p_total, 3),
        "p_breach_daily_dd": round(float(daily.mean()), 3),
        "p_breach_profit": round(float(profit.mean()), 3),
        "p_ruin": round(float((equity[:, -1] == 0).mean()), 3),
    }

    if p_pass < 0.5:
        return CheckResult(
            _ID, "C", "FAIL", f"Monte Carlo: P(pass challenge) = {p_pass:.1%} (< 50%)",
            f"{passed.sum()} / {n_sims} daily block paths meet the C2 constraints.", evidence,
        )
    if p_total > 0.2:
        return CheckResult(
            _ID, "C", "WARN", f"Monte Carlo: P(breach maxDD) = {p_total:.1%}",
            f"{total.sum()} / {n_sims} daily block paths touch the drawdown limit.", evidence,
        )
    return CheckResult(
        _ID, "C", "PASS", f"Monte Carlo robust: P(pass challenge) = {p_pass:.1%}",
        f"{passed.sum()} / {n_sims} daily block paths meet the C2 constraints.", evidence,
    )
