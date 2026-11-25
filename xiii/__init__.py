"""
XIII — the overfitting linter for backtests.
The Thirteenth Man Protocol, automated.

This is neither a robot, nor a signal, nor a promise of returns: it's a
peer reviewer that tells you how your backtest dies BEFORE you deploy.

Status v0.1: scope confirmed (see SPEC_v0.1.md). BrokerConfig + primitives
delivered; implementation of the 9 checks is the next session.
"""
from __future__ import annotations

from . import brokers, dataqc, metrics, registry
from .brokers import FTMO, VANTAGE, BrokerConfig, InstrumentCost
from .dataqc import DataQCReport
from .registry import ExperimentRecord, ExperimentRegistry
from .report import AuditReport, CheckResult

__version__ = "0.1.0.dev0"

__all__ = [
    "audit",
    "brokers",
    "dataqc",
    "DataQCReport",
    "metrics",
    "registry",
    "BrokerConfig",
    "InstrumentCost",
    "AuditReport",
    "CheckResult",
    "ExperimentRecord",
    "ExperimentRegistry",
    "FTMO",
    "VANTAGE",
]


import pandas as pd


def _to_returns(returns=None, equity=None, trades=None):
    """Normalizes input to series of ~daily returns (or None if impossible)."""

    if returns is not None:
        return pd.Series(returns).astype(float)
    if equity is not None:
        return pd.Series(equity).astype(float).pct_change().dropna()
    if trades is not None:
        # v0.1: trades->returns path not yet wired (requires capital).
        return None
    return None


def audit(
    trades=None,
    equity=None,
    returns=None,
    source_file=None,
    broker=None,
    deployed_sizing: float = 1.0,
    portfolio=None,
    sl_pips=None,
    tp_pips=None,
    win_rate_backtest: float | None = None,
    *,
    symbol: str = "GBPUSD",
    proxy: bool = False,
    proxy_data=None,
    reference_data=None,
    tick_spreads=None,
    spread_regime: str = "all",
    spread_percentile: int = 75,
    spread_source: str | None = None,
    spread_sensitive: bool = False,
    session_replay=None,
    session_periods_per_year: int = 252,
) -> AuditReport:
    """Launches the automatable checks of the Thirteenth Man on a backtest.

    Performance inputs (at least one): `returns` (~daily returns),
    `equity` (equity curve -> converted to returns), or `trades`.

    v0.1 checks (see SPEC §3), with their required inputs:
      A1_dummy_scan            <- source_file                                    [ACTIVE]
      A3_series_alignment      <- source_file                                    [ACTIVE]
      G1_lookahead_scan        <- source_file                                    [ACTIVE]
      B1_window_sensitivity    <- returns|equity      # catches the lucky-window mirage [ACTIVE]
      B3_yearly_breakdown      <- returns|equity (dated)                         [ACTIVE]
      C1_sizing_vs_maxdd       <- equity + broker + deployed_sizing             [ACTIVE]
      C2_montecarlo_constraints<- trades + broker, optional daily returns                               [ACTIVE]
      D2_spread_breakeven      <- sl_pips + tp_pips + broker                    [ACTIVE]
      DE_portfolio_correlation <- portfolio

    Missing input => the corresponding check returns SKIP (flagged), not an error.
    Returns an AuditReport (.print(), .to_json(), .passed, .verdicts).

    Declare proxy=True (or pass proxy_data) to require proxy/reference price
    comparison. Tick spreads optionally replace the broker's D2 fixed spread.
    Pass a session_replay(offset_minutes) callback for informative D3 diagnostics.

    STATUS v0.1.dev: 8/9 checks active (A1, A3, B1, B3, C1, C2, D2, G1). DE remaining.
    """
    from .checks.cost import d2_spread_breakeven
    from .checks.integrity import a1_dummy_scan, a3_series_alignment, g1_lookahead_scan
    from .checks.montecarlo import c2_montecarlo_constraints
    from .checks.sizing import c1_sizing_vs_maxdd
    from .checks.time_shift import time_shift_check
    from .checks.window import b1_window_sensitivity
    from .checks.yearly import b3_yearly_breakdown

    rets = _to_returns(returns=returns, equity=equity, trades=trades)

    # Calculate win_rate if not provided
    wr = win_rate_backtest
    if wr is None and trades is not None and isinstance(trades, pd.DataFrame):
        if "pnl_usd" in trades.columns:
            wr = (trades["pnl_usd"] > 0).sum() / len(trades) * 100

    verdicts = [
        a1_dummy_scan(source_file),
        a3_series_alignment(source_file),
        b1_window_sensitivity(rets),
        b3_yearly_breakdown(rets),
        c1_sizing_vs_maxdd(equity, broker, deployed_sizing),
        c2_montecarlo_constraints(
            trades, broker, deployed_sizing, daily_returns=rets if trades is not None else None,
        ),
        d2_spread_breakeven(
            sl_pips, tp_pips, win_rate_backtest=wr, broker=broker, symbol=symbol,
            tick_spreads=tick_spreads, spread_regime=spread_regime, spread_percentile=spread_percentile,
            spread_source=spread_source, spread_sensitive=spread_sensitive,
        ),
        g1_lookahead_scan(source_file),
    ]
    checks_total = 9
    if proxy or proxy_data is not None:
        verdicts.extend(dataqc.proxy_checks(proxy_data, reference_data))
        checks_total += 2
    if session_replay is not None:
        verdicts.append(time_shift_check(session_replay, periods_per_year=session_periods_per_year))
        checks_total += 1
    return AuditReport(verdicts=verdicts, checks_total=checks_total)
