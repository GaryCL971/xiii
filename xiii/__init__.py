"""
XIII — le linter d'overfitting pour backtests.
Le Protocole du Treizième Homme, automatisé.

Ce n'est ni un robot, ni un signal, ni une promesse de rendement : c'est un
pair reviewer qui te dit par quoi ton backtest meurt AVANT que tu déploies.

Statut v0.1 : périmètre confirmé (voir SPEC_v0.1.md). BrokerConfig + primitives
livrés ; l'implémentation des 9 checks est la session suivante.
"""
from __future__ import annotations

from . import brokers, metrics
from .brokers import FTMO, VANTAGE, BrokerConfig, InstrumentCost
from .report import AuditReport, CheckResult

__version__ = "0.1.0.dev0"

__all__ = [
    "audit",
    "brokers",
    "metrics",
    "BrokerConfig",
    "InstrumentCost",
    "AuditReport",
    "CheckResult",
    "FTMO",
    "VANTAGE",
]


import pandas as pd


def _to_returns(returns=None, equity=None, trades=None):
    """Normalise l'entrée en série de rendements ~quotidiens (ou None si impossible)."""

    if returns is not None:
        return pd.Series(returns).astype(float)
    if equity is not None:
        return pd.Series(equity).astype(float).pct_change().dropna()
    if trades is not None:
        # v0.1 : chemin trades->rendements pas encore branché (nécessite un capital).
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
) -> AuditReport:
    """Lance les checks automatisables du Treizième Homme sur un backtest.

    Entrées de performance (au moins une) : `returns` (rendements ~quotidiens),
    `equity` (courbe d'équité -> convertie en rendements), ou `trades`.

    Checks v0.1 (voir SPEC §3), avec leurs entrées requises :
      A1_dummy_scan            <- source_file                                    [ACTIF]
      A3_series_alignment      <- source_file                                    [ACTIF]
      G1_lookahead_scan        <- source_file                                    [ACTIF]
      B1_window_sensitivity    <- returns|equity      # rattrape le mirage 2.53->1.22  [ACTIF]
      B3_yearly_breakdown      <- returns|equity (datés)                         [ACTIF]
      C1_sizing_vs_maxdd       <- equity + broker + deployed_sizing             [ACTIF]
      C2_montecarlo_constraints<- trades + broker                               [ACTIF]
      D2_spread_breakeven      <- sl_pips + tp_pips + broker                    [ACTIF]
      DE_portfolio_correlation <- portfolio

    Une entrée absente => le check concerné renvoie SKIP (signalé), pas une erreur.
    Renvoie un AuditReport (.print(), .to_json(), .passed, .verdicts).

    STATUT v0.1.dev : 8/9 checks actifs (A1, A3, B1, B3, C1, C2, D2, G1). Reste DE.
    """
    from .checks.cost import d2_spread_breakeven
    from .checks.integrity import a1_dummy_scan, a3_series_alignment, g1_lookahead_scan
    from .checks.montecarlo import c2_montecarlo_constraints
    from .checks.sizing import c1_sizing_vs_maxdd
    from .checks.window import b1_window_sensitivity
    from .checks.yearly import b3_yearly_breakdown

    rets = _to_returns(returns=returns, equity=equity, trades=trades)

    # Calculer win_rate si pas fournie
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
        c2_montecarlo_constraints(trades, broker, deployed_sizing),
        d2_spread_breakeven(sl_pips, tp_pips, win_rate_backtest=wr, broker=broker, symbol="GBPUSD"),
        g1_lookahead_scan(source_file),
    ]
    return AuditReport(verdicts=verdicts, checks_total=9)
