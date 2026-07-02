"""
xiii.checks.cost — section D du protocole (honnêteté du coût d'exécution).

D2_spread_breakeven : le backtest suppose un spread (SL/TP pips). XIII calcule
le vrai breakeven win-rate APRÈS spread et commission, puis compare à la WR du backtest.

Logique : Sharpe 1.5 sur un backtest qui assume 0.5p spread, mais tu trades sur
Vantage à 1.6p → ton edge s'effondre. D2 crie danger.

Faille réelle : « spread assumption 0.5p » vs « réalité Vantage 1.6p sur entrée »
= breakeven WR passe de 33 % à 37 %.
"""
from __future__ import annotations

from ..brokers import BrokerConfig, InstrumentCost
from ..report import CheckResult

_ID = "D2_spread_breakeven"


def d2_spread_breakeven(
    sl_pips: float | None,
    tp_pips: float | None,
    win_rate_backtest: float | None,
    broker: BrokerConfig | None,
    symbol: str = "GBPUSD",
) -> CheckResult:
    """Calcule la win-rate de breakeven APRÈS spread réel du broker.

    Entrées :
      - sl_pips, tp_pips : les niveaux de la stratégie (ex: SL=15p TP=30p)
      - win_rate_backtest : WR du backtest (ex: 52%)
      - broker : BrokerConfig (FTMO, Vantage, etc.)
      - symbol : instrument (défaut: GBPUSD)

    Logique :
      1. Calcule breakeven WR *brut* (avant spread) : SL / (SL + TP)
      2. Calcule spread réel du broker pour ce symbole
      3. Recalcule breakeven WR *net* : (SL + spread) / (SL + TP + spread)
      4. Compare WR backtest à breakeven net
    """
    _id = "D2_spread_breakeven"

    # Validation des entrées
    if (sl_pips is None or tp_pips is None or win_rate_backtest is None
            or broker is None):
        return CheckResult(
            _id, "D", "SKIP",
            "Paramètres manquants pour calcul breakeven",
            "Passe sl_pips, tp_pips, win_rate_backtest, broker=FTMO|VANTAGE. "
            "D2 mesure le coût de spread sur ta stratégie.",
        )

    try:
        cost = broker.cost(symbol)
    except KeyError:
        return CheckResult(
            _id, "D", "SKIP",
            f"Symbole '{symbol}' non paramétré pour {broker.name}",
            f"Disponibles: {list(broker.instruments)}. "
            "v0.1 couvre GBPUSD/FTMO+Vantage. Autres arrivent en v1.0.",
        )

    # Calculs
    spread = cost.spread_pips
    commission = cost.commission_roundturn_usd / cost.pip_value_usd  # en pips
    total_cost = spread + commission

    be_gross = sl_pips / (sl_pips + tp_pips)
    tp_net = tp_pips - total_cost
    sl_net = sl_pips + total_cost
    be_net = sl_net / (sl_net + tp_net)

    margin_pct = (win_rate_backtest - be_net * 100) / (be_net * 100) * 100

    evidence = {
        "symbol": symbol,
        "broker": broker.name,
        "sl_pips": sl_pips,
        "tp_pips": tp_pips,
        "spread_pips": round(spread, 2),
        "commission_pips": round(commission, 2),
        "total_cost_pips": round(total_cost, 2),
        "be_gross_pct": round(be_gross * 100, 1),
        "be_net_pct": round(be_net * 100, 1),
        "wr_backtest_pct": round(win_rate_backtest, 1),
        "margin_above_breakeven_pct": round(margin_pct, 1),
    }

    if tp_net <= 0:
        return CheckResult(
            _id, "D", "FAIL",
            "TP net effondré après spread",
            f"Backtest: TP={tp_pips}p. Spread+commission={total_cost:.1f}p. "
            f"TP net = {tp_net:.1f}p (négatif!). "
            f"La stratégie ne peut pas gagner: coût > objectif de profit.",
            evidence,
        )

    if win_rate_backtest < be_net * 100:
        return CheckResult(
            _id, "D", "FAIL",
            f"Win-rate < breakeven (backtest {win_rate_backtest:.1f}% vs BE net {be_net*100:.1f}%)",
            f"Ton backtest perd même AVANT le spread réel du broker. "
            f"Soit le backtest est trop optimiste, soit il y a une erreur de sizing.",
            evidence,
        )

    if margin_pct < 5:
        return CheckResult(
            _id, "D", "WARN",
            f"Marge serrée : {margin_pct:.0f}% au-dessus du breakeven net",
            f"Backtest WR {win_rate_backtest:.1f}% vs BE {be_net*100:.1f}%. "
            f"Un slippage de +0.2p ou une baisse de WR de 1% → tu reviens au breakeven. "
            f"Peu de marge pour les erreurs.",
            evidence,
        )

    return CheckResult(
        _id, "D", "PASS",
        f"Coût spread acceptable : marge {margin_pct:.0f}% au-dessus du breakeven",
        f"Backtest assume {total_cost:.1f}p (spread {spread:.1f}p + commission {commission:.1f}p). "
        f"WR {win_rate_backtest:.1f}% vs BE net {be_net*100:.1f}%. Bon buffer.",
        evidence,
    )
