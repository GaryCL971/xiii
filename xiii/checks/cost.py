"""
xiii.checks.cost — section D of protocol (execution cost honesty).

D2_spread_breakeven: the backtest assumes a spread (SL/TP pips). XIII calculates
the true breakeven win-rate AFTER spread and commission, then compares to backtest WR.

Logic: Sharpe 1.5 on a backtest that assumes 0.5p spread, but you trade on
Vantage at 1.6p → your edge collapses. D2 screams danger.

Real flaw: "spread assumption 0.5p" vs "Vantage reality 1.6p on entry"
= breakeven WR goes from 33% to 37%.
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
    """Calculates the breakeven win-rate AFTER real broker spread.

    Inputs:
      - sl_pips, tp_pips: strategy levels (e.g., SL=15p TP=30p)
      - win_rate_backtest: WR from backtest (e.g., 52%)
      - broker: BrokerConfig (FTMO, Vantage, etc.)
      - symbol: instrument (default: GBPUSD)

    Logic:
      1. Calculates gross breakeven WR (before spread): SL / (SL + TP)
      2. Calculates real broker spread for this symbol
      3. Recalculates net breakeven WR: (SL + spread) / (SL + TP + spread)
      4. Compares backtest WR to net breakeven
    """
    _id = "D2_spread_breakeven"

    # Validate inputs
    if (sl_pips is None or tp_pips is None or win_rate_backtest is None
            or broker is None):
        return CheckResult(
            _id, "D", "SKIP",
            "Missing parameters for breakeven calculation",
            "Pass sl_pips, tp_pips, win_rate_backtest, broker=FTMO|VANTAGE. "
            "D2 measures spread cost on your strategy.",
        )

    try:
        cost = broker.cost(symbol)
    except KeyError:
        return CheckResult(
            _id, "D", "SKIP",
            f"Symbol '{symbol}' not configured for {broker.name}",
            f"Available: {list(broker.instruments)}. "
            "v0.1 covers GBPUSD/FTMO+Vantage. Others come in v1.0.",
        )

    # Calculations
    spread = cost.spread_pips
    commission = cost.commission_roundturn_usd / cost.pip_value_usd  # in pips
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
            "Net TP collapsed after spread",
            f"Backtest: TP={tp_pips}p. Spread+commission={total_cost:.1f}p. "
            f"Net TP = {tp_net:.1f}p (negative!). "
            f"Strategy cannot win: cost > profit target.",
            evidence,
        )

    if win_rate_backtest < be_net * 100:
        return CheckResult(
            _id, "D", "FAIL",
            f"Win-rate < breakeven (backtest {win_rate_backtest:.1f}% vs net BE {be_net*100:.1f}%)",
            f"Your backtest loses BEFORE real broker spread. "
            f"Either backtest is too optimistic, or there's a sizing error.",
            evidence,
        )

    if margin_pct < 5:
        return CheckResult(
            _id, "D", "WARN",
            f"Tight margin: {margin_pct:.0f}% above net breakeven",
            f"Backtest WR {win_rate_backtest:.1f}% vs BE {be_net*100:.1f}%. "
            f"Slippage of +0.2p or 1% WR drop → you hit breakeven. "
            f"Little margin for error.",
            evidence,
        )

    return CheckResult(
        _id, "D", "PASS",
        f"Spread cost acceptable: {margin_pct:.0f}% margin above net breakeven",
        f"Backtest assumes {total_cost:.1f}p (spread {spread:.1f}p + commission {commission:.1f}p). "
        f"WR {win_rate_backtest:.1f}% vs net BE {be_net*100:.1f}%. Good buffer.",
        evidence,
    )
