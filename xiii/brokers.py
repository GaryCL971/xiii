"""
xiii.brokers — broker profiles for XIII v0.1 (scope: FTMO + Vantage).

A broker imposes two families of constraints that the Thirteenth Man validates:
  1. ACCOUNT RULES (prop-firm: drawdown limits)     -> checks C1, C2
  2. EXECUTION COSTS (spread, commission, slippage) -> checks D2

Confidence levels of figures:
  [MEASURED]    = from real ticks (spread_cost_macroedge.py)
  [RULE]        = official broker/prop-firm rule
  [PLACEHOLDER] = to be finalized before public launch (see SPEC §4)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InstrumentCost:
    """Execution costs for an instrument at a given broker."""
    symbol: str
    pip: float                        # size of 1 pip in price (GBPUSD = 0.0001)
    pip_value_usd: float              # value of 1 pip per standard lot in USD (GBPUSD = 10.0)
    spread_pips: float                # median spread at entry window
    spread_pips_p95: float            # worst-day spread (tail risk)
    commission_roundturn_usd: float   # round-trip commission per lot (0 if Standard account)


@dataclass(frozen=True)
class BrokerConfig:
    """Complete broker profile: account rules + costs per instrument."""
    name: str
    kind: str                                   # "prop_firm" | "retail_broker"
    # --- account rules (None if not applicable, e.g. real account with no limit) ---
    max_total_drawdown: float | None            # e.g. FTMO -0.10 (fraction of initial balance)
    max_daily_loss: float | None                # e.g. FTMO -0.05
    profit_target: float | None                 # e.g. FTMO +0.10 (challenge objective)
    min_trading_days: int | None
    # --- execution ---
    default_slippage_pips: float
    instruments: dict[str, InstrumentCost] = field(default_factory=dict)

    @property
    def has_risk_rules(self) -> bool:
        """True for prop-firm (DD constraints), False for unrestricted real account."""
        return self.max_total_drawdown is not None

    def cost(self, symbol: str) -> InstrumentCost:
        if symbol not in self.instruments:
            raise KeyError(
                f"{self.name}: instrument '{symbol}' not configured in v0.1 "
                f"(available: {list(self.instruments)}). "
                "GBPUSD is the pilot instrument; others arrive in v1.0."
            )
        return self.instruments[symbol]


# ─────────────────────────────────────────────────────────────────────────────
# GBPUSD — pilot instrument v0.1
# pip = 0.0001, pip_value = 10 USD/lot (pair quoted in USD, standard lot 100k)
# ─────────────────────────────────────────────────────────────────────────────

_VANTAGE_GBPUSD = InstrumentCost(
    symbol="GBPUSD",
    pip=0.0001,
    pip_value_usd=10.0,
    spread_pips=1.6,          # [MEASURED] entry window 07-08h UTC, GBPUSD_ticks_30d
    spread_pips_p95=3.0,      # [PLACEHOLDER] worst-day tail, to be finalized on ticks
    commission_roundturn_usd=6.0,  # [PLACEHOLDER] ~3$/side RAW ECN; 0 if Standard account
)

_FTMO_GBPUSD = InstrumentCost(
    symbol="GBPUSD",
    pip=0.0001,
    pip_value_usd=10.0,
    spread_pips=1.0,          # [PLACEHOLDER] FTMO ≈ raw + commission, to be measured
    spread_pips_p95=2.5,      # [PLACEHOLDER]
    commission_roundturn_usd=6.0,  # [PLACEHOLDER]
)


VANTAGE = BrokerConfig(
    name="Vantage (RAW ECN, real account)",
    kind="retail_broker",
    max_total_drawdown=None,   # real account: no DD rules imposed
    max_daily_loss=None,
    profit_target=None,
    min_trading_days=None,
    default_slippage_pips=0.5,   # [PLACEHOLDER]
    instruments={"GBPUSD": _VANTAGE_GBPUSD},
)

FTMO = BrokerConfig(
    name="FTMO Challenge",
    kind="prop_firm",
    max_total_drawdown=-0.10,   # [RULE] hard breach limit
    max_daily_loss=-0.05,       # [RULE] hard breach limit
    profit_target=0.10,         # [RULE] phase 1 target (phase 2 = +0.05)
    min_trading_days=4,         # [TO CONFIRM] varies by version
    default_slippage_pips=0.5,  # [PLACEHOLDER]
    instruments={"GBPUSD": _FTMO_GBPUSD},
)


# Registry for CLI (`--broker ftmo`)
REGISTRY: dict[str, BrokerConfig] = {
    "ftmo": FTMO,
    "vantage": VANTAGE,
}
