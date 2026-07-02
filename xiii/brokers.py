"""
xiii.brokers — profils broker pour XIII v0.1 (périmètre FTMO + Vantage).

Un broker impose deux familles de contraintes que le Treizième Homme vérifie :
  1. des RÈGLES DE COMPTE (prop-firm : limites de drawdown)  -> checks C1, C2
  2. des COÛTS D'EXÉCUTION (spread, commission, slippage)     -> checks D2

Niveaux de confiance des chiffres :
  [MESURÉ]      = tiré des ticks réels (spread_cost_macroedge.py)
  [RÈGLE]       = règle officielle du broker/prop-firm
  [PLACEHOLDER] = à figer avant le lancement public (voir SPEC §4)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InstrumentCost:
    """Coûts d'exécution d'un instrument chez un broker donné."""
    symbol: str
    pip: float                        # taille d'1 pip en prix (GBPUSD = 0.0001)
    pip_value_usd: float              # valeur d'1 pip / lot standard, en USD (GBPUSD = 10.0)
    spread_pips: float                # spread médian de la fenêtre d'entrée
    spread_pips_p95: float            # spread mauvais-jour (queue haute)
    commission_roundturn_usd: float   # commission aller-retour / lot standard (0 si compte Standard)


@dataclass(frozen=True)
class BrokerConfig:
    """Profil complet d'un broker : règles de compte + coûts par instrument."""
    name: str
    kind: str                                   # "prop_firm" | "retail_broker"
    # --- règles de compte (None si non applicable, ex. compte réel sans limite) ---
    max_total_drawdown: float | None            # ex FTMO -0.10 (fraction du solde initial)
    max_daily_loss: float | None                # ex FTMO -0.05
    profit_target: float | None                 # ex FTMO +0.10 (objectif challenge)
    min_trading_days: int | None
    # --- exécution ---
    default_slippage_pips: float
    instruments: dict[str, InstrumentCost] = field(default_factory=dict)

    @property
    def has_risk_rules(self) -> bool:
        """True pour une prop-firm (contraintes DD), False pour un compte réel libre."""
        return self.max_total_drawdown is not None

    def cost(self, symbol: str) -> InstrumentCost:
        if symbol not in self.instruments:
            raise KeyError(
                f"{self.name} : instrument '{symbol}' non paramétré en v0.1 "
                f"(disponibles : {list(self.instruments)}). "
                "GBPUSD est l'instrument-pilote ; les autres arrivent en v1.0."
            )
        return self.instruments[symbol]


# ─────────────────────────────────────────────────────────────────────────────
# GBPUSD — instrument-pilote v0.1
# pip = 0.0001, pip_value = 10 USD/lot (paire en USD quote, lot standard 100k)
# ─────────────────────────────────────────────────────────────────────────────

_VANTAGE_GBPUSD = InstrumentCost(
    symbol="GBPUSD",
    pip=0.0001,
    pip_value_usd=10.0,
    spread_pips=1.6,          # [MESURÉ] fenêtre d'entrée 07-08h UTC, GBPUSD_ticks_30d
    spread_pips_p95=3.0,      # [PLACEHOLDER] queue mauvais-jour, à figer sur les ticks
    commission_roundturn_usd=6.0,  # [PLACEHOLDER] ~3$/side RAW ECN ; 0 si compte Standard
)

_FTMO_GBPUSD = InstrumentCost(
    symbol="GBPUSD",
    pip=0.0001,
    pip_value_usd=10.0,
    spread_pips=1.0,          # [PLACEHOLDER] FTMO ≈ raw + commission, à mesurer
    spread_pips_p95=2.5,      # [PLACEHOLDER]
    commission_roundturn_usd=6.0,  # [PLACEHOLDER]
)


VANTAGE = BrokerConfig(
    name="Vantage (RAW ECN, compte réel)",
    kind="retail_broker",
    max_total_drawdown=None,   # compte réel : aucune règle DD imposée
    max_daily_loss=None,
    profit_target=None,
    min_trading_days=None,
    default_slippage_pips=0.5,   # [PLACEHOLDER]
    instruments={"GBPUSD": _VANTAGE_GBPUSD},
)

FTMO = BrokerConfig(
    name="FTMO Challenge",
    kind="prop_firm",
    max_total_drawdown=-0.10,   # [RÈGLE] breach dur
    max_daily_loss=-0.05,       # [RÈGLE] breach dur
    profit_target=0.10,         # [RÈGLE] objectif phase 1 (phase 2 = +0.05)
    min_trading_days=4,         # [À CONFIRMER] varie selon version
    default_slippage_pips=0.5,  # [PLACEHOLDER]
    instruments={"GBPUSD": _FTMO_GBPUSD},
)


# Registre pour la CLI (`--broker ftmo`)
REGISTRY: dict[str, BrokerConfig] = {
    "ftmo": FTMO,
    "vantage": VANTAGE,
}
