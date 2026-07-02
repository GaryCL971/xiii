"""
xiii.checks.sizing — section C du protocole (honnêteté du sizing).

C1_sizing_vs_maxdd : le check CRITIQUE du Treizième Homme.

La faille réelle : « mon backtest maxDD = -2.6 %, à 1.08x je passe sous -10% FTMO ».
Reason : le backtest mesurait sur une fenêtre courte ; le maxDD réel sur 16 ans = -10.9 %.
À 1.08×, tu touches -11.8 % → breach.

C1 prend le sizing déployé et valide qu'il respecte les limites du broker.
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
    """Valide que le sizing déployé respecte les limites de drawdown du broker.

    Entrées :
      - equity : courbe d'équité (rendements déduits)
      - broker : BrokerConfig (FTMO, Vantage, etc.)
      - deployed_sizing : multiplicateur actuellement déployé (ex: 1.08×)

    Logique :
      1. Reconstruit les rendements depuis equity
      2. Calcule le maxDD réel de la courbe
      3. Applique le sizing : maxDD * deployed_sizing
      4. Valide vs limites broker (FTMO: -10%, Vantage: n/a)
    """
    _id = "C1_sizing_vs_maxdd"

    if equity is None or broker is None:
        return CheckResult(
            _id, "C", "SKIP",
            "Entrées manquantes",
            "Passe equity (courbe d'équité) + broker (FTMO|VANTAGE). "
            "C1 valide que le sizing respecte les limites de drawdown.",
        )

    if not broker.has_risk_rules:
        return CheckResult(
            _id, "C", "SKIP",
            f"Broker '{broker.name}' n'a pas de règles de risque",
            "Vantage n'impose pas de limite DD (compte réel libre). "
            "FTMO impose -10% → C1 s'applique.",
        )

    # Reconstruire les rendements
    equity_clean = equity.dropna().astype(float)
    if len(equity_clean) < 252:
        return CheckResult(
            _id, "C", "SKIP",
            "Historique équité insuffisant",
            f"Besoin >= 1 an (~252 jours) ; reçu {len(equity_clean)}.",
        )

    returns = equity_clean.pct_change().dropna()

    # Mesurer maxDD réel
    dd_base = max_drawdown(returns)
    dd_sized = dd_base * deployed_sizing

    evidence = {
        "dd_base_pct": round(dd_base * 100, 1),
        "deployed_sizing": deployed_sizing,
        "dd_after_sizing_pct": round(dd_sized * 100, 1),
        "broker_max_dd_limit_pct": broker.max_total_drawdown * 100,
    }

    limit = broker.max_total_drawdown
    margin = dd_sized - limit  # marge (négatif = safe, positif = breach)

    if dd_sized >= limit:
        return CheckResult(
            _id, "C", "FAIL",
            f"Sizing dépasserait la limite {broker.name}",
            f"Base maxDD: {dd_base*100:.1f}%. "
            f"Après sizing ×{deployed_sizing}: {dd_sized*100:.1f}%. "
            f"Limite {broker.name}: {limit*100:.1f}%. "
            f"→ BREACH DE {margin*100:.1f} pp. Réduire le sizing ou l'edge.",
            evidence,
        )

    if margin > -0.02:  # moins de 2 pp de marge
        return CheckResult(
            _id, "C", "WARN",
            f"Marge DD serrée : {-margin*100:.1f} pp sous la limite",
            f"Un glissement du maxDD de +{-margin*100:.1f} pp → breach. "
            f"Peu de buffer. À {-margin*100:.1f} pp près du falaise.",
            evidence,
        )

    return CheckResult(
        _id, "C", "PASS",
        f"Sizing conforme : {-margin*100:.1f} pp de marge vs limite {broker.name}",
        f"Base maxDD {dd_base*100:.1f}%, "
        f"après sizing ×{deployed_sizing} → {dd_sized*100:.1f}%. "
        f"Limite: {limit*100:.1f}%. Sûr.",
        evidence,
    )
