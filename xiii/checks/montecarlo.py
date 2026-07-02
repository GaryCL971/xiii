"""
xiii.checks.montecarlo — section C du protocole (Monte Carlo des contraintes broker).

C2_montecarlo_constraints : bootstrap de la séquence de trades réelle pour valider
probabilités sous les contraintes du broker (FTMO : pass challenge, respirer max DD,
max daily loss, etc.).

Logique : un backtest 50 trades / 4 mois peut passer le portfolio en moyenne, mais
un ordre de magnitude faible du tirage (50 permutations) peut faire bugger les limites.
Monte Carlo simule 1000 trajectoires possibles avec la même séquence de P&L, révèle
le pire cas.

NB : v0.1 = version light (pas de prise en compte des autocorrélations dans l'ordre).
Full = pattern-aware shuffling pour crise.
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
    """Bootstrap Monte Carlo des contraintes broker.

    Entrées :
      - trades : DataFrame avec colonnes [datetime, pnl_usd] (au moins)
      - broker : BrokerConfig
      - deployed_sizing : multiplicateur (ex: 1.08)
      - n_sims : nombre de trajectoires à simuler (défaut 1000)

    Logique :
      1. Charge la séquence de P&L réels
      2. Ré-échantillonne (permute) N fois
      3. Pour chaque trajet, vérifie:
         - maxDD (FTMO: <= -10%)
         - maxDD quotidien (FTMO: <= -5%)
         - profit cumulé >= objectif (FTMO: +10%)
      4. Rapporte P(pass challenge), P(breach DD), P(breach daily)
    """
    _id = "C2_montecarlo_constraints"

    if trades is None or len(trades) < 10 or broker is None:
        n = 0 if trades is None else len(trades)
        return CheckResult(
            _id, "C", "SKIP",
            "Données trades insuffisantes",
            f"Besoin >= 10 trades ; reçu {n}. "
            "C2 simule 1000 réordonnances pour tester robustesse vs limites broker.",
        )

    if not broker.has_risk_rules:
        return CheckResult(
            _id, "C", "SKIP",
            f"Broker '{broker.name}' n'a pas de règles de risque",
            "Vantage n'a pas de limites FTMO. C2 s'applique à FTMO.",
        )

    # Valider que trades a une colonne pnl ou similaire
    pnl_col = None
    for col in ["pnl_usd", "pnl", "profit"]:
        if col in trades.columns:
            pnl_col = col
            break
    if pnl_col is None:
        return CheckResult(
            _id, "C", "SKIP",
            "Colonne P&L non trouvée",
            "Trades doit contenir pnl_usd, pnl, ou profit.",
        )

    pnl = trades[pnl_col].values * deployed_sizing
    n_trades = len(pnl)

    # Paramètres FTMO
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
        # Permutation aléatoire de la séquence P&L
        shuffled = rng.permutation(pnl)

        # Trajectoire cumulée
        equity = initial_capital + np.cumsum(shuffled)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / initial_capital
        total_dd = dd.min()

        # Daily max loss (simulation simplifiée : pire trade en un jour)
        daily_dd = pnl.min() / initial_capital

        # Profit final
        final_pnl = shuffled.sum()

        # Vérify contraintes
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

    # Probabilités
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
            f"Monte Carlo : P(pass challenge) = {p_pass:.1%} (< 50%)",
            f"Sur 1000 réordonnances, seules {results['pass_challenge']} passent FTMO. "
            f"Votre séquence de trades est trop sensible à l'ordre. "
            f"Risque : si le marché perm favorise une mauvaise séquence, breach probable.",
            evidence,
        )

    if p_breach_total_dd > 0.2:
        return CheckResult(
            _id, "C", "WARN",
            f"Monte Carlo : P(breach maxDD) = {p_breach_total_dd:.1%}",
            f"{int(results['breach_total_dd'])} / {n_sims} runs dépassent -10%. "
            f"Un risque significatif : la séquence compte.",
            evidence,
        )

    return CheckResult(
        _id, "C", "PASS",
        f"Monte Carlo robuste : P(pass challenge) = {p_pass:.1%}",
        f"{results['pass_challenge']} / {n_sims} trajectoires passent FTMO. "
        f"Votre séquence de trades est robuste aux réordonnances.",
        evidence,
    )
