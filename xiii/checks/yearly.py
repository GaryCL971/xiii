"""
xiii.checks.yearly — section B du protocole (honnêteté année par année).

B3_yearly_breakdown : détecte les brèches annuelles (années négatives, breaches
de drawdown) qui révèlent un edge fragile ou régime-dépendant.

Logique : si une stratégie a 16 ans de données mais 2 années sont catastrophiques,
le Sharpe moyen cache le vrai risque.
"""
from __future__ import annotations

import pandas as pd

from ..metrics import ANN, max_drawdown, sharpe, annualized_return
from ..report import CheckResult

_ID = "B3_yearly_breakdown"


def b3_yearly_breakdown(
    returns: pd.Series | None,
    min_obs_per_year: int = 50,
) -> CheckResult:
    """Décompte année par année. Signale chaque année négative / chaque breach.

    Détecte : une stratégie « rentable en moyenne » mais cassée certaines années.
    Typique du mirage : bon sur 2023-2025, catastrophique sur 2015.
    """
    _id = "B3_yearly_breakdown"
    if returns is None or len(returns.dropna()) < min_obs_per_year:
        n = 0 if returns is None else len(returns.dropna())
        return CheckResult(
            _id, "B", "SKIP",
            "Historique insuffisant pour décompte année par année",
            f"Requiert >= {min_obs_per_year} points (~3 mois quotidien) ; reçu {n}.",
        )

    r = returns.dropna()
    if not isinstance(r.index, pd.DatetimeIndex):
        return CheckResult(
            _id, "B", "SKIP",
            "Index non daté",
            "Besoin d'un index datetime pour grouper par année. "
            "Passe `equity` plutôt que `returns` si l'index est déduit d'une courbe.",
        )

    yearly = {}
    bad_years = []  # années négatives ou breaches
    for year, group in r.groupby(r.index.year):
        if len(group) < min_obs_per_year:
            continue
        sh = sharpe(group)
        ann_ret = annualized_return(group)
        dd = max_drawdown(group)
        yearly[year] = {
            "trades": len(group),
            "sharpe": round(sh, 2),
            "ret_ann": round(ann_ret * 100, 1),
            "maxdd": round(dd * 100, 1),
        }
        if ann_ret < 0 or dd < -0.10:
            bad_years.append(
                (year, ann_ret, dd,
                 "negative return" if ann_ret < 0 else "DD < -10%")
            )

    evidence = {"yearly": yearly, "bad_years_count": len(bad_years)}

    if not yearly:
        return CheckResult(
            _id, "B", "SKIP",
            "Pas d'année complète",
            "Chaque année doit avoir >= {min_obs_per_year} points.",
        )

    if bad_years:
        summary = []
        for year, ann_ret, dd, reason in bad_years[:2]:
            summary.append(f"{year}: {reason} (return {ann_ret*100:+.1f}%, DD {dd*100:.1f}%)")
        detail = ", ".join(summary)
        if len(bad_years) > 2:
            detail += f", +{len(bad_years)-2} autres"

        status = "FAIL" if len(bad_years) >= 2 else "WARN"
        return CheckResult(
            _id, "B", status,
            f"{len(bad_years)} année(s) problématique(s) : {detail}",
            f"Une stratégie 'robuste' ne devrait pas avoir d'années en perte totale. "
            f"2+ mauvaises années → edge fragile ou régime-dépendant. "
            f"Investiguer : est-ce une crise de marché (2008, 2020) ou un vrai problème du signal?",
            evidence,
        )

    return CheckResult(
        _id, "B", "PASS",
        f"Décompte année par année : toutes positives, maxDD acceptable",
        f"{len(yearly)} années analysées, toutes avec Sharpe >= 0 et DD >= -10%. "
        f"Lire le détail pour identifier les années faibles (< Sharpe 1.0).",
        evidence,
    )
