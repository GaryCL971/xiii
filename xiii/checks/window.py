"""
xiii.checks.window — section B du protocole (honnêteté de la fenêtre).

B1_window_sensitivity : LE check qui rattrape le mirage.
Recalcule le Sharpe sur le plein historique vs des sous-fenêtres récentes.
Si une fenêtre courte et favorable gonfle le Sharpe, le chiffre "vendable"
est un artefact de fenêtre — exactement la faille Sharpe 2.53 -> honnête 1.22.

Logique de référence : audit 13e homme du 2026-06-21
("2.3 ans 2023-2025 gonflaient le Sharpe de +49% vs 2021-2025").

Entrée : une série de rendements ~quotidiens (pd.Series). B1 raisonne en nombre
d'observations (≈252/an) ; le support des séries irrégulières viendra plus tard.
"""
from __future__ import annotations

import pandas as pd

from ..metrics import ANN, sharpe
from ..report import CheckResult

_ID = "B1_window_sensitivity"


def _yr(y: int) -> str:
    return "1 an" if y == 1 else f"{y} ans"


def b1_window_sensitivity(
    returns: pd.Series | None,
    windows_years: tuple[int, ...] = (1, 2, 3, 5),
    warn_pct: float = 20.0,
    fail_pct: float = 50.0,
    min_obs: int = 252,
) -> CheckResult:
    """Détecte un Sharpe gonflé par une sous-fenêtre récente favorable.

    warn_pct / fail_pct : seuils de gonflement (%) du Sharpe d'une sous-fenêtre
    vs le plein historique. 2.53/1.22 = +107% -> FAIL ; +49% -> WARN.
    """
    if returns is None or len(returns.dropna()) < min_obs:
        n = 0 if returns is None else len(returns.dropna())
        return CheckResult(
            _ID, "B", "SKIP",
            "Historique insuffisant pour tester la sensibilité de fenêtre",
            f"Requiert >= {min_obs} points datés (~1 an quotidien) ; reçu {n}.",
        )

    r = returns.dropna()
    n = len(r)
    sh_full = sharpe(r)
    years_full = round(n / ANN, 1)

    windows = {"full": {"years": years_full, "sharpe": round(sh_full, 3)}}
    worst_infl = 0.0          # gonflement max (%) d'une sous-fenêtre
    worst_y: int | None = None
    worst_regime = False      # True si full<=0 mais fenêtre courte>0 (edge de régime)

    for y in windows_years:
        w = int(y * ANN)
        if w >= n or w < min_obs:
            continue
        sh_w = sharpe(r.iloc[-w:])
        windows[f"last_{y}y"] = {"years": y, "sharpe": round(sh_w, 3)}
        if sh_full > 0:
            infl = (sh_w / sh_full - 1.0) * 100.0
        else:
            # plein historique non rentable : toute fenêtre positive est un edge de régime
            infl = float("inf") if sh_w > 0 else 0.0
        if infl > worst_infl:
            worst_infl = infl
            worst_y = y
            worst_regime = sh_full <= 0 and sh_w > 0

    evidence = {
        "sharpe_full": round(sh_full, 3),
        "years_full": years_full,
        "windows": windows,
        "worst_window_years": worst_y,
        "inflation_pct": None if worst_y is None else (
            "inf" if worst_infl == float("inf") else round(worst_infl, 1)
        ),
        "thresholds": {"warn_pct": warn_pct, "fail_pct": fail_pct},
    }

    if worst_y is None:
        return CheckResult(
            _ID, "B", "PASS",
            "Fenêtre stable : aucune sous-fenêtre courte plus flatteuse",
            f"Sharpe plein historique {sh_full:.2f} sur {years_full} ans ; "
            f"les sous-fenêtres testées ne le dépassent pas.",
            evidence,
        )

    if worst_regime:
        return CheckResult(
            _ID, "B", "FAIL",
            f"Mirage : la stratégie ne 'marche' que sur la période récente ({_yr(worst_y)})",
            f"Sharpe plein historique {sh_full:.2f} (<= 0), mais positif sur la "
            f"fenêtre {_yr(worst_y)}. C'est un edge de RÉGIME, pas un edge robuste : "
            f"il disparaît hors de sa fenêtre favorable.",
            evidence,
        )

    sh_short = windows[f"last_{worst_y}y"]["sharpe"]
    if worst_infl >= fail_pct:
        status, verdict = "FAIL", "mirage"
    elif worst_infl >= warn_pct:
        status, verdict = "WARN", "à surveiller"
    else:
        return CheckResult(
            _ID, "B", "PASS",
            f"Fenêtre honnête : gonflement max +{worst_infl:.0f}% (< {warn_pct:.0f}%)",
            f"Sharpe plein {sh_full:.2f} ; meilleure sous-fenêtre ({_yr(worst_y)}) "
            f"{sh_short:.2f}. Écart dans le bruit.",
            evidence,
        )

    return CheckResult(
        _ID, "B", status,
        f"Sharpe gonflé de +{worst_infl:.0f}% par la fenêtre {_yr(worst_y)} ({verdict})",
        f"Plein historique ({years_full} ans) : Sharpe {sh_full:.2f}. "
        f"Fenêtre {_yr(worst_y)} : Sharpe {sh_short:.2f}. "
        f"Le chiffre 'vendable' vient de la fenêtre favorable — c'est la faille "
        f"type 2.53 -> 1.22. Sizer et décider sur {sh_full:.2f}, pas sur {sh_short:.2f}.",
        evidence,
    )
