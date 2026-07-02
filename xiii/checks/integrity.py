"""
xiii.checks.integrity — sections A et G du protocole (analyse statique du source).

Ces trois checks lisent le CODE, pas les données : ils attrapent le mensonge
avant même qu'un chiffre soit calculé.

  A1_dummy_scan       : stand-ins factices dans le script qui produit LE chiffre.
                        (Faille réelle : la brique CPI du livrable était
                        np.random.normal(0.0005, 0.0008) — un générateur aléatoire
                        vendu comme un backtest.)
  A3_series_alignment : concat().dropna() / join="inner" qui effondre la fenêtre
                        quand on aligne une série sparse sur une série dense.
                        (Faille réelle : fenêtre tombée à 23 jours.)
  G1_lookahead_scan   : références au futur — .shift(-n), rolling(center=True).

Suppression volontaire : ajouter `# xiii: ok` en fin de ligne pour marquer un
usage légitime (ex. np.random dans un bootstrap Monte Carlo assumé).
"""
from __future__ import annotations

import re
from pathlib import Path

from ..report import CheckResult

_SUPPRESS = "xiii: ok"


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires de scan
# ─────────────────────────────────────────────────────────────────────────────

def _read_lines(source_file) -> list[str] | None:
    try:
        return Path(source_file).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None


def _scan_lines(lines: list[str], patterns: list[tuple[str, re.Pattern]]):
    """Scan ligne à ligne. -> [(lineno, extrait, nom_pattern)]"""
    hits = []
    for i, line in enumerate(lines, 1):
        if _SUPPRESS in line:
            continue
        for name, rx in patterns:
            if rx.search(line):
                hits.append((i, line.strip()[:80], name))
                break
    return hits


def _scan_window(lines: list[str], patterns: list[tuple[str, re.Pattern]],
                 window: int = 3, anchor: re.Pattern | None = None):
    """Scan sur fenêtre glissante de `window` lignes jointes (attrape le multi-ligne).
    Déduplique les fenêtres qui se chevauchent ; `anchor` désigne, dans la fenêtre,
    la ligne à laquelle attribuer le hit (ex. celle qui contient `concat`)."""
    hits = []
    last: dict[str, int] = {}
    for i in range(len(lines)):
        if _SUPPRESS in lines[i]:
            continue
        chunk = " ".join(l.strip() for l in lines[i:i + window])
        for name, rx in patterns:
            if rx.search(chunk):
                j = i  # ligne attribuée : celle de l'ancre si trouvée dans la fenêtre
                if anchor is not None:
                    for k in range(i, min(i + window, len(lines))):
                        if anchor.search(lines[k]):
                            j = k
                            break
                if name in last and j + 1 - last[name] < window:
                    last[name] = j + 1
                    continue
                last[name] = j + 1
                hits.append((j + 1, lines[j].strip()[:80], name))
    return hits


def _skip(check_id: str, section: str) -> CheckResult:
    return CheckResult(
        check_id, section, "SKIP",
        "Pas de code source fourni",
        "Passe source_file=<chemin du script qui produit le chiffre headline> "
        "pour activer ce check.",
    )


def _unreadable(check_id: str, section: str, source_file) -> CheckResult:
    return CheckResult(
        check_id, section, "SKIP",
        "Code source illisible",
        f"Impossible de lire {source_file!r}.",
    )


def _evidence(hits) -> dict:
    return {"hits": [{"line": ln, "code": code, "pattern": name} for ln, code, name in hits]}


# ─────────────────────────────────────────────────────────────────────────────
# A1 — stand-ins factices
# ─────────────────────────────────────────────────────────────────────────────

_A1_FAIL = [
    ("np.random", re.compile(r"\bnp\.random\.|\bnumpy\.random\.")),
    ("dummy", re.compile(r"dummy", re.I)),
    ("placeholder", re.compile(r"placeholder", re.I)),
    ("mock", re.compile(r"\bmock", re.I)),
]
_A1_WARN = [
    ("TODO", re.compile(r"\bTODO\b")),
    ("FIXME", re.compile(r"\bFIXME\b")),
]


def a1_dummy_scan(source_file=None) -> CheckResult:
    """§A : le chiffre headline vient-il de données RÉELLES ?"""
    _id = "A1_dummy_scan"
    if source_file is None:
        return _skip(_id, "A")
    lines = _read_lines(source_file)
    if lines is None:
        return _unreadable(_id, "A", source_file)

    hard = _scan_lines(lines, _A1_FAIL)
    soft = _scan_lines(lines, _A1_WARN)
    ev = {"hard": _evidence(hard)["hits"], "soft": _evidence(soft)["hits"]}

    if hard:
        ln, code, name = hard[0]
        return CheckResult(
            _id, "A", "FAIL",
            f"Stand-in factice détecté : {name} (l.{ln}" +
            (f", +{len(hard)-1} autres" if len(hard) > 1 else "") + ")",
            f"l.{ln}: `{code}`\n"
            "Le chiffre headline risque de sortir d'un générateur, pas du marché — "
            "c'est la faille de la brique CPI (np.random vendu comme backtest). "
            "Si l'usage est légitime (bootstrap Monte Carlo), marque la ligne "
            "`# xiii: ok` et relance.",
            ev,
        )
    if soft:
        ln, code, name = soft[0]
        return CheckResult(
            _id, "A", "WARN",
            f"{len(soft)} marqueur(s) de chantier ({name}…) dans le script du chiffre",
            f"l.{ln}: `{code}`\n"
            "Un TODO/FIXME dans le chemin qui produit LE chiffre = le livrable "
            "n'est pas fini. Résous ou justifie avant de sizer dessus.",
            ev,
        )
    return CheckResult(
        _id, "A", "PASS",
        "Aucun stand-in factice (np.random/dummy/mock/placeholder/TODO/FIXME)",
        "Le script du chiffre headline ne contient pas de données de substitution.",
        ev,
    )


# ─────────────────────────────────────────────────────────────────────────────
# A3 — alignement de séries (l'inner-join qui effondre la fenêtre)
# ─────────────────────────────────────────────────────────────────────────────

_A3_PATTERNS = [
    ("concat().dropna()", re.compile(r"concat\s*\([^)]*\)[^)]*\.\s*dropna\s*\(")),
    ("concat(join='inner')", re.compile(r"concat\s*\([^)]*join\s*=\s*['\"]inner['\"]")),
    ("merge().dropna()", re.compile(r"merge\s*\([^)]*\)[^)]*\.\s*dropna\s*\(")),
]


def a3_series_alignment(source_file=None) -> CheckResult:
    """§A : les séries sparse/denses sont-elles alignées sans effondrer la fenêtre ?"""
    _id = "A3_series_alignment"
    if source_file is None:
        return _skip(_id, "A")
    lines = _read_lines(source_file)
    if lines is None:
        return _unreadable(_id, "A", source_file)

    hits = _scan_window(lines, _A3_PATTERNS, window=3,
                        anchor=re.compile(r"concat|merge"))
    ev = _evidence(hits)

    if hits:
        ln, code, name = hits[0]
        return CheckResult(
            _id, "A", "WARN",
            f"Alignement suspect : {name} (l.{ln}" +
            (f", +{len(hits)-1} autres" if len(hits) > 1 else "") + ")",
            f"l.{ln}: `{code}`\n"
            "Si une série est sparse (ex. mensuelle) et l'autre dense (quotidienne), "
            "l'inner-join réduit la fenêtre à l'intersection — faille réelle : fenêtre "
            "tombée à 23 jours. Préférer `reindex(index_commun).fillna(0)` et vérifier "
            "len() avant/après l'alignement. Légitime ? Marque `# xiii: ok`.",
            ev,
        )
    return CheckResult(
        _id, "A", "PASS",
        "Aucun anti-pattern d'alignement (concat/merge + dropna, join inner)",
        "Pas d'effondrement de fenêtre détecté à l'alignement des séries.",
        ev,
    )


# ─────────────────────────────────────────────────────────────────────────────
# G1 — lookahead (le futur qui fuite dans le signal)
# ─────────────────────────────────────────────────────────────────────────────

_G1_PATTERNS = [
    (".shift(-n)", re.compile(r"\.shift\s*\(\s*-\d")),
    ("rolling(center=True)", re.compile(r"rolling\s*\([^)]*center\s*=\s*True")),
]


def g1_lookahead_scan(source_file=None) -> CheckResult:
    """§G : le signal de la barre N n'utilise-t-il QUE de l'information <= N ?"""
    _id = "G1_lookahead_scan"
    if source_file is None:
        return _skip(_id, "G")
    lines = _read_lines(source_file)
    if lines is None:
        return _unreadable(_id, "G", source_file)

    hits = _scan_lines(lines, _G1_PATTERNS)
    ev = _evidence(hits)

    if hits:
        ln, code, name = hits[0]
        return CheckResult(
            _id, "G", "FAIL",
            f"Lookahead probable : {name} (l.{ln}" +
            (f", +{len(hits)-1} autres" if len(hits) > 1 else "") + ")",
            f"l.{ln}: `{code}`\n"
            "`.shift(-n)` aligne le FUTUR sur la barre courante ; "
            "`rolling(center=True)` centre la fenêtre sur la barre (moitié future). "
            "Un backtest qui voit le futur est irréproduisible en live. "
            "Usage légitime (labels d'entraînement, etc.) ? Marque `# xiii: ok`.",
            ev,
        )
    return CheckResult(
        _id, "G", "PASS",
        "Aucune référence au futur détectée (.shift(-n), rolling centré)",
        "Le signal semble causal — à confirmer par relecture (le scan statique "
        "ne voit pas tous les canaux de fuite).",
        ev,
    )
