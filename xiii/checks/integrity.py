"""
xiii.checks.integrity — sections A and G of protocol (static source analysis).

These three checks read CODE, not data: they catch the lie
before any number is even calculated.

  A1_dummy_scan       : dummy stand-ins in the script that produces THE number.
                        (Real flaw: the CPI brick of the deliverable was
                        np.random.normal(0.0005, 0.0008) — a random number generator
                        sold as a backtest.)
  A3_series_alignment : concat().dropna() / join="inner" that collapses the window
                        when aligning a sparse series to a dense one.
                        (Real flaw: window collapsed to 23 days.)
  G1_lookahead_scan   : future references — .shift(-n), rolling(center=True).

Intentional suppression: add `# xiii: ok` at end of line to mark legitimate
usage (e.g., np.random in an intended Monte Carlo bootstrap).
"""
from __future__ import annotations

import re
from pathlib import Path

from ..report import CheckResult

_SUPPRESS = "xiii: ok"


# ─────────────────────────────────────────────────────────────────────────────
# Scan utilities
# ─────────────────────────────────────────────────────────────────────────────

def _read_lines(source_file) -> list[str] | None:
    try:
        return Path(source_file).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None


def _scan_lines(lines: list[str], patterns: list[tuple[str, re.Pattern]]):
    """Line-by-line scan. -> [(lineno, excerpt, pattern_name)]"""
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
    """Scan over sliding window of `window` joined lines (catches multi-line).
    Deduplicates overlapping windows; `anchor` designates, in the window,
    the line to attribute the hit to (e.g., the one containing `concat`)."""
    hits = []
    last: dict[str, int] = {}
    for i in range(len(lines)):
        if _SUPPRESS in lines[i]:
            continue
        chunk = " ".join(l.strip() for l in lines[i:i + window])
        for name, rx in patterns:
            if rx.search(chunk):
                j = i  # line attributed: anchor's line if found in window
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
        "No source code provided",
        "Pass source_file=<path to script that produces the headline number> "
        "to enable this check.",
    )


def _unreadable(check_id: str, section: str, source_file) -> CheckResult:
    return CheckResult(
        check_id, section, "SKIP",
        "Source code unreadable",
        f"Cannot read {source_file!r}.",
    )


def _evidence(hits) -> dict:
    return {"hits": [{"line": ln, "code": code, "pattern": name} for ln, code, name in hits]}


# ─────────────────────────────────────────────────────────────────────────────
# A1 — dummy stand-ins
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
    """§A : does the headline number come from REAL data?"""
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
            f"Dummy stand-in detected: {name} (l.{ln}" +
            (f", +{len(hard)-1} others" if len(hard) > 1 else "") + ")",
            f"l.{ln}: `{code}`\n"
            "The headline number risks coming from a generator, not the market — "
            "this is the flaw of the CPI brick (np.random sold as backtest). "
            "If the usage is legitimate (Monte Carlo bootstrap), mark the line "
            "`# xiii: ok` and rerun.",
            ev,
        )
    if soft:
        ln, code, name = soft[0]
        return CheckResult(
            _id, "A", "WARN",
            f"{len(soft)} work-in-progress marker(s) ({name}...) in the number script",
            f"l.{ln}: `{code}`\n"
            "A TODO/FIXME in the path that produces THE number = the deliverable "
            "is not finished. Resolve or justify before sizing on it.",
            ev,
        )
    return CheckResult(
        _id, "A", "PASS",
        "No dummy stand-ins (np.random/dummy/mock/placeholder/TODO/FIXME)",
        "The headline number script contains no substitute data.",
        ev,
    )


# ─────────────────────────────────────────────────────────────────────────────
# A3 — series alignment (the inner-join that collapses the window)
# ─────────────────────────────────────────────────────────────────────────────

_A3_PATTERNS = [
    ("concat().dropna()", re.compile(r"concat\s*\([^)]*\)[^)]*\.\s*dropna\s*\(")),
    ("concat(join='inner')", re.compile(r"concat\s*\([^)]*join\s*=\s*['\"]inner['\"]")),
    ("merge().dropna()", re.compile(r"merge\s*\([^)]*\)[^)]*\.\s*dropna\s*\(")),
]


def a3_series_alignment(source_file=None) -> CheckResult:
    """§A : are sparse/dense series aligned without collapsing the window?"""
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
            f"Suspicious alignment: {name} (l.{ln}" +
            (f", +{len(hits)-1} others" if len(hits) > 1 else "") + ")",
            f"l.{ln}: `{code}`\n"
            "If one series is sparse (e.g., monthly) and the other dense (daily), "
            "the inner-join reduces the window to the intersection — real flaw: window "
            "collapsed to 23 days. Prefer `reindex(common_index).fillna(0)` and verify "
            "len() before/after alignment. Legitimate? Mark `# xiii: ok`.",
            ev,
        )
    return CheckResult(
        _id, "A", "PASS",
        "No alignment anti-patterns (concat/merge + dropna, join inner)",
        "No window collapse detected at series alignment.",
        ev,
    )


# ─────────────────────────────────────────────────────────────────────────────
# G1 — lookahead (future that leaks into the signal)
# ─────────────────────────────────────────────────────────────────────────────

_G1_PATTERNS = [
    (".shift(-n)", re.compile(r"\.shift\s*\(\s*-\d")),
    ("rolling(center=True)", re.compile(r"rolling\s*\([^)]*center\s*=\s*True")),
]


def g1_lookahead_scan(source_file=None) -> CheckResult:
    """§G : does the signal at bar N use ONLY information <= N?"""
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
            f"Probable lookahead: {name} (l.{ln}" +
            (f", +{len(hits)-1} others" if len(hits) > 1 else "") + ")",
            f"l.{ln}: `{code}`\n"
            "`.shift(-n)` aligns the FUTURE onto the current bar; "
            "`rolling(center=True)` centers the window on the bar (half future). "
            "A backtest that sees the future is irreproducible live. "
            "Legitimate usage (training labels, etc.)? Mark `# xiii: ok`.",
            ev,
        )
    return CheckResult(
        _id, "G", "PASS",
        "No future references detected (.shift(-n), centered rolling)",
        "The signal appears causal — to be confirmed by code review (static scan "
        "does not see all leakage channels).",
        ev,
    )
