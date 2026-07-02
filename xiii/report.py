"""
xiii.report — format de verdict d'un audit XIII.

Un audit = une liste de CheckResult. Chaque check répond PASS / WARN / FAIL / SKIP.
Règle : un seul FAIL => l'audit échoue (`passed == False`). C'est un gate, pas un score.
"""
from __future__ import annotations

import json
import sys
import textwrap
from dataclasses import asdict, dataclass, field

STATUSES = ("PASS", "WARN", "FAIL", "SKIP")
_MARK = {"PASS": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}


@dataclass
class CheckResult:
    check_id: str
    section: str                    # "A".."H"
    status: str                     # PASS | WARN | FAIL | SKIP
    headline: str
    detail: str = ""
    evidence: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"status invalide: {self.status!r} (attendu {STATUSES})")


@dataclass
class AuditReport:
    verdicts: list[CheckResult] = field(default_factory=list)
    checks_total: int = 9           # nb de checks prévus au périmètre v0.1

    @property
    def passed(self) -> bool:
        """True si aucun FAIL. Un gate binaire, pas un score."""
        return not any(v.status == "FAIL" for v in self.verdicts)

    def _count(self, status: str) -> int:
        return sum(v.status == status for v in self.verdicts)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {
                "passed": self.passed,
                "summary": {s: self._count(s) for s in STATUSES},
                "checks_active": len(self.verdicts),
                "checks_total": self.checks_total,
                "verdicts": [asdict(v) for v in self.verdicts],
            },
            ensure_ascii=False,
            indent=indent,
        )

    def print(self, stream=None) -> None:
        stream = stream or sys.stdout
        # Robustesse console Windows (comme les scripts du labo)
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

        W = 64
        line = "=" * W
        sub = "-" * W
        out = [
            line,
            f"  XIII — Audit du Treizième Homme".ljust(W - 12) + "v0.1.0.dev0",
            line,
        ]
        # ordre d'affichage : FAIL puis WARN puis SKIP puis PASS
        order = {"FAIL": 0, "WARN": 1, "SKIP": 2, "PASS": 3}
        for v in sorted(self.verdicts, key=lambda x: order[x.status]):
            code = v.check_id.split("_", 1)[0]   # "B1_window_sensitivity" -> "B1"
            out.append(f"  {_MARK[v.status]} {code} · {v.headline}")
            for para in (v.detail or "").split("\n"):
                for wl in textwrap.wrap(para, W - 9) or [""]:
                    out.append(" " * 9 + wl)
            out.append("")
        out.append(sub)
        verdict = "NE PAS DÉPLOYER" if not self.passed else "OK (sous réserve des checks restants)"
        out.append(
            f"  Résultat : {self._count('FAIL')} FAIL · {self._count('WARN')} WARN · "
            f"{self._count('SKIP')} SKIP   ->  {verdict}"
        )
        out.append(
            f"  Checks actifs : {len(self.verdicts)}/{self.checks_total} (v0.1 en cours)"
        )
        out.append(line)
        stream.write("\n".join(out) + "\n")
