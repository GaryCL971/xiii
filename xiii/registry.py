"""
xiii.registry — a queryable ledger of falsification experiments.

The Thirteenth Man kills the large majority of hypotheses. That graveyard is
an asset: every dead experiment is a lesson you already paid for. This module
turns a pile of prose post-mortems into a structured, queryable registry — so
you can ask "every experiment KILLED on XAUUSD" or "the Sharpe distribution of
the graveyard" instead of re-reading dozens of notes by hand.

Design — one source of truth, not two:
  * One ExperimentRecord per experiment, one JSONL file.
  * The registry is a PROJECTION. `source_memory` backlinks each row to the
    narrative post-mortem it was distilled from. Structured truth lives here;
    narrative truth stays in the note it points at. The registry never tries to
    be the story — only the index of it.
  * `verify()` is the Thirteenth Man turned on the ledger itself: it refuses to
    hold un-auditable records (a KILL with no reason, a PASS with no metrics,
    a dangling backlink).

Zero dependencies beyond the stdlib (pandas only for `to_dataframe`), matching
the rest of the package. dataclasses, not Pydantic.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# Outcome taxonomy. A gate vocabulary, not a score.
STATUSES = (
    "IN_PROGRESS",           # running, no verdict yet
    "PASSED",                # cleared the checks, not (yet) deployed
    "DEPLOYED",              # live (demo or real capital)
    "KILLED_BY_13TH_MAN",    # falsified — the common case
    "INCONCLUSIVE",          # not refuted, not confirmed (n too small, etc.)
    "PARKED",                # promising but shelved; may be re-opened
)

# A performance claim MUST carry metrics, or the record is not auditable.
_NEEDS_METRICS = ("PASSED", "DEPLOYED")

_ID_RE = re.compile(r"^exp[\s\-_]*0*(\d+)$", re.IGNORECASE)


def normalize_id(raw: str) -> str:
    """"exp51", "EXP-51", "Exp 051" -> "EXP-051". Non-exp ids pass through trimmed.

    This normalizes an *identifier string*, never prose — regex here is safe.
    """
    s = str(raw).strip()
    m = _ID_RE.match(s)
    if m:
        return f"EXP-{int(m.group(1)):03d}"
    return s


@dataclass
class ExperimentRecord:
    """One falsification experiment, distilled to structured fields.

    id            canonical id, e.g. "EXP-051" (see `normalize_id`).
    metrics       free-form {name: value}, e.g. {"sharpe": 0.50, "maxdd": -0.33}.
    deployed      convenience mirror: did this reach live capital?
    fail_reason   required when status is KILLED_BY_13TH_MAN.
    source_memory backlink slug to the narrative post-mortem (the `name:` of the
                  native memory card). The anti-drift tether — structured truth
                  here, narrative truth there.
    origins       who/what suggested the hypothesis (e.g. an assistant, a
                  research partner, an analyst). A list — threads are often
                  crossed. Enables scoring how deployable each source's ideas are.
    """
    id: str
    title: str
    status: str
    date: str = ""                                  # ISO close date, "" if open
    assets: list[str] = field(default_factory=list)
    theme: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    deployed: bool = False
    fail_reason: str | None = None
    source_memory: str = ""
    origins: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self):
        self.id = normalize_id(self.id)
        if self.status not in STATUSES:
            raise ValueError(f"invalid status: {self.status!r} (expected one of {STATUSES})")
        self.assets = [a.strip() for a in self.assets if a and str(a).strip()]
        self.origins = [o.strip() for o in self.origins if o and str(o).strip()]


@dataclass
class VerifyIssue:
    """One integrity problem found by ExperimentRegistry.verify()."""
    level: str          # "FAIL" | "WARN"
    record_id: str
    message: str


_FIELD_NAMES = {f.name for f in fields(ExperimentRecord)}


def _coerce(record) -> ExperimentRecord:
    """Accept an ExperimentRecord or a dict; ignore unknown keys (forward-compat)."""
    if isinstance(record, ExperimentRecord):
        return record
    return ExperimentRecord(**{k: v for k, v in dict(record).items() if k in _FIELD_NAMES})


class ExperimentRegistry:
    """An in-memory, queryable collection of ExperimentRecord, backed by JSONL.

    Dedup is by canonical `id` (last write wins), so re-seeding is idempotent.
    """

    def __init__(self, records=None):
        self._records: dict[str, ExperimentRecord] = {}
        for r in (records or []):
            self.add(r)

    # ---- persistence -------------------------------------------------------
    @classmethod
    def load(cls, path) -> "ExperimentRegistry":
        """Load a JSONL ledger. Missing file => empty registry (not an error)."""
        path = Path(path)
        reg = cls()
        if not path.exists():
            return reg
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{lineno}: bad JSON: {e}") from e
                reg.add(d)
        return reg

    def save(self, path) -> None:
        """Write the ledger as JSONL (one record per line), sorted by id."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in sorted(self.records, key=lambda x: x.id):
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # ---- mutation ----------------------------------------------------------
    def add(self, record) -> ExperimentRecord:
        rec = _coerce(record)
        self._records[rec.id] = rec         # dedup by id, last write wins
        return rec

    # ---- access ------------------------------------------------------------
    @property
    def records(self) -> list[ExperimentRecord]:
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self.records)

    def get(self, id) -> "ExperimentRecord | None":
        return self._records.get(normalize_id(id))

    # ---- query -------------------------------------------------------------
    def query(self, status=None, asset=None, theme=None, deployed=None, origin=None) -> list[ExperimentRecord]:
        """Filter records. All criteria are ANDed; None means 'don't care'."""
        out = self.records
        if status is not None:
            out = [r for r in out if r.status == status]
        if asset is not None:
            a = asset.strip().upper()
            out = [r for r in out if any(x.upper() == a for x in r.assets)]
        if theme is not None:
            t = theme.strip().lower()
            out = [r for r in out if t in r.theme.lower()]
        if deployed is not None:
            out = [r for r in out if r.deployed is bool(deployed)]
        if origin is not None:
            o = origin.strip().lower()
            out = [r for r in out if any(x.lower() == o for x in r.origins)]
        return out

    def to_dataframe(self):
        """Flatten to a pandas DataFrame; metrics become `m_<name>` columns."""
        import pandas as pd

        rows = []
        for r in self.records:
            row = {k: v for k, v in asdict(r).items() if k != "metrics"}
            row["assets"] = ", ".join(r.assets)
            row["origins"] = ", ".join(r.origins)
            for mk, mv in r.metrics.items():
                row[f"m_{mk}"] = mv
            rows.append(row)
        return pd.DataFrame(rows)

    def origin_breakdown(self) -> dict:
        """Status tally per origin — who suggested what, and how it ended.

        A record with several origins counts under each. Records with no origin
        fall under '(unattributed)'. This is the falsification scoreboard: how
        deployable each source's suggestions actually turned out.
        """
        out: dict[str, dict[str, int]] = {}
        for r in self.records:
            for k in (r.origins or ["(unattributed)"]):
                bucket = out.setdefault(k, {})
                bucket[r.status] = bucket.get(r.status, 0) + 1
        return out

    def print_origins(self, stream=None) -> None:
        """Print the origin scoreboard: per source, status tally + deployed count."""
        stream = stream or sys.stdout
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

        bd = self.origin_breakdown()
        W = 64
        out = ["=" * W, "  XIII — Experiment origins · suggestion scoreboard", "=" * W]
        for origin in sorted(bd, key=lambda k: sum(bd[k].values()), reverse=True):
            tally = bd[origin]
            n = sum(tally.values())
            live = tally.get("DEPLOYED", 0)
            parts = " · ".join(f"{s}:{c}" for s, c in sorted(tally.items()))
            out.append(f"  {origin}  (n={n}, deployed={live})")
            out.append(f"      {parts}")
        out.append("=" * W)
        stream.write("\n".join(out) + "\n")

    # ---- 13th-man self-audit ----------------------------------------------
    def verify(self, memory_dir=None) -> list[VerifyIssue]:
        """Audit the ledger's own integrity. A gate, not a score.

        FAIL: un-auditable records (bad shape, KILL without reason, evidence-
              bearing status without metrics).
        WARN: soft smells (missing/dangling backlink, deployed/status mismatch).

        If `memory_dir` is given, backlinks are checked for existence — a cheap
        file test, no YAML parsing, so no false positives on formatting.
        """
        issues: list[VerifyIssue] = []
        for r in self.records:
            if not r.title.strip():
                issues.append(VerifyIssue("FAIL", r.id, "empty title"))
            if r.status in _NEEDS_METRICS and not r.metrics:
                issues.append(VerifyIssue("FAIL", r.id, f"status {r.status} but no metrics"))
            if r.status == "KILLED_BY_13TH_MAN" and not (r.fail_reason and r.fail_reason.strip()):
                issues.append(VerifyIssue("FAIL", r.id, "KILLED_BY_13TH_MAN but no fail_reason"))
            if r.status == "KILLED_BY_13TH_MAN" and not r.metrics:
                issues.append(VerifyIssue("WARN", r.id, "KILLED but no metric recorded (qualitative kill)"))
            if r.deployed and r.status not in ("DEPLOYED", "PASSED"):
                issues.append(VerifyIssue("WARN", r.id, f"deployed=True but status={r.status}"))
            if not r.source_memory.strip():
                issues.append(VerifyIssue("WARN", r.id, "no source_memory backlink"))
            elif memory_dir is not None:
                p = Path(memory_dir) / f"{r.source_memory}.md"
                if not p.exists():
                    issues.append(
                        VerifyIssue("WARN", r.id, f"source_memory '{r.source_memory}.md' not found")
                    )
        return issues

    def backlink_coverage(self) -> float:
        """Fraction of records carrying a source_memory backlink (0..1)."""
        if not self._records:
            return 1.0
        linked = sum(1 for r in self.records if r.source_memory.strip())
        return linked / len(self._records)

    def print_verify(self, memory_dir=None, stream=None) -> bool:
        """Print a compact verify report. Returns True iff no FAIL (the gate)."""
        stream = stream or sys.stdout
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

        issues = self.verify(memory_dir=memory_dir)
        fails = [i for i in issues if i.level == "FAIL"]
        warns = [i for i in issues if i.level == "WARN"]

        W = 64
        out = ["=" * W, "  XIII — Experiment Registry · integrity check", "=" * W]
        out.append(f"  records: {len(self)}   backlink coverage: {self.backlink_coverage():.0%}")
        by_status = {s: len(self.query(status=s)) for s in STATUSES}
        out.append("  " + " · ".join(f"{s}:{n}" for s, n in by_status.items() if n))
        out.append("-" * W)
        for i in fails + warns:
            mark = "[FAIL]" if i.level == "FAIL" else "[WARN]"
            out.append(f"  {mark} {i.record_id}: {i.message}")
        if not issues:
            out.append("  [ OK ] no integrity issues")
        out.append("-" * W)
        verdict = "LEDGER OK" if not fails else "LEDGER NOT AUDITABLE"
        out.append(f"  {len(fails)} FAIL · {len(warns)} WARN   ->  {verdict}")
        out.append("=" * W)
        stream.write("\n".join(out) + "\n")
        return not fails
