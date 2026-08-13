"""
xiii.dataqc — data quality gate, upstream of any backtest audit.

A backtest can lie for a reason that has nothing to do with the strategy:
the DATA is wrong. This module assumes every dataset is potentially
misleading until proven otherwise, and looks for the ways it can
invalidate a research result.

Real flaws that motivated each check:
  DQ8_lead_lag    : a free daily FX feed published START-OF-DAY snapshots
                    labeled as daily closes. A lag-1 "edge" with t~5 passed
                    every split and placebo — and was pure timestamp artifact.
  DQ9_divergence  : a signal validated on continuous futures proxies lost
                    ~0.5 Sharpe once re-measured on the broker's CFD feed.
                    Relative rankings across assets flipped too.
  DQ2/DQ5         : CFD feeds have session holes and frozen quotes that
                    silently shrink or bias the effective sample.
  DQ6_length      : short windows inflate Sharpe. A 2.3-year window sold a
                    2.53 Sharpe that collapsed on the honest 16-year history.

Verdict vocabulary (a gate, not a score):
  usable                -> no FAIL, no WARN
  usable_with_warning   -> no FAIL, at least one WARN (read them)
  reject_data           -> at least one FAIL: do not backtest on this
                           dataset until fixed.

Usage:
    import pandas as pd
    from xiii import dataqc

    df = pd.read_parquet("eurusd_d1.parquet")       # OHLC or close-only
    rep = dataqc.qc(df, symbol="EURUSD")
    rep.print()

    # Two-series checks (snapshot detection, proxy divergence):
    rep = dataqc.qc(proxy_df, symbol="EURUSD_yahoo", reference=broker_df)
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .report import CheckResult

VERDICTS = ("usable", "usable_with_warning", "reject_data")
_MARK = {"PASS": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}


# ─────────────────────────────────────────────────────────────────────────────
# Input normalization
# ─────────────────────────────────────────────────────────────────────────────

_OHLC = ("open", "high", "low", "close")
_TIME_COLS = ("time", "date", "datetime", "timestamp")


def _match_col(name, vocab) -> str | None:
    """Maps a column name onto `vocab`, tolerating MultiIndex tuples and
    yfinance-style stringified tuples like "('close', '^ndx')"."""
    parts = name if isinstance(name, tuple) else (name,)
    for p in parts:
        s = str(p).strip().lower()
        if s in vocab:
            return s
    joined = " ".join(str(p) for p in parts).lower()
    for k in vocab:
        if re.search(rf"\b{k}\b", joined) and "adj" not in joined:
            return k
    return None


def _pick_columns(df: pd.DataFrame, vocab) -> dict:
    """{canonical_name: original_column}, exact matches taking priority."""
    mapping: dict = {}
    for c in df.columns:  # pass 1: exact
        parts = c if isinstance(c, tuple) else (c,)
        for p in parts:
            s = str(p).strip().lower()
            if s in vocab and s not in mapping:
                mapping[s] = c
    for c in df.columns:  # pass 2: fuzzy, only for still-missing keys
        k = _match_col(c, vocab)
        if k and k not in mapping:
            mapping[k] = c
    return mapping


def _normalize(data) -> pd.DataFrame:
    """Accepts a DataFrame (OHLC or close-only, case-insensitive, MultiIndex
    columns tolerated) or a Series. Promotes a time/date column to index if
    needed. Returns a DataFrame with lowercase columns and a naive
    DatetimeIndex."""
    if isinstance(data, pd.Series):
        df = data.to_frame("close")
    elif isinstance(data, pd.DataFrame):
        if not isinstance(data.index, pd.DatetimeIndex):
            tmap = _pick_columns(data, _TIME_COLS)
            if tmap:
                tcol = next(iter(tmap.values()))
                data = data.set_index(pd.to_datetime(data[tcol]))
        omap = _pick_columns(data, _OHLC)
        if not omap:
            raise ValueError(
                f"no OHLC/close column found (got {list(data.columns)[:8]})"
            )
        df = pd.DataFrame({k: data[c] for k, c in omap.items()}, index=data.index)
    else:
        raise TypeError(f"expected DataFrame or Series, got {type(data).__name__}")

    if not isinstance(df.index, pd.DatetimeIndex):
        if pd.api.types.is_numeric_dtype(df.index):
            raise ValueError(
                "index is numeric and no time/date column was found — refusing "
                "to guess timestamps (an integer index converted to epoch would "
                "silently fabricate 1970 dates)"
            )
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def _median_spacing(idx: pd.DatetimeIndex) -> pd.Timedelta:
    if len(idx) < 3:
        return pd.Timedelta(days=1)
    return pd.Series(idx).diff().dropna().median()


def _is_daily(spacing: pd.Timedelta) -> bool:
    return spacing >= pd.Timedelta(hours=12)


def _log_returns(close: pd.Series) -> pd.Series:
    c = close[close > 0]
    return np.log(c).diff().dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Single-series checks
# ─────────────────────────────────────────────────────────────────────────────

def dq1_index_sanity(df: pd.DataFrame) -> CheckResult:
    """Timestamps must be unique and ordered before anything else is trusted."""
    _id = "DQ1_index_sanity"
    dup = int(df.index.duplicated().sum())
    unsorted = not df.index.is_monotonic_increasing
    ev = {"duplicates": dup, "sorted": not unsorted, "n_bars": len(df)}

    if dup:
        return CheckResult(
            _id, "DQ", "FAIL",
            f"{dup} duplicated timestamp(s)",
            "Duplicated bars double-count some days and silently reweight the "
            "sample. Deduplicate at the source (keep last) and rerun.",
            ev,
        )
    if unsorted:
        return CheckResult(
            _id, "DQ", "WARN",
            "Index not sorted (auto-sorted for the remaining checks)",
            "An unsorted feed often means several files were concatenated; "
            "check for overlaps at the seams.",
            ev,
        )
    return CheckResult(_id, "DQ", "PASS",
                       f"{len(df)} bars, unique and ordered timestamps", "", ev)


def dq2_gaps(df: pd.DataFrame) -> CheckResult:
    """Holes in the history shrink or bias the effective sample."""
    _id = "DQ2_gaps"
    idx = df.index
    if len(idx) < 10:
        return CheckResult(_id, "DQ", "SKIP", "Too few bars to assess gaps", "", {})

    spacing = _median_spacing(idx)
    daily = _is_daily(spacing)
    diffs = pd.Series(idx).diff().dropna()
    # Weekends are normal for daily and intraday market data: tolerate up to
    # 4 calendar days (long weekend + holiday) before calling it a hole.
    tol = pd.Timedelta(days=4) if daily else max(pd.Timedelta(days=4), 6 * spacing)
    gaps = diffs[diffs > tol]
    largest = diffs.max()

    missing_share = 0.0
    if daily:
        expected = np.busday_count(idx[0].date(), idx[-1].date()) + 1
        missing_share = max(0.0, 1.0 - len(idx) / max(expected, 1))

    worst = [
        f"{idx[i - 1].date()} -> {idx[i].date()} ({d.days}d)"
        for i, d in zip(gaps.nlargest(3).index, gaps.nlargest(3))
    ]
    ev = {"n_gaps": int(len(gaps)), "largest_gap_days": float(largest.days),
          "missing_share_vs_busdays": round(missing_share, 4), "worst": worst}

    if largest > pd.Timedelta(days=30) or missing_share > 0.10:
        return CheckResult(
            _id, "DQ", "FAIL",
            f"History has holes: largest gap {largest.days}d, "
            f"~{missing_share:.0%} of business days missing",
            "A multi-week hole (feed outage, delisting, symbol change) makes "
            "period metrics incomparable. Fix the source or restrict the window "
            "to the clean segment — and say so in the report.\n"
            + "; ".join(worst),
            ev,
        )
    if len(gaps):
        return CheckResult(
            _id, "DQ", "WARN",
            f"{len(gaps)} gap(s) beyond {tol.days} calendar days "
            f"(largest {largest.days}d)",
            "Check whether these holes overlap the periods that drive the "
            "backtest PnL.\n" + "; ".join(worst),
            ev,
        )
    return CheckResult(_id, "DQ", "PASS",
                       "No abnormal holes in the history", "", ev)


def dq3_ohlc_coherence(df: pd.DataFrame) -> CheckResult:
    """high >= max(open, close), low <= min(open, close), prices > 0."""
    _id = "DQ3_ohlc_coherence"
    if not all(c in df.columns for c in _OHLC):
        return CheckResult(
            _id, "DQ", "SKIP", "Close-only input, OHLC coherence not assessable",
            "Pass full OHLC bars to enable this check.", {},
        )
    o, h, l, c = (df[k].astype(float) for k in _OHLC)
    eps = 1e-9
    bad = (
        (h < l - eps) | (h + eps < o) | (h + eps < c)
        | (l - eps > o) | (l - eps > c)
        | (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)
    )
    n_bad = int(bad.sum())
    share = n_bad / max(len(df), 1)
    worst = [str(t.date()) for t in df.index[bad][:5]]
    ev = {"violations": n_bad, "share": round(share, 6), "examples": worst}

    if share > 0.001:
        return CheckResult(
            _id, "DQ", "FAIL",
            f"{n_bad} incoherent OHLC bar(s) ({share:.2%})",
            "high<low or close outside [low, high]: the feed is corrupt at a "
            "scale that will distort ranges, stops and ATR-style logic.\n"
            "Examples: " + ", ".join(worst),
            ev,
        )
    if n_bad:
        return CheckResult(
            _id, "DQ", "WARN",
            f"{n_bad} incoherent OHLC bar(s) (isolated)",
            "Inspect and drop/repair them at the source rather than in the "
            "backtest.\nExamples: " + ", ".join(worst),
            ev,
        )
    return CheckResult(_id, "DQ", "PASS", "OHLC bars are internally coherent", "", ev)


def dq4_spikes(df: pd.DataFrame) -> CheckResult:
    """Returns far outside the distribution: bad ticks vs real events."""
    _id = "DQ4_spikes"
    rets = _log_returns(df["close"].astype(float))
    if len(rets) < 100:
        return CheckResult(_id, "DQ", "SKIP", "Too few bars to assess spikes", "", {})

    sigma = 1.4826 * float((rets - rets.median()).abs().median())
    if sigma == 0:
        return CheckResult(
            _id, "DQ", "FAIL", "Zero robust volatility (constant series?)",
            "More than half of the returns are identical — dead or synthetic feed.",
            {"robust_sigma": 0.0},
        )
    z = (rets - rets.median()).abs() / sigma
    spikes = z[z > 10]
    share = len(spikes) / len(rets)
    top = [f"{t.date()}: {rets[t]:+.2%} (z={z[t]:.0f})" for t in spikes.nlargest(3).index]
    ev = {"n_spikes_z10": int(len(spikes)), "share": round(share, 5),
          "robust_sigma": round(sigma, 6), "top": top}

    if share > 0.003:
        return CheckResult(
            _id, "DQ", "FAIL",
            f"{len(spikes)} extreme return(s) beyond 10 robust sigmas "
            f"({share:.2%} of bars)",
            "At this frequency it is data corruption, not market events. "
            "Clean the feed before measuring anything on it.\n" + "; ".join(top),
            ev,
        )
    if len(spikes):
        return CheckResult(
            _id, "DQ", "WARN",
            f"{len(spikes)} extreme return(s) beyond 10 robust sigmas",
            "Each one is either a real crisis bar (keep it) or a bad tick "
            "(kill it). Decide explicitly, per bar — this is where max-drawdown "
            "numbers are made and broken.\n" + "; ".join(top),
            ev,
        )
    return CheckResult(_id, "DQ", "PASS",
                       "No returns beyond 10 robust sigmas", "", ev)


def dq5_frozen(df: pd.DataFrame) -> CheckResult:
    """Frozen quotes: the feed repeats itself instead of trading."""
    _id = "DQ5_frozen"
    close = df["close"].astype(float)
    if len(close) < 50:
        return CheckResult(_id, "DQ", "SKIP", "Too few bars to assess freezes", "", {})

    same = (close.diff() == 0)
    runs = same.groupby((~same).cumsum()).sum()
    longest = int(runs.max()) if len(runs) else 0
    zero_share = float(same.mean())

    spacing = _median_spacing(df.index)
    daily = _is_daily(spacing)
    warn_run, fail_run = (5, 15) if daily else (
        int(pd.Timedelta(days=2) / spacing), int(pd.Timedelta(days=10) / spacing))
    ev = {"longest_frozen_run": longest, "zero_return_share": round(zero_share, 4),
          "warn_run": warn_run, "fail_run": fail_run}

    if longest > fail_run or zero_share > 0.60:
        return CheckResult(
            _id, "DQ", "FAIL",
            f"Feed frozen: longest identical-close run {longest} bars, "
            f"{zero_share:.0%} zero-return bars",
            "Long frozen stretches mean no real quotes; every statistic "
            "computed across them (vol, Sharpe, DD) is fiction.",
            ev,
        )
    if longest > warn_run:
        return CheckResult(
            _id, "DQ", "WARN",
            f"Longest identical-close run: {longest} bars",
            "Check whether these stretches are holidays (fine) or feed "
            "outages recorded as flat quotes (not fine).",
            ev,
        )
    return CheckResult(_id, "DQ", "PASS", "No abnormal frozen stretches", "", ev)


def dq6_length(df: pd.DataFrame, min_years_warn: float,
               min_years_fail: float) -> CheckResult:
    """Short windows inflate Sharpe. Insist on the longest available history."""
    _id = "DQ6_length"
    span = (df.index[-1] - df.index[0]).days / 365.25
    ev = {"span_years": round(span, 2), "n_bars": len(df),
          "min_years_warn": min_years_warn, "min_years_fail": min_years_fail}

    if span < min_years_fail or len(df) < 300:
        return CheckResult(
            _id, "DQ", "FAIL",
            f"History too short: {span:.1f} years ({len(df)} bars)",
            f"Below {min_years_fail:g} years nothing survives a regime change. "
            "Real flaw: a 2.3-year window sold a Sharpe of 2.53 that collapsed "
            "once the full 16-year history was measured. Get more history first.",
            ev,
        )
    if span < min_years_warn:
        return CheckResult(
            _id, "DQ", "WARN",
            f"History spans {span:.1f} years — below the {min_years_warn:g}y "
            "comfort bar",
            "Usable for exploration, but headline metrics must be labeled "
            "with the window and re-checked when longer history is available "
            "(15+ years preferred).",
            ev,
        )
    return CheckResult(_id, "DQ", "PASS",
                       f"History spans {span:.1f} years ({len(df)} bars)", "", ev)


def dq7_clock(df: pd.DataFrame) -> CheckResult:
    """Daily bars should share one clock time; a mixed grid hints at
    timezone/DST mangling or resampled snapshots."""
    _id = "DQ7_clock"
    spacing = _median_spacing(df.index)
    if not _is_daily(spacing):
        return CheckResult(
            _id, "DQ", "SKIP",
            "Intraday data: clock-grid check applies to daily bars",
            "For intraday feeds, validate session boundaries against the "
            "broker's published schedule instead.", {},
        )
    times = pd.Series(df.index.time)
    dominant_share = float(times.value_counts(normalize=True).iloc[0])
    n_distinct = int(times.nunique())
    ev = {"dominant_time_share": round(dominant_share, 4),
          "distinct_times": n_distinct}

    if dominant_share < 0.90:
        return CheckResult(
            _id, "DQ", "WARN",
            f"Daily bars carry {n_distinct} distinct clock times "
            f"(dominant one only {dominant_share:.0%})",
            "Mixed stamps usually mean DST shifts or a feed that resamples "
            "with a moving anchor. Any open/close-anchored signal measured on "
            "this grid must be re-validated on the broker's own bars.",
            ev,
        )
    return CheckResult(_id, "DQ", "PASS",
                       f"Daily bars share one clock time ({dominant_share:.0%})",
                       "", ev)


# ─────────────────────────────────────────────────────────────────────────────
# Two-series checks (candidate vs reference/broker feed)
# ─────────────────────────────────────────────────────────────────────────────

def _aligned_returns(cand: pd.DataFrame, ref: pd.DataFrame):
    """Close-to-close log returns on the intersection of dates.
    Inner join is DELIBERATE here: these checks compare the two feeds where
    both exist; they do not measure performance on the joined window."""
    a = _log_returns(cand["close"].astype(float))
    b = _log_returns(ref["close"].astype(float))
    if _is_daily(_median_spacing(cand.index)):
        a.index = a.index.normalize()
        b.index = b.index.normalize()
    common = a.index.intersection(b.index)
    return a.loc[common], b.loc[common]


def dq8_lead_lag(cand: pd.DataFrame, ref: pd.DataFrame | None) -> CheckResult:
    """Are the candidate's timestamps what they claim to be?

    A feed that stamps start-of-day snapshots as daily closes correlates
    best with the reference at lag +/-1, not 0. Any lag-1 'edge' found on
    such a feed is a timestamp artifact — this exact trap produced a t~5
    signal that survived every split and placebo."""
    _id = "DQ8_lead_lag"
    if ref is None:
        return CheckResult(
            _id, "DQ", "SKIP", "No reference series provided",
            "Pass reference=<broker feed> to test timestamp alignment. "
            "Mandatory before trusting any open/close/gap-anchored signal.", {},
        )
    a, b = _aligned_returns(cand, ref)
    if len(a) < 200:
        return CheckResult(
            _id, "DQ", "SKIP",
            f"Only {len(a)} overlapping bars with the reference",
            "Not enough overlap to assess alignment.", {"overlap": len(a)},
        )
    corrs = {}
    for lag in range(-2, 3):
        corrs[lag] = float(a.corr(b.shift(lag)))
    best = max(corrs, key=lambda k: abs(corrs[k]))
    ev = {"corr_by_lag": {str(k): round(v, 4) for k, v in corrs.items()},
          "best_lag": best, "overlap": len(a)}

    if best != 0 and abs(corrs[best]) - abs(corrs[0]) > 0.02 and abs(corrs[best]) > 0.20:
        return CheckResult(
            _id, "DQ", "FAIL",
            f"Timestamp misalignment: max correlation at lag {best:+d} "
            f"({corrs[best]:.2f} vs {corrs[0]:.2f} at lag 0)",
            "The candidate feed is shifted vs the reference — snapshot quotes "
            "labeled as closes, or a timezone offset. Every lagged signal "
            "measured on it is suspect. Rebuild the dataset from the "
            "reference feed.",
            ev,
        )
    return CheckResult(
        _id, "DQ", "PASS",
        f"Correlation peaks at lag 0 ({corrs[0]:.2f})",
        "Timestamps are consistent with the reference feed.", ev,
    )


def dq9_divergence(cand: pd.DataFrame, ref: pd.DataFrame | None) -> CheckResult:
    """How far is the proxy from the instrument actually traded?"""
    _id = "DQ9_divergence"
    if ref is None:
        return CheckResult(
            _id, "DQ", "SKIP", "No reference series provided",
            "Pass reference=<broker feed> to quantify proxy divergence.", {},
        )
    a, b = _aligned_returns(cand, ref)
    if len(a) < 200:
        return CheckResult(
            _id, "DQ", "SKIP",
            f"Only {len(a)} overlapping bars with the reference",
            "Not enough overlap to quantify divergence.", {"overlap": len(a)},
        )
    corr = float(a.corr(b))
    te = float((a - b).std() * np.sqrt(252))
    sign_agree = float((np.sign(a) == np.sign(b)).mean())
    ev = {"corr_lag0": round(corr, 4), "tracking_error_ann": round(te, 4),
          "sign_agreement": round(sign_agree, 4), "overlap": len(a)}

    if corr < 0.90:
        return CheckResult(
            _id, "DQ", "FAIL",
            f"Proxy diverges from reference: corr {corr:.2f}, "
            f"sign agreement {sign_agree:.0%}",
            "At this distance, backtest conclusions do not transfer to the "
            "traded instrument. Real flaw: a strategy validated on futures "
            "proxies lost ~0.5 Sharpe re-measured on the broker feed. "
            "Re-run the backtest on the reference data.",
            ev,
        )
    if corr < 0.97:
        return CheckResult(
            _id, "DQ", "WARN",
            f"Proxy is close but not identical: corr {corr:.2f}, "
            f"tracking error {te:.1%}/yr",
            "Fine for exploration. Before sizing or deployment, re-measure "
            "the final signal on the broker feed — small divergences are "
            "enough to flip relative rankings between assets.",
            ev,
        )
    return CheckResult(
        _id, "DQ", "PASS",
        f"Proxy tracks the reference (corr {corr:.2f}, TE {te:.1%}/yr)", "", ev,
    )


def dq10_cost_shift(cost_series: pd.Series | None) -> CheckResult:
    """Spreads/swaps drift: a cost measured today is not the cost of 2010."""
    _id = "DQ10_cost_shift"
    if cost_series is None:
        return CheckResult(
            _id, "DQ", "SKIP", "No cost series provided",
            "Pass cost_series=<spread or swap history> to test for structural "
            "shifts in trading costs.", {},
        )
    s = pd.Series(cost_series).dropna().astype(float)
    if len(s) < 100:
        return CheckResult(_id, "DQ", "SKIP",
                           f"Only {len(s)} cost observations", "", {"n": len(s)})
    half = len(s) // 2
    m1, m2 = float(s.iloc[:half].median()), float(s.iloc[half:].median())
    ratio = m2 / m1 if m1 else np.inf
    ev = {"median_first_half": round(m1, 6), "median_second_half": round(m2, 6),
          "ratio": round(ratio, 3)}

    if ratio > 1.5 or ratio < 0.67:
        return CheckResult(
            _id, "DQ", "WARN",
            f"Cost regime shift: median moved {m1:.4g} -> {m2:.4g} (x{ratio:.2f})",
            "Backtests priced at one flat cost misstate half the history. "
            "Use period-appropriate costs, or at least the WORST half.",
            ev,
        )
    return CheckResult(_id, "DQ", "PASS",
                       f"No structural cost shift (x{ratio:.2f} across halves)",
                       "", ev)


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataQCReport:
    symbol: str | None
    n_bars: int
    start: str
    end: str
    verdicts: list[CheckResult] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if any(v.status == "FAIL" for v in self.verdicts):
            return "reject_data"
        if any(v.status == "WARN" for v in self.verdicts):
            return "usable_with_warning"
        return "usable"

    def _count(self, status: str) -> int:
        return sum(v.status == status for v in self.verdicts)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {
                "symbol": self.symbol,
                "verdict": self.verdict,
                "n_bars": self.n_bars,
                "window": [self.start, self.end],
                "summary": {s: self._count(s) for s in ("PASS", "WARN", "FAIL", "SKIP")},
                "verdicts": [asdict(v) for v in self.verdicts],
            },
            ensure_ascii=False, indent=indent,
        )

    def print(self, stream=None) -> None:
        stream = stream or sys.stdout
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
        W = 64
        line = "=" * W
        head = f"  XIII — Data QC" + (f" · {self.symbol}" if self.symbol else "")
        out = [line, head, f"  {self.n_bars} bars · {self.start} -> {self.end}", line]
        order = {"FAIL": 0, "WARN": 1, "SKIP": 2, "PASS": 3}
        import textwrap
        for v in sorted(self.verdicts, key=lambda x: order[x.status]):
            code = v.check_id.split("_", 1)[0]
            out.append(f"  {_MARK[v.status]} {code} · {v.headline}")
            for para in (v.detail or "").split("\n"):
                for wl in textwrap.wrap(para, W - 9) or [""]:
                    out.append(" " * 9 + wl)
            out.append("")
        out.append("-" * W)
        out.append(
            f"  Result: {self._count('FAIL')} FAIL · {self._count('WARN')} WARN · "
            f"{self._count('SKIP')} SKIP   ->  {self.verdict.upper()}"
        )
        out.append(line)
        stream.write("\n".join(out) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def qc(
    data,
    *,
    symbol: str | None = None,
    reference=None,
    cost_series=None,
    min_years_warn: float = 10.0,
    min_years_fail: float = 3.0,
) -> DataQCReport:
    """Runs the data quality gate on one dataset.

    data       : DataFrame (OHLC or close-only, case-insensitive) or Series.
    reference  : broker/authoritative feed for the same instrument. Enables
                 DQ8 (timestamp alignment) and DQ9 (proxy divergence) —
                 mandatory before trusting open/close/gap-anchored signals
                 or cross-asset rankings built on proxies.
    cost_series: spread or swap history, enables DQ10.

    Missing input => the corresponding check returns SKIP, not an error.
    A crashed check returns FAIL (no silent crash).
    Returns a DataQCReport (.print(), .to_json(), .verdict).
    """
    df = _normalize(data)
    df = df[~df.index.duplicated(keep="last")].sort_index() if not df.index.is_unique \
        else df.sort_index()
    ref = _normalize(reference) if reference is not None else None
    if ref is not None:
        ref = ref[~ref.index.duplicated(keep="last")].sort_index()

    # DQ1 runs on the RAW index (before dedup/sort) to report what it saw.
    raw = _normalize(data)

    steps = [
        lambda: dq1_index_sanity(raw),
        lambda: dq2_gaps(df),
        lambda: dq3_ohlc_coherence(df),
        lambda: dq4_spikes(df),
        lambda: dq5_frozen(df),
        lambda: dq6_length(df, min_years_warn, min_years_fail),
        lambda: dq7_clock(df),
        lambda: dq8_lead_lag(df, ref),
        lambda: dq9_divergence(df, ref),
        lambda: dq10_cost_shift(cost_series),
    ]
    verdicts = []
    for step in steps:
        try:
            verdicts.append(step())
        except Exception as e:  # no silent crash: a broken check is a FAIL
            verdicts.append(CheckResult(
                "DQx_crashed", "DQ", "FAIL",
                f"Check crashed: {type(e).__name__}",
                f"{e} — a check that cannot run on this dataset is itself "
                "a data-quality signal.",
                {"error": str(e)},
            ))

    return DataQCReport(
        symbol=symbol,
        n_bars=len(df),
        start=str(df.index[0].date()) if len(df) else "",
        end=str(df.index[-1].date()) if len(df) else "",
        verdicts=verdicts,
    )
