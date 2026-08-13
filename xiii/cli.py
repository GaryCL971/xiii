"""Command-line interface for XIII.

    xiii audit equity.csv [trades.csv] [options]

Exit codes are what make this usable in CI: 0 when the audit passes, 1 when any
check FAILs, 2 on a usage or input error. A linter that always exits 0 gates nothing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import __version__, audit
from .brokers import REGISTRY as BROKER_REGISTRY


class CliError(Exception):
    """Usage or input problem — reported without a traceback, exit code 2."""


def _load_equity(path: Path) -> pd.Series:
    """Dated CSV (date,equity) -> numeric Series. First data column wins."""
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        raise CliError(f"equity file not found: {path}") from None
    except Exception as exc:
        raise CliError(f"cannot read equity csv '{path}': {exc}") from exc

    if df.shape[1] < 1:
        raise CliError(f"equity csv '{path}' has no data column (expected: date,equity)")

    series = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    if series.empty:
        raise CliError(f"equity csv '{path}': no numeric values in column '{df.columns[0]}'")
    return series


def _load_trades(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise CliError(f"trades file not found: {path}") from None
    except Exception as exc:
        raise CliError(f"cannot read trades csv '{path}': {exc}") from exc

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def _cmd_audit(args: argparse.Namespace) -> int:
    if args.source is not None and not args.source.is_file():
        raise CliError(f"source file not found: {args.source}")

    report = audit(
        equity=_load_equity(args.equity),
        trades=_load_trades(args.trades) if args.trades else None,
        source_file=args.source,
        broker=BROKER_REGISTRY[args.broker] if args.broker else None,
        deployed_sizing=args.deployed_sizing,
        sl_pips=args.sl_pips,
        tp_pips=args.tp_pips,
        win_rate_backtest=args.win_rate,
    )

    if args.json:
        # Windows consoles default to cp1252 and the report carries arrows (U+2192).
        # AuditReport.print() already guards for this; the JSON path needs it too.
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
        print(report.to_json())
    else:
        report.print()
    return 0 if report.passed else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xiii",
        description="The overfitting linter for trading backtests.",
    )
    parser.add_argument("--version", action="version", version=f"xiii {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("audit", help="run the Thirteenth Man checks on a backtest")
    p.add_argument("equity", type=Path, help="CSV: date,equity")
    p.add_argument("trades", type=Path, nargs="?", help="CSV with a pnl column (optional)")
    p.add_argument(
        "--source", type=Path, metavar="FILE",
        help="strategy source file — unlocks the integrity checks (A1, A3, G1)",
    )
    p.add_argument(
        "--broker", choices=sorted(BROKER_REGISTRY),
        help="broker rules — unlocks the cost and sizing checks (C1, C2, D2)",
    )
    p.add_argument(
        "--deployed-sizing", type=float, default=1.0, metavar="X",
        help="sizing multiplier actually deployed, vs the backtest (default: 1.0)",
    )
    p.add_argument("--sl-pips", type=float, metavar="N", help="stop loss, in pips (with --tp-pips: D2)")
    p.add_argument("--tp-pips", type=float, metavar="N", help="take profit, in pips (with --sl-pips: D2)")
    p.add_argument("--win-rate", type=float, metavar="R", help="backtest win rate, 0-1 (refines D2)")
    p.add_argument("--json", action="store_true", help="emit JSON instead of the readable report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "audit":
            return _cmd_audit(args)
    except CliError as exc:
        print(f"xiii: error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
