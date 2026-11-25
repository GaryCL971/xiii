"""Calibrate execution spreads from timestamped bid/ask ticks or tick samples."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpreadQuantiles:
    n: int
    p50: float
    p75: float
    p95: float

    def quantile(self, percentile: int) -> float:
        if percentile not in (50, 75, 95):
            raise ValueError("spread percentile must be 50, 75, or 95")
        return getattr(self, f"p{percentile}")


@dataclass(frozen=True)
class TickSpreadProfile:
    symbol: str
    regimes: dict[str, SpreadQuantiles]
    source: str

    def spread_price(self, regime: str, percentile: int) -> float:
        if regime not in self.regimes:
            raise ValueError(f"No tick spread calibration for {self.symbol}, regime {regime!r}")
        return self.regimes[regime].quantile(percentile)


def window_mask(index: pd.DatetimeIndex, start: int, end: int, offset: int = 0) -> np.ndarray:
    """Select wall-clock minutes [start,end), optionally shifted, including overnight windows."""
    if not isinstance(index, pd.DatetimeIndex) or index.hasnans:
        raise ValueError("an explicitly clock-aligned DatetimeIndex is required")
    if not (0 <= start < 1440 and 0 <= end <= 1440 and start != end):
        raise ValueError("session bounds must delimit a nonempty interval in minutes")
    minute = np.asarray(index.hour * 60 + index.minute)
    duration = end - start if end > start else 1440 + end - start
    return (minute - start - offset) % 1440 < duration


def calibrate_spread_samples(
    symbol: str, spread: pd.Series, *, regimes: dict[str, tuple[int, int]] | None = None,
    source: str, min_samples: int = 20,
) -> TickSpreadProfile:
    """Calibrate price-unit quantiles from observed tick spreads in the feed clock.

    A first-tick-per-bar archive can be used if declared as such in source.
    Regimes are local minutes, e.g. 08:00–08:15 = (480, 495).
    Missing or undersampled regimes are refused instead of falling back.
    """
    if not symbol.strip() or not source.strip() or min_samples < 1:
        raise ValueError("symbol, tick source and positive min_samples are required")
    values = spread.to_numpy(dtype=float)
    if not isinstance(spread.index, pd.DatetimeIndex) or spread.index.hasnans:
        raise ValueError("tick spread samples need valid timestamps")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("spread samples must be finite and nonnegative")
    masks = {"all": np.ones(values.size, dtype=bool)}
    for name, (start, end) in (regimes or {}).items():
        if not name or name == "all":
            raise ValueError("use a nonempty regime name other than 'all'")
        masks[name] = window_mask(spread.index, start, end)
    result = {}
    for name, mask in masks.items():
        sample = values[mask]
        if sample.size < min_samples:
            raise ValueError(f"{symbol}/{name}: only {sample.size} tick samples, need {min_samples}")
        result[name] = SpreadQuantiles(sample.size, *map(float, np.quantile(sample, [0.50, 0.75, 0.95])))
    return TickSpreadProfile(symbol, result, source)


def calibrate_tick_spreads(
    symbol: str, ticks: pd.DataFrame, *, source: str,
    regimes: dict[str, tuple[int, int]] | None = None, min_samples: int = 20,
) -> TickSpreadProfile:
    """Read bid/ask prices in their original units and timestamps in the feed clock."""
    bid, ask = ticks["bid"].astype(float), ticks["ask"].astype(float)
    if not np.isfinite(bid).all() or not np.isfinite(ask).all() or (bid <= 0).any():
        raise ValueError("tick bid/ask require finite prices and strictly positive bid")
    return calibrate_spread_samples(symbol, ask - bid, regimes=regimes, source=source, min_samples=min_samples)
