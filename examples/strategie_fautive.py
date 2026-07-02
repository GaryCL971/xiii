"""
SPECIMEN — intentionally faulty strategy for XIII demo.

This file concentrates the 3 classic faults XIII detects by static analysis.
It must NOT be executed or traded: it's the demo bait.
Each fault here is a flaw ACTUALLY found in the lab in June 2026.
"""
import numpy as np
import pandas as pd


def build_cpi_brick(dates):
    """CPI brick... except it never saw a real inflation number."""
    # TODO: wire real CPI data
    ret = pd.Series(np.random.normal(0.0005, 0.0008, len(dates)), index=dates)  # dummy
    return ret


def build_portfolio(cpi, nasdaq):
    """Aligns a monthly series (CPI) to a daily series (NASDAQ)."""
    df = pd.concat([cpi, nasdaq], axis=1).dropna()   # silent inner-join
    return df.mean(axis=1)


def signal(close):
    """Moving average crossover… that looks ahead twice."""
    ma = close.rolling(100, center=True).mean()      # centered window = half future
    pos = (close > ma).astype(float).shift(-1)       # aligns tomorrow's position to today
    return pos
