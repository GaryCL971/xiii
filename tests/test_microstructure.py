"""Tests for proxy quality, tick spread calibration and session replay."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from xiii import FTMO, audit, dataqc
from xiii.brokers import InstrumentCost
from xiii.spreads import calibrate_tick_spreads, window_mask


def feed(returns):
    return pd.Series(np.exp(np.cumsum(returns)), index=pd.bdate_range("2022-01-03", periods=len(returns)))


def by_id(report):
    return {v.check_id: v for v in report.verdicts}


def test_aligned_proxy_passes_both_checks():
    broker = feed(np.random.default_rng(20).normal(0, 0.01, 500))
    result = by_id(dataqc.qc(broker, reference=broker.copy(), proxy=True))
    assert result["DQ8_lead_lag"].status == result["DQ9_divergence"].status == "PASS"
    assert audit(proxy_data=broker, reference_data=broker.copy()).passed


def test_proxy_lead_lag_is_blocking():
    rng = np.random.default_rng(21)
    changes = rng.normal(0, 0.01, 500)
    report = audit(proxy_data=feed(np.r_[0, changes[:-1]]), reference_data=feed(changes))
    dq8 = by_id(report)["DQ8_lead_lag"]
    assert dq8.status == "FAIL"
    assert abs(dq8.evidence["best_lag"]) == 1
    assert not report.passed


def test_soft_divergence_blocks_only_declared_proxy():
    rng = np.random.default_rng(21)
    changes = rng.normal(0, 0.01, 600)
    broker, candidate = feed(changes), feed(changes + rng.normal(0, 0.0035, 600))
    legacy = by_id(dataqc.qc(candidate, reference=broker))
    strict = by_id(dataqc.qc(candidate, reference=broker, proxy=True))
    assert legacy["DQ9_divergence"].status == "WARN"
    assert strict["DQ9_divergence"].status == "FAIL"
    assert strict["DQ8_lead_lag"].status == "PASS"
    assert not audit(proxy_data=candidate, reference_data=broker).passed
    assert not any(v.section == "DQ" for v in audit().verdicts)


def test_missing_short_or_degenerate_reference_fails_only_in_proxy_mode():
    broker = feed(np.random.default_rng(3).normal(0, 0.01, 500))
    for reference in (None, broker.iloc[:150], pd.Series(1.0, index=broker.index)):
        strict = by_id(dataqc.qc(broker, reference=reference, proxy=True))
        assert strict["DQ8_lead_lag"].status == strict["DQ9_divergence"].status == "FAIL"
    native = by_id(dataqc.qc(broker))
    assert native["DQ8_lead_lag"].status == native["DQ9_divergence"].status == "SKIP"
    assert not audit(proxy=True).passed


def test_tick_calibration_is_in_price_units_and_changes_d2():
    times = pd.date_range("2024-01-01", periods=50, freq="D") + pd.Timedelta(hours=8)
    index = times.append(times + pd.Timedelta(hours=4)).append(times + pd.Timedelta(hours=15))
    widths = np.r_[np.linspace(0.60, 1.20, 50), np.full(50, 0.10), np.full(50, 0.45)]
    ticks = pd.DataFrame({"bid": 2000, "ask": 2000 + widths}, index=index)
    windows = {"opening": (480, 495), "liquid": (720, 735), "rollover": (1380, 15)}
    for symbol in ("XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "US100"):
        profile = calibrate_tick_spreads(symbol, ticks, regimes=windows, source="synthetic ticks")
        assert profile.symbol == symbol
        assert profile.spread_price("liquid", 95) == pytest.approx(0.10)
        assert profile.spread_price("rollover", 50) == pytest.approx(0.45)
        assert profile.spread_price("opening", 95) > profile.spread_price("opening", 75) > profile.spread_price("opening", 50)

    instrument = InstrumentCost("XAUUSD", 0.1, 10, 1, 2, 0)
    broker = replace(FTMO, instruments={"XAUUSD": instrument})
    args = dict(broker=broker, symbol="XAUUSD", sl_pips=10, tp_pips=15, win_rate_backtest=60)
    fixed = by_id(audit(**args))["D2_spread_breakeven"]
    blocked = by_id(audit(**args, spread_source="bar_minimum", spread_sensitive=True))["D2_spread_breakeven"]
    profile = calibrate_tick_spreads("XAUUSD", ticks, regimes=windows, source="synthetic ticks")
    real = by_id(audit(**args, tick_spreads=profile, spread_regime="opening"))["D2_spread_breakeven"]
    assert fixed.status == "PASS"
    assert blocked.status == real.status == "FAIL"
    assert (fixed.evidence["spread_pips"], real.evidence["spread_pips"]) == (1, 10.5)
    assert real.evidence["spread_tick_samples"] == 50


def test_shift_flags_one_minute_signal_but_not_broad_window():
    index = pd.date_range("2024-01-01", freq="15min", periods=96 * 20)
    gain = pd.Series(0.0, index=index)
    for hour, minute in ((8, 0), (11, 30)):
        gain.loc[(index.hour == hour) & (index.minute == minute)] = np.linspace(0.001, 0.002, 20)

    def replay(start, end):
        def run(offset):
            return gain.where(window_mask(index, start, end, offset), 0).groupby(index.normalize()).sum()
        return run

    fragile = by_id(audit(session_replay=replay(480, 481)))["D3_time_shift_sensitivity"]
    stable = by_id(audit(session_replay=replay(675, 720)))["D3_time_shift_sensitivity"]
    assert fragile.status == "WARN"
    assert fragile.evidence["offsets"]["15"]["expectancy_degradation"] == pytest.approx(1)
    assert fragile.evidence["offsets"]["0"]["expectancy"] == pytest.approx(0.0015)
    assert stable.status == "PASS"
    assert stable.evidence["offsets"]["-15"]["sharpe_degradation"] == pytest.approx(0)


def test_cli_blocks_declared_proxy_without_reference(tmp_path, capsys):
    from xiii.cli import main

    prices = feed(np.random.default_rng(23).normal(0, 0.01, 500))
    path = tmp_path / "feed.csv"
    prices.rename("close").to_csv(path, index_label="date")
    assert main(["audit", str(path), "--proxy-data", str(path), "--json"]) == 1
    output = capsys.readouterr().out
    assert "DQ8_lead_lag" in output and "DQ9_divergence" in output


def test_proxy_aware_feeds_use_same_clock():
    prices = feed(np.random.default_rng(24).normal(0, 0.01, 500)).tz_localize("UTC")
    shifted_clock = prices.tz_convert("America/New_York")
    assert all(c.status == "PASS" for c in dataqc.proxy_checks(shifted_clock, prices))
    assert all(c.status == "FAIL" for c in dataqc.proxy_checks(shifted_clock, prices.tz_localize(None)))
