# XIII — The Thirteenth Man Linter for Backtests

**Stop lying to yourself about your backtest. XIII catches the 8 ways you're doing it.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-v0.1.0.dev0-yellow.svg)](CHANGELOG.md)

---

## What is XIII?

A linter for overfitting in trading backtests. Inspired by 3 years of backtest debt and the realization that the only profitable trades in my account were bugs in the code.

XIII checks your backtest against the **Thirteenth Man Protocol** — 8 gates designed to catch every type of lie a backtest can tell:

```
A. Livrable integrity       → no fake stand-ins (np.random, TODO, FIXME)
B. Window honesty           → no Sharpe inflation from lucky periods
C. Sizing honesty           → broker limits respected under stress
D. Brick-by-brick audit     → every component actually helps
E. Crisis correlations      → components stay uncorrelated when it matters
F. Falsify your fears       → test with data, not theory
G. Execution fidelity       → EA reproduces the exact rule
H. Pre-mortem              → write "how this dies" BEFORE going live
```

---

## Quick Start

### Install

```bash
pip install xiii
```

### Audit Your Backtest

```bash
# Minimal (checks B1: window mirage, G1: lookahead)
xiii.audit equity.csv

# Full (with broker rules, spread cost, sizing)
xiii.audit equity.csv trades.csv --broker ftmo --deployed-sizing 1.08
```

### Python API

```python
import xiii
import pandas as pd

# Load your data
equity = pd.Series(...)  # your backtest equity curve
trades = pd.DataFrame(...)  # your trade log

# Run the audit
report = xiii.audit(
    equity=equity,
    trades=trades,
    source_file="my_strategy.py",
    broker=xiii.FTMO,
    deployed_sizing=1.08,
    sl_pips=15,
    tp_pips=30,
)

# Print the verdict
report.print()

# Or export as JSON
import json
print(json.dumps(json.loads(report.to_json()), indent=2))
```

---

## What XIII Catches

**Real examples from live trading:**

| Check | The Lie | What XIII Does |
|-------|---------|---|
| **B1** | Sharpe 2.53 on 2.3 years, but 1.22 on 16 years | Compares full history vs lucky windows; flags +49% inflation |
| **A1** | Backtest data is `np.random.normal(0.0005, 0.0008)` | Scans code for dummy data, generators, placeholders |
| **A3** | `concat().dropna()` collapses monthly×daily to 23 days | Detects silent inner-joins that erase history |
| **C1** | Sizing 1.08× → drawdown hits −11.8% (FTMO limit −10%) | Validates sizing respects broker rules under stress |
| **C2** | 50 trades look good, but Monte Carlo shows 0% pass FTMO | Bootstrap reorders to test robustness |
| **D2** | Assumes 0.5p spread, broker is 1.6p → breakeven WR jumps 33%→37% | Recalculates profit targets after real costs |
| **G1** | Signal uses `rolling(center=True)` = half future | Scans for lookahead bias, `.shift(-n)` |

---

## Status: v0.1.0.dev0

**Active checks:** 8 of 9

- ✅ A1, A3, B1, B3 (static code analysis + data checks)
- ✅ C1, C2, D2 (broker constraints + costs)
- ✅ G1 (lookahead detection)
- ⏳ DE (multi-brick correlation — coming soon)

**Supported brokers:** FTMO, Vantage (GBPUSD)

**Brokers coming in v1.0:** Interactive Brokers, Ninjatrader, others (via customizable `BrokerConfig`)

---

## The Story

Read **[The Essay](content/ESSAI_PHARE_DRAFT_EN.md)** to understand why XIII exists:

> I spent 3 years building trading robots. The only strategies that ever made real money were bugs in the code. That taught me exactly what to ask a backtest to make it stop lying.

XIII is the linter that asks those questions.

---

## FAQ

### Is XIII a trading signal?
No. It doesn't predict price direction. It predicts *whether your backtest is honest*.

### Will XIII make me money?
No. XIII will stop you from *believing* you're making money when you're losing it.

### Is this financial advice?
No. It's educational software. Use at your own risk.

### Why open-source?
Because the Thirteenth Man Protocol is an attitude, not IP. Once you know to hunt for your own blindness, no one can stop you.

---

## Contributing

Found a false positive? A missing check? A better way to detect overfitting?

[Open an issue](https://github.com/GaryCL971/xiii/issues) or [submit a PR](https://github.com/GaryCL971/xiii/pulls).

---

## License

Apache 2.0 — see [LICENSE](LICENSE)

---

## Author

Built by [GaryCL971](https://github.com/GaryCL971) from 3 years of backtest mistakes.

If XIII saves you 6 months of wasted work, consider [subscribing to the research](https://substack.com/GaryCL971) or [sending me your backtest for a red-team audit](mailto:redacted@example.com).

---

**Made from backtest debt and zero regrets.**
