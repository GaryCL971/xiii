"""Tests for daily C2 and its shared numerical kernel."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

from xiii import FTMO, audit
from xiii._risk_paths import block_bootstrap, breach_paths, courbe_equite, path_drawdowns
from xiii.checks.montecarlo import c2_montecarlo_constraints


class MonteCarloTests(unittest.TestCase):
    def test_lab_parity(self):
        root = Path(__file__).resolve().parents[2]
        if not (root / "lab_agents" / "statistiques.py").is_file():
            self.skipTest("Parity test requires the parent laboratory checkout.")
        sys.path.insert(0, str(root))
        try:
            from lab_agents import statistiques
        finally:
            sys.path.pop(0)
        returns = np.tile(np.r_[np.full(12, -0.008), np.full(18, 0.005)], 4)
        paths = block_bootstrap(returns, graine=41)
        np.testing.assert_array_equal(paths, statistiques.block_bootstrap(returns, graine=41))
        np.testing.assert_array_equal(courbe_equite(paths), statistiques.courbe_equite(paths))
        prob = statistiques.probabilite_breach_chemin(paths)
        self.assertEqual(c2_montecarlo_constraints(None, FTMO, seed=41, daily_returns=returns)
                         .evidence["p_breach_total_dd"], round(prob, 3))

    def test_seed_and_generator_are_equivalent(self):
        returns = np.arange(45) / 1000
        a = block_bootstrap(returns, graine=8)
        b = block_bootstrap(returns, graine=np.random.default_rng(8))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (2000, 45))

    def test_circular_blocks_preserve_order_and_serial_structure(self):
        index = np.arange(61, dtype=float)
        blocks = block_bootstrap(index, n_chemins=200, graine=9)
        for start in range(0, index.size, 20):
            np.testing.assert_array_equal(np.diff(blocks[:, start:start+20], axis=1) % index.size, 1)
        self.assertTrue(np.any(np.diff(blocks[:, :20], axis=1) < 0))

        returns = np.tile(np.r_[np.ones(10), -np.ones(10)], 5)
        ordered = block_bootstrap(returns, n_chemins=200, graine=9)
        rng = np.random.default_rng(9)
        shuffled = np.array([rng.permutation(returns) for _ in ordered])
        lag = lambda a: np.corrcoef(a[:, :-1].ravel(), a[:, 1:].ravel())[0, 1]
        self.assertGreater(lag(ordered), 0.7)
        self.assertLess(abs(lag(shuffled)), 0.1)

    def test_composition_starting_peak_and_absorbing_ruin(self):
        returns = np.array([[-0.12, 0.20, -1.10, 0.50]])
        equity = courbe_equite(returns)
        np.testing.assert_allclose(equity, [[0.88, 1.056, 0.0, 0.0]])
        np.testing.assert_allclose(path_drawdowns(equity), [[-0.12, 0.0, -1.0, -1.0]])
        self.assertFalse(np.allclose(equity, 1 + np.cumsum(returns, axis=1)))
        np.testing.assert_array_equal(breach_paths(courbe_equite([[-0.10], [-0.099]])), [True, False])
        result = c2_montecarlo_constraints(None, FTMO, daily_returns=np.full(40, -1.10))
        self.assertEqual((result.evidence["p_ruin"], result.evidence["p_breach_total_dd"]), (1, 1))

    def test_short_empty_and_nonfinite(self):
        for n in (0, 19, 39):
            with self.subTest(n=n):
                returns = np.zeros(n)
                with self.assertRaisesRegex(ValueError, "trop courte"):
                    block_bootstrap(returns)
                self.assertEqual(c2_montecarlo_constraints(None, FTMO, daily_returns=returns).status, "SKIP")
        a = block_bootstrap(np.r_[np.zeros(40), np.nan, np.inf])
        self.assertEqual(a.shape, (2000, 40))

    def test_trade_api_aggregates_dates_and_counts_daily_loss(self):
        dates = pd.bdate_range("2024-01-01", periods=40)
        trades = pd.DataFrame({"datetime": dates.repeat(2), "pnl_usd": -3000.0})
        result = c2_montecarlo_constraints(trades, FTMO, 1.0, 200, 13)
        self.assertEqual((result.evidence["n_days"], result.evidence["n_trades"]), (40, 80))
        self.assertEqual(result.evidence["p_breach_daily_dd"], 1.0)
        self.assertEqual(c2_montecarlo_constraints(trades.drop(columns="datetime"), FTMO).status, "SKIP")

    def test_audit_uses_daily_returns_with_trades_and_preserves_structural_skip(self):
        dates = pd.bdate_range("2024-01-01", periods=40)
        returns = pd.Series(0.004, index=dates)
        trades = pd.DataFrame({"datetime": dates, "pnl_usd": -3000.0})

        def c2_report(**kwargs):
            return next(v for v in audit(returns=returns, broker=FTMO, **kwargs).verdicts
                        if v.check_id == "C2_montecarlo_constraints")

        result = c2_report(trades=trades)
        self.assertEqual((result.status, result.evidence["returns_source"]), ("PASS", "daily_returns"))
        self.assertEqual(c2_report().status, "SKIP")


if __name__ == "__main__":
    unittest.main()
