"""Offline order-regeneration tests (the 1B closed-loop weight-learning path).

These exercise `regenerate_strategy` and the BacktestEngine `regenerate_orders`
hook against the committed ORCL strategy fixtures. They need cached ORCL OHLCV
under ~/.tradingagents/cache (anchors are recomputed from it); when the cache is
absent `regenerate_strategy` returns None and the test self-skips rather than
failing, so the suite stays green on a network-less machine.
"""

import glob
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORCL_DIR = ROOT / "back_test" / "strategy" / "ORCL"
ORDER_KEYS = ("action", "entry", "add_position", "take_profit", "reduce_stop", "stop_loss")


def _load_orcl():
    return [json.load(open(f)) for f in sorted(glob.glob(str(ORCL_DIR / "ORCL_*.json")))]


@unittest.skipUnless(ORCL_DIR.exists(), "ORCL strategy fixtures not present")
class RegenerateStrategyTest(unittest.TestCase):
    def test_default_config_reproduces_flat_entry_orders(self):
        """A saved fresh `entry` implies flat holdings at generation time.

        Regenerating those dates with flat holdings and the policy's default
        config must reproduce the saved orders byte-for-byte — proof the
        reconstruction feeds policy_from_market_state the same inputs the
        generator did.
        """
        from tradingagents.agents.managers.portfolio_state_manager import (
            regenerate_strategy,
            _load_recent_phases,
        )

        flat_entry = [
            d for d in _load_orcl()
            if d.get("market_state") and (d.get("entry", {}).get("size_pct", 0) or 0) > 0
        ]
        self.assertGreater(len(flat_entry), 0, "no flat-entry fixtures to check")

        checked = 0
        for d in flat_entry:
            regen = regenerate_strategy(
                ticker="ORCL",
                as_of_date=d["as_of_date"],
                market_state_dict=d["market_state"],
                holdings_info={"quantity": 0.0},
                recent_phases=_load_recent_phases("ORCL", d["as_of_date"], n=2),
                policy_config=None,
            )
            if regen is None:
                self.skipTest("ORCL OHLCV cache unavailable; cannot recompute anchors")
            for key in ORDER_KEYS:
                self.assertEqual(regen.get(key), d.get(key), f"{d['as_of_date']} {key}")
            checked += 1
        self.assertGreater(checked, 0)

    def test_risk_weight_changes_at_least_one_order(self):
        """The whole point of 1B: a scoring weight must move some order."""
        from tradingagents.agents.managers.portfolio_state_manager import regenerate_strategy

        changed = False
        any_run = False
        for d in _load_orcl():
            ms = d.get("market_state")
            if not ms:
                continue
            common = dict(
                ticker="ORCL",
                as_of_date=d["as_of_date"],
                market_state_dict=ms,
                holdings_info={"quantity": 0.0},
                recent_phases=None,
            )
            lo = regenerate_strategy(**common, policy_config={"trend_score_weight": 1.0, "risk_score_weight": 0.1})
            hi = regenerate_strategy(**common, policy_config={"trend_score_weight": 1.0, "risk_score_weight": 8.0})
            if lo is None or hi is None:
                self.skipTest("ORCL OHLCV cache unavailable; cannot recompute anchors")
            any_run = True
            if lo["action"] != hi["action"] or lo["entry"] != hi["entry"] or lo["add_position"] != hi["add_position"]:
                changed = True
                break
        self.assertTrue(any_run)
        self.assertTrue(changed, "risk_score_weight should change at least one regenerated order")


@unittest.skipUnless(ORCL_DIR.exists(), "ORCL strategy fixtures not present")
class EngineRegenerationTest(unittest.TestCase):
    def test_run_backtest_metrics_enables_regeneration_with_weights(self):
        """When weight keys are present, run_backtest_metrics must regenerate."""
        from back_test.optimize_policy import run_backtest_metrics, default_policy_params

        params = dict(default_policy_params())
        params.update({"momentum_score_weight": 0.5, "event_score_weight": 0.3, "risk_score_weight": 1.0})
        _metrics, report, _trades = run_backtest_metrics(
            "ORCL", "2026-04-01", "2026-05-28", params, initial_capital=100_000.0,
        )
        if not report:
            self.skipTest("ORCL OHLCV cache unavailable; backtest produced no report")
        self.assertGreater(report.get("strategies_regenerated", 0), 0)

    def test_execution_only_params_do_not_regenerate(self):
        """Without weight keys, behaviour is unchanged (pure replay)."""
        from back_test.optimize_policy import run_backtest_metrics, default_policy_params

        _metrics, report, _trades = run_backtest_metrics(
            "ORCL", "2026-04-01", "2026-05-28", default_policy_params(), initial_capital=100_000.0,
        )
        if not report:
            self.skipTest("ORCL OHLCV cache unavailable; backtest produced no report")
        self.assertEqual(report.get("strategies_regenerated", 0), 0)


if __name__ == "__main__":
    unittest.main()
