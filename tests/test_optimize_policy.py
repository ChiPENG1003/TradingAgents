import pandas as pd
import unittest

from back_test.optimize_policy import (
    DECISION_PARAM_KEYS,
    PARAMETER_GROUPS,
    build_walk_forward_folds,
    default_policy_params,
    resolve_parameter_names,
    score_metrics,
    suggest_policy_params,
)


class FakeTrial:
    def suggest_float(self, name, low, high, step=None):
        return low if step is None else low

    def suggest_int(self, name, low, high):
        return low

    def suggest_categorical(self, name, choices):
        return choices[0]


class MidpointTrial(FakeTrial):
    """Returns the midpoint of each float range so floor/cap ordering is exercised."""

    def suggest_float(self, name, low, high, step=None):
        return (low + high) / 2.0


class OptimizePolicyTest(unittest.TestCase):
    def test_suggest_policy_params_contains_engine_search_space(self):
        params = suggest_policy_params(FakeTrial())

        expected = {
            "max_trade_risk_pct",
            "max_single_add_pct",
            "max_position_after_add_pct",
            "max_adds_per_trade",
            "min_days_between_adds",
            "max_entry_gap_above_plan_pct",
            "max_add_gap_above_plan_pct",
            "obvious_bull_max_entry_gap_pct",
            "obvious_bull_max_add_gap_pct",
            "entry_signal_ttl_trading_days",
            "add_signal_ttl_trading_days",
            "block_shrinking_volume_adds",
            "shrinking_volume_close_hold_days",
            "add_key_level_tolerance_pct",
        }
        self.assertEqual(set(params), expected)

    def test_learn_weights_adds_only_scoring_weights(self):
        base = set(suggest_policy_params(FakeTrial()))
        learned = set(suggest_policy_params(FakeTrial(), learn_weights=True))

        self.assertEqual(
            learned - base,
            {
                "momentum_score_weight", "event_score_weight", "risk_score_weight",
                "trend_continuation_weight", "reversal_risk_weight",
                "breakout_quality_weight", "pullback_quality_weight",
                "volume_support_weight", "event_risk_weight",
                "reward_risk_quality_weight",
            },
        )
        # trend_score_weight is pinned to 1.0 downstream, never searched.
        self.assertNotIn("trend_score_weight", learned)

    def test_learn_decision_adds_decision_params(self):
        base = set(suggest_policy_params(FakeTrial()))
        learned = set(suggest_policy_params(FakeTrial(), learn_decision=True))
        self.assertEqual(learned - base, set(DECISION_PARAM_KEYS))

    def test_learn_decision_keeps_caps_at_or_above_floors(self):
        params = suggest_policy_params(MidpointTrial(), learn_decision=True)
        self.assertGreaterEqual(params["strong_uptrend_cap"], params["strong_uptrend_floor"])
        self.assertGreaterEqual(params["weak_uptrend_cap"], params["weak_uptrend_floor"])

    def test_default_policy_params_match_current_risk_budget(self):
        params = default_policy_params()

        self.assertEqual(params["max_trade_risk_pct"], 0.020)
        self.assertIs(params["block_shrinking_volume_adds"], True)

    def test_score_metrics_rewards_return_and_penalizes_drawdown(self):
        good = score_metrics(
            {"total_return": 0.10, "sharpe_ratio": 1.0, "max_drawdown": -0.03, "n_trades": 4},
            {},
        )
        fragile = score_metrics(
            {"total_return": 0.10, "sharpe_ratio": 1.0, "max_drawdown": -0.12, "n_trades": 4},
            {},
        )

        self.assertGreater(good, fragile)

    def test_parameter_groups_are_composable_and_deduplicated(self):
        names = resolve_parameter_names(
            ["execution", "setup_quality"], ["max_trade_risk_pct"]
        )
        self.assertEqual(names.count("max_trade_risk_pct"), 1)
        self.assertIn("breakout_quality_weight", names)
        self.assertEqual(set(PARAMETER_GROUPS["setup_quality"]) - set(names), set())

    def test_selected_parameter_names_limit_trial_search_space(self):
        params = suggest_policy_params(
            FakeTrial(),
            parameter_names=("max_trade_risk_pct", "breakout_quality_weight"),
        )
        self.assertEqual(
            set(params), {"max_trade_risk_pct", "breakout_quality_weight"}
        )

    def test_oracle_objective_hard_rejects_drawdown_breach(self):
        score = score_metrics(
            {"total_return": 0.4, "max_drawdown": -0.30, "n_trades": 5},
            {"oracle": {"capture_ratio": 0.8}},
            objective="oracle_capture",
            max_drawdown_limit=0.25,
            drawdown_mode="hard",
        )
        self.assertLess(score, -999_999)

    def test_build_walk_forward_folds_uses_train_then_next_test_window(self):
        days = pd.Series(pd.date_range("2025-01-01", periods=10, freq="D"))

        folds = build_walk_forward_folds(days, train_days=4, test_days=2, step_days=2)

        self.assertEqual(len(folds), 3)
        self.assertEqual(folds[0].train_start, "2025-01-01")
        self.assertEqual(folds[0].train_end, "2025-01-04")
        self.assertEqual(folds[0].test_start, "2025-01-05")
        self.assertEqual(folds[0].test_end, "2025-01-06")
        self.assertEqual(folds[1].train_start, "2025-01-03")


if __name__ == "__main__":
    unittest.main()
