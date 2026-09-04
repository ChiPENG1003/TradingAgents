import pandas as pd

from back_test.oracle import constraint_matched_oracle, oracle_capture_ratio


def test_oracle_uses_hindsight_but_respects_position_cap():
    prices = pd.Series([100.0, 110.0, 90.0, 120.0])
    full = constraint_matched_oracle(
        prices, initial_capital=100_000, max_position_weight=1.0, weight_steps=10
    )
    half = constraint_matched_oracle(
        prices, initial_capital=100_000, max_position_weight=0.5, weight_steps=10
    )
    assert full.total_return > half.total_return > 0
    assert max(half.target_weights) <= 0.5


def test_oracle_charges_costs_and_capture_ratio_is_normalized():
    prices = pd.Series([100.0, 110.0, 90.0, 120.0])
    free = constraint_matched_oracle(prices, initial_capital=100_000)
    costly = constraint_matched_oracle(
        prices, initial_capital=100_000, commission=10.0, slippage_bps=20.0
    )
    assert costly.total_return < free.total_return
    assert oracle_capture_ratio(free.total_return, free.total_return) == 1.0
