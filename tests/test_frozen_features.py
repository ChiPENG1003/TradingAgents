import json
from pathlib import Path
from unittest.mock import patch

from tradingagents.agents.managers.portfolio_state_manager import (
    freeze_feature_snapshot,
    policy_from_frozen_snapshot,
)
from tradingagents.agents.schemas import (
    LiveDecisionRow,
    LivePortfolioDecision,
    render_live_portfolio_decision,
)


ROOT = Path(__file__).resolve().parents[1]


def _sample_v1():
    path = sorted((ROOT / "back_test" / "features" / "AAPL").glob("*.json"))[0]
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_snapshot_is_quant_only_and_hash_stable():
    first = freeze_feature_snapshot(_sample_v1())
    second = freeze_feature_snapshot(_sample_v1())
    assert first == second
    assert first["schema_version"] == "frozen_feature_v2"
    assert first["snapshot_hash"] == second["snapshot_hash"]
    state = first["market_state"]
    assert "state_summary" not in state
    assert "key_risks" not in state
    assert "evidence" not in state
    assert "invalidation" not in state
    assert "prior_policy_output_do_not_use_as_label" not in first
    assert "event_features" not in first
    assert "execution_context" not in first
    assert first["feature_columns"]["ohlcv_anchors"] == list(first["ohlcv_anchors"])
    assert first["structure_features"]["long_term_structure"]["trend"] is not None


def test_policy_from_frozen_snapshot_performs_no_market_io():
    frozen = freeze_feature_snapshot(_sample_v1())
    with patch(
        "tradingagents.agents.managers.portfolio_state_manager.load_ohlcv",
        side_effect=AssertionError("market I/O is forbidden"),
    ):
        strategy = policy_from_frozen_snapshot(frozen, holdings_info={"quantity": 0})
    assert strategy["ticker"] == "AAPL"
    assert strategy["schema_version"] == "v3"


def test_breakout_and_pullback_weights_change_at_least_one_frozen_order():
    paths = sorted((ROOT / "back_test" / "features" / "AAPL").glob("*.json"))
    for weight in ("breakout_quality_weight", "pullback_quality_weight"):
        changed = False
        for path in paths:
            frozen = freeze_feature_snapshot(json.loads(path.read_text(encoding="utf-8")))
            baseline = policy_from_frozen_snapshot(
                frozen, holdings_info={"quantity": 0}, policy_config={weight: 0.0}
            )
            learned = policy_from_frozen_snapshot(
                frozen, holdings_info={"quantity": 0}, policy_config={weight: 1.0}
            )
            if any(
                baseline[key] != learned[key]
                for key in ("action", "entry", "add_position")
            ):
                changed = True
                break
        assert changed, f"{weight} must affect at least one frozen order"


def test_live_decision_renders_table_first_and_computes_nav():
    decision = LivePortfolioDecision(
        ticker="SOUN",
        as_of_date="2026-08-19",
        decision="BUY",
        current_price=8.02,
        plan_rows=[LiveDecisionRow(
            item="底仓",
            operation="HOLD",
            trigger_description="现价 8.02",
            shares=260,
            reference_price=8.02,
        )],
        holding_period="days to weeks",
        rationale="test",
    )
    rendered = render_live_portfolio_decision(decision, {"equity": 20_000})
    assert rendered.startswith("| 项目 | 触发条件/价格 | 股数 | 参考金额 | 占 NAV |")
    assert "260 股" in rendered
    assert "约 2,085" in rendered
    assert "10.4%" in rendered
