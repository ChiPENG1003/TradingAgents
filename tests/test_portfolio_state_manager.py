from types import SimpleNamespace

from back_test.policy_config import (
    PortfolioStatePolicyConfig,
    coerce_portfolio_state_policy_config,
)
from tradingagents.agents.managers.portfolio_state_manager import (
    MarketState,
    PortfolioStatePolicyConfig as ManagerPortfolioStatePolicyConfig,
    _fallback_market_state,
    _invoke_market_state,
    _market_state_response_to_model,
    policy_from_market_state,
)


def _market_state_json() -> str:
    return """{
      "schema_version": "state_v1",
      "ticker": "SPY",
      "as_of_date": "2024-01-02",
      "regime": "strong_uptrend",
      "market_phase": "healthy_bull_trend",
      "trend_score": 0.7,
      "risk_score": 0.25,
      "momentum_score": 0.6,
      "event_score": 0.1,
      "confidence": 0.8,
      "horizon_days": 10,
      "thesis": "Trend structure remains constructive.",
      "invalidation_condition": "Loses SMA50 on a closing basis.",
      "key_risks": ["Macro shock", "Volume fades"]
    }"""


def _anchors() -> dict:
    return {
        "current_price": 459.9912,
        "atr14": 3.863,
        "nearest_support": 459.7966,
        "nearest_resistance": 460.3124,
        "recent_high_20d": 464.76,
        "sma20": 454.7778,
        "sma50": 436.8196,
        "sma200": 420.1105,
        "volume_ratio": 1.528,
    }


def test_market_state_response_parses_free_text_json_content():
    response = SimpleNamespace(content=f"```json\n{_market_state_json()}\n```")

    state = _market_state_response_to_model(response)

    assert isinstance(state, MarketState)
    assert state.ticker == "SPY"
    assert state.market_phase == "healthy_bull_trend"


def test_invoke_market_state_falls_back_when_structured_output_unsupported():
    class ReasonerLikeLLM:
        def with_structured_output(self, _schema):
            raise NotImplementedError("deepseek-reasoner does not support tool_choice")

        def invoke(self, _prompt):
            return SimpleNamespace(content=_market_state_json())

    state = _invoke_market_state(
        ReasonerLikeLLM(),
        "classify",
        "SPY",
        "2024-01-02",
        _anchors(),
        "expanding",
    )

    assert state.regime == "strong_uptrend"
    assert state.confidence == 0.8


def test_invoke_market_state_skips_structured_for_deepseek_thinking_mode():
    class ThinkingLLM:
        model_name = "deepseek-v4-flash"
        extra_body = {"thinking": {"type": "enabled"}}

        def with_structured_output(self, _schema):
            raise AssertionError("structured output should be skipped")

        def invoke(self, _prompt):
            return SimpleNamespace(content=_market_state_json())

    state = _invoke_market_state(
        ThinkingLLM(),
        "classify",
        "SPY",
        "2024-01-02",
        _anchors(),
        "expanding",
    )

    assert state.market_phase == "healthy_bull_trend"


def test_invoke_market_state_uses_anchor_fallback_for_invalid_json():
    class BadJsonLLM:
        def with_structured_output(self, _schema):
            raise NotImplementedError("unsupported")

        def invoke(self, _prompt):
            return SimpleNamespace(content="I cannot produce JSON here.")

    state = _invoke_market_state(
        BadJsonLLM(),
        "classify",
        "SPY",
        "2024-01-02",
        _anchors(),
        "expanding",
    )

    assert state.ticker == "SPY"
    assert state.regime == "strong_uptrend"
    assert state.market_phase == "accelerating_bull"
    assert state.confidence == 0.50


def test_fallback_market_state_is_valid_market_state():
    state = _fallback_market_state("SPY", "2024-01-02", _anchors(), "expanding")

    assert isinstance(state, MarketState)
    assert state.as_of_date == "2024-01-02"


def test_policy_config_moved_to_back_test_module_with_manager_compat_export():
    config = coerce_portfolio_state_policy_config(
        {"volume_multipliers": {"soft": 0.6}, "range_cap": 0.25}
    )

    assert ManagerPortfolioStatePolicyConfig is PortfolioStatePolicyConfig
    assert config.range_cap == 0.25
    assert config.volume_multipliers["soft"] == 0.6
    assert config.volume_multipliers["normal"] == 1.0


def test_market_context_bearish_blocks_new_stock_entry():
    stock_state = MarketState.model_validate_json(_market_state_json())
    index_state = MarketState.model_validate_json(_market_state_json())
    index_state.regime = "breakdown_risk"
    index_state.market_phase = "early_bear_reversal"
    index_state.trend_score = -0.35
    index_state.momentum_score = -0.25
    index_state.risk_score = 0.65

    strategy = policy_from_market_state(
        stock_state,
        _anchors(),
        holdings_info={},
        constraints={"bearish_volume_divergence": False},
        volume_regime="normal",
        market_context_state=index_state,
        market_context_ticker="^GSPC",
    )

    assert strategy.action == "HOLD"
    assert "market_context bearish blocks new entries" in strategy.rationale_summary
