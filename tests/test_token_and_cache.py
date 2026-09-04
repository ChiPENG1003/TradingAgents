"""Token accounting and provider-aware prompt caching."""

import pytest
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from tradingagents.agents.utils.prompt_cache import (
    MIN_CACHEABLE_CHARS,
    build_cached_prompt,
)
from tradingagents.agents.utils.token_usage import TokenUsageTracker, attribute_to


# --------------------------------------------------------------- prompt cache

class _FakeOpenAI:
    pass


class _FakeAnthropic:
    pass


_FakeAnthropic.__module__ = "langchain_anthropic.chat_models"

STATIC = "S" * (MIN_CACHEABLE_CHARS + 10)


def test_openai_style_provider_gets_one_concatenated_string():
    """Automatic prefix caching needs no markers, only static-first ordering."""
    out = build_cached_prompt(_FakeOpenAI(), STATIC, "VARIABLE")
    assert isinstance(out, str)
    assert out.startswith(STATIC) and out.endswith("VARIABLE")


def test_anthropic_gets_a_cache_breakpoint_after_the_static_block():
    out = build_cached_prompt(_FakeAnthropic(), STATIC, "VARIABLE")
    assert isinstance(out, list) and isinstance(out[0], HumanMessage)
    blocks = out[0].content
    assert blocks[0]["text"] == STATIC
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert blocks[1]["text"] == "VARIABLE"
    assert "cache_control" not in blocks[1]


def test_short_static_prefix_is_not_worth_a_breakpoint():
    out = build_cached_prompt(_FakeAnthropic(), "too short", "VARIABLE")
    assert out == "too shortVARIABLE"


def test_market_state_static_prefix_carries_no_interpolation():
    """A per-run value in the static half would stop the prefix from matching."""
    import re
    from pathlib import Path

    src = Path("tradingagents/agents/managers/portfolio_state_manager.py").read_text()
    start = src.index('state_prompt_static = """') + len('state_prompt_static = """')
    static = src[start:src.index('"""', start)]

    assert len(static) > MIN_CACHEABLE_CHARS
    assert not re.search(r"\{\w+\}", static), "static prefix must not interpolate"


# ------------------------------------------------------------- token tracking

def _response(input_tokens, output_tokens, cache_read=0):
    message = SimpleNamespace(usage_metadata={
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_token_details": {"cache_read": cache_read},
    })
    return SimpleNamespace(generations=[[SimpleNamespace(message=message)]], llm_output=None)


def test_usage_is_attributed_to_the_running_node():
    tracker = TokenUsageTracker()
    with attribute_to("Market Analyst"):
        tracker.on_llm_end(_response(1000, 200, cache_read=800))
        tracker.on_llm_end(_response(1500, 300))
    with attribute_to("Bull Researcher"):
        tracker.on_llm_end(_response(400, 100))

    market = tracker.by_node["Market Analyst"]
    assert market.calls == 2
    assert market.input_tokens == 2500
    assert market.output_tokens == 500
    assert market.cache_read_tokens == 800
    assert market.cache_hit_rate == 800 / 2500
    assert tracker.totals.input_tokens == 2900
    assert tracker.by_node["Bull Researcher"].calls == 1


def test_usage_outside_a_node_is_not_silently_dropped():
    tracker = TokenUsageTracker()
    tracker.on_llm_end(_response(10, 5))
    assert tracker.by_node["unattributed"].calls == 1


def test_llm_output_fallback_when_the_message_carries_no_usage():
    tracker = TokenUsageTracker()
    response = SimpleNamespace(
        generations=[[SimpleNamespace(message=SimpleNamespace(usage_metadata=None))]],
        llm_output={"token_usage": {"prompt_tokens": 70, "completion_tokens": 30}},
    )
    with attribute_to("Trader"):
        tracker.on_llm_end(response)
    assert tracker.by_node["Trader"].input_tokens == 70
    assert tracker.by_node["Trader"].output_tokens == 30


def test_responses_without_usage_are_counted_not_guessed():
    tracker = TokenUsageTracker()
    tracker.on_llm_end(SimpleNamespace(generations=[], llm_output=None))
    assert tracker.by_node == {}
    assert tracker.responses_without_usage == 1


def test_report_and_render_survive_an_empty_run():
    tracker = TokenUsageTracker()
    assert "no LLM usage recorded" in tracker.render()
    assert tracker.report()["totals"]["calls"] == 0


# ------------------------------------------------------------------- pricing

from tradingagents.llm_clients.model_catalog import (  # noqa: E402
    MODEL_PRICING,
    estimate_cost,
    normalize_model_id,
)


def test_dated_snapshot_ids_resolve_to_family_pricing():
    """Providers echo the resolved snapshot, prices are published per family."""
    assert normalize_model_id("gpt-5.4-mini-2026-03-17") == "gpt-5.4-mini"
    assert normalize_model_id("gpt-5.6-luna") == "gpt-5.6-luna"
    assert estimate_cost("gpt-5.4-mini-2026-03-17", 1_000_000, 0) == pytest.approx(0.75)


def test_cached_tokens_are_a_discount_not_an_addition():
    fresh_only = estimate_cost("gpt-5.6-luna", 1_000_000, 0, cached_input_tokens=0)
    all_cached = estimate_cost("gpt-5.6-luna", 1_000_000, 0, cached_input_tokens=1_000_000)
    assert fresh_only == pytest.approx(0.20)
    assert all_cached == pytest.approx(0.02)


def test_unknown_model_yields_none_rather_than_a_wrong_number():
    assert estimate_cost("some-unlisted-model", 1000, 1000) is None


def test_tracker_reports_unknown_cost_as_none_not_a_partial_total():
    """One unpriced call must not leave a total that looks complete."""
    tracker = TokenUsageTracker()
    priced = _response(1000, 100)
    priced.llm_output = {"model_name": "gpt-5.6-luna"}
    unpriced = _response(1000, 100)
    unpriced.llm_output = {"model_name": "mystery-model"}
    with attribute_to("Market Analyst"):
        tracker.on_llm_end(priced)
        tracker.on_llm_end(unpriced)
    assert tracker.by_node["Market Analyst"].cost_usd is None
    assert tracker.totals.cost_usd is None
    assert "n/a" in tracker.render()


def test_every_catalogued_openai_model_that_we_default_to_has_a_price():
    from tradingagents.default_config import DEFAULT_CONFIG
    for key in ("quick_think_llm", "deep_think_llm"):
        assert DEFAULT_CONFIG[key] in MODEL_PRICING, key


# ------------------------------------------------------- DeepSeek peak/off-peak

from datetime import datetime, timezone  # noqa: E402

from tradingagents.llm_clients.model_catalog import is_peak  # noqa: E402

WED_PEAK = datetime(2026, 9, 2, 7, 0, tzinfo=timezone.utc)
WED_OFF = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
SAT_SAME_HOUR = datetime(2026, 9, 5, 7, 0, tzinfo=timezone.utc)


def test_peak_windows_are_weekday_only():
    assert is_peak(WED_PEAK)
    assert not is_peak(WED_OFF)
    assert not is_peak(SAT_SAME_HOUR), "weekends are off-peak all day"


def test_deepseek_costs_double_during_peak():
    off = estimate_cost("deepseek-v4-flash", 1_000_000, 1_000_000, at=WED_OFF)
    peak = estimate_cost("deepseek-v4-flash", 1_000_000, 1_000_000, at=WED_PEAK)
    assert off == pytest.approx(0.22 + 0.66)
    assert peak == pytest.approx(off * 2)


def test_openai_pricing_ignores_the_deepseek_peak_window():
    """Only DeepSeek has peak pricing; OpenAI must not be doubled with it."""
    at_peak = estimate_cost("gpt-5.6-luna", 1_000_000, 0, at=WED_PEAK)
    at_off = estimate_cost("gpt-5.6-luna", 1_000_000, 0, at=WED_OFF)
    assert at_peak == at_off == pytest.approx(0.20)


def test_deepseek_aliases_resolve_to_a_priced_api_model():
    """The factory bills the resolved id, so that is what must be priced."""
    from tradingagents.llm_clients.factory import _resolve_deepseek_alias
    from tradingagents.llm_clients.model_catalog import MODEL_PRICING

    for alias in ("deepseek-v4-pro", "deepseek-v4-flash-thinking",
                  "deepseek-v4-flash-instant"):
        api_model, _thinking = _resolve_deepseek_alias(alias)
        assert api_model in MODEL_PRICING, alias


def test_resolved_deepseek_ids_do_not_warn_as_unknown():
    from tradingagents.llm_clients.validators import validate_model
    assert validate_model("deepseek", "deepseek-v4-flash")
    assert validate_model("deepseek", "deepseek-v4-pro")
