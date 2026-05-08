"""Backtest-only Portfolio State Manager.

State-first refactor of portfolio decision-making for backtest mode:
- The LLM only emits a qualitative MarketState (regime + scores + thesis).
- Deterministic Python policy converts MarketState into the existing
  PortfolioStrategy order schema using anchors and rule constraints.

Live mode continues to use create_portfolio_manager from portfolio_manager.py.
"""

import json
import logging
import re
import warnings
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from back_test.policy_config import (
    PortfolioStatePolicyConfig,
    coerce_portfolio_state_policy_config,
    _DEFAULT_PHASE_MODIFIER,
    _DEFAULT_VOLUME_MULTIPLIER,
)
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.managers.portfolio_manager import (
    PortfolioStrategy,
    PriceSizeBlock,
    StopLossBlock,
    _classify_volume_regime,
    _enforce_strategy_rules,
    _is_broad_index_instrument,
)
from tradingagents.dataflows.stockstats_utils import load_ohlcv

logger = logging.getLogger(__name__)

__all__ = [
    "PortfolioStatePolicyConfig",
    "MarketState",
    "create_portfolio_state_manager",
    "create_market_aware_portfolio_state_manager",
    "policy_from_market_state",
]


def _apply_order_size_multiplier(strategy: dict, multiplier: float) -> dict:
    """Scale deterministic order sizes without exposing the multiplier to the LLM."""
    if multiplier <= 0:
        raise ValueError(f"order_size_multiplier must be > 0, got {multiplier}")
    if multiplier == 1.0:
        return strategy

    for key in ("entry", "add_position", "take_profit", "reduce_stop"):
        block = strategy.get(key)
        if not isinstance(block, dict):
            continue
        size_pct = float(block.get("size_pct") or 0.0)
        if size_pct <= 0:
            continue
        block["size_pct"] = round(min(size_pct * multiplier, 100.0), 1)

    rationale = strategy.get("rationale_summary") or ""
    strategy["rationale_summary"] = (
        f"{rationale} | hardcoded_order_size_multiplier={multiplier:g}"
    )
    return strategy


class MarketState(BaseModel):
    schema_version: Literal["state_v1"] = "state_v1"
    ticker: str
    as_of_date: str

    regime: Literal[
        "strong_uptrend",
        "weak_uptrend",
        "range",
        "breakdown_risk",
        "downtrend",
        "event_driven",
        "unclear",
    ]

    market_phase: Literal[
        # Bull
        "early_bull_reversal",
        "healthy_bull_trend",
        "accelerating_bull",
        "overextended_bull",
        "bull_pullback",
        "late_bull_distribution",
        # Bear
        "early_bear_reversal",
        "healthy_bear_trend",
        "accelerating_bear",
        "oversold_bear",
        "bear_rally",
        "late_bear_exhaustion",
        # Neutral
        "range_compression",
        "high_volatility_range",
        "macro_event_regime",
        "unclear",
    ]

    trend_score: float = Field(ge=-1.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    momentum_score: float = Field(ge=-1.0, le=1.0)
    event_score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    horizon_days: int = Field(ge=1, le=60)

    thesis: str
    invalidation_condition: str
    key_risks: list[str]


def _find_market_state(value, seen: Optional[set[int]] = None) -> Optional[MarketState]:
    if value is None:
        return None
    if isinstance(value, MarketState):
        return value
    if seen is None:
        seen = set()
    obj_id = id(value)
    if obj_id in seen:
        return None
    seen.add(obj_id)

    if isinstance(value, dict):
        try:
            return MarketState.model_validate(value)
        except Exception:
            pass
        for nested in value.values():
            found = _find_market_state(nested, seen)
            if found is not None:
                return found
        return None

    for attr in ("parsed", "content", "output"):
        nested = getattr(value, attr, None)
        found = _find_market_state(nested, seen)
        if found is not None:
            return found

    if isinstance(value, (list, tuple)):
        for nested in value:
            found = _find_market_state(nested, seen)
            if found is not None:
                return found

    if isinstance(value, str):
        return _market_state_from_text(value)

    additional_kwargs = getattr(value, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        found = _find_market_state(additional_kwargs, seen)
        if found is not None:
            return found

    return None


def _market_state_from_text(text: str) -> Optional[MarketState]:
    """Parse a free-text JSON MarketState fallback response."""
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return MarketState.model_validate_json(candidate)
        except Exception:
            try:
                return MarketState.model_validate(json.loads(candidate))
            except Exception:
                continue
    return None


def _market_state_response_to_model(response) -> MarketState:
    state = _find_market_state(response)
    if state is None:
        raise TypeError(
            f"Structured output did not contain MarketState: {type(response).__name__}"
        )
    return state


def _llm_disallows_structured_output(llm) -> bool:
    """Return True for known providers/modes that reject tool_choice."""
    model = (getattr(llm, "model_name", None) or getattr(llm, "model", "") or "").lower()
    if model == "deepseek-reasoner":
        return True

    extra_body = getattr(llm, "extra_body", None) or {}
    thinking = extra_body.get("thinking") if isinstance(extra_body, dict) else None
    return isinstance(thinking, dict) and thinking.get("type") == "enabled"


def _compute_short_term_market_anchors(
    ticker: str,
    trade_date: str,
    lookback_days: int = 80,
) -> Optional[dict]:
    """Return short-horizon numeric anchors for 1-5 trading day decisions."""
    try:
        df = load_ohlcv(ticker, trade_date)
    except Exception:
        return None
    if df is None or df.empty:
        return None

    df = df.sort_values("Date").tail(lookback_days).reset_index(drop=True)
    if len(df) < 2:
        return None

    last = df.iloc[-1]
    current_close = float(last["Close"])
    if current_close <= 0:
        return None

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    volume = pd.to_numeric(df.get("Volume"), errors="coerce") if "Volume" in df else None

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    def _sma(window: int) -> Optional[float]:
        if len(close) < window:
            return None
        return float(close.tail(window).mean())

    def _ema(window: int) -> Optional[float]:
        if len(close) < window:
            return None
        return float(close.ewm(span=window, adjust=False).mean().iloc[-1])

    def _nearest_resistance(window: int) -> Optional[float]:
        slice_ = high.tail(window)
        above = slice_[slice_ > current_close]
        return float(above.min()) if not above.empty else None

    def _nearest_support(window: int) -> Optional[float]:
        slice_ = low.tail(window)
        below = slice_[slice_ < current_close]
        return float(below.max()) if not below.empty else None

    atr5 = float(true_range.tail(5).mean()) if len(true_range) >= 5 else float(true_range.mean())
    atr14 = float(true_range.tail(14).mean()) if len(true_range) >= 14 else float(true_range.mean())

    resistance = _nearest_resistance(5) or _nearest_resistance(10) or _nearest_resistance(20)
    support = _nearest_support(5) or _nearest_support(10) or _nearest_support(20)
    latest_volume = float(volume.iloc[-1]) if volume is not None and pd.notna(volume.iloc[-1]) else None
    volume_20_sma = (
        float(volume.tail(20).mean())
        if volume is not None and len(volume.dropna()) >= 20
        else None
    )
    volume_ratio = (
        latest_volume / volume_20_sma
        if latest_volume is not None and volume_20_sma not in (None, 0)
        else None
    )
    volume_ratio_3d = None
    if volume is not None and volume_20_sma not in (None, 0) and len(volume.dropna()) >= 20:
        volume_20_series = volume.rolling(20).mean()
        ratio_series = volume / volume_20_series
        recent_ratios = ratio_series.tail(3).dropna()
        if len(recent_ratios) == 3:
            volume_ratio_3d = [round(float(v), 3) for v in recent_ratios.tolist()]

    return {
        "as_of_close_date": pd.to_datetime(last["Date"]).strftime("%Y-%m-%d"),
        "current_price": round(current_close, 4),
        "atr5": round(atr5, 4),
        "atr14": round(atr14, 4),
        "atr14_pct": round(atr14 / current_close * 100.0, 3),
        "ema5": round(_ema(5), 4) if _ema(5) is not None else None,
        "ema10": round(_ema(10), 4) if _ema(10) is not None else None,
        "ema20": round(_ema(20), 4) if _ema(20) is not None else None,
        "sma5": round(_sma(5), 4) if _sma(5) is not None else None,
        "sma10": round(_sma(10), 4) if _sma(10) is not None else None,
        "sma20": round(_sma(20), 4) if _sma(20) is not None else None,
        "sma50": round(_sma(50), 4) if _sma(50) is not None else None,
        "sma200": None,
        "recent_high_5d": round(float(high.tail(5).max()), 4),
        "recent_low_5d": round(float(low.tail(5).min()), 4),
        "recent_high_10d": round(float(high.tail(10).max()), 4),
        "recent_low_10d": round(float(low.tail(10).min()), 4),
        "recent_high_20d": round(float(high.tail(20).max()), 4),
        "recent_low_20d": round(float(low.tail(20).min()), 4),
        "nearest_resistance": round(resistance, 4) if resistance is not None else None,
        "nearest_support": round(support, 4) if support is not None else None,
        "latest_volume": round(latest_volume, 4) if latest_volume is not None else None,
        "volume_20_sma": round(volume_20_sma, 4) if volume_20_sma is not None else None,
        "volume_50_sma": round(volume_20_sma, 4) if volume_20_sma is not None else None,
        "volume_ratio": round(volume_ratio, 3) if volume_ratio is not None else None,
        "volume_ratio_3d": volume_ratio_3d,
    }


def _format_short_term_market_anchors(anchors: dict) -> str:
    def _fmt(value):
        if value is None:
            return "n/a"
        if isinstance(value, list):
            return "[" + ", ".join(f"{item:g}" for item in value) + "]"
        return f"{value:g}"

    return (
        "**Short-term market anchors (precomputed from OHLCV through "
        f"{anchors['as_of_close_date']} — DO NOT recompute, USE these numbers verbatim):**\n"
        f"- current_price: {_fmt(anchors['current_price'])}\n"
        f"- ATR(5) / ATR(14): {_fmt(anchors['atr5'])} / {_fmt(anchors['atr14'])}  "
        f"(ATR14 ≈ {_fmt(anchors['atr14_pct'])}% of price)\n"
        f"- EMA5 / EMA10 / EMA20: {_fmt(anchors['ema5'])} / {_fmt(anchors['ema10'])} / {_fmt(anchors['ema20'])}\n"
        f"- SMA5 / SMA10 / SMA20 / SMA50: {_fmt(anchors['sma5'])} / {_fmt(anchors['sma10'])} / {_fmt(anchors['sma20'])} / {_fmt(anchors['sma50'])}\n"
        f"- 5-day range: high {_fmt(anchors['recent_high_5d'])}, low {_fmt(anchors['recent_low_5d'])}\n"
        f"- 10-day range: high {_fmt(anchors['recent_high_10d'])}, low {_fmt(anchors['recent_low_10d'])}\n"
        f"- nearest resistance above current: {_fmt(anchors['nearest_resistance'])}\n"
        f"- nearest support below current: {_fmt(anchors['nearest_support'])}\n"
        f"- latest_volume / volume_20_sma / volume_ratio: {_fmt(anchors.get('latest_volume'))} / "
        f"{_fmt(anchors.get('volume_20_sma'))} / {_fmt(anchors.get('volume_ratio'))}\n"
        f"- last 3 volume ratios vs 20-day average: {_fmt(anchors.get('volume_ratio_3d'))}\n"
        "- proximity rule: a price P is \"within X%\" iff |P - current_price| / current_price <= X/100. "
        "Use this for all distance-to-current checks; do not estimate from the report.\n"
    )


def _is_short_term_uptrend(anchors: dict) -> bool:
    current = anchors.get("current_price")
    ema5 = anchors.get("ema5") or anchors.get("sma20")
    ema10 = anchors.get("ema10") or anchors.get("sma50")
    ema20 = anchors.get("ema20") or anchors.get("sma200")
    if current is None or ema10 is None or ema20 is None:
        return False
    return (ema5 is not None and current > ema5 > ema10 > ema20) or current > ema10 > ema20


def _is_new_short_high_with_weak_volume(anchors: dict) -> bool:
    current = anchors.get("current_price")
    recent_high = anchors.get("recent_high_10d")
    ratios = anchors.get("volume_ratio_3d")
    if current is None or recent_high is None or not ratios:
        return False
    return current >= recent_high and all(ratio < 0.8 for ratio in ratios)


def _derive_short_term_rule_constraints(anchors: Optional[dict], holdings_info: dict, ticker: str) -> dict:
    if not anchors:
        return {
            "available": False,
            "allowed_actions": ["BUY", "HOLD", "SELL"],
            "entry_mode": "llm_discretion",
            "max_entry_size_pct": 30,
            "max_add_position_size_pct": 30,
            "volume_regime": "unavailable",
            "notes": ["Short-term anchors unavailable; cap new/add exposure at 30%."],
        }

    has_position = float(holdings_info.get("quantity") or 0.0) > 0
    volume_regime = _classify_volume_regime(anchors.get("volume_ratio"))
    short_uptrend = _is_short_term_uptrend(anchors)
    broad_index = _is_broad_index_instrument(ticker)
    bearish_volume_divergence = _is_new_short_high_with_weak_volume(anchors)

    allowed_actions = ["BUY", "HOLD", "SELL"]
    entry_mode = "short_term_normal"
    max_entry_size_pct = 60
    max_add_position_size_pct = 40
    notes = []

    if volume_regime == "unavailable":
        max_entry_size_pct = 30
        max_add_position_size_pct = 30
        notes.append("20-day volume ratio unavailable; cap short-term entries and adds at 30%.")
    elif volume_regime == "shrinking":
        max_entry_size_pct = 30
        max_add_position_size_pct = 20
        entry_mode = "pullback_or_small_only"
        notes.append("Shrinking 20-day volume confirmation limits short-term exposure.")
    elif volume_regime == "soft":
        max_entry_size_pct = 45
        max_add_position_size_pct = 30
        entry_mode = "pullback_or_reduced_size"
        notes.append("Sub-normal 20-day volume caps short-term entries at 45% and adds at 30%.")

    if broad_index and short_uptrend:
        allowed_actions = ["BUY", "HOLD"] if not has_position else ["BUY", "HOLD", "SELL"]
        if volume_regime in ("normal", "expanding"):
            max_entry_size_pct = max(max_entry_size_pct, 70)
        notes.append("Broad index short-term uptrend favors BUY/HOLD; SELL needs clear short-term damage.")

    if bearish_volume_divergence:
        max_entry_size_pct = 0
        max_add_position_size_pct = 0
        entry_mode = "no_new_or_add"
        notes.append("10-day high on weak 3-day volume ratio blocks new entries and adds.")

    return {
        "available": True,
        "allowed_actions": allowed_actions,
        "entry_mode": entry_mode,
        "max_entry_size_pct": max_entry_size_pct,
        "max_add_position_size_pct": max_add_position_size_pct,
        "volume_regime": volume_regime,
        "strong_uptrend": short_uptrend,
        "short_term_uptrend": short_uptrend,
        "broad_index": broad_index,
        "bearish_volume_divergence": bearish_volume_divergence,
        "notes": notes,
    }


def _format_short_term_rule_constraints(constraints: dict) -> str:
    notes = constraints.get("notes") or []
    notes_text = "\n".join(f"- {note}" for note in notes) if notes else "- none"
    return (
        "\n\n**Deterministic short-term rule constraints (hard limits; obey these over debate wording):**\n"
        f"- allowed_actions: {', '.join(constraints['allowed_actions'])}\n"
        f"- entry_mode: {constraints['entry_mode']}\n"
        f"- max_entry_size_pct: {constraints['max_entry_size_pct']:g}\n"
        f"- max_add_position_size_pct: {constraints['max_add_position_size_pct']:g}\n"
        f"- volume_regime: {constraints['volume_regime']}\n"
        f"- short_term_uptrend: {constraints.get('short_term_uptrend', 'n/a')}\n"
        f"- broad_index: {constraints.get('broad_index', 'n/a')}\n"
        f"- bearish_volume_divergence: {constraints.get('bearish_volume_divergence', 'n/a')}\n"
        f"- notes:\n{notes_text}\n"
    )


def _fallback_market_state(
    ticker: str,
    as_of_date: str,
    anchors: dict,
    volume_regime: str,
) -> MarketState:
    """Conservative deterministic MarketState when the LLM cannot emit valid JSON."""
    current = float(anchors["current_price"])
    atr = float(anchors.get("atr5") or anchors["atr14"])
    ema5 = anchors.get("ema5") or anchors.get("sma20")
    ema10 = anchors.get("ema10") or anchors.get("sma50")
    ema20 = anchors.get("ema20") or anchors.get("sma200")
    volume_ratio = anchors.get("volume_ratio")

    strong_uptrend = _is_short_term_uptrend(anchors)
    above_ema10 = ema10 is not None and current > float(ema10)
    above_ema20 = ema20 is not None and current > float(ema20)
    weak_uptrend = bool(above_ema10 and above_ema20)
    below_ema10 = ema10 is not None and current < float(ema10)

    if strong_uptrend:
        regime = "strong_uptrend"
        market_phase = "healthy_bull_trend"
        trend_score = 0.65
        momentum_score = 0.55
        if ema5 is not None and current > float(ema5) + 2.0 * atr:
            market_phase = "overextended_bull"
            risk_score = 0.45
        elif volume_ratio is not None and float(volume_ratio) >= 1.5:
            market_phase = "accelerating_bull"
            risk_score = 0.35
        else:
            risk_score = 0.30
    elif weak_uptrend:
        regime = "weak_uptrend"
        market_phase = "early_bull_reversal"
        trend_score = 0.35
        momentum_score = 0.25
        risk_score = 0.45
    elif below_ema10:
        regime = "breakdown_risk"
        market_phase = "early_bear_reversal"
        trend_score = -0.35
        momentum_score = -0.25
        risk_score = 0.65
    else:
        regime = "range"
        market_phase = "range_compression" if volume_regime in {"soft", "shrinking"} else "unclear"
        trend_score = 0.0
        momentum_score = 0.0
        risk_score = 0.50

    return MarketState(
        ticker=ticker,
        as_of_date=as_of_date,
        regime=regime,
        market_phase=market_phase,
        trend_score=trend_score,
        risk_score=risk_score,
        momentum_score=momentum_score,
        event_score=0.0,
        confidence=0.50,
        horizon_days=5,
        thesis=(
            "LLM MarketState JSON was unavailable; fallback classification was "
            "derived from price, moving-average, ATR, support/resistance, and volume anchors."
        ),
        invalidation_condition="Anchor structure changes materially at the next review.",
        key_risks=[
            "Fallback state excludes qualitative news/fundamental nuance",
            "Single-date technical classification may be noisy",
        ],
    )


def _invoke_market_state(
    llm,
    state_prompt: str,
    ticker: str,
    as_of_date: str,
    anchors: dict,
    volume_regime: str,
) -> MarketState:
    """Invoke MarketState generation with structured output, JSON fallback, then anchors fallback."""
    structured_llm = None
    if _llm_disallows_structured_output(llm):
        logger.warning(
            "Portfolio State Manager: provider/mode does not support structured MarketState output; "
            "falling back to free-text JSON"
        )
    else:
        try:
            structured_llm = llm.with_structured_output(MarketState)
        except (NotImplementedError, AttributeError) as exc:
            logger.warning(
                "Portfolio State Manager: provider does not support structured MarketState output (%s); "
                "falling back to free-text JSON",
                exc,
            )

    if structured_llm is not None:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Pydantic serializer warnings:.*",
                    category=UserWarning,
                )
                response = structured_llm.invoke(state_prompt)
            return _market_state_response_to_model(response)
        except Exception as exc:
            logger.warning(
                "Portfolio State Manager: structured MarketState invocation failed (%s); "
                "retrying once as free-text JSON",
                exc,
            )

    response = llm.invoke(state_prompt)
    try:
        return _market_state_response_to_model(response)
    except Exception as exc:
        content = getattr(response, "content", "")
        snippet = str(content).replace("\n", " ")[:300]
        logger.warning(
            "Portfolio State Manager: free-text MarketState JSON parse failed (%s); "
            "using deterministic anchor fallback. Response snippet: %s",
            exc,
            snippet,
        )
        return _fallback_market_state(ticker, as_of_date, anchors, volume_regime)


# Bull-phase claims that require short EMA hierarchy confirmation; if EMA disagrees,
# downgrade phase to "unclear" so the policy doesn't act on a fabricated trend.
_TRENDING_BULL_PHASES = {"healthy_bull_trend", "accelerating_bull", "bull_pullback"}

# Phases that get a high target_weight floor (>=0.50) and would commit a fresh
# 50-60%+ position on first activation. Bull hysteresis applies only to these.
_BULL_FLOOR_PHASES = {"healthy_bull_trend", "accelerating_bull", "bull_pullback"}

# Phases that force a full liquidation of an existing position. Bear hysteresis
# applies only to these — the symmetric risk to "commit 60% on noise" is
# "liquidate 100% on noise". The softer bear phases (oversold_bear, bear_rally,
# late_bear_exhaustion) already only block new entries, so they don't need it.
_BEAR_FORCE_SELL_PHASES = {
    "early_bear_reversal", "healthy_bear_trend", "accelerating_bear",
}

# All bull-side phases for "was the prior state already bullish?" check.
_ANY_BULL_PHASE = {
    "early_bull_reversal", "healthy_bull_trend", "accelerating_bull",
    "overextended_bull", "bull_pullback", "late_bull_distribution",
}

# All bear-side phases for "was the prior state already bearish?" check.
_ANY_BEAR_PHASE = {
    "early_bear_reversal", "healthy_bear_trend", "accelerating_bear",
    "oversold_bear", "bear_rally", "late_bear_exhaustion",
}


def _market_state_bias(state: Optional[MarketState]) -> str:
    if state is None:
        return "unavailable"
    if state.regime in {"strong_uptrend", "weak_uptrend"} or state.market_phase in _ANY_BULL_PHASE:
        return "bullish"
    if state.regime in {"breakdown_risk", "downtrend"} or state.market_phase in _ANY_BEAR_PHASE:
        return "bearish"
    return "neutral"

# Project-root-relative path to saved per-ticker strategy JSONs.
_STRATEGY_ROOT = Path(__file__).resolve().parents[3] / "back_test" / "strategy"

# Fallback regex when older strategy JSONs lack the market_state field but
# encode regime/phase in rationale_summary text.
_RATIONALE_PHASE_RE = re.compile(r"market_phase=([a-z_]+)")


def _load_recent_phases(ticker: str, trade_date: str, n: int = 2) -> list[str]:
    """Return the market_phase of the most recent N strategies before trade_date.

    Used by policy_from_market_state to apply regime-change hysteresis: when
    today's phase is "healthy_bull_trend" / "accelerating_bull" / "bull_pullback"
    (which trigger a >=50% sizing floor) but the most recent N strategies were
    NOT bullish, we treat today's claim as a single noisy flip rather than a
    confirmed regime, and downgrade to "early_bull_reversal".

    Falls back to parsing market_phase out of rationale_summary when an older
    strategy JSON lacks the structured `market_state` field.
    """
    strategy_dir = _STRATEGY_ROOT / ticker
    if not strategy_dir.exists():
        return []

    # Filenames are "{TICKER}_YYYY-MM-DD.json"; lex sort matches chronological.
    candidates = []
    prefix = f"{ticker}_"
    for path in strategy_dir.glob(f"{ticker}_*.json"):
        stem_date = path.stem[len(prefix):]
        if stem_date < trade_date:
            candidates.append((stem_date, path))
    candidates.sort()  # ascending by date
    candidates = candidates[-n:]

    phases: list[str] = []
    for _, path in candidates:
        try:
            with open(path) as fp:
                data = json.load(fp)
        except Exception:
            continue
        ms = data.get("market_state") or {}
        phase = ms.get("market_phase")
        if not phase:
            rationale = data.get("rationale_summary") or ""
            match = _RATIONALE_PHASE_RE.search(rationale)
            if match:
                phase = match.group(1)
        if phase:
            phases.append(phase)
    return phases


def policy_from_market_state(
    state: MarketState,
    anchors: dict,
    holdings_info: dict,
    constraints: dict,
    volume_regime: str,
    recent_phases: Optional[list[str]] = None,
    policy_config: Optional[PortfolioStatePolicyConfig] = None,
    market_context_state: Optional[MarketState] = None,
    market_context_ticker: Optional[str] = None,
) -> PortfolioStrategy:
    """Deterministically convert MarketState → PortfolioStrategy.

    v1 weights are intentionally simple and pending calibration. Sizing/pricing
    logic lives here; volume regime and EMA structure are NOT re-judged from
    the LLM — they come from anchors via _classify_volume_regime and
    _is_short_term_uptrend.

    recent_phases: market_phase of the most recent prior strategies, oldest →
    newest. Used to apply hysteresis: a single LLM flip from bear/range to
    "core bull" gets downgraded to early_bull_reversal (probe size, no floor)
    until the new regime is confirmed by additional observations.
    """
    config = policy_config or PortfolioStatePolicyConfig()
    phase_modifiers = config.merged_phase_modifiers()

    current = float(anchors["current_price"])
    atr = float(anchors.get("atr5") or anchors["atr14"])
    support = anchors.get("nearest_support")
    resistance = anchors.get("nearest_resistance")
    support = float(support) if support is not None else None
    resistance = float(resistance) if resistance is not None else None

    has_position = float(holdings_info.get("quantity") or 0.0) > 0.0
    notes: list[str] = []
    stock_bias = _market_state_bias(state)
    market_context_bias = _market_state_bias(market_context_state)
    market_context_blocks_add = False
    if market_context_state is not None:
        context_name = market_context_ticker or market_context_state.ticker
        notes.append(
            f"market_context={context_name}: bias={market_context_bias}, "
            f"regime={market_context_state.regime}, "
            f"phase={market_context_state.market_phase}, "
            f"trend={market_context_state.trend_score:.2f}, "
            f"momentum={market_context_state.momentum_score:.2f}, "
            f"risk={market_context_state.risk_score:.2f}"
        )

    # A. Cross-check: if LLM claims uptrend but short EMA structure disagrees, downgrade.
    effective_regime = state.regime
    if state.regime in {"strong_uptrend", "weak_uptrend"} and not _is_short_term_uptrend(anchors):
        notes.append(
            f"LLM regime={state.regime} downgraded to range — EMA5/10/20 structure does not confirm short-term uptrend."
        )
        effective_regime = "range"

    # A'. Same cross-check for market_phase: trending bull phases require short EMA confirmation.
    effective_phase = state.market_phase
    if state.market_phase in _TRENDING_BULL_PHASES and not _is_short_term_uptrend(anchors):
        notes.append(
            f"LLM market_phase={state.market_phase} downgraded to unclear — EMA5/10/20 structure does not confirm trend."
        )
        effective_phase = "unclear"

    # A''. Objective check for overextended_bull. LLM tends to call "approaching
    # resistance" or "short-term momentum cooling" overextended; on broad bull
    # markets this is the norm, not a reason to exit. Require distance from
    # EMA5 to exceed 2 ATR before honoring the overextended call.
    ema5 = anchors.get("ema5")
    if state.market_phase == "overextended_bull" and ema5 is not None and atr > 0:
        distance_atr = (current - float(ema5)) / atr
        if distance_atr < config.overextended_sma20_atr_threshold:
            notes.append(
                f"LLM market_phase=overextended_bull downgraded to healthy_bull_trend — "
                f"distance from EMA5 is only {distance_atr:.2f} ATR "
                f"(< {config.overextended_sma20_atr_threshold:g} threshold)."
            )
            effective_phase = "healthy_bull_trend"

    # A'''. Bull-side hysteresis on regime change. The LLM can flip from
    # "breakdown_risk" to "healthy_bull_trend" in a single review (cf. AAPL
    # 2024-01-17 → 01-24). When that happens, an unconditional 50%+ floor
    # commits a fresh full-size position on a single noisy classification
    # right at a likely local top. Require at least one of the most recent
    # N phases to also have been bull-leaning before honoring core-bull
    # floors. Otherwise downgrade to early_bull_reversal: cap=0.40, no
    # floor, allow_add=False — i.e. probe.
    confirm_n = config.hysteresis_confirmation_count
    recent = recent_phases or []
    if (
        effective_phase in _BULL_FLOOR_PHASES
        and confirm_n > 0
        and len(recent) >= confirm_n
        and all(p not in _ANY_BULL_PHASE for p in recent[-confirm_n:])
    ):
        notes.append(
            f"hysteresis: phase={effective_phase} after recent_phases={recent[-confirm_n:]} "
            "(no prior bull confirmation); downgraded to early_bull_reversal — "
            "probe size only until trend is observed twice."
        )
        effective_phase = "early_bull_reversal"

    # A''''. Bear-side hysteresis (symmetric). The LLM can also flip from a
    # sustained bull/range into early_bear_reversal / healthy_bear_trend /
    # accelerating_bear in one review — and those phases force_sell_if_position,
    # liquidating 100% of the position on a single noisy bearish read. That
    # is the mirror of the bull-side over-commit: same noise, opposite sign.
    # When the prior 2 phases were not bear-leaning, downgrade to bear_rally —
    # block new entries, but DO NOT force_sell. Existing risk management
    # (stop_loss, take_profit) already protects the position; let it work
    # for one more cycle and only liquidate if bear is confirmed twice.
    if (
        effective_phase in _BEAR_FORCE_SELL_PHASES
        and confirm_n > 0
        and len(recent) >= confirm_n
        and all(p not in _ANY_BEAR_PHASE for p in recent[-confirm_n:])
    ):
        notes.append(
            f"hysteresis: phase={effective_phase} after recent_phases={recent[-confirm_n:]} "
            "(no prior bear confirmation); downgraded to bear_rally — "
            "block new entries but defer force_sell until bear is observed twice."
        )
        effective_phase = "bear_rally"

    # C. Broad-index uptrend override: when the ticker is a broad index ETF and
    # short EMA structure confirms uptrend, skip phase-driven block_new logic.
    # In long-running broad-market bulls, sitting out is the bigger risk than
    # adding too aggressively; defer sizing to short-term rule constraints which
    # already knows broad-index rules.
    broad_uptrend_override = (
        _is_broad_index_instrument(state.ticker) and _is_short_term_uptrend(anchors)
    )
    if broad_uptrend_override:
        notes.append("broad-index strong uptrend: phase block_new bypassed.")

    phase_mod = phase_modifiers.get(effective_phase, {})

    def _rationale(extra: Optional[list[str]] = None) -> str:
        all_notes = notes + (extra or [])
        parts = [
            state.thesis,
            f"regime={state.regime}",
            f"effective_regime={effective_regime}",
            f"market_phase={state.market_phase}",
            f"effective_phase={effective_phase}",
            f"trend={state.trend_score:.2f}",
            f"momentum={state.momentum_score:.2f}",
            f"risk={state.risk_score:.2f}",
            f"event={state.event_score:.2f}",
            f"confidence={state.confidence:.2f}",
            f"volume_regime={volume_regime}",
            f"invalidation={state.invalidation_condition}",
            f"risks={'; '.join(state.key_risks) if state.key_risks else 'none'}",
        ]
        if all_notes:
            parts.append("notes=" + " | ".join(all_notes))
        return " | ".join(parts)

    # B. Hard regime overrides take priority over the linear formula.
    if effective_regime == "downtrend" and has_position:
        return PortfolioStrategy(
            ticker=state.ticker,
            as_of_date=state.as_of_date,
            action="SELL",
            entry=PriceSizeBlock(),
            add_position=PriceSizeBlock(),
            take_profit=PriceSizeBlock(),
            reduce_stop=PriceSizeBlock(),
            stop_loss=StopLossBlock(price=None),
            rationale_summary=_rationale(["regime=downtrend forces SELL on existing position."]),
        )

    if effective_regime == "breakdown_risk" and has_position:
        return PortfolioStrategy(
            ticker=state.ticker,
            as_of_date=state.as_of_date,
            action="SELL",
            entry=PriceSizeBlock(),
            add_position=PriceSizeBlock(),
            take_profit=PriceSizeBlock(),
            reduce_stop=PriceSizeBlock(),
            stop_loss=StopLossBlock(price=None),
            rationale_summary=_rationale(
                ["regime=breakdown_risk forces SELL on existing position."]
            ),
        )

    if effective_regime in {"breakdown_risk", "downtrend"} and not has_position:
        return PortfolioStrategy(
            ticker=state.ticker,
            as_of_date=state.as_of_date,
            action="HOLD",
            entry=PriceSizeBlock(),
            add_position=PriceSizeBlock(),
            take_profit=PriceSizeBlock(),
            reduce_stop=PriceSizeBlock(),
            stop_loss=StopLossBlock(price=None),
            rationale_summary=_rationale([f"regime={effective_regime} blocks new entries."]),
        )

    # B'. Phase-driven SELL on existing position (early/healthy/accelerating bear).
    if phase_mod.get("force_sell_if_position") and has_position:
        return PortfolioStrategy(
            ticker=state.ticker,
            as_of_date=state.as_of_date,
            action="SELL",
            entry=PriceSizeBlock(),
            add_position=PriceSizeBlock(),
            take_profit=PriceSizeBlock(),
            reduce_stop=PriceSizeBlock(),
            stop_loss=StopLossBlock(price=None),
            rationale_summary=_rationale(
                [f"market_phase={effective_phase} forces SELL on existing position."]
            ),
        )

    # B''. Phase blocks new positions (bear phases, oversold_bear, bear_rally,
    # high_volatility_range, macro_event_regime). Without an existing position,
    # there is nothing to trim — return HOLD early. Skip when broad-index uptrend.
    if phase_mod.get("block_new_position") and not has_position and not broad_uptrend_override:
        return PortfolioStrategy(
            ticker=state.ticker,
            as_of_date=state.as_of_date,
            action="HOLD",
            entry=PriceSizeBlock(),
            add_position=PriceSizeBlock(),
            take_profit=PriceSizeBlock(),
            reduce_stop=PriceSizeBlock(),
            stop_loss=StopLossBlock(price=None),
            rationale_summary=_rationale(
                [f"market_phase={effective_phase} blocks new entries."]
            ),
        )

    if market_context_bias == "bearish" and not has_position:
        return PortfolioStrategy(
            ticker=state.ticker,
            as_of_date=state.as_of_date,
            action="HOLD",
            entry=PriceSizeBlock(),
            add_position=PriceSizeBlock(),
            take_profit=PriceSizeBlock(),
            reduce_stop=PriceSizeBlock(),
            stop_loss=StopLossBlock(price=None),
            rationale_summary=_rationale(["market_context bearish blocks new entries."]),
        )

    if market_context_bias == "bearish" and has_position and stock_bias != "bullish":
        return PortfolioStrategy(
            ticker=state.ticker,
            as_of_date=state.as_of_date,
            action="SELL",
            entry=PriceSizeBlock(),
            add_position=PriceSizeBlock(),
            take_profit=PriceSizeBlock(),
            reduce_stop=PriceSizeBlock(),
            stop_loss=StopLossBlock(price=None),
            rationale_summary=_rationale(
                ["market_context bearish plus non-bullish stock state forces SELL."]
            ),
        )

    if market_context_bias == "bearish":
        market_context_blocks_add = True

    # C. Linear signal + regime ceilings/floors.
    # Bull/bear-leaning score weights are intentionally halved relative to v1:
    # the LLM tends to amplify bullish/bearish advocacy from analyst reports
    # and risk-debate framing into score extremes, which then propagates into
    # target_weight. By halving these coefficients we keep the LLM's
    # qualitative judgment as a tilt, not a driver — regime/phase floors and
    # caps (set by SMA-anchored Python) carry the structural sizing decision.
    # risk_score weight is preserved so genuinely high-risk states still cut.
    raw_signal = (
        config.trend_score_weight * state.trend_score
        + config.momentum_score_weight * state.momentum_score
        + config.event_score_weight * state.event_score
        - config.risk_score_weight * state.risk_score
    )
    target_weight = max(0.0, raw_signal) * state.confidence

    if effective_regime == "strong_uptrend":
        target_weight = max(target_weight, config.strong_uptrend_floor)
        target_weight = min(target_weight, config.strong_uptrend_cap)
    elif effective_regime == "weak_uptrend":
        target_weight = max(target_weight, config.weak_uptrend_floor)
        target_weight = min(target_weight, config.weak_uptrend_cap)
    elif effective_regime in {"range", "unclear"}:
        target_weight = min(target_weight, config.range_cap)
    elif effective_regime == "event_driven":
        target_weight = min(target_weight, config.event_driven_cap)

    # D. Volume regime multiplier (deterministic, anchored).
    multiplier = config.volume_multipliers.get(
        volume_regime,
        _DEFAULT_VOLUME_MULTIPLIER["unavailable"],
    )
    target_weight *= multiplier
    if volume_regime == "unavailable":
        target_weight = min(target_weight, config.unavailable_volume_cap)

    if market_context_bias == "bearish":
        target_weight *= config.market_context_bearish_weight_multiplier
        notes.append(
            "market_context bearish: target_weight multiplied by "
            f"{config.market_context_bearish_weight_multiplier:g} and adds blocked."
        )
    elif market_context_bias == "bullish" and stock_bias == "bullish":
        target_weight *= config.market_context_bullish_weight_multiplier
        notes.append(
            "market_context bullish with stock bullish: target_weight multiplied by "
            f"{config.market_context_bullish_weight_multiplier:g}."
        )

    # D'. Phase floor/cap. Floor implements 核心持仓 (e.g. healthy_bull_trend
    # holds >=0.50 even when raw_signal is weak); cap enforces phase-specific
    # ceilings (e.g. overextended_bull caps at 0.30).
    phase_floor = phase_mod.get("floor")
    phase_cap = phase_mod.get("cap")
    if phase_floor is not None:
        target_weight = max(target_weight, phase_floor)
    if phase_cap is not None:
        target_weight = min(target_weight, phase_cap)

    bearish_div = bool(constraints.get("bearish_volume_divergence"))
    if bearish_div:
        notes.append("bearish_volume_divergence: blocking new entries.")
        target_weight = 0.0
    
    target_weight = max(0.0, min(target_weight, config.max_target_weight))

    # Bearish divergence + has_position → defensive reduce_stop, not new orders.
    if bearish_div and has_position:
        stop_base = (
            support
            if support is not None
            else current - config.bearish_divergence_fallback_stop_atr * atr
        )
        stop_price = round(
            min(stop_base, current - config.bearish_divergence_stop_atr * atr), 2
        )
        reduce_price = round((stop_price + current) / 2.0, 2)
        return PortfolioStrategy(
            ticker=state.ticker,
            as_of_date=state.as_of_date,
            action="HOLD",
            entry=PriceSizeBlock(),
            add_position=PriceSizeBlock(),
            take_profit=PriceSizeBlock(),
            reduce_stop=PriceSizeBlock(
                price=reduce_price,
                size_pct=config.bearish_divergence_reduce_pct,
            ),
            stop_loss=StopLossBlock(price=stop_price),
            rationale_summary=_rationale(
                [
                    "bearish_volume_divergence: scaling out "
                    f"{config.bearish_divergence_reduce_pct:g}% via reduce_stop above hard stop."
                ]
            ),
        )

    # E. Action selection — SELL is regime-driven (handled above), not weight-driven.
    if target_weight <= config.min_trade_weight:
        return PortfolioStrategy(
            ticker=state.ticker,
            as_of_date=state.as_of_date,
            action="HOLD",
            entry=PriceSizeBlock(),
            add_position=PriceSizeBlock(),
            take_profit=PriceSizeBlock(),
            reduce_stop=PriceSizeBlock(),
            stop_loss=StopLossBlock(price=None),
            rationale_summary=_rationale(
                [f"target_weight={target_weight:.2f} below threshold; HOLD."]
            ),
        )

    # F. Build BUY orders. Phase decides entry mode and take-profit aggressiveness.
    trend_market_entry = (
        phase_mod.get("trend_market_entry") or effective_regime == "strong_uptrend"
    )
    pullback_buy = bool(phase_mod.get("pullback_buy"))
    block_new = bool(phase_mod.get("block_new_position")) and not broad_uptrend_override
    # broad_uptrend_override only relaxes the new-entry block. It must NOT force
    # allow_add=True, otherwise overextended_bull / late_bull_distribution lose
    # their explicit "no add" semantics and the policy adds against LLM warnings.
    allow_add = phase_mod.get("allow_add", True) and not market_context_blocks_add

    if block_new:
        # has_position branch — keep core, no new entry, no add. Stop and TP still set.
        entry_price = None
        entry_size = 0.0
        add_size = 0.0
    elif pullback_buy:
        # bull_pullback: we ARE in the pullback, fill at current rather than chase deeper.
        entry_price = round(current, 2)
        entry_size = round(target_weight * 100.0, 1) if not has_position else 0.0
        add_size = (
            round(
                min(
                    config.pullback_entry_add_max_pct,
                    target_weight * config.pullback_entry_add_weight_multiplier,
                ),
                1,
            )
            if has_position and allow_add
            else 0.0
        )
    elif trend_market_entry and volume_regime in {"normal", "expanding"} and not has_position:
        # 允许 trend following: market entry rather than wait for deep pullback.
        entry_price = None
        entry_size = round(target_weight * 100.0, 1)
        add_size = 0.0
    else:
        floor_price = support if support is not None else current * 0.98
        entry_price = round(max(current - 0.5 * atr, floor_price), 2)
        entry_size = round(target_weight * 100.0, 1) if not has_position else 0.0
        add_size = (
            round(
                min(
                    config.default_add_max_pct,
                    target_weight * config.default_add_weight_multiplier,
                ),
                1,
            )
            if has_position and allow_add
            else 0.0
        )

    # Stop loss: 2.5 ATR floor in trend regimes so normal volatility doesn't
    # whipsaw out. Use whichever (support or 2.5*ATR-below) is FURTHER, not closer.
    stop_base = support if support is not None else current - config.stop_loss_atr_multiple * atr
    stop_price = round(min(stop_base, current - config.stop_loss_atr_multiple * atr), 2)

    # Take-profit: for short-term trend phases, let winners clear the recent
    # 10-day high instead of taking profit too close to current price.
    recent_high_10d = anchors.get("recent_high_10d") or anchors.get("recent_high_20d")
    tp_far_phases = {"healthy_bull_trend", "accelerating_bull", "bull_pullback"}
    if effective_phase in tp_far_phases or effective_regime == "strong_uptrend":
        atr_target = current + config.trend_take_profit_atr_multiple * atr
        if recent_high_10d is not None:
            atr_target = max(
                atr_target,
                float(recent_high_10d) * config.trend_take_profit_recent_high_multiplier,
            )
        take_profit_price = round(atr_target, 2)
    else:
        take_profit_price = round(
            resistance
            if resistance is not None
            else current + config.default_take_profit_atr_multiple * atr,
            2,
        )

    phase_tp_size = phase_mod.get("tp_size")
    if phase_tp_size is not None:
        take_profit_size = phase_tp_size
    elif effective_regime == "strong_uptrend":
        take_profit_size = config.strong_uptrend_take_profit_size_pct
    else:
        take_profit_size = config.default_take_profit_size_pct

    # If phase blocks new positions but we have one, the resulting BUY-with-no-orders
    # is converted to HOLD by _enforce_strategy_rules downstream. We still emit
    # take_profit and stop_loss so risk management remains active.
    return PortfolioStrategy(
        ticker=state.ticker,
        as_of_date=state.as_of_date,
        action="BUY",
        entry=PriceSizeBlock(price=entry_price, size_pct=entry_size),
        add_position=PriceSizeBlock(price=None, size_pct=add_size),
        take_profit=PriceSizeBlock(price=take_profit_price, size_pct=take_profit_size),
        reduce_stop=PriceSizeBlock(),
        stop_loss=StopLossBlock(price=stop_price),
        rationale_summary=_rationale(
            [f"target_weight={target_weight:.2f}, phase_tp_size={take_profit_size:g}."]
        ),
    )


def _format_holdings_section(holdings_info: dict) -> str:
    if not holdings_info:
        return ""
    quantity = float(holdings_info.get("quantity") or 0.0)
    cash = holdings_info.get("cash")
    avg_buy_price = holdings_info.get("avg_buy_price")
    mark_price = holdings_info.get("mark_price")
    equity = holdings_info.get("equity")
    stop_loss = holdings_info.get("stop_loss")
    if quantity > 0:
        return (
            "- Current simulated holdings: "
            f"{quantity:g} shares"
            + (f", average buy price {float(avg_buy_price):g}" if avg_buy_price is not None else "")
            + (f", mark price {float(mark_price):g}" if mark_price is not None else "")
            + (f", active stop {float(stop_loss):g}" if stop_loss is not None else "")
            + (f", cash {float(cash):g}" if cash is not None else "")
            + (f", equity {float(equity):g}" if equity is not None else "")
            + ". Manage this existing position; do not behave as if the portfolio is flat.\n"
        )
    return (
        "- Current simulated holdings: no open position"
        + (f", cash {float(cash):g}" if cash is not None else "")
        + (f", equity {float(equity):g}" if equity is not None else "")
        + ". If the regime is favorable, prioritize establishing a starter position.\n"
    )


def _passthrough_debate_state(risk_debate_state: dict, decision_text: str) -> dict:
    return {
        "judge_decision": decision_text,
        "history": risk_debate_state["history"],
        "aggressive_history": risk_debate_state["aggressive_history"],
        "conservative_history": risk_debate_state["conservative_history"],
        "neutral_history": risk_debate_state["neutral_history"],
        "latest_speaker": "Judge",
        "current_aggressive_response": risk_debate_state["current_aggressive_response"],
        "current_conservative_response": risk_debate_state["current_conservative_response"],
        "current_neutral_response": risk_debate_state["current_neutral_response"],
        "count": risk_debate_state["count"],
    }


def _compute_market_context_state(
    context_ticker: str,
    trade_date: str,
) -> tuple[Optional[MarketState], Optional[dict], str]:
    context_anchors = _compute_short_term_market_anchors(context_ticker, trade_date)
    if context_anchors is None:
        return None, None, "unavailable"
    context_volume_regime = _classify_volume_regime(context_anchors.get("volume_ratio"))
    context_state = _fallback_market_state(
        context_ticker,
        context_anchors["as_of_close_date"],
        context_anchors,
        context_volume_regime,
    )
    return context_state, context_anchors, context_volume_regime


def create_portfolio_state_manager(
    llm,
    memory,
    policy_config: Optional[dict[str, Any] | PortfolioStatePolicyConfig] = None,
):
    """Backtest-only Portfolio Manager that uses MarketState + deterministic policy.

    The LLM emits a qualitative MarketState; policy_from_market_state builds the
    PortfolioStrategy from anchors and rule constraints; _enforce_strategy_rules
    runs as the final gate.
    """

    resolved_policy_config = coerce_portfolio_state_policy_config(policy_config)

    def portfolio_state_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])
        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]
        holdings_info = state.get("holdings_info") or {}
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]

        curr_situation = (
            f"{state['market_report']}\n\n{state['sentiment_report']}\n\n"
            f"{state['news_report']}\n\n{state['fundamentals_report']}"
        )
        past_memories = memory.get_memories(curr_situation, n_matches=2)
        past_memory_str = "".join(rec["recommendation"] + "\n\n" for rec in past_memories)
        lessons_section = (
            f"- Lessons from past decisions: **{past_memory_str}**\n" if past_memory_str else ""
        )

        anchors = _compute_short_term_market_anchors(ticker, trade_date)
        if anchors is not None:
            anchor_date = anchors.get("as_of_close_date")
            staleness = ""
            if anchor_date and anchor_date != trade_date:
                staleness = f" ⚠ STALE (Δ vs trade_date={trade_date})"
            print(
                f"[portfolio_state_manager] anchors {ticker} "
                f"trade_date={trade_date} as_of={anchor_date}{staleness} "
                f"current={anchors.get('current_price')} atr5/14={anchors.get('atr5')}/{anchors.get('atr14')} "
                f"support={anchors.get('nearest_support')} "
                f"resistance={anchors.get('nearest_resistance')} "
                f"recent_high_5d/10d={anchors.get('recent_high_5d')}/{anchors.get('recent_high_10d')} "
                f"ema5/10/20={anchors.get('ema5')}/{anchors.get('ema10')}/{anchors.get('ema20')} "
                f"vol_ratio={anchors.get('volume_ratio')}",
                flush=True,
            )
        if anchors is None:
            empty = PortfolioStrategy(
                ticker=ticker,
                as_of_date=trade_date,
                action="HOLD",
                entry=PriceSizeBlock(),
                add_position=PriceSizeBlock(),
                take_profit=PriceSizeBlock(),
                reduce_stop=PriceSizeBlock(),
                stop_loss=StopLossBlock(price=None),
                rationale_summary="No OHLCV anchors available; defaulting to HOLD.",
            ).model_dump()
            decision_text = (
                f"Decision: HOLD\n"
                f"MarketState: unavailable (no anchors)\n"
                f"Rationale: {empty['rationale_summary']}\n"
                f"Structured strategy schema_version={empty['schema_version']}"
            )
            return {
                "risk_debate_state": _passthrough_debate_state(risk_debate_state, decision_text),
                "final_trade_decision": decision_text,
                "market_state": None,
                "structured_strategy": empty,
            }

        constraints = _derive_short_term_rule_constraints(anchors, holdings_info, ticker)
        volume_regime = _classify_volume_regime(anchors.get("volume_ratio"))
        anchors_block = "\n\n" + _format_short_term_market_anchors(anchors)
        constraints_block = _format_short_term_rule_constraints(constraints)
        as_of_date = anchors["as_of_close_date"]
        holdings_section = _format_holdings_section(holdings_info)

        state_prompt = f"""You are the Portfolio Manager. In backtest mode your job is NOT to create executable orders. You only classify the market state.

        {instrument_context}

        You are NOT allowed to output:
        - entry / add / take-profit / reduce-stop / stop-loss prices
        - position sizes
        - BUY / SELL / HOLD trade orders
        - target prices or allocation percentages

        Your only job: classify the market state and compress the analyst debate into stable, testable state variables.

        Return only a MarketState object through the configured schema. If schema/tool
        output is unavailable, return only a raw JSON object with these exact fields.

        **Anti-advocacy directive — read this BEFORE scoring:**
        The inputs below contain three advocacy channels that you must NOT take at face value:
        (a) Bull / Bear researcher arguments embedded in the Research Manager's plan.
        (b) Aggressive / Conservative / Neutral analysts in the Risk Analysts Debate.
        (c) Subjective adjectives like "bullish" / "bearish" / "strong upside" in news, sentiment, or fundamentals reports.
        These channels are RHETORIC, not evidence. They are designed to take a side. Do NOT let the volume, intensity, or polarity of bullish or bearish framing decide your scores.

        When scoring, base your judgment on:
        1. Anchors (price structure, EMA5/10/20 hierarchy, 20-day volume ratio, ATR, support/resistance) — primary.
        2. Concrete events (filings, earnings results, macro prints, policy decisions) — supporting.
        3. Substantive analyst observations grounded in (1) or (2).
        NOT on:
        - How confidently the bull researcher / aggressive analyst phrased their case.
        - Whether news headlines used the word "bullish" or "bearish".
        - The number of paragraphs each side spent advocating.

        If the anchors say "price < EMA10" or "EMA5 < EMA10 < EMA20", you must NOT label trend_score > 0 just because the bull researcher made a passionate case. If anchors say "EMA5 > EMA10 > EMA20 + volume normal", you must NOT label trend_score < 0 just because the bear researcher cited a tail-risk scenario.

        Equal advocacy on both sides → that is evidence of "unclear" / "range", not a reason to pick the louder side.

        Definitions:
        - regime: current 1-5 trading day market condition. Choose strong_uptrend / weak_uptrend only when EMA hierarchy and momentum agree; otherwise prefer range or unclear.
        - market_phase: finer-grained phase within the regime. Choose ONE:
            BULL phases:
            - early_bull_reversal: early recovery after bearish period; trend damage stabilizing; first higher-highs/higher-lows; risk = fake breakout / dead-cat bounce.
            - healthy_bull_trend: stable price > EMA5 > EMA10 > EMA20, shallow pullbacks, persistent short-term trend. STRONGEST short-term trend-following regime.
            - accelerating_bull: rapid price expansion, momentum increasing fast, expanding volume. Risk = blow-off top.
            - overextended_bull: trend intact but distance above EMA5 large, RSI elevated, weakening volume confirmation. Risk/reward deteriorated; NOT necessarily bearish.
            - bull_pullback: healthy correction within intact bull trend, declining volume on pullback, structure still bullish. Favored re-entry/addition zone.
            - late_bull_distribution: trend alive but breadth weakening, leadership narrowing, volume divergence. Possible transition.
            BEAR phases:
            - early_bear_reversal: support breakdown, failed rebounds, deteriorating breadth.
            - healthy_bear_trend: stable downtrend, lower highs/lower lows, rallies fail quickly.
            - accelerating_bear: panic-like acceleration, volatility spike, forced liquidation, heavy volume.
            - oversold_bear: bearish regime intact but downside extension extreme. Oversold ≠ bottom.
            - bear_rally: countertrend rally inside larger bear; TRAP for trend-followers. Do not buy.
            - late_bear_exhaustion: selling pressure weakening, volatility stabilizing. Observation only.
            NEUTRAL / SPECIAL phases:
            - range_compression: low-vol sideways; trend-following fails here.
            - high_volatility_range: sideways with high vol, fake breakouts, whipsaws. Hardest environment.
            - macro_event_regime: dominated by FOMC / CPI / earnings / geopolitical / policy shocks; event risk > technicals.
            - unclear: genuinely mixed evidence; use sparingly.
        - trend_score [-1, 1]: directional trend strength.
        - risk_score [0, 1]: downside, event, macro, technical, and liquidity risk combined.
        - momentum_score [-1, 1]: recent price/volume momentum.
        - event_score [-1, 1]: whether news, sentiment, and fundamentals support or hurt the long thesis.
        - confidence [0, 1]: reliability of the state classification, NOT how much to buy.
        - horizon_days: expected validity period of this state.
        - invalidation_condition: qualitative ("loses EMA10 on closing basis"), not a precise stop price.
        - key_risks: short bullet phrases.

        Use ticker exactly: {ticker}.
        Use as_of_date exactly: {as_of_date}.

        Anchors and rule constraints below are CONTEXT for forming your thesis. DO NOT echo numbers as orders.

        **Context:**
        - Research Manager's plan: {research_plan}
        - Trader's proposal: {trader_plan}
        {lessons_section}{holdings_section}**Risk Analysts Debate:**
        {history}
        {anchors_block}{constraints_block}

        Do not output trading orders, prices, sizes, markdown, or fenced code blocks.
        Do not output text outside the schema/JSON object.{get_language_instruction()}"""

        market_state = _invoke_market_state(
            llm,
            state_prompt,
            ticker,
            as_of_date,
            anchors,
            volume_regime,
        )

        market_context_state = None
        market_context_volume_regime = "unavailable"
        if resolved_policy_config.market_context_enabled:
            context_ticker = resolved_policy_config.market_context_ticker
            if context_ticker:
                market_context_state, context_anchors, market_context_volume_regime = (
                    _compute_market_context_state(context_ticker, trade_date)
                )
                if context_anchors is not None and market_context_state is not None:
                    print(
                        f"[portfolio_state_manager] market_context {context_ticker} "
                        f"trade_date={trade_date} "
                        f"as_of={context_anchors.get('as_of_close_date')} "
                        f"bias={_market_state_bias(market_context_state)} "
                        f"regime={market_context_state.regime} "
                        f"phase={market_context_state.market_phase} "
                        f"current={context_anchors.get('current_price')} "
                        f"ema5/10/20={context_anchors.get('ema5')}/"
                        f"{context_anchors.get('ema10')}/{context_anchors.get('ema20')} "
                        f"vol_ratio={context_anchors.get('volume_ratio')}",
                        flush=True,
                    )

        # Hysteresis input: load most recent N prior strategies' market_phase
        # so policy can detect single-flip regime changes (e.g. 3 weeks of
        # breakdown_risk → 1 week of healthy_bull_trend = probe, not commit).
        recent_phases = _load_recent_phases(
            ticker,
            trade_date,
            n=resolved_policy_config.recent_phase_lookback,
        )

        strategy_dict = policy_from_market_state(
            market_state, anchors, holdings_info, constraints, volume_regime,
            recent_phases=recent_phases,
            policy_config=resolved_policy_config,
            market_context_state=market_context_state,
            market_context_ticker=resolved_policy_config.market_context_ticker,
        ).model_dump()
        strategy_dict = _enforce_strategy_rules(
            strategy_dict, anchors, constraints, holdings_info
        )
        strategy_dict = _apply_order_size_multiplier(
            strategy_dict,
            resolved_policy_config.order_size_multiplier,
        )

        if market_context_state is not None:
            market_context_text = (
                f"MarketContext: ticker={resolved_policy_config.market_context_ticker}, "
                f"bias={_market_state_bias(market_context_state)}, "
                f"regime={market_context_state.regime}, "
                f"phase={market_context_state.market_phase}, "
                f"trend={market_context_state.trend_score:.2f}, "
                f"momentum={market_context_state.momentum_score:.2f}, "
                f"risk={market_context_state.risk_score:.2f}, "
                f"volume_regime={market_context_volume_regime}\n"
            )
        else:
            market_context_text = "MarketContext: unavailable\n"

        decision_text = (
            f"Decision: {strategy_dict['action']}\n"
            f"MarketState: regime={market_state.regime}, "
            f"phase={market_state.market_phase}, "
            f"trend={market_state.trend_score:.2f}, "
            f"momentum={market_state.momentum_score:.2f}, "
            f"risk={market_state.risk_score:.2f}, "
            f"event={market_state.event_score:.2f}, "
            f"confidence={market_state.confidence:.2f}, "
            f"volume_regime={volume_regime}\n"
            f"{market_context_text}"
            f"Rationale: {strategy_dict.get('rationale_summary', '')}\n"
            f"Structured strategy schema_version={strategy_dict['schema_version']}"
        )

        return {
            "risk_debate_state": _passthrough_debate_state(risk_debate_state, decision_text),
            "final_trade_decision": decision_text,
            "market_state": market_state.model_dump(),
            "market_context_state": (
                market_context_state.model_dump()
                if market_context_state is not None
                else None
            ),
            "structured_strategy": strategy_dict,
        }

    return portfolio_state_manager_node


def create_market_aware_portfolio_state_manager(
    llm,
    memory,
    policy_config: Optional[dict[str, Any] | PortfolioStatePolicyConfig] = None,
):
    """Backtest PortfolioState manager with stock + index bullish/bearish context."""
    return create_portfolio_state_manager(llm, memory, policy_config=policy_config)
