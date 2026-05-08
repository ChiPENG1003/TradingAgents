"""Backtest-only Portfolio State Manager.

State-first refactor of portfolio decision-making for backtest mode:
- The LLM only emits a qualitative MarketState (regime + scores + thesis).
- Deterministic Python policy converts MarketState into the existing
  PortfolioStrategy order schema using anchors and rule constraints.

Live mode continues to use create_portfolio_manager from portfolio_manager.py.
"""

import warnings
from typing import Literal, Optional

from pydantic import BaseModel, Field

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.managers.portfolio_manager import (
    PortfolioStrategy,
    PriceSizeBlock,
    StopLossBlock,
    _classify_volume_regime,
    _compute_market_anchors,
    _derive_rule_constraints,
    _enforce_strategy_rules,
    _format_market_anchors,
    _format_rule_constraints,
    _is_broad_index_instrument,
    _is_strong_uptrend,
)


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

    additional_kwargs = getattr(value, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        found = _find_market_state(additional_kwargs, seen)
        if found is not None:
            return found

    return None


def _market_state_response_to_model(response) -> MarketState:
    state = _find_market_state(response)
    if state is None:
        raise TypeError(
            f"Structured output did not contain MarketState: {type(response).__name__}"
        )
    return state


_VOLUME_MULTIPLIER = {
    "expanding": 1.0,
    "normal": 1.0,
    "soft": 0.7,
    "shrinking": 0.5,
    "unavailable": 0.5,
}

# Phase modifiers — applied AFTER regime ceil/floor and volume multiplier.
# Encodes the four operating principles:
#   - 核心持仓: high floor in healthy_bull_trend / bull_pullback
#   - 不要频繁止盈: low tp_size in trend phases (15-20 vs default 25-40)
#   - 允许 trend following: market entry permitted in trend phases
#   - pullback 买入: bull_pullback gets aggressive add + at-current entry
# Keys not in dict default to no modification.
_PHASE_MODIFIER: dict[str, dict] = {
    # ----- Bull -----
    "early_bull_reversal":    {"cap": 0.40, "tp_size": 30.0, "allow_add": False},
    "healthy_bull_trend":     {"floor": 0.50, "cap": 0.85, "tp_size": 15.0, "allow_add": True,  "trend_market_entry": True},
    "accelerating_bull":      {"floor": 0.40, "cap": 0.80, "tp_size": 20.0, "allow_add": False, "trend_market_entry": True},
    # overextended_bull: keep core, allow trimmed add. NOT block_new_position —
    # in long bull markets "approaching resistance" is the norm, not a reason to exit.
    "overextended_bull":      {"cap": 0.55, "tp_size": 30.0, "allow_add": True},
    "bull_pullback":          {"floor": 0.50, "cap": 0.85, "tp_size": 15.0, "allow_add": True,  "pullback_buy": True},
    "late_bull_distribution": {"cap": 0.25, "tp_size": 50.0, "allow_add": False},
    # ----- Bear (force SELL existing, block new) -----
    "early_bear_reversal":    {"force_sell_if_position": True, "block_new_position": True},
    "healthy_bear_trend":     {"force_sell_if_position": True, "block_new_position": True},
    "accelerating_bear":      {"force_sell_if_position": True, "block_new_position": True},
    "oversold_bear":          {"cap": 0.0, "block_new_position": True},
    "bear_rally":             {"cap": 0.0, "block_new_position": True},      # trap for trend-followers
    "late_bear_exhaustion":   {"cap": 0.20, "tp_size": 30.0},
    # ----- Neutral -----
    "range_compression":      {"cap": 0.25, "tp_size": 60.0},                 # full TP near range top
    "high_volatility_range":  {"cap": 0.15, "block_new_position": True},
    "macro_event_regime":     {"cap": 0.10, "block_new_position": True},
    "unclear":                {"cap": 0.20},
}

# Bull-phase claims that require SMA hierarchy confirmation; if SMA disagrees,
# downgrade phase to "unclear" so the policy doesn't act on a fabricated trend.
_TRENDING_BULL_PHASES = {"healthy_bull_trend", "accelerating_bull", "bull_pullback"}


def policy_from_market_state(
    state: MarketState,
    anchors: dict,
    holdings_info: dict,
    constraints: dict,
    volume_regime: str,
) -> PortfolioStrategy:
    """Deterministically convert MarketState → PortfolioStrategy.

    v1 weights are intentionally simple and pending calibration. Sizing/pricing
    logic lives here; volume regime and SMA structure are NOT re-judged from
    the LLM — they come from anchors via _classify_volume_regime and
    _is_strong_uptrend.
    """
    current = float(anchors["current_price"])
    atr = float(anchors["atr14"])
    support = anchors.get("nearest_support")
    resistance = anchors.get("nearest_resistance")
    support = float(support) if support is not None else None
    resistance = float(resistance) if resistance is not None else None

    has_position = float(holdings_info.get("quantity") or 0.0) > 0.0
    notes: list[str] = []

    # A. Cross-check: if LLM claims uptrend but SMA structure disagrees, downgrade.
    effective_regime = state.regime
    if state.regime in {"strong_uptrend", "weak_uptrend"} and not _is_strong_uptrend(anchors):
        notes.append(
            f"LLM regime={state.regime} downgraded to range — SMA hierarchy does not confirm uptrend."
        )
        effective_regime = "range"

    # A'. Same cross-check for market_phase: trending bull phases require SMA confirmation.
    effective_phase = state.market_phase
    if state.market_phase in _TRENDING_BULL_PHASES and not _is_strong_uptrend(anchors):
        notes.append(
            f"LLM market_phase={state.market_phase} downgraded to unclear — SMA hierarchy does not confirm trend."
        )
        effective_phase = "unclear"

    # A''. Objective check for overextended_bull. LLM tends to call "approaching
    # resistance" or "short-term momentum cooling" overextended; on broad bull
    # markets this is the norm, not a reason to exit. Require distance from
    # SMA20 to exceed 2 ATR before honoring the overextended call.
    sma20 = anchors.get("sma20")
    if state.market_phase == "overextended_bull" and sma20 is not None and atr > 0:
        distance_atr = (current - float(sma20)) / atr
        if distance_atr < 2.0:
            notes.append(
                f"LLM market_phase=overextended_bull downgraded to healthy_bull_trend — "
                f"distance from SMA20 is only {distance_atr:.2f} ATR (< 2.0 threshold)."
            )
            effective_phase = "healthy_bull_trend"

    # C. Broad-index uptrend override: when the ticker is a broad index ETF and
    # SMA structure confirms strong uptrend, skip phase-driven block_new logic.
    # In long-running broad-market bulls, sitting out is the bigger risk than
    # adding too aggressively; defer sizing to _derive_rule_constraints which
    # already knows broad-index rules.
    broad_uptrend_override = (
        _is_broad_index_instrument(state.ticker) and _is_strong_uptrend(anchors)
    )
    if broad_uptrend_override:
        notes.append("broad-index strong uptrend: phase block_new bypassed.")

    phase_mod = _PHASE_MODIFIER.get(effective_phase, {})

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

    # C. Linear signal + regime ceilings/floors.
    # Bull/bear-leaning score weights are intentionally halved relative to v1:
    # the LLM tends to amplify bullish/bearish advocacy from analyst reports
    # and risk-debate framing into score extremes, which then propagates into
    # target_weight. By halving these coefficients we keep the LLM's
    # qualitative judgment as a tilt, not a driver — regime/phase floors and
    # caps (set by SMA-anchored Python) carry the structural sizing decision.
    # risk_score weight is preserved so genuinely high-risk states still cut.
    raw_signal = (
        0.25 * state.trend_score      # was 0.50
        + 0.125 * state.momentum_score  # was 0.25
        + 0.075 * state.event_score     # was 0.15
        - 0.40 * state.risk_score       # unchanged: risk should still bite
    )
    target_weight = max(0.0, raw_signal) * state.confidence

    if effective_regime == "strong_uptrend":
        target_weight = max(target_weight, 0.60)
        target_weight = min(target_weight, 0.90)
    elif effective_regime == "weak_uptrend":
        target_weight = max(target_weight, 0.35)
        target_weight = min(target_weight, 0.70)
    elif effective_regime in {"range", "unclear"}:
        target_weight = min(target_weight, 0.35)
    elif effective_regime == "event_driven":
        target_weight = min(target_weight, 0.50)

    # D. Volume regime multiplier (deterministic, anchored).
    multiplier = _VOLUME_MULTIPLIER.get(volume_regime, 0.5)
    target_weight *= multiplier
    if volume_regime == "unavailable":
        target_weight = min(target_weight, 0.35)

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
    
    target_weight = max(0.0, min(target_weight, 0.90))

    # Bearish divergence + has_position → defensive reduce_stop, not new orders.
    if bearish_div and has_position:
        stop_base = support if support is not None else current - 2.0 * atr
        stop_price = round(min(stop_base, current - 1.5 * atr), 2)
        reduce_price = round((stop_price + current) / 2.0, 2)
        return PortfolioStrategy(
            ticker=state.ticker,
            as_of_date=state.as_of_date,
            action="HOLD",
            entry=PriceSizeBlock(),
            add_position=PriceSizeBlock(),
            take_profit=PriceSizeBlock(),
            reduce_stop=PriceSizeBlock(price=reduce_price, size_pct=30.0),
            stop_loss=StopLossBlock(price=stop_price),
            rationale_summary=_rationale(
                ["bearish_volume_divergence: scaling out 30% via reduce_stop above hard stop."]
            ),
        )

    # E. Action selection — SELL is regime-driven (handled above), not weight-driven.
    if target_weight <= 0.05:
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
    allow_add = phase_mod.get("allow_add", True)

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
            round(min(40.0, target_weight * 60.0), 1)
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
            round(min(30.0, target_weight * 50.0), 1)
            if has_position and allow_add
            else 0.0
        )

    # Stop loss: 2.5 ATR floor in trend regimes so normal volatility doesn't
    # whipsaw out. Use whichever (support or 2.5*ATR-below) is FURTHER, not closer.
    stop_base = support if support is not None else current - 2.5 * atr
    stop_price = round(min(stop_base, current - 2.5 * atr), 2)

    # Take-profit: 不要频繁止盈 in trend phases — small size, far target.
    # In long-running broad bulls, 3*ATR can be too close (ATR compresses),
    # so floor TP at recent_high_20d * 1.05 to let the trend run.
    recent_high_20d = anchors.get("recent_high_20d")
    tp_far_phases = {"healthy_bull_trend", "accelerating_bull", "bull_pullback"}
    if effective_phase in tp_far_phases or effective_regime == "strong_uptrend":
        atr_target = current + 3.0 * atr
        if recent_high_20d is not None:
            atr_target = max(atr_target, float(recent_high_20d) * 1.05)
        take_profit_price = round(atr_target, 2)
    else:
        take_profit_price = round(
            resistance if resistance is not None else current + 2.0 * atr, 2
        )

    phase_tp_size = phase_mod.get("tp_size")
    if phase_tp_size is not None:
        take_profit_size = phase_tp_size
    elif effective_regime == "strong_uptrend":
        take_profit_size = 25.0
    else:
        take_profit_size = 40.0

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


def create_portfolio_state_manager(llm, memory):
    """Backtest-only Portfolio Manager that uses MarketState + deterministic policy.

    The LLM emits a qualitative MarketState; policy_from_market_state builds the
    PortfolioStrategy from anchors and rule constraints; _enforce_strategy_rules
    runs as the final gate.
    """

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

        anchors = _compute_market_anchors(ticker, trade_date)
        if anchors is not None:
            anchor_date = anchors.get("as_of_close_date")
            staleness = ""
            if anchor_date and anchor_date != trade_date:
                staleness = f" ⚠ STALE (Δ vs trade_date={trade_date})"
            print(
                f"[portfolio_state_manager] anchors {ticker} "
                f"trade_date={trade_date} as_of={anchor_date}{staleness} "
                f"current={anchors.get('current_price')} atr14={anchors.get('atr14')} "
                f"support={anchors.get('nearest_support')} "
                f"resistance={anchors.get('nearest_resistance')} "
                f"recent_high_20d={anchors.get('recent_high_20d')} "
                f"sma20/50/200={anchors.get('sma20')}/{anchors.get('sma50')}/{anchors.get('sma200')} "
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

        constraints = _derive_rule_constraints(anchors, holdings_info, ticker)
        volume_regime = _classify_volume_regime(anchors.get("volume_ratio"))
        anchors_block = "\n\n" + _format_market_anchors(anchors)
        constraints_block = _format_rule_constraints(constraints)
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

Return only a MarketState object through the configured schema.

**Anti-advocacy directive — read this BEFORE scoring:**
The inputs below contain three advocacy channels that you must NOT take at face value:
  (a) Bull / Bear researcher arguments embedded in the Research Manager's plan.
  (b) Aggressive / Conservative / Neutral analysts in the Risk Analysts Debate.
  (c) Subjective adjectives like "bullish" / "bearish" / "strong upside" in news, sentiment, or fundamentals reports.
These channels are RHETORIC, not evidence. They are designed to take a side. Do NOT let the volume, intensity, or polarity of bullish or bearish framing decide your scores.

When scoring, base your judgment on:
  1. Anchors (price structure, SMA hierarchy, volume ratio, ATR, support/resistance) — primary.
  2. Concrete events (filings, earnings results, macro prints, policy decisions) — supporting.
  3. Substantive analyst observations grounded in (1) or (2).
NOT on:
  - How confidently the bull researcher / aggressive analyst phrased their case.
  - Whether news headlines used the word "bullish" or "bearish".
  - The number of paragraphs each side spent advocating.

If the anchors say "price < SMA50", you must NOT label trend_score > 0 just because the bull researcher made a passionate case. If anchors say "SMA hierarchy bullish + volume normal", you must NOT label trend_score < 0 just because the bear researcher cited a tail-risk scenario.

Equal advocacy on both sides → that is evidence of "unclear" / "range", not a reason to pick the louder side.

Definitions:
- regime: current market condition. Choose strong_uptrend / weak_uptrend only when SMA hierarchy and momentum agree; otherwise prefer range or unclear.
- market_phase: finer-grained phase within the regime. Choose ONE:
    BULL phases:
      - early_bull_reversal: early recovery after bearish period; trend damage stabilizing; first higher-highs/higher-lows; risk = fake breakout / dead-cat bounce.
      - healthy_bull_trend: stable price > SMA20 > SMA50 > SMA200, shallow pullbacks, persistent trend. STRONGEST trend-following regime.
      - accelerating_bull: rapid price expansion, momentum increasing fast, expanding volume. Risk = blow-off top.
      - overextended_bull: trend intact but distance above SMA20 large, RSI elevated, weakening volume confirmation. Risk/reward deteriorated; NOT necessarily bearish.
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
- invalidation_condition: qualitative ("loses SMA50 on closing basis"), not a precise stop price.
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

Do not output trading orders, prices, sizes, or markdown. Do not output text outside the schema.{get_language_instruction()}"""

        structured_llm = llm.with_structured_output(MarketState)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Pydantic serializer warnings:.*",
                category=UserWarning,
            )
            response = structured_llm.invoke(state_prompt)
        market_state = _market_state_response_to_model(response)

        strategy_dict = policy_from_market_state(
            market_state, anchors, holdings_info, constraints, volume_regime
        ).model_dump()
        strategy_dict = _enforce_strategy_rules(
            strategy_dict, anchors, constraints, holdings_info
        )

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
            f"Rationale: {strategy_dict.get('rationale_summary', '')}\n"
            f"Structured strategy schema_version={strategy_dict['schema_version']}"
        )

        return {
            "risk_debate_state": _passthrough_debate_state(risk_debate_state, decision_text),
            "final_trade_decision": decision_text,
            "market_state": market_state.model_dump(),
            "structured_strategy": strategy_dict,
        }

    return portfolio_state_manager_node
