import warnings
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from tradingagents.agents.utils.agent_utils import build_instrument_context, get_language_instruction
from tradingagents.dataflows.stockstats_utils import load_ohlcv


BROAD_INDEX_TICKERS = {
    "SPY",
    "VOO",
    "IVV",
    "QQQ",
    "QQQM",
    "DIA",
    "IWM",
    "VTI",
    "VT",
    "^GSPC",
    "^IXIC",
    "^DJI",
    "^RUT",
}


class PriceSizeBlock(BaseModel):
    price: Optional[float] = Field(default=None, description="Plain numeric limit price, or null when unused.")
    size_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class StopLossBlock(BaseModel):
    price: Optional[float] = Field(default=None, description="Plain numeric stop price, or null when unused.")


class PortfolioStrategy(BaseModel):
    schema_version: Literal["v3"] = "v3"
    ticker: str
    as_of_date: str = Field(description="YYYY-MM-DD analysis date.")
    action: Literal["BUY", "HOLD", "SELL"]
    entry: PriceSizeBlock
    add_position: PriceSizeBlock
    take_profit: PriceSizeBlock = Field(
        description="Sell on rise (high >= price). size_pct=100 means full take-profit close."
    )
    reduce_stop: PriceSizeBlock = Field(
        description="Partial defensive sell on drop (low <= price). Price must sit above stop_loss."
    )
    stop_loss: StopLossBlock = Field(
        description="Full-close stop. Triggered when low <= price; closes 100% of the position."
    )
    rationale_summary: str

    @model_validator(mode="after")
    def normalize_sell_orders(self):
        if self.action == "SELL":
            self.entry = PriceSizeBlock()
            self.add_position = PriceSizeBlock()
            self.take_profit = PriceSizeBlock()
            self.reduce_stop = PriceSizeBlock()
        return self

    @model_validator(mode="after")
    def validate_risk_levels(self):
        if self.action == "SELL":
            return self
        rs_price = self.reduce_stop.price
        sl_price = self.stop_loss.price
        if (
            rs_price is not None
            and sl_price is not None
            and self.reduce_stop.size_pct > 0
            and rs_price <= sl_price
        ):
            raise ValueError(
                f"reduce_stop.price ({rs_price}) must be ABOVE stop_loss.price ({sl_price}); "
                "otherwise the partial trim has no defensive value over the hard stop."
            )
        if self.action == "BUY" and self.entry.size_pct > 0 and sl_price is None:
            raise ValueError("BUY with a non-zero entry must define stop_loss.price.")
        if self.add_position.size_pct > 0 and sl_price is None:
            raise ValueError("add_position with size_pct > 0 must be paired with stop_loss.price.")
        return self


def _compute_market_anchors(ticker: str, trade_date: str, lookback_days: int = 220) -> Optional[dict]:
    """Return precomputed numeric anchors for the prompt.

    The Portfolio Manager prompt previously asked the LLM to extract current
    price / ATR / support / resistance from the free-form market_report. That
    forces the model to do arithmetic on prose. Instead, compute them here
    deterministically and inject as structured fields.

    Returns None if no usable price data is available so the caller can omit
    the anchors block gracefully.
    """
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
    atr14 = float(true_range.tail(14).mean()) if len(true_range) >= 14 else float(true_range.mean())

    def _sma(window: int) -> Optional[float]:
        if len(close) < window:
            return None
        return float(close.tail(window).mean())

    def _nearest_resistance(window: int) -> Optional[float]:
        slice_ = high.tail(window)
        above = slice_[slice_ > current_close]
        return float(above.min()) if not above.empty else None

    def _nearest_support(window: int) -> Optional[float]:
        slice_ = low.tail(window)
        below = slice_[slice_ < current_close]
        return float(below.max()) if not below.empty else None

    resistance = _nearest_resistance(20) or _nearest_resistance(60)
    support = _nearest_support(20) or _nearest_support(60)
    latest_volume = float(volume.iloc[-1]) if volume is not None and pd.notna(volume.iloc[-1]) else None
    volume_50_sma = (
        float(volume.tail(50).mean())
        if volume is not None and len(volume.dropna()) >= 50
        else None
    )
    volume_ratio = (
        latest_volume / volume_50_sma
        if latest_volume is not None and volume_50_sma not in (None, 0)
        else None
    )
    volume_ratio_3d = None
    if volume is not None and volume_50_sma not in (None, 0) and len(volume.dropna()) >= 50:
        volume_50_series = volume.rolling(50).mean()
        ratio_series = volume / volume_50_series
        recent_ratios = ratio_series.tail(3).dropna()
        if len(recent_ratios) == 3:
            volume_ratio_3d = [round(float(v), 3) for v in recent_ratios.tolist()]

    return {
        "as_of_close_date": pd.to_datetime(last["Date"]).strftime("%Y-%m-%d"),
        "current_price": round(current_close, 4),
        "atr14": round(atr14, 4),
        "atr14_pct": round(atr14 / current_close * 100.0, 3),
        "sma20": round(_sma(20), 4) if _sma(20) is not None else None,
        "sma50": round(_sma(50), 4) if _sma(50) is not None else None,
        "sma200": round(_sma(200), 4) if _sma(200) is not None else None,
        "recent_high_20d": round(float(high.tail(20).max()), 4),
        "recent_low_20d": round(float(low.tail(20).min()), 4),
        "nearest_resistance": round(resistance, 4) if resistance is not None else None,
        "nearest_support": round(support, 4) if support is not None else None,
        "latest_volume": round(latest_volume, 4) if latest_volume is not None else None,
        "volume_50_sma": round(volume_50_sma, 4) if volume_50_sma is not None else None,
        "volume_ratio": round(volume_ratio, 3) if volume_ratio is not None else None,
        "volume_ratio_3d": volume_ratio_3d,
    }


def _format_market_anchors(anchors: dict) -> str:
    def _fmt(value):
        if value is None:
            return "n/a"
        if isinstance(value, list):
            return "[" + ", ".join(f"{item:g}" for item in value) + "]"
        return f"{value:g}"

    return (
        "**Market anchors (precomputed from OHLCV through "
        f"{anchors['as_of_close_date']} — DO NOT recompute, USE these numbers verbatim):**\n"
        f"- current_price: {_fmt(anchors['current_price'])}\n"
        f"- ATR(14): {_fmt(anchors['atr14'])}  (≈ {_fmt(anchors['atr14_pct'])}% of price; use as the unit of \"normal\" daily move)\n"
        f"- SMA20 / SMA50 / SMA200: {_fmt(anchors['sma20'])} / {_fmt(anchors['sma50'])} / {_fmt(anchors['sma200'])}\n"
        f"- 20-day range: high {_fmt(anchors['recent_high_20d'])}, low {_fmt(anchors['recent_low_20d'])}\n"
        f"- nearest resistance above current: {_fmt(anchors['nearest_resistance'])}\n"
        f"- nearest support below current: {_fmt(anchors['nearest_support'])}\n"
        f"- latest_volume / volume_50_sma / volume_ratio: {_fmt(anchors.get('latest_volume'))} / "
        f"{_fmt(anchors.get('volume_50_sma'))} / {_fmt(anchors.get('volume_ratio'))}\n"
        f"- last 3 volume ratios vs 50-day average: {_fmt(anchors.get('volume_ratio_3d'))}\n"
        "- proximity rule: a price P is \"within X%\" iff |P - current_price| / current_price <= X/100. "
        "Use this for all distance-to-current checks; do not estimate from the report.\n"
    )


def _is_broad_index_instrument(ticker: str) -> bool:
    normalized = ticker.upper()
    return normalized in BROAD_INDEX_TICKERS or normalized.endswith((".INDEX", ".IDX"))


def _classify_volume_regime(volume_ratio: Optional[float]) -> str:
    if volume_ratio is None:
        return "unavailable"
    if volume_ratio >= 1.5:
        return "expanding"
    if volume_ratio < 0.7:
        return "shrinking"
    if volume_ratio >= 0.9:
        return "normal"
    return "soft"


def _is_strong_uptrend(anchors: dict) -> bool:
    current = anchors.get("current_price")
    sma20 = anchors.get("sma20")
    sma50 = anchors.get("sma50")
    sma200 = anchors.get("sma200")
    if current is None or sma50 is None or sma200 is None:
        return False
    return (
        sma20 is not None and current > sma20 > sma50 > sma200
    ) or current > sma50 > sma200


def _is_new_high_with_weak_volume(anchors: dict) -> bool:
    current = anchors.get("current_price")
    recent_high = anchors.get("recent_high_20d")
    ratios = anchors.get("volume_ratio_3d")
    if current is None or recent_high is None or not ratios:
        return False
    return current >= recent_high and all(ratio < 0.8 for ratio in ratios)


def _derive_rule_constraints(anchors: Optional[dict], holdings_info: dict, ticker: str) -> dict:
    if not anchors:
        return {
            "available": False,
            "allowed_actions": ["BUY", "HOLD", "SELL"],
            "entry_mode": "llm_discretion",
            "max_entry_size_pct": 35,
            "max_add_position_size_pct": 35,
            "volume_regime": "unavailable",
            "notes": ["Market anchors unavailable; cap new/add exposure at 35%."],
        }

    has_position = float(holdings_info.get("quantity") or 0.0) > 0
    volume_ratio = anchors.get("volume_ratio")
    volume_regime = _classify_volume_regime(volume_ratio)
    strong_uptrend = _is_strong_uptrend(anchors)
    broad_index = _is_broad_index_instrument(ticker)
    bearish_volume_divergence = _is_new_high_with_weak_volume(anchors)

    allowed_actions = ["BUY", "HOLD", "SELL"]
    entry_mode = "normal"
    max_entry_size_pct = 70
    max_add_position_size_pct = 50
    notes = []

    if volume_regime == "unavailable":
        max_entry_size_pct = 35
        max_add_position_size_pct = 35
        notes.append("Volume ratio unavailable; cap entry and add-position size at 35%.")
    elif volume_regime == "shrinking":
        max_entry_size_pct = 35
        max_add_position_size_pct = 25
        entry_mode = "pullback_or_small_only"
        notes.append("Shrinking volume limits new/add exposure.")
    elif volume_regime == "soft":
        max_entry_size_pct = 55
        max_add_position_size_pct = 35
        entry_mode = "pullback_or_reduced_size"
        notes.append("Sub-normal volume confirmation caps entries at 55% and adds at 35%.")

    if broad_index and strong_uptrend:
        allowed_actions = ["BUY", "HOLD"] if not has_position else ["BUY", "HOLD", "SELL"]
        if volume_regime in ("normal", "expanding"):
            max_entry_size_pct = max(max_entry_size_pct, 80)
        notes.append("Broad index uptrend forbids fresh bearish positioning; SELL needs existing position management.")

    if bearish_volume_divergence:
        max_entry_size_pct = 0
        max_add_position_size_pct = 0
        entry_mode = "no_new_or_add"
        notes.append("New high on weak 3-day volume ratio blocks new entries and adds.")

    return {
        "available": True,
        "allowed_actions": allowed_actions,
        "entry_mode": entry_mode,
        "max_entry_size_pct": max_entry_size_pct,
        "max_add_position_size_pct": max_add_position_size_pct,
        "volume_regime": volume_regime,
        "strong_uptrend": strong_uptrend,
        "broad_index": broad_index,
        "bearish_volume_divergence": bearish_volume_divergence,
        "notes": notes,
    }


def _format_rule_constraints(constraints: dict) -> str:
    notes = constraints.get("notes") or []
    notes_text = "\n".join(f"- {note}" for note in notes) if notes else "- none"
    return (
        "\n\n**Deterministic rule constraints (hard limits; obey these over debate wording):**\n"
        f"- allowed_actions: {', '.join(constraints['allowed_actions'])}\n"
        f"- entry_mode: {constraints['entry_mode']}\n"
        f"- max_entry_size_pct: {constraints['max_entry_size_pct']:g}\n"
        f"- max_add_position_size_pct: {constraints['max_add_position_size_pct']:g}\n"
        f"- volume_regime: {constraints['volume_regime']}\n"
        f"- strong_uptrend: {constraints.get('strong_uptrend', 'n/a')}\n"
        f"- broad_index: {constraints.get('broad_index', 'n/a')}\n"
        f"- bearish_volume_divergence: {constraints.get('bearish_volume_divergence', 'n/a')}\n"
        f"- notes:\n{notes_text}\n"
    )


def _clamp_size(block: dict, max_size: float) -> None:
    block["size_pct"] = min(float(block.get("size_pct") or 0.0), float(max_size))
    if block["size_pct"] <= 0:
        block["size_pct"] = 0.0
        block["price"] = None


def _distance_pct(price: Optional[float], current_price: Optional[float]) -> Optional[float]:
    if price is None or current_price in (None, 0):
        return None
    return abs(float(price) - float(current_price)) / float(current_price) * 100.0


def _append_rule_note(strategy: dict, note: str) -> None:
    rationale = strategy.get("rationale_summary") or ""
    if note not in rationale:
        strategy["rationale_summary"] = (rationale + " Rule adjustment: " + note).strip()


def _clear_entry_orders(strategy: dict) -> None:
    strategy["entry"] = PriceSizeBlock().model_dump()
    strategy["add_position"] = PriceSizeBlock().model_dump()


def _enforce_strategy_rules(strategy: dict, anchors: Optional[dict], constraints: dict, holdings_info: dict) -> dict:
    strategy = PortfolioStrategy.model_validate(strategy).model_dump()
    has_position = float(holdings_info.get("quantity") or 0.0) > 0
    current_price = anchors.get("current_price") if anchors else None

    if strategy["action"] not in constraints["allowed_actions"]:
        original_action = strategy["action"]
        strategy["action"] = "HOLD" if original_action == "SELL" or has_position else "BUY"
        if strategy["action"] not in constraints["allowed_actions"]:
            strategy["action"] = constraints["allowed_actions"][0]
        _append_rule_note(
            strategy,
            f"{original_action} was outside allowed_actions={constraints['allowed_actions']}; action set to {strategy['action']}.",
        )
        if original_action == "SELL":
            _clear_entry_orders(strategy)

    _clamp_size(strategy["entry"], constraints["max_entry_size_pct"])
    _clamp_size(strategy["add_position"], constraints["max_add_position_size_pct"])

    if constraints["entry_mode"] == "no_new_or_add":
        _clear_entry_orders(strategy)
        if not has_position and strategy["action"] == "BUY":
            strategy["action"] = "HOLD"
        _append_rule_note(strategy, "new entries and adds blocked by deterministic volume divergence rule.")

    entry_distance = _distance_pct(strategy["entry"].get("price"), current_price)
    if entry_distance is not None and entry_distance > 10:
        strategy["entry"]["size_pct"] = 0.0
        strategy["entry"]["price"] = None
        _append_rule_note(strategy, "entry more than 10% from current price was removed.")

    add_distance = _distance_pct(strategy["add_position"].get("price"), current_price)
    if add_distance is not None and add_distance > 10:
        strategy["add_position"]["size_pct"] = 0.0
        strategy["add_position"]["price"] = None
        _append_rule_note(strategy, "add-position level more than 10% from current price was removed.")

    if strategy["action"] == "BUY" and strategy["entry"]["size_pct"] <= 0 and strategy["add_position"]["size_pct"] <= 0:
        strategy["action"] = "HOLD"
        _append_rule_note(strategy, "BUY without an executable entry/add was converted to HOLD.")

    if strategy["action"] in ("BUY", "HOLD") and (
        strategy["entry"]["size_pct"] > 0 or strategy["add_position"]["size_pct"] > 0
    ):
        if strategy["stop_loss"]["price"] is None and anchors:
            reference = strategy["entry"]["price"] or current_price
            stop = min(
                anchors.get("nearest_support") or reference,
                float(reference) - 1.5 * float(anchors["atr14"]),
            )
            strategy["stop_loss"]["price"] = round(max(stop, 0.01), 4)
            _append_rule_note(strategy, "missing stop_loss was filled from support/ATR anchor.")

    return PortfolioStrategy.model_validate(strategy).model_dump()


def _strategy_response_to_dict(response) -> dict:
    """Extract PortfolioStrategy from provider-specific structured-output wrappers."""
    strategy = _find_portfolio_strategy(response)
    if strategy is None:
        raise TypeError(f"Structured output did not contain PortfolioStrategy: {type(response).__name__}")
    return strategy.model_dump()


def _find_portfolio_strategy(value, seen: Optional[set[int]] = None) -> Optional[PortfolioStrategy]:
    if value is None:
        return None
    if isinstance(value, PortfolioStrategy):
        return value
    if seen is None:
        seen = set()
    obj_id = id(value)
    if obj_id in seen:
        return None
    seen.add(obj_id)

    if isinstance(value, dict):
        try:
            return PortfolioStrategy.model_validate(value)
        except Exception:
            pass
        for nested in value.values():
            found = _find_portfolio_strategy(nested, seen)
            if found is not None:
                return found
        return None

    for attr in ("parsed", "content", "output"):
        nested = getattr(value, attr, None)
        found = _find_portfolio_strategy(nested, seen)
        if found is not None:
            return found

    if isinstance(value, (list, tuple)):
        for nested in value:
            found = _find_portfolio_strategy(nested, seen)
            if found is not None:
                return found

    additional_kwargs = getattr(value, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        found = _find_portfolio_strategy(additional_kwargs, seen)
        if found is not None:
            return found

    return None


def create_portfolio_manager(llm, memory):
    def portfolio_manager_node(state) -> dict:

        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]
        trading_mode = state.get("trading_mode", "live")
        holdings_info = state.get("holdings_info") or {}

        curr_situation = (
            f"{state['market_report']}\n\n{state['sentiment_report']}\n\n"
            f"{state['news_report']}\n\n{state['fundamentals_report']}"
        )
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        lessons_section = f"- Lessons from past decisions: **{past_memory_str}**\n" if past_memory_str else ""
        holdings_section = ""
        if holdings_info:
            quantity = float(holdings_info.get("quantity") or 0.0)
            cash = holdings_info.get("cash")
            avg_buy_price = holdings_info.get("avg_buy_price")
            mark_price = holdings_info.get("mark_price")
            equity = holdings_info.get("equity")
            stop_loss = holdings_info.get("stop_loss")
            if quantity > 0:
                holdings_section = (
                    "- Current simulated holdings: "
                    f"{quantity:g} shares"
                    + (f", average buy price {float(avg_buy_price):g}" if avg_buy_price is not None else "")
                    + (f", mark price {float(mark_price):g}" if mark_price is not None else "")
                    + (f", active stop {float(stop_loss):g}" if stop_loss is not None else "")
                    + (f", cash {float(cash):g}" if cash is not None else "")
                    + (f", equity {float(equity):g}" if equity is not None else "")
                    + ". Manage this existing position; do not behave as if the portfolio is flat.\n"
                )
            else:
                holdings_section = (
                    "- Current simulated holdings: no open position"
                    + (f", cash {float(cash):g}" if cash is not None else "")
                    + (f", equity {float(equity):g}" if equity is not None else "")
                    + ". If the regime is favorable, prioritize establishing a starter position.\n"
                )

        core_prompt = f"""You are the Portfolio Manager. Synthesize the risk analysts' debate and deliver the final short-term trading decision.

{instrument_context}

**Decision** (choose one): **BUY** | **HOLD** | **SELL**

**Context:**
- Research Manager's plan: {research_plan}
- Trader's proposal: {trader_plan}
{lessons_section}
{holdings_section}
**Risk Analysts Debate:**
{history}

Be decisive and ground every parameter in specific evidence from the debate.{get_language_instruction()}"""

        live_prompt = core_prompt + """

**Required Output:**
1. **Decision**: BUY / HOLD / SELL
2. **Trade Parameters**: Initial entry, add-position level, take-profit level, reduce-stop level, stop-loss level, expected holding period (days to weeks)
3. **Position Sizing**: Suggested allocation as % of portfolio and rationale
4. **Rationale**: Key reasoning grounded in the analysts' debate"""

        if trading_mode == "backtest":
            anchors = _compute_market_anchors(state["company_of_interest"], state["trade_date"])
            anchors_block = ("\n\n" + _format_market_anchors(anchors)) if anchors else ""
            constraints = _derive_rule_constraints(
                anchors,
                holdings_info,
                state["company_of_interest"],
            )
            constraints_block = _format_rule_constraints(constraints)
            structured_prompt = core_prompt + anchors_block + constraints_block + f"""

Backtest structured-output rules:
- Emit the structured strategy object only through the configured schema.
- Use ticker exactly: {state["company_of_interest"]}.
- Use as_of_date exactly: {state["trade_date"]}.
- action is a directional label; execution is driven by entry/add/take_profit/reduce_stop/stop_loss fields.
- SELL only closes an existing long position; this system does not support opening short positions.
- entry.size_pct and add_position.size_pct are percentages of available cash.
- take_profit and reduce_stop size_pct are percentages of current shares.
- price=null with size_pct>0 means execute that order at the next trading day's open.
- price=null with size_pct=0 means no order in that block.
- take_profit fires when price rises to its level (sell-on-rise). Its price MUST be above the current price.
- reduce_stop fires when price drops to its level (partial defensive trim before the full stop). Its price MUST sit between stop_loss.price and the current price (stop_loss < reduce_stop < current).
- stop_loss is a full-close stop; a separate reduce_stop can scale out a portion before that level is hit.
- For no order in a given block, set price to null and size_pct to 0.
- For SELL on an existing position, set entry/add/take_profit/reduce_stop price to null and size_pct to 0.
- Prices must be plain numbers, not strings.

Aggressiveness rules (lean aggressive; HOLD is reserved for genuinely balanced setups, not weak conviction):
- Default to a strong long bias whenever the thesis is even mildly positive and downside risk is bounded by a defined stop.
- Treat HOLD as a fallback only when bull and bear cases are roughly symmetric AND no entry within 10% of current price is justifiable. Indecision is not a HOLD; pick a side.
- Pull entry.price within 6% of the current price by default so the order actually fills; only stretch beyond 6% when there is a clear technical reason (support shelf, prior breakout retest).
- In a confirmed uptrend, missed participation is a real risk. Prefer entry.price=null with a meaningful size_pct over a pullback limit that is unlikely to fill.
- In a confirmed uptrend, prefer next-open entry over waiting for a deep pullback.
- For broad index ETFs or highly diversified instruments, if current_price > SMA20 > SMA50 > SMA200, or current_price > SMA50 and SMA50 > SMA200, default action should be BUY or HOLD-with-position-management, not SELL.
- For broad index ETFs or highly diversified instruments in that regime, if no position exists, use entry.price=null with entry.size_pct between 50 and 80 unless downside risk is extreme.
- For broad index ETFs or highly diversified instruments in that regime, do not require a pullback entry in a strong trend.
- For broad index ETFs or highly diversified instruments in that regime, SELL requires clear trend damage or severe macro/technical deterioration; ordinary overbought readings are not enough.
- Volume confirmation (放量站稳) is a HARD gate. Compute volume_ratio = latest daily volume / volume_50_sma from the market analyst's Volume confirmation section, and apply the next four rules BEFORE the confidence-based size bands below. Volume can downgrade size or veto an entry; it can never upgrade beyond the bands.
- Breakout-class entries — a close above multi-week resistance, an SMA20 reclaim from below, a touch/close above boll_ub, or a gap-up open above the prior range — REQUIRE volume_ratio >= 1.5. If the setup is breakout-class but volume_ratio < 1.5, do NOT chase: emit HOLD, or place entry.price as a pullback limit BELOW the breakout level so it only fills on a retest.
- Trend-continuation entries (price already above SMA20 > SMA50 and you are buying a shallow pullback rather than a fresh breakout) only require volume_ratio >= 0.9; do not impose the 1.5x gate on these.
- For add_position above the current cost basis when volume_ratio < 0.7 (price extending on shrinking participation), cut the planned add_position.size_pct in half — e.g., 50 becomes 25.
- If price prints a new high but volume_ratio has stayed below 0.8 for the last 3 sessions, treat it as bearish divergence: prefer a reduce_stop sized 25-30% over outright SELL, and do NOT initiate a new add_position regardless of confidence.
- If the market analyst report does not include a usable volume_ratio, state that volume confirmation is unavailable in rationale_summary and conservatively cap entry.size_pct and add_position.size_pct at 35.
- If confidence is low/medium and entry.price is within 8% of current, allow entry.size_pct between 35 and 55.
- If confidence is medium and entry.price is within 6% of current, allow entry.size_pct between 50 and 70.
- If confidence is high and the setup is trend-following, breakout, or strong momentum, allow entry.size_pct between 70 and 90.
- If already holding a long position and the bullish thesis remains valid, default to add_position rather than HOLD-doing-nothing.
- For add_position, allow size_pct between 25 and 50 when price is within 6% of current; up to 60 when within 3% and momentum is confirming.
- In a strong uptrend, take_profit is optional. If used, set it farther than the nearest minor resistance and usually sell only 20-30%; do not repeatedly cap upside in a trending market.
- Use larger take_profit sizes only when the upside thesis is mature, momentum is fading, or price is materially extended from SMA20/ATR bands.
- Use reduce_stop sparingly: only when the thesis is fragile and you want to trim before the hard stop. Typical size_pct is 25-50.
- Do not emit reduce_stop just for "feeling cautious"; if the bull case holds, lean on stop_loss alone.
- Stop loss must always be present for BUY or ADD actions, but place it at a structural level, not so tight that normal volatility shakes you out.
- For index ETFs, stop_loss should normally be at least 1.5 ATR below entry unless the thesis is explicitly short-lived; in strong uptrends, prefer a stop below a recent swing low, SMA20/SMA50, or another structural level rather than a tight nearest-support stop.
- Avoid oversized entries only when the stop distance is genuinely wide (>8% of entry price); otherwise, larger sizes are preferred."""
            
            structured_llm = llm.with_structured_output(PortfolioStrategy)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Pydantic serializer warnings:.*",
                    category=UserWarning,
                )
                response = structured_llm.invoke(structured_prompt)
            strategy = _strategy_response_to_dict(response)
            strategy = _enforce_strategy_rules(strategy, anchors, constraints, holdings_info)
            decision_text = (
                f"Decision: {strategy['action']}\n"
                f"Rationale: {strategy.get('rationale_summary', '')}\n"
                f"Structured strategy schema_version={strategy['schema_version']}"
            )
        else:
            response = llm.invoke(live_prompt)
            strategy = None
            decision_text = response.content


        new_risk_debate_state = {
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

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": decision_text,
            "structured_strategy": strategy,
        }

    return portfolio_manager_node
