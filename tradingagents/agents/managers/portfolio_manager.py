import warnings
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from tradingagents.agents.utils.agent_utils import build_instrument_context, get_language_instruction
from tradingagents.dataflows.stockstats_utils import load_ohlcv


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
    }


def _format_market_anchors(anchors: dict) -> str:
    def _fmt(value):
        return "n/a" if value is None else f"{value:g}"

    return (
        "**Market anchors (precomputed from OHLCV through "
        f"{anchors['as_of_close_date']} — DO NOT recompute, USE these numbers verbatim):**\n"
        f"- current_price: {_fmt(anchors['current_price'])}\n"
        f"- ATR(14): {_fmt(anchors['atr14'])}  (≈ {_fmt(anchors['atr14_pct'])}% of price; use as the unit of \"normal\" daily move)\n"
        f"- SMA20 / SMA50 / SMA200: {_fmt(anchors['sma20'])} / {_fmt(anchors['sma50'])} / {_fmt(anchors['sma200'])}\n"
        f"- 20-day range: high {_fmt(anchors['recent_high_20d'])}, low {_fmt(anchors['recent_low_20d'])}\n"
        f"- nearest resistance above current: {_fmt(anchors['nearest_resistance'])}\n"
        f"- nearest support below current: {_fmt(anchors['nearest_support'])}\n"
        "- proximity rule: a price P is \"within X%\" iff |P - current_price| / current_price <= X/100. "
        "Use this for all distance-to-current checks; do not estimate from the report.\n"
    )


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
            structured_prompt = core_prompt + anchors_block + f"""

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
