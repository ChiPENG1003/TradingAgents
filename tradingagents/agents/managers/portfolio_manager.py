from tradingagents.agents.utils.decision_context import bounded_text, risk_handoff
from tradingagents.agents.managers.live_allocation import enforce_live_allocation
import json
import logging
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from tradingagents.agents.utils.agent_utils import (
    build_capital_context,
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.conflict_detector import format_conflict_report_for_prompt
from tradingagents.agents.schemas import (
    LiveDecisionRow,
    LivePortfolioDecision,
    render_live_portfolio_decision,
)
from tradingagents.agents.utils.structured import bind_structured


logger = logging.getLogger(__name__)


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


def create_portfolio_manager(llm, memory):
    structured_llm = bind_structured(llm, LivePortfolioDecision, "Live Portfolio Manager")

    def invoke_live_decision(prompt: str) -> LivePortfolioDecision:
        if structured_llm is not None:
            try:
                return LivePortfolioDecision.model_validate(structured_llm.invoke(prompt))
            except Exception as exc:
                logger.warning("Live Portfolio Manager structured output failed: %s", exc)
        response = llm.invoke(prompt)
        content = str(getattr(response, "content", ""))
        fenced = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
        candidate = fenced.group(1) if fenced else content
        try:
            return LivePortfolioDecision.model_validate(json.loads(candidate))
        except Exception as exc:
            raise ValueError("Live Portfolio Manager did not return valid structured JSON") from exc

    def portfolio_manager_node(state) -> dict:
        # Lazy import: portfolio_state_manager imports schemas from this module,
        # so a top-level import here would be circular.
        from tradingagents.agents.managers.portfolio_state_manager import (
            _compute_short_term_market_anchors,
            _derive_short_term_rule_constraints,
            _format_short_term_market_anchors,
            _format_short_term_rule_constraints,
        )

        instrument_context = build_instrument_context(state["company_of_interest"])

        history = risk_handoff(state["risk_debate_state"])
        risk_debate_state = state["risk_debate_state"]
        research_plan = bounded_text(state["investment_plan"], 2200)
        trader_plan = bounded_text(state["trader_investment_plan"], 2400)
        holdings_info = state.get("holdings_info") or {}

        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        anchors = state["decision_anchors"] if "decision_anchors" in state else _compute_short_term_market_anchors(ticker, trade_date)
        holdings_info = {**holdings_info, "mark_price": (anchors or {}).get("current_price") or holdings_info.get("mark_price")}
        capital_context = build_capital_context(holdings_info)
        anchors_block = (
            "\n\n" + _format_short_term_market_anchors(anchors) if anchors else ""
        )
        constraints = _derive_short_term_rule_constraints(anchors, holdings_info, ticker)
        constraints_block = "\n\n" + _format_short_term_rule_constraints(constraints)
        conflict_block = format_conflict_report_for_prompt(state.get("conflict_report"))

        # P0.4 — required-data degradation gate. Core technical anchors must be
        # present to make a directional short-term call; without them the LLM
        # would be guessing levels from prose. Force HOLD instead of fabricating.
        required_anchors = ("current_price", "atr14", "ema10", "ema20")
        missing_required = (
            list(required_anchors)
            if anchors is None
            else [k for k in required_anchors if anchors.get(k) is None]
        )
        if missing_required:
            mark = holdings_info.get("mark_price")
            forced_model = LivePortfolioDecision(
                ticker=ticker,
                as_of_date=trade_date,
                decision="HOLD",
                current_price=mark,
                plan_rows=[LiveDecisionRow(
                    item="数据保护",
                    operation="HOLD",
                    trigger_description="核心市场数据恢复前不执行新交易",
                )],
                holding_period="until data is restored",
                rationale=(
                    "FORCED HOLD — required market anchors were unavailable; "
                    "directional levels must not be fabricated."
                ),
                missing_data=list(missing_required),
                watch_list=["restore OHLCV / indicator data feed, then re-run analysis"],
            )
            forced = render_live_portfolio_decision(forced_model, holdings_info)
            return {
                "risk_debate_state": {
                    **risk_debate_state,
                    "judge_decision": forced,
                    "latest_speaker": "Judge",
                },
                "final_trade_decision": forced,
                "live_portfolio_decision": forced_model.model_dump(),
                "structured_strategy": None,
            }

        curr_situation = (
            f"{state['market_report']}\n\n{state['sentiment_report']}\n\n"
            f"{state['news_report']}\n\n{state['fundamentals_report']}"
        )
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += bounded_text(rec["recommendation"], 600) + "\n\n"

        lessons_section = f"- Lessons from past decisions: **{past_memory_str}**\n" if past_memory_str else ""
        capital_block = f"\n\n{capital_context}" if capital_context else ""
        prompt = f"""You are the Portfolio Manager. Resolve the remaining disagreements in the risk handoffs and deliver the final short-term trading decision. Research Manager owns the thesis, Trader owns execution levels, and risk analysts own upside/downside/invalidation checks. Do not repeat their reports; keep rationale within 180 words. Python owns portfolio arithmetic and hard limits.

{instrument_context}{capital_block}

**Decision** (choose one): **BUY** | **HOLD** | **SELL**

**Context:**
- Research Manager's plan: {research_plan}
- Trader's proposal: {trader_plan}
{lessons_section}
**Latest compact risk assessments (one per role):**
{history}{anchors_block}{constraints_block}{conflict_block}

Use the precomputed market anchors verbatim — do not re-derive prices, ATR, support/resistance, or volume_ratio from the analyst reports. Treat the rule constraints as hard caps: position sizing and allowed_actions must respect them even when the debate suggests otherwise; note any constraint that overrode the debate wording in your rationale.

Be decisive and ground every parameter in specific evidence from the debate.{get_language_instruction()}

Return the configured LivePortfolioDecision schema only. plan_rows must be ordered as an executable ladder: existing/core position, entries/adds, defensive reduction, hard stop, profit-taking stages, then trend tail when applicable. Use numeric shares and prices; do not calculate reference_amount or NAV percentage because Python will calculate those. Represent a full exit with operation=CLEAR. Set scenario=base for current holdings and buys, defensive for reductions/stops on weakness, and profit for take-profit rows/trend tail. Defensive and profit branches are alternatives, not sequential exits. Size exits against current actual holdings only; pending buys may not fill. Include a numeric defensive CLEAR stop below every proposed buy. Entry cap applies to the first entry; add cap applies to all adds combined; both are percentages of NAV. Put conditional variants in conditional_notes. Decision audit fields must distinguish data-supported facts from inference."""

        try:
            decision_model = invoke_live_decision(prompt)
        except ValueError as exc:
            logger.error("Live Portfolio Manager output rejected: %s", exc)
            decision_model = LivePortfolioDecision(
                ticker=ticker,
                as_of_date=trade_date,
                decision="HOLD",
                current_price=float(anchors["current_price"]),
                plan_rows=[LiveDecisionRow(
                    item="输出保护",
                    operation="HOLD",
                    trigger_description="结构化决策生成失败，本轮不执行新交易",
                )],
                holding_period="until next review",
                rationale="FORCED HOLD — structured live decision validation failed.",
                missing_data=["valid LivePortfolioDecision output"],
                watch_list=["review provider structured-output logs and rerun"],
            )
        decision_model.ticker = ticker
        decision_model.as_of_date = trade_date
        decision_model.current_price = float(anchors["current_price"])
        decision_model = enforce_live_allocation(decision_model, holdings_info, constraints)
        decision_text = render_live_portfolio_decision(decision_model, holdings_info)

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
            "live_portfolio_decision": decision_model.model_dump(),
            "structured_strategy": None,
        }

    return portfolio_manager_node
