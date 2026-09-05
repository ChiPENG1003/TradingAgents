"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    position_sizing: Optional[str] = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: Optional[float] = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Live Portfolio Manager execution plan
# ---------------------------------------------------------------------------


class LiveDecisionRow(BaseModel):
    scenario: Literal["base", "defensive", "profit"] = "base"
    item: str = Field(description="Short row label, e.g. core, add 1, hard stop, take profit 1.")
    operation: Literal["HOLD", "BUY", "SELL", "CLEAR"]
    trigger_description: str
    price_low: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    price_high: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    confirmation_rule: Optional[str] = None
    shares: Optional[int] = Field(default=None, ge=0)
    target_shares_after: Optional[int] = Field(default=None, ge=0)
    reference_price: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)


class LivePortfolioDecision(BaseModel):
    ticker: str
    as_of_date: str
    decision: Literal["BUY", "HOLD", "SELL"]
    current_price: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    plan_rows: list[LiveDecisionRow] = Field(min_length=1)
    conditional_notes: list[str] = Field(default_factory=list)
    holding_period: str
    rationale: str
    data_supported: list[str] = Field(default_factory=list)
    inferred: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    invalidation_triggers: list[str] = Field(default_factory=list)
    watch_list: list[str] = Field(default_factory=list)


def render_live_portfolio_decision(
    decision: LivePortfolioDecision,
    holdings_info: Optional[dict] = None,
) -> str:
    """Render a stable table first; all amounts/NAV values are computed in Python."""
    holdings = holdings_info or {}
    equity = holdings.get("nav") or holdings.get("equity")
    equity = float(equity) if equity not in (None, 0) else None

    def money(value: Optional[float]) -> str:
        return "—" if value is None else f"约 {value:,.0f}"

    def pct(value: Optional[float]) -> str:
        return "—" if value is None else f"{value:.1f}%"

    rows = [
        "| 项目 | 触发条件/价格 | 股数 | 参考金额 | 占 NAV |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in decision.plan_rows:
        trigger = row.trigger_description
        if row.confirmation_rule:
            trigger = f"{trigger}；{row.confirmation_rule}"
        shares = row.shares
        if row.operation == "CLEAR":
            shares_text = f"清仓全部剩余（当前 {shares:,} 股）" if shares is not None else "清仓全部剩余"
        elif shares is None:
            shares_text = "—"
        elif row.operation == "BUY":
            shares_text = f"加 {shares:,} 股"
        elif row.operation == "SELL":
            shares_text = f"减 {shares:,} 股"
        else:
            shares_text = f"{shares:,} 股"
        ref_price = row.reference_price
        if ref_price is None:
            if row.price_low is not None and row.price_high is not None:
                ref_price = (row.price_low + row.price_high) / 2.0
            elif row.price_low is not None:
                ref_price = row.price_low
            elif row.price_high is not None:
                ref_price = row.price_high
            else:
                ref_price = decision.current_price
        amount = abs(shares * ref_price) if shares is not None and ref_price is not None else None
        nav_pct = amount / equity * 100.0 if amount is not None and equity else None
        rows.append(
            f"| {row.item} ({row.scenario}) | {trigger} | {shares_text} | {money(amount)} | {pct(nav_pct)} |"
        )

    if holdings.get("nav"):
        from tradingagents.live_portfolio import allocation_context
        allocation = allocation_context(holdings, decision.current_price)
        rows.extend(["", f"**组合占比**：目标股票当前 {pct(allocation['target_weight_pct'])}；其他持仓已知市值占比 {pct(allocation['other_known_weight_pct'])}（{allocation['other_position_count']} 只）。"])
        if allocation["other_unpriced_count"]:
            rows.append(f"另有 {allocation['other_unpriced_count']} 只其他持仓缺少当前价格；其成本合计 {money(allocation['other_unpriced_cost_basis'])}，不作为市值占比。")
        buy_shares = sum(r.shares or 0 for r in decision.plan_rows if r.operation == "BUY")
        if buy_shares and decision.current_price and equity:
            projected_weight = (float(holdings.get("quantity") or 0) + buy_shares) * decision.current_price / equity * 100
            rows.append(f"全部买入行成交后，按当前价及不变 NAV 估算目标持仓占比 {projected_weight:.1f}%；配置上限 {allocation['max_position_pct']:g}%。")
        rows.append("表中占 NAV 为该行交易/持仓的参考金额比例；defensive 与 profit 是互斥情景。其他持仓保持不变。")
    parts = rows
    for note in decision.conditional_notes:
        parts.extend(["", f"> {note}"])
    parts.extend([
        "", f"**Decision**: {decision.decision}",
        "", f"**Holding Period**: {decision.holding_period}",
        "", f"**Rationale**: {decision.rationale}",
        "", "**Decision Audit**:",
        f"- **Data-supported**: {'; '.join(decision.data_supported) or 'none'}",
        f"- **Inferred**: {'; '.join(decision.inferred) or 'none'}",
        f"- **Missing data**: {'; '.join(decision.missing_data) or 'none'}",
        f"- **Invalidation triggers**: {'; '.join(decision.invalidation_triggers) or 'none'}",
        f"- **Watch-list**: {'; '.join(decision.watch_list) or 'none'}",
    ])
    return "\n".join(parts)
