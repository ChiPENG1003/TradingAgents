"""Deterministic limits for a single-target live recommendation (never execution)."""
from __future__ import annotations

import math

from tradingagents.agents.schemas import LiveDecisionRow, LivePortfolioDecision
from tradingagents.live_portfolio import allocation_context


def enforce_live_allocation(decision: LivePortfolioDecision, holdings: dict, constraints: dict) -> LivePortfolioDecision:
    decision = decision.model_copy(deep=True)
    if decision.current_price is None:
        decision.decision = "HOLD"
        decision.plan_rows = [LiveDecisionRow(item="数据保护", operation="HOLD", trigger_description="价格缺失，重新获取行情后分析")]
        decision.missing_data.append("current price")
        return decision
    allocation = allocation_context(holdings, decision.current_price)
    nav = allocation["nav"]
    initial = int(float(holdings.get("quantity") or 0))
    budget = allocation["buy_budget"]
    entry_budget = nav * constraints["max_entry_size_pct"] / 100
    add_budget = nav * constraints["max_add_position_size_pct"] / 100
    notes = []
    if allocation["cash"] is None:
        notes.append("可用现金未知：买入股数设为 0；请在持仓 JSON 中提供 cash 或全部持仓的 market_price 后重新分析。")
        decision.missing_data.append("available cash")
    if allocation["other_unpriced_count"]:
        decision.missing_data.append("market prices for other holdings; cost basis is not current weight")
    stops = [r for r in decision.plan_rows if r.operation == "CLEAR" and r.scenario != "profit"]
    stop_prices = [p for r in stops for p in (r.price_low, r.price_high, r.reference_price) if p is not None]
    min_buy = min([decision.current_price] + [p for r in decision.plan_rows if r.operation == "BUY" for p in (r.price_low, r.price_high, r.reference_price) if p is not None])
    has_stop = bool(stop_prices) and max(stop_prices) < min_buy
    buy_total = 0
    buy_count = 0
    for row in decision.plan_rows:
        if row.operation != "BUY":
            continue
        row.scenario = "base"
        requested = row.shares or 0
        # Use the highest stated price for reservation, so a range cannot hide
        # an order more expensive than its displayed midpoint/reference.
        price = max(p for p in (row.price_low, row.price_high, row.reference_price, decision.current_price) if p is not None)
        cap = entry_budget if initial == 0 and buy_count == 0 else add_budget
        allowed = min(requested, max(0, math.floor(min(budget, cap) / price)))
        if (constraints["entry_mode"] == "no_new_or_add" or "BUY" not in constraints["allowed_actions"] or decision.decision == "SELL" or not has_stop):
            allowed = 0
        if requested != allowed or row.shares is None:
            notes.append(f"{row.item}：买入由 {requested} 股调整为 {allowed} 股（现金、组合占比、累计仓位限制或止损条件）。")
        row.shares = allowed
        budget = max(0, budget - allowed * price)
        if initial == 0 and buy_count == 0:
            entry_budget -= allowed * price
        else:
            add_budget -= allowed * price
        buy_total += allowed
        row.target_shares_after = initial + buy_total
        buy_count += 1
    # Exit branches are alternatives, never a serial stop-then-profit path.
    # Reserve sells against current holdings only: pending buys may never fill.
    remaining = {"defensive": initial, "profit": initial}
    for row in decision.plan_rows:
        if row.operation in ("SELL", "CLEAR"):
            if row.scenario == "base":
                row.scenario = "defensive" if row.operation == "CLEAR" else "profit"
            balance = remaining[row.scenario]
            requested = balance if row.operation == "CLEAR" else (row.shares or 0)
            shares = min(requested, balance) if "SELL" in constraints["allowed_actions"] else 0
            if shares != requested:
                notes.append(f"{row.item}：卖出股数限制为当前分支可持有的 {shares} 股。")
            row.shares = shares
            remaining[row.scenario] -= shares
            row.target_shares_after = remaining[row.scenario]
        elif row.operation == "HOLD":
            row.shares = remaining.get(row.scenario, initial)
            row.target_shares_after = row.shares
    if decision.decision not in constraints["allowed_actions"]:
        decision.decision = "HOLD"
        notes.append("原决策不在 allowed_actions 内，已调整为 HOLD。")
    if decision.decision == "BUY" and buy_total == 0:
        decision.decision = "HOLD"
    if decision.decision == "SELL" and not any(r.shares for r in decision.plan_rows if r.operation in ("SELL", "CLEAR")):
        decision.decision = "HOLD"
    if not any((r.shares or 0) > 0 for r in decision.plan_rows):
        decision.plan_rows = [LiveDecisionRow(item="等待", operation="HOLD", trigger_description="资金或交易条件满足后重新分析", shares=initial)]
    if buy_total:
        notes.append("买入行按全部触发累计预留资金；卖出分支按当前实际持股计算，买入成交后需重新分析。")
    if notes:
        decision.conditional_notes.extend(notes)
        decision.rationale += " Python 仓位校验已应用；最终股数和决策以上表为准。"
    return decision
