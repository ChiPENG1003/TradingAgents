from tradingagents.agents.utils.decision_context import bounded_text, evidence_brief, risk_handoff

from tradingagents.agents.utils.agent_utils import (
    DEBATE_EVIDENCE_GUARDRAIL,
    build_capital_context,
    get_language_instruction,
)


def create_aggressive_debator(llm):
    def aggressive_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_handoff(risk_debate_state)
        aggressive_history = risk_debate_state.get("aggressive_history", "")


        trader_decision = bounded_text(state["trader_investment_plan"], 2400)
        evidence = evidence_brief(state)
        capital_context = build_capital_context(state.get("holdings_info"))
        capital_block = f"\n\n{capital_context}" if capital_context else ""

        prompt = f"""You are the Aggressive Risk Analyst. Champion the trader's decision by arguing for its upside potential and countering conservative/neutral objections with specific rebuttals. Write a compact handoff, at most 180 words: proposed action/size, two sourced facts, strongest counter-evidence, invalidation trigger, and missing data. Carry forward any unresolved material objection from earlier rounds.

Trader's decision: {trader_decision}{capital_block}

Evidence available to all risk analysts:
{evidence}

Latest assessment from each role: {history}{DEBATE_EVIDENCE_GUARDRAIL}{get_language_instruction()}"""

        response = llm.invoke(prompt)

        argument = f"Aggressive Analyst: {response.content}"

        new_risk_debate_state = {
            "history": risk_debate_state.get("history", "") + "\n" + argument,
            "aggressive_history": aggressive_history + "\n" + argument,
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Aggressive",
            "current_aggressive_response": argument,
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return aggressive_node
