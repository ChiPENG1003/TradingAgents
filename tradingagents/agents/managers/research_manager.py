
from tradingagents.agents.utils.agent_utils import build_instrument_context, get_language_instruction


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])
        history = state["investment_debate_state"].get("history", "")
        investment_debate_state = state["investment_debate_state"]

        curr_situation = (
            f"{state['market_report']}\n\n{state['sentiment_report']}\n\n"
            f"{state['news_report']}\n\n{state['fundamentals_report']}"
        )
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        memory_section = (
            f"Here are your past reflections on mistakes:\n\"{past_memory_str}\"\n\n"
            if past_memory_str else ""
        )

        prompt = f"""You are the Research Manager. Review the bull/bear debate and deliver a decisive investment recommendation: Buy, Sell, or Hold (only if strongly justified — do not default to Hold). Include your recommendation, rationale, and concrete next steps for the trader.

{memory_section}{instrument_context}

Debate history:
{history}

{get_language_instruction()}"""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
