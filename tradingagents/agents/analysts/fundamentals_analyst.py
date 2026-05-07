from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
    get_language_instruction,
)
from tradingagents.dataflows.y_finance import _is_historical_curr_date
from tradingagents.dataflows.config import get_config


def _sanitize_fundamentals_report(report: str, current_date: str) -> str:
    if not report:
        return report

    lines = report.splitlines()
    if _is_historical_curr_date(current_date):
        blocked_phrases = ("52 Week High", "52 Week Low")
        lines = [
            line for line in lines
            if not any(phrase.lower() in line.lower() for phrase in blocked_phrases)
        ]

    if not any(line.strip().startswith("# As-of date:") for line in lines):
        lines.insert(0, f"# As-of date: {current_date}")

    return "\n".join(lines).strip()


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_message = (
            "You are a fundamentals analyst. Use the available tools to retrieve financial statements and company data. "
            f"Begin the report with exactly this line: # As-of date: {current_date}. "
            "Write a report covering key financial metrics and fundamental health. "
            "For historical dates, do not include real-time snapshot fields such as 52 Week High or 52 Week Low. "
            "Append a Markdown summary table at the end."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Use the provided tools to gather data and write your report."
                    " If you cannot fully answer, that's OK — your report will be used by downstream agents."
                    " Tools: {tool_names}.\n{system_message}"
                    " Current date: {current_date}. Strict-cutoff: use only filings, data, and events dated on or before this current date; never use later information. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content
            report = _sanitize_fundamentals_report(report, current_date)

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
