import re

from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor

@tool
def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """
    Retrieve a single technical indicator for a given ticker symbol.
    Uses the configured technical_indicators vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        indicator (str): A single technical indicator name, e.g. 'rsi', 'macd'. Call this tool once per indicator.
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted dataframe containing the technical indicators for the specified ticker symbol and indicator.
    """
    # LLMs sometimes pass multiple indicators as a comma-separated string;
    # split and process each individually.
    indicators = [i.strip().lower() for i in indicator.split(",") if i.strip()]
    results = []
    for ind in indicators:
        try:
            results.append(route_to_vendor("get_indicators", symbol, ind, curr_date, look_back_days))
        except ValueError as e:
            results.append(str(e))
    return "\n\n".join(results)

_DATE_VALUE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}):\s*(.*)$")


def _parse_indicator_series(text: str) -> tuple[dict[str, str], str]:
    """Split a vendor indicator payload into {date: value} plus its description.

    Vendors return one "YYYY-MM-DD: value" line per calendar day followed by a
    prose description. Non-trading days come back as an "N/A: ..." sentence.
    """
    values: dict[str, str] = {}
    description: list[str] = []
    for line in text.splitlines():
        match = _DATE_VALUE_RE.match(line.strip())
        if match:
            values[match.group(1)] = match.group(2).strip()
        elif line.strip():
            description.append(line.strip())
    return values, " ".join(description)


def _compact(value: str) -> str:
    """Round a numeric reading to 4 significant digits; pass text through."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return f"{number:.4g}"


@tool
def get_indicators_table(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicators: Annotated[str, "comma-separated indicator names, e.g. 'rsi,macd,atr'"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """Retrieve several technical indicators at once as a single aligned table.

    Prefer this over calling get_indicators once per indicator: one call returns
    every requested series sharing one date column, so the conversation carries
    one table instead of N separate ones.

    Args:
        symbol (str): Ticker symbol, e.g. AAPL
        indicators (str): Comma-separated indicator names, e.g. 'rsi,macd,atr'
        curr_date (str): The current trading date, YYYY-mm-dd
        look_back_days (int): How many days to look back, default 30
    Returns:
        str: One table with a date column and one column per indicator, newest
        first, non-trading days omitted, followed by each indicator's description.
    """
    names = [i.strip().lower() for i in indicators.split(",") if i.strip()]
    if not names:
        return "No indicators requested."

    series: dict[str, dict[str, str]] = {}
    descriptions: list[str] = []
    errors: list[str] = []
    for name in names:
        try:
            payload = route_to_vendor(
                "get_indicators", symbol, name, curr_date, look_back_days
            )
        except ValueError as e:
            errors.append(f"{name}: {e}")
            continue
        values, description = _parse_indicator_series(payload)
        series[name] = values
        if description:
            descriptions.append(f"- {name}: {description}")

    if not series:
        return "\n".join(errors) or "No indicator data available."

    # Non-trading days carry no reading for any indicator; dropping them removes
    # roughly 30% of the rows without losing information.
    dates = sorted(
        {
            date
            for values in series.values()
            for date, value in values.items()
            if not value.startswith("N/A")
        },
        reverse=True,
    )
    columns = [name for name in names if name in series]
    lines = [
        f"## {symbol} indicators through {curr_date} "
        f"({len(dates)} trading days; non-trading days omitted)",
        "",
        "date | " + " | ".join(columns),
        "--- | " + " | ".join("---" for _ in columns),
    ]
    for date in dates:
        row = [_compact(series[name].get(date, "")) for name in columns]
        lines.append(f"{date} | " + " | ".join(row))
    if descriptions:
        lines += ["", "Indicator notes:"] + descriptions
    if errors:
        lines += ["", "Unavailable:"] + [f"- {e}" for e in errors]
    return "\n".join(lines)
