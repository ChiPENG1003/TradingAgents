import json

import pandas as pd

from .alpha_vantage_common import _make_api_request


def _parse_payload(result):
    """Return (obj, was_str). _make_api_request returns the raw response *text*
    (a JSON string for these endpoints), so callers that only guarded on
    isinstance(result, dict) silently skipped filtering and leaked future data
    into backtests. Parse the string first; non-JSON bodies pass through."""
    if isinstance(result, dict):
        return result, False
    if isinstance(result, str):
        try:
            return json.loads(result), True
        except (json.JSONDecodeError, ValueError):
            return None, True
    return None, False


def _filter_reports_by_date(result, curr_date: str):
    """Filter annualReports/quarterlyReports to exclude entries after curr_date.

    Prevents look-ahead bias by removing fiscal periods that end after
    the simulation's current date. Handles both dict and JSON-string payloads
    (the API helper returns a string), preserving the caller's return type.
    """
    if not curr_date:
        return result
    obj, was_str = _parse_payload(result)
    if not isinstance(obj, dict):
        return result
    for key in ("annualReports", "quarterlyReports"):
        if key in obj:
            obj[key] = [
                r for r in obj[key]
                if r.get("fiscalDateEnding", "") <= curr_date
            ]
    return json.dumps(obj) if was_str else obj


# Time-invariant OVERVIEW fields. Static descriptors of the company that don't
# depend on "now" — safe at any date.
_OVERVIEW_STRUCTURAL_KEYS = {
    "Symbol", "AssetType", "Name", "Description", "CIK", "Exchange",
    "Currency", "Country", "Sector", "Industry", "Address",
    "FiscalYearEnd", "OfficialSite", "OfficialName",
}


def _is_historical_curr_date(curr_date: str | None) -> bool:
    """Same heuristic as y_finance._is_historical_curr_date — duplicated to
    avoid an import cycle between the two vendor modules."""
    if not curr_date:
        return False
    try:
        target = pd.to_datetime(curr_date).normalize()
        today = pd.Timestamp.today().normalize()
        return (today - target).days > 2
    except Exception:
        return False


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    """
    Retrieve comprehensive fundamental data for a given ticker symbol using Alpha Vantage.

    Alpha Vantage's OVERVIEW endpoint is a real-time snapshot (52W high/low,
    market cap, TTM ratios, MAs, latest quarter EPS) — every numeric field
    is measured against TODAY. During a historical backtest only the
    structural identifier fields are returned to prevent look-ahead bias.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd

    Returns:
        str: Company overview data including financial ratios and key metrics
    """
    params = {
        "symbol": ticker,
    }
    result = _make_api_request("OVERVIEW", params)

    if _is_historical_curr_date(curr_date):
        # _make_api_request returns the raw response text (a JSON string), so
        # parse before sanitizing — the previous isinstance(result, dict) guard
        # never matched and let real-time OVERVIEW fields leak into backtests.
        obj, was_str = _parse_payload(result)
        if isinstance(obj, dict):
            sanitized = {k: v for k, v in obj.items() if k in _OVERVIEW_STRUCTURAL_KEYS}
            sanitized["_as_of_date"] = curr_date
            sanitized["_note"] = (
                "Real-time OVERVIEW fields (52W high/low, market cap, TTM ratios, "
                "MAs, etc.) omitted to prevent look-ahead bias. Use balance_sheet/"
                "cashflow/income_statement for filed historical figures."
            )
            return json.dumps(sanitized) if was_str else sanitized

    return result


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve balance sheet data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("BALANCE_SHEET", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve cash flow statement data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("CASH_FLOW", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve income statement data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("INCOME_STATEMENT", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)

