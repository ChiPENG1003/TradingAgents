"""Regression tests for the upstream fixes ported into this fork.

Covers the two silent look-ahead / stale-data bugs that would otherwise
corrupt backtest results without any visible error:

  * yfinance occasionally returns a year-old OHLCV frame that still has rows,
    which used to pass the empty-check and feed wrong prices into a report.
  * Alpha Vantage fundamentals arrive as a JSON *string*, so the old
    isinstance(result, dict) guard skipped the look-ahead filter entirely and
    let future fiscal periods / real-time OVERVIEW fields leak into backtests.
"""

import json

import pandas as pd
import pytest

from tradingagents.dataflows.stockstats_utils import (
    StaleMarketDataError,
    assert_ohlcv_not_stale,
)
from tradingagents.dataflows.alpha_vantage_fundamentals import _filter_reports_by_date


def test_stale_frame_rejected():
    df = pd.DataFrame(
        {"Date": pd.to_datetime(["2025-06-01", "2025-06-02"]), "Close": [1.0, 2.0]}
    )
    with pytest.raises(StaleMarketDataError):
        assert_ohlcv_not_stale(df, "2026-06-14", "TEST")


def test_fresh_frame_passes():
    df = pd.DataFrame(
        {"Date": pd.to_datetime(["2026-06-12", "2026-06-13"]), "Close": [1.0, 2.0]}
    )
    assert_ohlcv_not_stale(df, "2026-06-14", "TEST")  # must not raise


def test_holiday_gap_within_tolerance_passes():
    # 9 calendar days < MAX_OHLCV_STALE_DAYS (10): a long weekend, not stale.
    df = pd.DataFrame({"Date": pd.to_datetime(["2026-06-05"]), "Close": [1.0]})
    assert_ohlcv_not_stale(df, "2026-06-14", "TEST")


def test_empty_frame_left_to_caller():
    assert_ohlcv_not_stale(pd.DataFrame(), "2026-06-14", "TEST")  # no raise


def test_alpha_vantage_json_string_is_filtered():
    """The payload is a JSON string; future fiscal periods must be dropped."""
    payload = json.dumps(
        {
            "symbol": "AAPL",
            "annualReports": [
                {"fiscalDateEnding": "2025-09-30", "x": 1},
                {"fiscalDateEnding": "2026-09-30", "x": 2},  # future vs curr_date
            ],
        }
    )
    out = _filter_reports_by_date(payload, "2026-01-01")
    assert isinstance(out, str)  # return type preserved
    kept = [r["fiscalDateEnding"] for r in json.loads(out)["annualReports"]]
    assert kept == ["2025-09-30"]


def test_alpha_vantage_non_json_passthrough():
    body = "Open,High,Low\n1,2,3"
    assert _filter_reports_by_date(body, "2026-01-01") == body


def test_alpha_vantage_no_curr_date_passthrough():
    payload = json.dumps({"annualReports": [{"fiscalDateEnding": "2030-01-01"}]})
    assert _filter_reports_by_date(payload, None) == payload
