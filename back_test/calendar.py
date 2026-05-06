"""Trading-day helpers for backtest date ranges."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def normalize_trading_days(dates: Iterable) -> pd.Series:
    """Return sorted normalized trading dates from a Date-like iterable."""
    values = pd.to_datetime(pd.Series(list(dates)), errors="coerce").dropna()
    if values.empty:
        return values
    return values.dt.normalize().drop_duplicates().sort_values().reset_index(drop=True)


def adjust_backtest_window(
    trading_dates: Iterable,
    requested_start: str,
    requested_end: str,
) -> tuple[str, str]:
    """Adjust requested boundaries to actual trading days.

    Start moves to the first trading day on or after requested_start.
    End moves to the last trading day on or before requested_end.
    """
    dates = normalize_trading_days(trading_dates)
    start = pd.to_datetime(requested_start).normalize()
    end = pd.to_datetime(requested_end).normalize()

    if start > end:
        raise ValueError(f"Backtest start date {requested_start} is after end date {requested_end}.")

    effective_start = dates[dates >= start]
    effective_end = dates[dates <= end]
    if effective_start.empty or effective_end.empty:
        raise ValueError(
            f"No trading days found between {requested_start} and {requested_end}."
        )

    start_date = effective_start.iloc[0]
    end_date = effective_end.iloc[-1]
    if start_date > end_date:
        raise ValueError(
            f"No trading days found between {requested_start} and {requested_end}."
        )

    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def first_trading_day_on_or_after(
    trading_dates: Iterable,
    date: str,
    latest: str,
) -> str | None:
    """Return first trading day on/after date and on/before latest."""
    dates = normalize_trading_days(trading_dates)
    target = pd.to_datetime(date).normalize()
    cutoff = pd.to_datetime(latest).normalize()
    matches = dates[(dates >= target) & (dates <= cutoff)]
    if matches.empty:
        return None
    return matches.iloc[0].strftime("%Y-%m-%d")
