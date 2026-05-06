"""Performance metrics for the backtest engine.

All functions accept either pandas Series or numpy arrays where natural.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def total_return(equity: pd.Series) -> float:
    """Return cumulative return: equity[-1] / equity[0] - 1."""
    if equity.empty:
        return 0.0
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start == 0:
        return 0.0
    return end / start - 1.0


def annualized_return(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized geometric return derived from the equity curve."""
    n = len(equity)
    if n < 2:
        return 0.0
    tr = total_return(equity)
    years = n / periods_per_year
    if years <= 0:
        return 0.0
    return (1.0 + tr) ** (1.0 / years) - 1.0


def daily_returns(equity: pd.Series) -> pd.Series:
    """Simple daily returns from the equity curve."""
    return equity.pct_change().dropna()


def sharpe_ratio(
    returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio. rf is annual risk-free rate."""
    if returns.empty:
        return 0.0
    excess = returns - rf / periods_per_year
    std = excess.std(ddof=1)
    if not std or math.isnan(std) or std == 0:
        return 0.0
    return float(excess.mean() / std * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline as a negative number (e.g. -0.23 = -23%)."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def win_rate(trades: List[dict]) -> Optional[float]:
    """Fraction of round-trip trades whose realized PnL was positive.

    `trades` is a list of dicts each with a 'pnl' key (set when the position
    is closed). Open positions without a close are ignored.
    """
    closed = [t for t in trades if t.get("pnl") is not None]
    if not closed:
        return None
    wins = sum(1 for t in closed if t["pnl"] > 0)
    return wins / len(closed)


def summarize(equity: pd.Series, trades: Optional[List[dict]] = None) -> dict:
    """Return a dict of all metrics suitable for JSON serialization."""
    rets = daily_returns(equity)
    summary = {
        "total_return":       total_return(equity),
        "annualized_return":  annualized_return(equity),
        "sharpe_ratio":       sharpe_ratio(rets),
        "max_drawdown":       max_drawdown(equity),
        "n_observations":     int(len(equity)),
    }
    if trades is not None:
        wr = win_rate(trades)
        summary["win_rate"] = wr
        summary["n_trades"] = len([t for t in trades if t.get("pnl") is not None])
    return summary
