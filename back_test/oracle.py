"""Perfect-foresight benchmark under explicit portfolio/execution constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OracleResult:
    total_return: float
    final_equity: float
    equity_curve: list[float]
    target_weights: list[float]
    turnover: float


def constraint_matched_oracle(
    prices: pd.Series,
    *,
    initial_capital: float,
    max_position_weight: float = 1.0,
    commission: float = 0.0,
    slippage_bps: float = 0.0,
    weight_steps: int = 20,
) -> OracleResult:
    """Return the best hindsight close-to-close path on a discrete weight grid.

    The oracle is long-only, cannot exceed ``max_position_weight``, pays the
    configured proportional slippage and flat commission whenever its target
    weight changes, and can only use one target weight for each full daily
    close-to-close interval. It is an evaluation upper bound, never a feature.
    """
    series = pd.Series(prices, dtype=float).dropna().reset_index(drop=True)
    if len(series) < 2 or initial_capital <= 0:
        return OracleResult(0.0, float(initial_capital), [float(initial_capital)], [0.0], 0.0)
    if (series <= 0).any():
        raise ValueError("oracle prices must be positive")
    cap = min(max(float(max_position_weight), 0.0), 1.0)
    steps = max(1, int(weight_steps))
    weights = np.linspace(0.0, cap, steps + 1)
    n = len(series)
    wealth = np.full(len(weights), -np.inf)
    wealth[0] = float(initial_capital)
    parents: list[np.ndarray] = []

    for t in range(n - 1):
        ratio = float(series.iloc[t + 1] / series.iloc[t])
        nxt = np.full(len(weights), -np.inf)
        parent = np.full(len(weights), -1, dtype=int)
        for old_i, old_wealth in enumerate(wealth):
            if not np.isfinite(old_wealth) or old_wealth <= 0:
                continue
            for new_i, new_weight in enumerate(weights):
                turnover = abs(float(new_weight - weights[old_i]))
                cost = old_wealth * turnover * float(slippage_bps) / 10_000.0
                if turnover > 1e-12:
                    cost += float(commission)
                after_cost = old_wealth - cost
                if after_cost <= 0:
                    continue
                candidate = after_cost * (1.0 + float(new_weight) * (ratio - 1.0))
                if candidate > nxt[new_i]:
                    nxt[new_i] = candidate
                    parent[new_i] = old_i
        wealth = nxt
        parents.append(parent)

    final_i = int(np.nanargmax(wealth))
    final_equity = float(wealth[final_i])
    indices = [final_i]
    for parent in reversed(parents):
        indices.append(int(parent[indices[-1]]))
    indices.reverse()
    path_weights = [float(weights[index]) for index in indices[1:]]

    equity = [float(initial_capital)]
    old_weight = 0.0
    total_turnover = 0.0
    for t, new_weight in enumerate(path_weights):
        turnover = abs(new_weight - old_weight)
        total_turnover += turnover
        cost = equity[-1] * turnover * float(slippage_bps) / 10_000.0
        if turnover > 1e-12:
            cost += float(commission)
        after_cost = equity[-1] - cost
        ratio = float(series.iloc[t + 1] / series.iloc[t])
        equity.append(after_cost * (1.0 + new_weight * (ratio - 1.0)))
        old_weight = new_weight

    return OracleResult(
        total_return=final_equity / float(initial_capital) - 1.0,
        final_equity=final_equity,
        equity_curve=equity,
        target_weights=path_weights,
        turnover=total_turnover,
    )


def oracle_capture_ratio(policy_return: float, oracle_return: float) -> float:
    """Log-return capture; zero when the oracle has no positive opportunity."""
    if oracle_return <= 0.0 or policy_return <= -1.0:
        return 0.0
    return float(np.log1p(policy_return) / np.log1p(oracle_return))
