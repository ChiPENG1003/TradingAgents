from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

import pandas as pd

from back_test.engine import BacktestEngine, PROJECT_ROOT
from back_test.metrics import summarize
from back_test.oracle import constraint_matched_oracle, oracle_capture_ratio
from back_test.policy_config import (
    PortfolioStatePolicyConfig,
    optimizable_bounds,
    optimizable_dtypes,
    optimizable_field_names,
)


OPTIMIZATION_DIR = PROJECT_ROOT / "back_test" / "results" / "optimization"


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str

SCORE_WEIGHT_KEYS = (
    "momentum_score_weight", "event_score_weight", "risk_score_weight",
    "trend_continuation_weight", "reversal_risk_weight",
    "breakout_quality_weight", "pullback_quality_weight",
    "volume_support_weight", "event_risk_weight",
    "reward_risk_quality_weight",
)
DEFAULT_SCALED_WEIGHTS = {
    "momentum_score_weight": 0.5,
    "event_score_weight": 0.3,
    "risk_score_weight": 1.92,
    "trend_continuation_weight": 0.0,
    "reversal_risk_weight": 0.0,
    "breakout_quality_weight": 0.0,
    "pullback_quality_weight": 0.0,
    "volume_support_weight": 0.0,
    "event_risk_weight": 0.0,
    "reward_risk_quality_weight": 0.0,
}
DECISION_PARAM_KEYS = (
    "stop_loss_atr_multiple",
    "trend_take_profit_atr_multiple",
    "default_take_profit_atr_multiple",
    "strong_uptrend_floor",
    "strong_uptrend_cap",
    "weak_uptrend_floor",
    "weak_uptrend_cap",
    "range_cap",
    "max_target_weight",
    "min_trade_weight",
    "order_size_multiplier",
    "strong_uptrend_take_profit_size_pct",
    "default_take_profit_size_pct",
)

POLICY_ONLY_KEYS = SCORE_WEIGHT_KEYS + DECISION_PARAM_KEYS

EXECUTION_PARAM_KEYS = (
    "max_trade_risk_pct", "max_single_add_pct", "max_position_after_add_pct",
    "max_adds_per_trade", "min_days_between_adds",
    "max_entry_gap_above_plan_pct", "max_add_gap_above_plan_pct",
    "obvious_bull_max_entry_gap_pct", "obvious_bull_max_add_gap_pct",
    "entry_signal_ttl_trading_days", "add_signal_ttl_trading_days",
    "block_shrinking_volume_adds", "shrinking_volume_close_hold_days",
    "add_key_level_tolerance_pct",
)

PARAMETER_GROUPS = {
    "execution": EXECUTION_PARAM_KEYS,
    "core_state": (
        "momentum_score_weight", "risk_score_weight",
        "trend_continuation_weight", "reversal_risk_weight",
    ),
    "setup_quality": (
        "breakout_quality_weight", "pullback_quality_weight",
        "volume_support_weight", "reward_risk_quality_weight",
    ),
    "event": ("event_score_weight", "event_risk_weight"),
    "risk_exit": (
        "stop_loss_atr_multiple", "trend_take_profit_atr_multiple",
        "default_take_profit_atr_multiple", "strong_uptrend_take_profit_size_pct",
        "default_take_profit_size_pct",
    ),
    "position_sizing": (
        "strong_uptrend_floor", "strong_uptrend_cap",
        "weak_uptrend_floor", "weak_uptrend_cap", "range_cap",
        "max_target_weight", "min_trade_weight", "order_size_multiplier",
    ),
}
PARAMETER_GROUPS["all"] = tuple(dict.fromkeys(
    key for keys in PARAMETER_GROUPS.values() for key in keys
))


def resolve_parameter_names(
    groups: Optional[list[str]] = None,
    params: Optional[list[str]] = None,
) -> tuple[str, ...]:
    """Resolve ordered, de-duplicated search parameters from CLI-style selections."""
    groups = groups or ["execution"]
    unknown_groups = sorted(set(groups) - set(PARAMETER_GROUPS))
    if unknown_groups:
        raise ValueError(f"unknown parameter groups: {unknown_groups}")
    valid = set(EXECUTION_PARAM_KEYS) | set(POLICY_ONLY_KEYS)
    unknown_params = sorted(set(params or []) - valid)
    if unknown_params:
        raise ValueError(f"unknown optimization parameters: {unknown_params}")
    selected = [key for group in groups for key in PARAMETER_GROUPS[group]]
    selected.extend(params or [])
    return tuple(dict.fromkeys(selected))


def _policy_param_metadata() -> dict[str, tuple[float, float, str]]:
    """Map each optimizable PortfolioStatePolicyConfig field to (low, high, dtype)."""
    return {
        name: (low, high, dtype)
        for name, (low, high), dtype in zip(
            optimizable_field_names(), optimizable_bounds(), optimizable_dtypes()
        )
    }

_ORDERED_FLOOR_CAP_PAIRS = (
    ("strong_uptrend_floor", "strong_uptrend_cap"),
    ("weak_uptrend_floor", "weak_uptrend_cap"),
)


def _suggest_decision_params(trial) -> dict[str, Any]:
    """Suggest the decision-layer params using bounds from the dataclass metadata.

    Caps in ``_ORDERED_FLOOR_CAP_PAIRS`` are sampled with their lower bound
    raised to the matching floor, guaranteeing ``cap >= floor``.
    """
    meta = _policy_param_metadata()
    cap_to_floor = {cap: floor for floor, cap in _ORDERED_FLOOR_CAP_PAIRS}
    params: dict[str, Any] = {}
    for name in DECISION_PARAM_KEYS:
        low, high, dtype = meta[name]
        if name in cap_to_floor:
            # The floor is earlier in DECISION_PARAM_KEYS, so already sampled.
            low = max(low, float(params[cap_to_floor[name]]))
        if dtype == "int":
            params[name] = trial.suggest_int(name, int(low), int(high))
        else:
            params[name] = trial.suggest_float(name, low, high)
    return params


def default_decision_params() -> dict[str, Any]:
    """Dataclass-default values for DECISION_PARAM_KEYS, for the closed-loop baseline."""
    defaults = PortfolioStatePolicyConfig()
    return {name: getattr(defaults, name) for name in DECISION_PARAM_KEYS}


def suggest_policy_params(
    trial,
    *,
    learn_weights: bool = False,
    learn_decision: bool = False,
    parameter_names: Optional[tuple[str, ...]] = None,
) -> dict[str, Any]:
    if parameter_names is None:
        selected = list(EXECUTION_PARAM_KEYS)
        if learn_weights:
            selected.extend(SCORE_WEIGHT_KEYS)
        if learn_decision:
            selected.extend(DECISION_PARAM_KEYS)
        parameter_names = tuple(dict.fromkeys(selected))

    float_specs = {
        "max_trade_risk_pct": (0.008, 0.030, 0.002),
        "max_single_add_pct": (3.0, 12.0, 1.0),
        "max_position_after_add_pct": (0.35, 0.85, 0.05),
        "max_entry_gap_above_plan_pct": (0.000, 0.025, 0.0025),
        "max_add_gap_above_plan_pct": (0.000, 0.015, 0.0025),
        "obvious_bull_max_entry_gap_pct": (0.005, 0.030, 0.0025),
        "obvious_bull_max_add_gap_pct": (0.000, 0.010, 0.0025),
        "add_key_level_tolerance_pct": (0.000, 0.015, 0.0025),
        "momentum_score_weight": (0.0, 1.5, 0.05),
        "event_score_weight": (0.0, 0.8, 0.025),
        "risk_score_weight": (0.0, 2.0, 0.05),
        "trend_continuation_weight": (0.0, 1.0, 0.05),
        "reversal_risk_weight": (0.0, 1.0, 0.05),
        "breakout_quality_weight": (0.0, 1.0, 0.05),
        "pullback_quality_weight": (0.0, 1.0, 0.05),
        "volume_support_weight": (0.0, 1.0, 0.05),
        "event_risk_weight": (0.0, 1.0, 0.05),
        "reward_risk_quality_weight": (0.0, 1.0, 0.05),
    }
    int_specs = {
        "max_adds_per_trade": (0, 4),
        "min_days_between_adds": (1, 5),
        "entry_signal_ttl_trading_days": (1, 4),
        "add_signal_ttl_trading_days": (1, 3),
        "shrinking_volume_close_hold_days": (1, 4),
    }
    params: dict[str, Any] = {}
    meta = _policy_param_metadata()
    cap_to_floor = {cap: floor for floor, cap in _ORDERED_FLOOR_CAP_PAIRS}
    for name in parameter_names:
        if name in float_specs:
            low, high, step = float_specs[name]
            params[name] = trial.suggest_float(name, low, high, step=step)
        elif name in int_specs:
            low, high = int_specs[name]
            params[name] = trial.suggest_int(name, low, high)
        elif name == "block_shrinking_volume_adds":
            params[name] = trial.suggest_categorical(name, [True, False])
        elif name in meta:
            low, high, dtype = meta[name]
            floor_name = cap_to_floor.get(name)
            if floor_name and floor_name in params:
                low = max(low, float(params[floor_name]))
            params[name] = (
                trial.suggest_int(name, int(low), int(high))
                if dtype == "int"
                else trial.suggest_float(name, low, high)
            )
        else:
            raise ValueError(f"unsupported optimization parameter: {name}")
    return params


def default_policy_params() -> dict[str, Any]:
    """Current hand-tuned defaults used as a stable baseline."""
    return {
        "max_trade_risk_pct": 0.020,
        "max_single_add_pct": 8.0,
        "max_position_after_add_pct": 0.60,
        "max_adds_per_trade": 2,
        "min_days_between_adds": 2,
        "max_entry_gap_above_plan_pct": 0.010,
        "max_add_gap_above_plan_pct": 0.008,
        "obvious_bull_max_entry_gap_pct": 0.015,
        "obvious_bull_max_add_gap_pct": 0.005,
        "entry_signal_ttl_trading_days": 2,
        "add_signal_ttl_trading_days": 1,
        "block_shrinking_volume_adds": True,
        "shrinking_volume_close_hold_days": 2,
        "add_key_level_tolerance_pct": 0.005,
    }


def baseline_params_for(parameter_names: tuple[str, ...]) -> dict[str, Any]:
    engine_defaults = default_policy_params()
    policy_defaults = PortfolioStatePolicyConfig()
    baseline: dict[str, Any] = {}
    for name in parameter_names:
        if name in engine_defaults:
            baseline[name] = engine_defaults[name]
        elif name in DEFAULT_SCALED_WEIGHTS:
            baseline[name] = DEFAULT_SCALED_WEIGHTS[name]
        elif hasattr(policy_defaults, name):
            baseline[name] = getattr(policy_defaults, name)
        else:
            raise ValueError(f"no baseline value for parameter {name}")
    return baseline


def _replace_nonfinite(value):
    if isinstance(value, dict):
        return {k: _replace_nonfinite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_nonfinite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 0.0
    return value


OBJECTIVE_VERSION = "oracle_capture_v1"
SEARCH_SPACE_VERSION = "parameter_groups_v1"


def _short_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def dataset_hash(
    ticker: str,
    strategies_dir: Optional[Path] = None,
) -> str:
    root = strategies_dir or (PROJECT_ROOT / "back_test" / "strategy" / ticker)
    digest = hashlib.sha256()
    paths = sorted(Path(root).glob(f"{ticker}_*.json"))
    if not paths:
        raise ValueError(f"no strategy snapshots found for dataset hash under {root}")
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def score_metrics(
    metrics: dict[str, Any],
    report: Optional[dict[str, Any]] = None,
    *,
    objective: str = "legacy",
    max_drawdown_limit: Optional[float] = None,
    drawdown_mode: str = "penalty",
    drawdown_penalty: float = 2.0,
) -> float:
    """Score one replay, optionally against the constraint-matched oracle."""
    report = report or {}
    total = float(metrics.get("total_return") or 0.0)
    sharpe = float(metrics.get("sharpe_ratio") or 0.0)
    max_dd = abs(float(metrics.get("max_drawdown") or 0.0))
    n_trades = int(metrics.get("n_trades") or 0)
    risk_rejects = int(report.get("buy_rejected_risk_budget") or 0)
    gap_rejects = int(report.get("gap_buy_rejected") or 0)
    ttl_expired = int(report.get("signal_ttl_expired") or 0)

    oracle = report.get("oracle") or {}
    if objective == "oracle_capture":
        base_score = float(oracle.get("capture_ratio") or 0.0)
    elif objective == "total_return":
        base_score = total
    elif objective == "legacy":
        base_score = total + 0.15 * sharpe - 1.5 * max_dd
    else:
        raise ValueError(f"unknown objective: {objective}")

    if max_drawdown_limit is not None and max_dd > float(max_drawdown_limit):
        excess = max_dd - float(max_drawdown_limit)
        if drawdown_mode == "hard":
            return -1_000_000.0 - excess
        if drawdown_mode != "penalty":
            raise ValueError(f"unknown drawdown_mode: {drawdown_mode}")
        base_score -= float(drawdown_penalty) * excess

    trade_penalty = 0.02 if n_trades == 0 else 0.0
    operational_penalty = 0.001 * risk_rejects + 0.0002 * (gap_rejects + ttl_expired)
    return base_score - trade_penalty - operational_penalty


def run_backtest_metrics(
    ticker: str,
    start: str,
    end: str,
    params: dict[str, Any],
    *,
    initial_capital: float,
    strategies_dir: Optional[Path] = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    engine_params = dict(params)
    policy_overrides = {
        key: engine_params.pop(key)
        for key in POLICY_ONLY_KEYS
        if key in engine_params
    }
    regenerate_orders = bool(policy_overrides)
    policy_config = None
    if regenerate_orders:
        policy_config = dict(policy_overrides)
        if any(key in policy_overrides for key in SCORE_WEIGHT_KEYS):
            policy_config.setdefault("trend_score_weight", 1.0)
    engine = BacktestEngine(
        ticker=ticker,
        start_date=start,
        end_date=end,
        initial_capital=initial_capital,
        strategies_dir=strategies_dir,
        regenerate_orders=regenerate_orders,
        policy_config=policy_config,
        **engine_params,
    )
    result = engine.run()
    metrics = summarize(result.equity_curve["Equity"], result.trades)
    report = dict(result.report or {})
    oracle = constraint_matched_oracle(
        result.equity_curve["MarkPrice"],
        initial_capital=initial_capital,
        max_position_weight=float(params.get("max_position_after_add_pct", 1.0)),
        commission=float(params.get("commission", 0.0)),
        slippage_bps=float(params.get("slippage_bps", 0.0)),
    )
    report["oracle"] = {
        "version": OBJECTIVE_VERSION,
        "total_return": oracle.total_return,
        "final_equity": oracle.final_equity,
        "turnover": oracle.turnover,
        "capture_ratio": oracle_capture_ratio(metrics["total_return"], oracle.total_return),
    }
    return metrics, report, result.trades


def optimize_train_window(
    ticker: str,
    train_start: str,
    train_end: str,
    *,
    n_trials: int,
    seed: int,
    initial_capital: float,
    strategies_dir: Optional[Path] = None,
    score_fn: Optional[Callable[[dict[str, Any], dict[str, Any]], float]] = None,
    learn_weights: bool = False,
    learn_decision: bool = False,
    parameter_names: Optional[tuple[str, ...]] = None,
    objective_name: str = "oracle_capture",
    max_drawdown_limit: Optional[float] = 0.25,
    drawdown_mode: str = "penalty",
    drawdown_penalty: float = 2.0,
) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Optuna is not installed. "
            "Run: pip install 'tradingagents[optimize]'  or: pip install 'optuna>=4.0.0'"
        ) from exc

    def objective(trial) -> float:
        params = suggest_policy_params(
            trial, learn_weights=learn_weights, learn_decision=learn_decision,
            parameter_names=parameter_names,
        )
        metrics, report, _trades = run_backtest_metrics(
            ticker,
            train_start,
            train_end,
            params,
            initial_capital=initial_capital,
            strategies_dir=strategies_dir,
        )
        score = (
            score_fn(metrics, report)
            if score_fn is not None
            else score_metrics(
                metrics, report, objective=objective_name,
                max_drawdown_limit=max_drawdown_limit,
                drawdown_mode=drawdown_mode,
                drawdown_penalty=drawdown_penalty,
            )
        )
        trial.set_user_attr("metrics", _replace_nonfinite(metrics))
        trial.set_user_attr("report", _replace_nonfinite(report))
        return score

    safe_ticker = ticker.replace("^", "X").replace("/", "_").replace(".", "_")
    if parameter_names is None:
        resolved = list(EXECUTION_PARAM_KEYS)
        if learn_weights:
            resolved.extend(SCORE_WEIGHT_KEYS)
        if learn_decision:
            resolved.extend(DECISION_PARAM_KEYS)
        resolved_names = tuple(dict.fromkeys(resolved))
    else:
        resolved_names = parameter_names
    data_tag = dataset_hash(ticker, strategies_dir)
    objective_tag = _short_hash({
        "version": OBJECTIVE_VERSION, "objective": objective_name,
        "max_drawdown_limit": max_drawdown_limit, "drawdown_mode": drawdown_mode,
        "drawdown_penalty": drawdown_penalty,
    })
    search_tag = _short_hash({
        "version": SEARCH_SPACE_VERSION, "parameter_names": resolved_names,
    })
    study_name = (
        f"{safe_ticker}_{train_start}_{train_end}_s{seed}_"
        f"d{data_tag}_o{objective_tag}_p{search_tag}"
    )
    OPTIMIZATION_DIR.mkdir(parents=True, exist_ok=True)
    # A SQLite busy timeout lets concurrent fold workers retry instead of
    # erroring with "database is locked" when they write trials at the same time.
    storage = optuna.storages.RDBStorage(
        url="sqlite:///" + str(OPTIMIZATION_DIR / "optuna.db"),
        engine_kwargs={"connect_args": {"timeout": 60}},
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    finished = sum(1 for t in study.trials if t.value is not None)
    remaining = max(0, n_trials - finished)
    study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    trials = [
        {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "metrics": trial.user_attrs.get("metrics", {}),
        }
        for trial in study.trials
    ]
    return dict(study.best_params), float(study.best_value), trials


def _optimize_fold_worker(kwargs: dict[str, Any]) -> tuple[int, tuple]:
    """ProcessPool entry point: run one fold's train-window optimization.

    Returns (fold_index, optimize_train_window result) so the parent can
    reassemble fold results in order. Module-level so it is picklable.
    """
    fold_index = kwargs.pop("fold_index")
    return fold_index, optimize_train_window(**kwargs)


def build_walk_forward_folds(
    trading_days: pd.Series,
    *,
    train_days: int,
    test_days: int,
    step_days: Optional[int] = None,
) -> list[Fold]:
    days = pd.Series(pd.to_datetime(trading_days).drop_duplicates()).sort_values().reset_index(drop=True)
    step = step_days or test_days
    if train_days <= 1 or test_days <= 0 or step <= 0:
        raise ValueError("train_days must be > 1 and test_days/step_days must be positive")
    if step < test_days:
        warnings.warn(
            f"step_days={step} < test_days={test_days}: consecutive fold test windows will "
            "overlap and mean_test_score will double-count shared trading days. "
            "Use step_days >= test_days for non-overlapping out-of-sample evaluation.",
            UserWarning,
            stacklevel=2,
        )

    folds: list[Fold] = []
    start_idx = 0
    fold_index = 0
    while start_idx + train_days + test_days <= len(days):
        train = days.iloc[start_idx:start_idx + train_days]
        test = days.iloc[start_idx + train_days:start_idx + train_days + test_days]
        folds.append(
            Fold(
                index=fold_index,
                train_start=train.iloc[0].strftime("%Y-%m-%d"),
                train_end=train.iloc[-1].strftime("%Y-%m-%d"),
                test_start=test.iloc[0].strftime("%Y-%m-%d"),
                test_end=test.iloc[-1].strftime("%Y-%m-%d"),
            )
        )
        fold_index += 1
        start_idx += step
    return folds


def _load_trading_days(ticker: str, start: str, end: str, initial_capital: float) -> pd.Series:
    engine = BacktestEngine(ticker, start, end, initial_capital=initial_capital)
    prices = engine.load_prices()
    if prices.empty:
        raise ValueError(f"No price data found for {ticker} between {start} and {end}")
    return prices["Date"]


def walk_forward_optimize(
    ticker: str,
    start: str,
    end: str,
    *,
    n_trials: int,
    train_days: int,
    test_days: int,
    step_days: Optional[int],
    seed: int,
    initial_capital: float,
    strategies_dir: Optional[Path] = None,
    learn_weights: bool = False,
    learn_decision: bool = False,
    fold_workers: int = 1,
    parameter_names: Optional[tuple[str, ...]] = None,
    objective_name: str = "oracle_capture",
    max_drawdown_limit: Optional[float] = 0.25,
    drawdown_mode: str = "penalty",
    drawdown_penalty: float = 2.0,
) -> dict[str, Any]:
    trading_days = _load_trading_days(ticker, start, end, initial_capital)
    folds = build_walk_forward_folds(
        trading_days,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
    )
    if not folds:
        raise ValueError(
            f"Not enough trading days for train_days={train_days}, test_days={test_days}"
        )

    # When learning weights and/or decision params, the baseline must also
    # regenerate (with the policy's default values) so it is compared on the same
    # closed-loop footing as the optimized params rather than against frozen
    # replayed orders.
    if parameter_names is None:
        selected = list(EXECUTION_PARAM_KEYS)
        if learn_weights:
            selected.extend(SCORE_WEIGHT_KEYS)
        if learn_decision:
            selected.extend(DECISION_PARAM_KEYS)
        parameter_names = tuple(dict.fromkeys(selected))
    baseline_params = baseline_params_for(parameter_names)

    # Phase 1 (expensive): optimize each fold's train window. Folds are
    # independent (separate Optuna studies), so they parallelize cleanly across
    # processes. Phase 2 (metrics/assembly below) stays sequential and cheap.
    fold_opt_kwargs = [
        {
            "fold_index": fold.index,
            "ticker": ticker,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "n_trials": n_trials,
            "seed": seed + fold.index,
            "initial_capital": initial_capital,
            "strategies_dir": strategies_dir,
            "learn_weights": learn_weights,
            "learn_decision": learn_decision,
            "parameter_names": parameter_names,
            "objective_name": objective_name,
            "max_drawdown_limit": max_drawdown_limit,
            "drawdown_mode": drawdown_mode,
            "drawdown_penalty": drawdown_penalty,
        }
        for fold in folds
    ]
    opt_by_fold: dict[int, tuple] = {}
    workers = max(1, min(int(fold_workers), len(folds)))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for fold_index, result in executor.map(_optimize_fold_worker, fold_opt_kwargs):
                opt_by_fold[fold_index] = result
    else:
        for kwargs in fold_opt_kwargs:
            fold_index, result = _optimize_fold_worker(dict(kwargs))
            opt_by_fold[fold_index] = result

    fold_results = []
    test_scores = []
    baseline_test_scores = []
    for fold in folds:
        best_params, best_train_score, trials = opt_by_fold[fold.index]
        train_metrics, train_report, _ = run_backtest_metrics(
            ticker,
            fold.train_start,
            fold.train_end,
            best_params,
            initial_capital=initial_capital,
            strategies_dir=strategies_dir,
        )
        test_metrics, test_report, _ = run_backtest_metrics(
            ticker,
            fold.test_start,
            fold.test_end,
            best_params,
            initial_capital=initial_capital,
            strategies_dir=strategies_dir,
        )
        baseline_test_metrics, baseline_test_report, _ = run_backtest_metrics(
            ticker,
            fold.test_start,
            fold.test_end,
            baseline_params,
            initial_capital=initial_capital,
            strategies_dir=strategies_dir,
        )
        test_score = score_metrics(
            test_metrics, test_report, objective=objective_name,
            max_drawdown_limit=max_drawdown_limit, drawdown_mode=drawdown_mode,
            drawdown_penalty=drawdown_penalty,
        )
        test_scores.append(test_score)
        baseline_score = score_metrics(
            baseline_test_metrics, baseline_test_report, objective=objective_name,
            max_drawdown_limit=max_drawdown_limit, drawdown_mode=drawdown_mode,
            drawdown_penalty=drawdown_penalty,
        )
        baseline_test_scores.append(baseline_score)
        fold_results.append({
            "fold": fold.__dict__,
            "best_params": best_params,
            "best_train_score": best_train_score,
            "train_metrics": train_metrics,
            "train_report": train_report,
            "test_score": test_score,
            "test_metrics": test_metrics,
            "test_report": test_report,
            "baseline_test_score": baseline_score,
            "baseline_test_metrics": baseline_test_metrics,
            "baseline_test_report": baseline_test_report,
            "test_score_improvement": test_score - baseline_score,
            "n_trials": n_trials,
            "trials": trials,
        })

    mean_test_score = sum(test_scores) / len(test_scores)
    mean_baseline_test_score = sum(baseline_test_scores) / len(baseline_test_scores)
    return {
        "schema_version": "policy_optimization_v1",
        "optimizer": "optuna_tpe",
        "ticker": ticker,
        "start": start,
        "end": end,
        "initial_capital": initial_capital,
        "n_trials": n_trials,
        "train_days": train_days,
        "test_days": test_days,
        "step_days": step_days or test_days,
        "seed": seed,
        "dataset_hash": dataset_hash(ticker, strategies_dir),
        "objective": {
            "name": objective_name,
            "version": OBJECTIVE_VERSION,
            "max_drawdown_limit": max_drawdown_limit,
            "drawdown_mode": drawdown_mode,
            "drawdown_penalty": drawdown_penalty,
        },
        "search_space": {
            "source": "back_test.optimize_policy.suggest_policy_params",
            "default_params": default_policy_params(),
            "learn_weights": learn_weights,
            "learn_decision": learn_decision,
            "scoring_weights_searched": list(SCORE_WEIGHT_KEYS) if learn_weights else [],
            "decision_params_searched": list(DECISION_PARAM_KEYS) if learn_decision else [],
            "parameter_names": list(parameter_names),
            "search_space_hash": _short_hash({
                "version": SEARCH_SPACE_VERSION,
                "parameter_names": parameter_names,
            }),
        },
        "summary": {
            "n_folds": len(fold_results),
            "mean_test_score": mean_test_score,
            "mean_baseline_test_score": mean_baseline_test_score,
            "mean_test_score_improvement": mean_test_score - mean_baseline_test_score,
            "improved_folds": sum(
                score > baseline for score, baseline in zip(test_scores, baseline_test_scores)
            ),
            "best_fold_index": max(
                range(len(fold_results)),
                key=lambda i: fold_results[i]["test_score"],
            ),
        },
        "folds": _replace_nonfinite(fold_results),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize backtest execution-policy parameters with Optuna TPE and walk-forward validation."
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--train-days", type=int, default=120)
    parser.add_argument("--test-days", type=int, default=20)
    parser.add_argument("--step-days", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--strategies-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--param-group", action="append", choices=sorted(PARAMETER_GROUPS),
        help=(
            "Parameter bundle to optimize; repeat to combine. Available: "
            + ", ".join(sorted(PARAMETER_GROUPS))
            + ". Defaults to execution when no group/legacy flag is supplied."
        ),
    )
    parser.add_argument(
        "--param", action="append", dest="individual_params",
        help="Add one named parameter to the selected bundles; repeat as needed.",
    )
    parser.add_argument(
        "--objective", choices=("oracle_capture", "total_return", "legacy"),
        default="oracle_capture",
    )
    parser.add_argument(
        "--max-drawdown-limit", type=float, default=0.25,
        help="Absolute maximum drawdown threshold, e.g. 0.25. Use a negative value to disable.",
    )
    parser.add_argument(
        "--drawdown-mode", choices=("hard", "penalty"), default="penalty",
    )
    parser.add_argument("--drawdown-penalty", type=float, default=2.0)
    parser.add_argument(
        "--learn-weights",
        action="store_true",
        help=(
            "Also search MarketState scoring weights (momentum/event/risk; trend "
            "pinned to 1.0). Enables order regeneration from each strategy's saved "
            "MarketState so the weights affect the backtest."
        ),
    )
    parser.add_argument(
        "--learn-decision",
        action="store_true",
        help=(
            "Also search decision-layer policy params (stop/take-profit ATR "
            "distances, target-weight bands, sizing multiplier) with bounds from "
            "PortfolioStatePolicyConfig. Enables order regeneration; can be "
            "combined with --learn-weights."
        ),
    )
    parser.add_argument(
        "--fold-workers", type=int, default=1,
        help=(
            "Number of folds to optimize in parallel processes (default 1). Each "
            "fold is an independent Optuna study; capped at the fold count."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    groups = list(args.param_group or [])
    if args.learn_weights:
        groups.extend(["core_state", "setup_quality", "event"])
    if args.learn_decision:
        groups.extend(["risk_exit", "position_sizing"])
    if not groups:
        groups = ["execution"]
    parameter_names = resolve_parameter_names(groups, args.individual_params)
    max_drawdown_limit = (
        None if args.max_drawdown_limit is not None and args.max_drawdown_limit < 0
        else args.max_drawdown_limit
    )
    result = walk_forward_optimize(
        args.ticker.upper(),
        args.start,
        args.end,
        n_trials=args.n_trials,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        seed=args.seed,
        initial_capital=args.initial_capital,
        strategies_dir=args.strategies_dir,
        learn_weights=args.learn_weights,
        learn_decision=args.learn_decision,
        fold_workers=args.fold_workers,
        parameter_names=parameter_names,
        objective_name=args.objective,
        max_drawdown_limit=max_drawdown_limit,
        drawdown_mode=args.drawdown_mode,
        drawdown_penalty=args.drawdown_penalty,
    )
    OPTIMIZATION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output
    if out_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OPTIMIZATION_DIR / (
            f"{args.ticker.upper()}_{args.start}_{args.end}_optuna_tpe_{stamp}.json"
        )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_replace_nonfinite(result), f, indent=2)

    print(f"Optimization written to: {out_path}")
    print(f"Folds: {result['summary']['n_folds']}")
    print(f"Mean test score: {result['summary']['mean_test_score']:.6f}")
    best = result["folds"][result["summary"]["best_fold_index"]]
    print(f"Best fold: {best['fold']['index']} test_score={best['test_score']:.6f}")
    print("Best-fold params:")
    for key, value in best["best_params"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
