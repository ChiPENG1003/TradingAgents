"""PortfolioState policy configuration.

Holds the dataclass + CLI argparse glue for the deterministic policy used
by the backtest-only portfolio_state_manager. Kept separate from the agent
implementation so the agent module stays focused on decisions, not on how
to be CLI-configured.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


_DEFAULT_VOLUME_MULTIPLIER = {
    "expanding": 1.0,
    "normal": 1.0,
    "soft": 0.7,
    "shrinking": 0.5,
    "unavailable": 0.5,
}

# Phase modifiers — applied AFTER regime ceil/floor and volume multiplier.
# Encodes the four operating principles:
#   - 核心持仓: high floor in healthy_bull_trend / bull_pullback
#   - 不要频繁止盈: low tp_size in trend phases (15-20 vs default 25-40)
#   - 允许 trend following: market entry permitted in trend phases
#   - pullback 买入: bull_pullback gets aggressive add + at-current entry
# Keys not in dict default to no modification.
_DEFAULT_PHASE_MODIFIER: dict[str, dict] = {
    # ----- Bull -----
    "early_bull_reversal":    {"cap": 0.40, "tp_size": 30.0, "allow_add": False},
    "healthy_bull_trend":     {"floor": 0.50, "cap": 0.85, "tp_size": 15.0, "allow_add": True,  "trend_market_entry": True},
    "accelerating_bull":      {"floor": 0.40, "cap": 0.80, "tp_size": 20.0, "allow_add": False, "trend_market_entry": True},
    # overextended_bull: keep core, allow trimmed add. NOT block_new_position —
    # in long bull markets "approaching resistance" is the norm, not a reason to exit.
    "overextended_bull":      {"cap": 0.55, "tp_size": 30.0, "allow_add": True},
    "bull_pullback":          {"floor": 0.50, "cap": 0.85, "tp_size": 15.0, "allow_add": True,  "pullback_buy": True},
    "late_bull_distribution": {"cap": 0.25, "tp_size": 50.0, "allow_add": False},
    # ----- Bear (force SELL existing, block new) -----
    "early_bear_reversal":    {"force_sell_if_position": True, "block_new_position": True},
    "healthy_bear_trend":     {"force_sell_if_position": True, "block_new_position": True},
    "accelerating_bear":      {"force_sell_if_position": True, "block_new_position": True},
    "oversold_bear":          {"cap": 0.0, "block_new_position": True},
    "bear_rally":             {"cap": 0.0, "block_new_position": True},      # trap for trend-followers
    "late_bear_exhaustion":   {"cap": 0.20, "tp_size": 30.0},
    # ----- Neutral -----
    "range_compression":      {"cap": 0.25, "tp_size": 60.0},                 # full TP near range top
    "high_volatility_range":  {"cap": 0.15, "block_new_position": True},
    "macro_event_regime":     {"cap": 0.10, "block_new_position": True},
    "unclear":                {"cap": 0.20},
}


@dataclass(frozen=True)
class PortfolioStatePolicyConfig:
    """Tunable deterministic policy parameters for backtest PortfolioState mode."""

    trend_score_weight: float = 0.25
    momentum_score_weight: float = 0.125
    event_score_weight: float = 0.075
    risk_score_weight: float = 0.40

    strong_uptrend_floor: float = 0.60
    strong_uptrend_cap: float = 0.90
    weak_uptrend_floor: float = 0.35
    weak_uptrend_cap: float = 0.70
    range_cap: float = 0.35
    event_driven_cap: float = 0.50
    unavailable_volume_cap: float = 0.35
    max_target_weight: float = 0.90
    min_trade_weight: float = 0.05
    order_size_multiplier: float = 1.0

    volume_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "expanding": 1.0,
            "normal": 1.0,
            "soft": 0.7,
            "shrinking": 0.5,
            "unavailable": 0.5,
        }
    )
    phase_modifiers: dict[str, dict[str, Any]] = field(default_factory=dict)

    recent_phase_lookback: int = 3
    hysteresis_confirmation_count: int = 2
    overextended_sma20_atr_threshold: float = 2.0

    pullback_entry_add_max_pct: float = 40.0
    pullback_entry_add_weight_multiplier: float = 60.0
    default_add_max_pct: float = 30.0
    default_add_weight_multiplier: float = 50.0

    bearish_divergence_reduce_pct: float = 30.0
    bearish_divergence_stop_atr: float = 1.5
    bearish_divergence_fallback_stop_atr: float = 2.0
    stop_loss_atr_multiple: float = 2.5
    trend_take_profit_atr_multiple: float = 3.0
    trend_take_profit_recent_high_multiplier: float = 1.05
    default_take_profit_atr_multiple: float = 2.0
    strong_uptrend_take_profit_size_pct: float = 25.0
    default_take_profit_size_pct: float = 40.0
    market_context_enabled: bool = True
    market_context_ticker: str = "^GSPC"
    market_context_bearish_weight_multiplier: float = 0.50
    market_context_bullish_weight_multiplier: float = 1.10

    def merged_phase_modifiers(self) -> dict[str, dict[str, Any]]:
        merged = {phase: values.copy() for phase, values in _DEFAULT_PHASE_MODIFIER.items()}
        for phase, values in self.phase_modifiers.items():
            base = merged.setdefault(phase, {})
            base.update(values)
        return merged


def default_portfolio_state_policy_config() -> dict[str, Any]:
    """Return a serializable default config for TradingAgentsGraph config dicts."""
    return asdict(PortfolioStatePolicyConfig())


def add_portfolio_state_policy_args(parser) -> None:
    """Attach PortfolioStatePolicyConfig argparse options to a parser."""
    group = parser.add_argument_group("组合状态策略参数")
    group.add_argument("--ps-trend-weight", type=float, default=0.25,
        dest="ps_trend_score_weight", help="趋势评分对目标仓位的影响权重；越大越追随趋势。默认 0.25。",)
    group.add_argument("--ps-momentum-weight", type=float, default=0.125,
        dest="ps_momentum_score_weight", help="动量评分对目标仓位的影响权重；越大越重视近期涨跌和量价动能。默认 0.125。",)
    group.add_argument("--ps-event-weight", type=float, default=0.075,
        dest="ps_event_score_weight", help="新闻、财报、宏观事件评分对目标仓位的影响权重。默认 0.075。",)
    group.add_argument("--ps-risk-weight", type=float, default=0.40,
        dest="ps_risk_score_weight", help="风险评分对目标仓位的扣减权重；越大越保守。默认 0.40。",)
    group.add_argument("--ps-strong-floor", type=float, default=0.60,
        dest="ps_strong_uptrend_floor", help="强上升趋势下的最低目标仓位，0.60 表示至少 60%%。默认 0.60。",)
    group.add_argument("--ps-strong-cap", type=float, default=0.90,
        dest="ps_strong_uptrend_cap", help="强上升趋势下的最高目标仓位，0.90 表示最多 90%%。默认 0.9。",)
    group.add_argument("--ps-weak-floor", type=float, default=0.20,
        dest="ps_weak_uptrend_floor", help="弱上升趋势下的最低目标仓位。默认 0.20。",)
    group.add_argument("--ps-weak-cap", type=float, default=0.45,
        dest="ps_weak_uptrend_cap", help="弱上升趋势下的最高目标仓位。默认 0.45。",)
    group.add_argument("--ps-range-cap", type=float, default=0.35,
        dest="ps_range_cap", help="震荡或方向不明行情下的最高目标仓位。默认 0.35。",)
    group.add_argument("--ps-event-cap", type=float, default=0.50,
        dest="ps_event_driven_cap", help="事件驱动行情下的最高目标仓位，用来限制财报、宏观事件等不确定性。默认 0.5。",)
    group.add_argument("--ps-max-weight", type=float, default=0.90,
        dest="ps_max_target_weight", help="任何情况下允许的全局最高目标仓位。默认 0.9。",)
    group.add_argument("--ps-min-trade-weight", type=float, default=0.03,
        dest="ps_min_trade_weight", help="低于该目标仓位时不新开仓，直接 HOLD；0.05 表示 5%%。默认 0.05。",)
    
    # ================ 决策放大倍数 ================
    group.add_argument("--ps-order-size-mult", type=float, default=1.0,
        dest="ps_order_size_multiplier", help="硬编码订单数量倍率；2.0 表示策略原本买/卖 N 股时改为 2N 股，最高封顶 100%%。默认 1.0。",)
    # ===========================================

    group.add_argument("--ps-recent-phase-lookback", type=int, default=3,
        dest="ps_recent_phase_lookback", help="读取最近多少期市场阶段，用于判断趋势切换是否可靠。默认 3。",)
    group.add_argument("--ps-hysteresis-confirm", type=int, default=1,
        dest="ps_hysteresis_confirmation_count", help="趋势或熊市切换需要最近多少期确认；越大越防抖、越慢反应。默认 2。",)
    group.add_argument("--ps-overextended-atr", type=float, default=2.0,
        dest="ps_overextended_sma20_atr_threshold", help="价格高于 EMA5 多少个 ATR 才承认短线过度延伸。默认 2。",)
    group.add_argument("--ps-stop-atr", type=float, default=2.5,
        dest="ps_stop_loss_atr_multiple", help="止损距离的 ATR 倍数；越大止损越宽。默认 2.5。",)
    group.add_argument("--ps-trend-tp-atr", type=float, default=3.0,
        dest="ps_trend_take_profit_atr_multiple", help="趋势行情止盈目标距离，按 ATR 倍数计算。默认 3。",)
    group.add_argument("--ps-default-tp-atr", type=float, default=2.0,
        dest="ps_default_take_profit_atr_multiple", help="非趋势行情止盈目标距离，按 ATR 倍数计算。默认 2。",)
    group.add_argument("--ps-trend-tp-high-mult", type=float, default=1.05,
        dest="ps_trend_take_profit_recent_high_multiplier", help="趋势行情止盈价相对 20 日高点的最低倍数；1.05 表示高点上方 5%%。默认 1.05。",)
    group.add_argument("--ps-default-add-max", type=float, default=20.0,
        dest="ps_default_add_max_pct", help="普通情况下已有持仓时的单次最大加仓比例，单位是百分比。默认 20。",)
    group.add_argument("--ps-pullback-add-max", type=float, default=25.0,
        dest="ps_pullback_entry_add_max_pct", help="牛市回调阶段已有持仓时的单次最大加仓比例，单位是百分比。默认 25。",)
    group.add_argument("--ps-bearish-div-reduce", type=float, default=30.0,
        dest="ps_bearish_divergence_reduce_pct", help="出现看跌量价背离时的防御性减仓比例，单位是百分比。默认 30。",)
    group.add_argument("--ps-soft-volume-mult", type=float, default=0.7,
        dest="ps_volume_soft", help="成交量偏弱时的仓位折扣系数；0.7 表示目标仓位乘以 70%%。默认 0.7。")
    group.add_argument("--ps-shrinking-volume-mult", type=float, default=0.5,
        dest="ps_volume_shrinking", help="成交量萎缩时的仓位折扣系数。默认 0.5。",)
    group.add_argument("--ps-unavailable-volume-mult", type=float, default=0.5,
        dest="ps_volume_unavailable", help="成交量数据不可用时的仓位折扣系数。默认 0.5。",)
    group.add_argument("--ps-index-context", default="^GSPC",
        dest="ps_market_context_ticker", help="用于共同判断市场 bullish/bearish 的指数 ticker。默认 ^GSPC。",)
    group.add_argument("--ps-disable-index-context", action="store_true",
        dest="ps_disable_market_context", help="关闭指数 bullish/bearish 上下文，只使用个股状态。",)
    group.add_argument("--ps-index-bear-mult", type=float, default=0.50,
        dest="ps_market_context_bearish_weight_multiplier", help="指数 bearish 时目标仓位折扣。默认 0.50。",)
    group.add_argument("--ps-index-bull-mult", type=float, default=1.10,
        dest="ps_market_context_bullish_weight_multiplier", help="指数 bullish 且个股 bullish 时目标仓位放大。默认 1.10。",)


def portfolio_state_policy_config_from_args(args) -> dict[str, Any]:
    """Build a sparse portfolio_state_policy config dict from argparse args."""
    mapping = {
        "ps_trend_score_weight": "trend_score_weight",
        "ps_momentum_score_weight": "momentum_score_weight",
        "ps_event_score_weight": "event_score_weight",
        "ps_risk_score_weight": "risk_score_weight",
        "ps_strong_uptrend_floor": "strong_uptrend_floor",
        "ps_strong_uptrend_cap": "strong_uptrend_cap",
        "ps_weak_uptrend_floor": "weak_uptrend_floor",
        "ps_weak_uptrend_cap": "weak_uptrend_cap",
        "ps_range_cap": "range_cap",
        "ps_event_driven_cap": "event_driven_cap",
        "ps_max_target_weight": "max_target_weight",
        "ps_min_trade_weight": "min_trade_weight",
        "ps_order_size_multiplier": "order_size_multiplier",
        "ps_recent_phase_lookback": "recent_phase_lookback",
        "ps_hysteresis_confirmation_count": "hysteresis_confirmation_count",
        "ps_overextended_sma20_atr_threshold": "overextended_sma20_atr_threshold",
        "ps_stop_loss_atr_multiple": "stop_loss_atr_multiple",
        "ps_trend_take_profit_atr_multiple": "trend_take_profit_atr_multiple",
        "ps_default_take_profit_atr_multiple": "default_take_profit_atr_multiple",
        "ps_trend_take_profit_recent_high_multiplier": (
            "trend_take_profit_recent_high_multiplier"
        ),
        "ps_default_add_max_pct": "default_add_max_pct",
        "ps_pullback_entry_add_max_pct": "pullback_entry_add_max_pct",
        "ps_bearish_divergence_reduce_pct": "bearish_divergence_reduce_pct",
        "ps_market_context_ticker": "market_context_ticker",
        "ps_market_context_bearish_weight_multiplier": (
            "market_context_bearish_weight_multiplier"
        ),
        "ps_market_context_bullish_weight_multiplier": (
            "market_context_bullish_weight_multiplier"
        ),
    }
    config: dict[str, Any] = {}
    for arg_name, config_name in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            config[config_name] = value

    volume_overrides = {
        "soft": getattr(args, "ps_volume_soft", None),
        "shrinking": getattr(args, "ps_volume_shrinking", None),
        "unavailable": getattr(args, "ps_volume_unavailable", None),
    }
    volume_multipliers = {
        key: value for key, value in volume_overrides.items() if value is not None
    }
    if volume_multipliers:
        config["volume_multipliers"] = volume_multipliers

    if getattr(args, "ps_disable_market_context", False):
        config["market_context_enabled"] = False

    return config


def coerce_portfolio_state_policy_config(
    value: Optional[dict[str, Any] | PortfolioStatePolicyConfig],
) -> PortfolioStatePolicyConfig:
    if isinstance(value, PortfolioStatePolicyConfig):
        return value
    if not value:
        return PortfolioStatePolicyConfig()

    defaults = asdict(PortfolioStatePolicyConfig())
    merged = defaults.copy()
    merged.update(value)
    volume_multipliers = defaults["volume_multipliers"].copy()
    volume_multipliers.update(value.get("volume_multipliers") or {})
    merged["volume_multipliers"] = volume_multipliers
    merged["phase_modifiers"] = value.get("phase_modifiers") or {}
    return PortfolioStatePolicyConfig(**merged)
