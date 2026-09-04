"""Shared model catalog for CLI selections and validation."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

ModelOption = Tuple[str, str]
ProviderModeOptions = Dict[str, Dict[str, List[ModelOption]]]

# (display_label, tier_key, quick_model, deep_model)
ModelTier = Tuple[str, str, str, str]


MODEL_OPTIONS: ProviderModeOptions = {
    "openai": {
        "quick": [
            ("GPT-5.6 Luna - Cheapest current-gen ($0.20/$1.20 per 1M)", "gpt-5.6-luna"),
            ("GPT-5.4 Mini - Previous gen ($0.75/$4.50 per 1M)", "gpt-5.4-mini"),
            ("GPT-5.4 Nano - Previous gen, cheapest ($0.20/$1.25 per 1M)", "gpt-5.4-nano"),
            ("GPT-5.6 Terra - Mid tier ($2/$12 per 1M)", "gpt-5.6-terra"),
            ("GPT-4.1 - Smartest non-reasoning model", "gpt-4.1"),
        ],
        "deep": [
            ("GPT-5.6 Terra - Mid tier, current gen ($2/$12 per 1M)", "gpt-5.6-terra"),
            ("GPT-5.6 Sol - Frontier, current gen ($5/$30 per 1M)", "gpt-5.6-sol"),
            ("GPT-5.6 Luna - Cheapest current-gen ($0.20/$1.20 per 1M)", "gpt-5.6-luna"),
            ("GPT-5.5 - Previous frontier ($5/$30 per 1M)", "gpt-5.5"),
            ("GPT-5.4 - Previous gen ($2.50/$15 per 1M)", "gpt-5.4"),
            ("GPT-5.4 Pro - Most capable, expensive ($30/$180 per 1M)", "gpt-5.4-pro"),
        ],
    },
    "anthropic": {
        "quick": [
            ("Claude Sonnet 4.6 - Best speed and intelligence balance", "claude-sonnet-4-6"),
            ("Claude Haiku 4.5 - Fast, near-instant responses", "claude-haiku-4-5"),
            ("Claude Sonnet 4.5 - Agents and coding", "claude-sonnet-4-5"),
        ],
        "deep": [
            ("Claude Opus 4.6 - Most intelligent, agents and coding", "claude-opus-4-6"),
            ("Claude Opus 4.5 - Premium, max intelligence", "claude-opus-4-5"),
            ("Claude Sonnet 4.6 - Best speed and intelligence balance", "claude-sonnet-4-6"),
            ("Claude Sonnet 4.5 - Agents and coding", "claude-sonnet-4-5"),
        ],
    },
    "google": {
        "quick": [
            ("Gemini 3 Flash - Next-gen fast", "gemini-3-flash-preview"),
            ("Gemini 2.5 Flash - Balanced, stable", "gemini-2.5-flash"),
            ("Gemini 3.1 Flash Lite - Most cost-efficient", "gemini-3.1-flash-lite-preview"),
            ("Gemini 2.5 Flash Lite - Fast, low-cost", "gemini-2.5-flash-lite"),
        ],
        "deep": [
            ("Gemini 3.1 Pro - Reasoning-first, complex workflows", "gemini-3.1-pro-preview"),
            ("Gemini 3 Flash - Next-gen fast", "gemini-3-flash-preview"),
            ("Gemini 2.5 Pro - Stable pro model", "gemini-2.5-pro"),
            ("Gemini 2.5 Flash - Balanced, stable", "gemini-2.5-flash"),
        ],
    },
    "xai": {
        "quick": [
            ("Grok 4.1 Fast (Non-Reasoning) - Speed optimized, 2M ctx", "grok-4-1-fast-non-reasoning"),
            ("Grok 4 Fast (Non-Reasoning) - Speed optimized", "grok-4-fast-non-reasoning"),
            ("Grok 4.1 Fast (Reasoning) - High-performance, 2M ctx", "grok-4-1-fast-reasoning"),
        ],
        "deep": [
            ("Grok 4 - Flagship model", "grok-4-0709"),
            ("Grok 4.1 Fast (Reasoning) - High-performance, 2M ctx", "grok-4-1-fast-reasoning"),
            ("Grok 4 Fast (Reasoning) - High-performance", "grok-4-fast-reasoning"),
            ("Grok 4.1 Fast (Non-Reasoning) - Speed optimized, 2M ctx", "grok-4-1-fast-non-reasoning"),
        ],
    },
    # No DeepSeek model accepts a json_schema response_format yet ("This
    # response_format type is unavailable now", verified 2026-09-02), so
    # MarketState falls back to free-text JSON parsing on this provider.
    # Tool calling works on every tier, with or without thinking enabled.
    "deepseek": {
        "quick": [
            ("DeepSeek V4 Flash-Instant - Cheapest ($0.22/$0.66 off-peak)", "deepseek-v4-flash-instant"),
            ("DeepSeek V4 Flash-Thinking - Same price, reasoning on", "deepseek-v4-flash-thinking"),
            ("DeepSeek V4 Pro - Most capable ($0.66/$1.98 off-peak)", "deepseek-v4-pro"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("DeepSeek V4 Pro - Most capable ($0.66/$1.98 off-peak)", "deepseek-v4-pro"),
            ("DeepSeek V4 Flash-Thinking - Same price as instant, reasoning on", "deepseek-v4-flash-thinking"),
            ("DeepSeek V4 Flash-Instant - Cheapest ($0.22/$0.66 off-peak)", "deepseek-v4-flash-instant"),
            ("Custom model ID", "custom"),
        ],
    },
    "qwen": {
        "quick": [
            ("Qwen 3.5 Flash", "qwen3.5-flash"),
            ("Qwen Plus", "qwen-plus"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("Qwen 3.6 Plus", "qwen3.6-plus"),
            ("Qwen 3.5 Plus", "qwen3.5-plus"),
            ("Qwen 3 Max", "qwen3-max"),
            ("Custom model ID", "custom"),
        ],
    },
    "glm": {
        "quick": [
            ("GLM-4.7", "glm-4.7"),
            ("GLM-5", "glm-5"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("GLM-5.1", "glm-5.1"),
            ("GLM-5", "glm-5"),
            ("Custom model ID", "custom"),
        ],
    },
    # OpenRouter models are fetched dynamically at CLI runtime.
    # No static entries needed; any model ID is accepted by the validator.
    "ollama": {
        "quick": [
            ("Qwen3:latest (8B, local)", "qwen3:latest"),
            ("GPT-OSS:latest (20B, local)", "gpt-oss:latest"),
            ("GLM-4.7-Flash:latest (30B, local)", "glm-4.7-flash:latest"),
        ],
        "deep": [
            ("GLM-4.7-Flash:latest (30B, local)", "glm-4.7-flash:latest"),
            ("GPT-OSS:latest (20B, local)", "gpt-oss:latest"),
            ("Qwen3:latest (8B, local)", "qwen3:latest"),
        ],
    },
}


MODEL_TIERS: Dict[str, List[ModelTier]] = {
    # Prices verified 2026-09 across three sources; see MODEL_PRICING below.
    "openai": [
        ("Budget   — gpt-5.6-luna × 2 (both roles)",               "budget",   "gpt-5.6-luna",               "gpt-5.6-luna"),
        ("Standard — gpt-5.6-luna (quick) + gpt-5.6-terra (deep)", "standard", "gpt-5.6-luna",               "gpt-5.6-terra"),
        ("Premium  — gpt-5.6-terra (quick) + gpt-5.6-sol (deep)",  "premium",  "gpt-5.6-terra",              "gpt-5.6-sol"),
    ],
    "anthropic": [
        ("Budget   — haiku-4-5 (quick) + sonnet-4-6 (deep)",       "budget",   "claude-haiku-4-5",           "claude-sonnet-4-6"),
        ("Standard — sonnet-4-6 (quick) + opus-4-6 (deep)",        "standard", "claude-sonnet-4-6",          "claude-opus-4-6"),
        ("Premium  — opus-4-6 × 2 (both roles)",                   "premium",  "claude-opus-4-6",            "claude-opus-4-6"),
    ],
    "google": [
        ("Budget   — 2.5-flash-lite (quick) + 2.5-flash (deep)",   "budget",   "gemini-2.5-flash-lite",      "gemini-2.5-flash"),
        ("Standard — 2.5-flash (quick) + 2.5-pro (deep)",          "standard", "gemini-2.5-flash",           "gemini-2.5-pro"),
        ("Premium  — 3-flash (quick) + 3.1-pro (deep)",            "premium",  "gemini-3-flash-preview",     "gemini-3.1-pro-preview"),
    ],
    "xai": [
        ("Budget   — grok-4.1-fast-non-reasoning × 2",             "budget",   "grok-4-1-fast-non-reasoning", "grok-4-1-fast-non-reasoning"),
        ("Standard — grok-4.1-fast-non-reasoning + grok-4",        "standard", "grok-4-1-fast-non-reasoning", "grok-4-0709"),
        ("Premium  — grok-4.1-fast-reasoning + grok-4",            "premium",  "grok-4-1-fast-reasoning",     "grok-4-0709"),
    ],
    "ollama": [
        ("Budget   — qwen3:latest × 2",                            "budget",   "qwen3:latest",               "qwen3:latest"),
        ("Standard — qwen3:latest (quick) + glm-4.7-flash (deep)", "standard", "qwen3:latest",               "glm-4.7-flash:latest"),
        ("Premium  — gpt-oss:latest (quick) + glm-4.7-flash (deep)", "premium", "gpt-oss:latest",            "glm-4.7-flash:latest"),
    ],
    # instant and thinking bill identically (both resolve to deepseek-v4-flash),
    # so Standard buys reasoning on the deep role at no extra token cost.
    "deepseek": [
        ("Budget   — flash-instant × 2",                           "budget",   "deepseek-v4-flash-instant",  "deepseek-v4-flash-instant"),
        ("Standard — flash-instant (quick) + flash-thinking (deep)", "standard", "deepseek-v4-flash-instant", "deepseek-v4-flash-thinking"),
        ("Premium  — flash-thinking (quick) + v4-pro (deep)",      "premium",  "deepseek-v4-flash-thinking", "deepseek-v4-pro"),
    ],
    "qwen": [
        ("Budget   — qwen-plus × 2",                               "budget",   "qwen-plus",                  "qwen-plus"),
        ("Standard — qwen3.5-flash (quick) + qwen3.5-plus (deep)", "standard", "qwen3.5-flash",             "qwen3.5-plus"),
        ("Premium  — qwen-plus (quick) + qwen3-max (deep)",        "premium",  "qwen-plus",                  "qwen3-max"),
    ],
    "glm": [
        ("Budget   — glm-4.7 × 2",                                  "budget",   "glm-4.7",                   "glm-4.7"),
        ("Standard — glm-4.7 (quick) + glm-5 (deep)",               "standard", "glm-4.7",                   "glm-5"),
        ("Premium  — glm-5 (quick) + glm-5.1 (deep)",              "premium",  "glm-5",                     "glm-5.1"),
    ],
}


def get_model_options(provider: str, mode: str) -> List[ModelOption]:
    """Return shared model options for a provider and selection mode."""
    return MODEL_OPTIONS[provider.lower()][mode]


def get_model_tiers(provider: str) -> List[ModelTier]:
    """Return tier presets (display, key, quick_model, deep_model) for a provider."""
    return MODEL_TIERS.get(provider.lower(), [])


# API ids a provider resolves an alias to. The catalog lists what a user
# picks; validation also has to accept what the factory actually sends, or
# every DeepSeek run warns that its own resolved model is unknown.
RESOLVED_MODEL_IDS: Dict[str, List[str]] = {
    "deepseek": ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4-flash-vision-exp"],
}


def get_known_models() -> Dict[str, List[str]]:
    """Build known model names from the shared CLI catalog."""
    return {
        provider: sorted(
            {
                value
                for options in mode_options.values()
                for _, value in options
            }
            | set(RESOLVED_MODEL_IDS.get(provider, []))
        )
        for provider, mode_options in MODEL_OPTIONS.items()
    }


# (input, cached_input, output) US$ per 1M tokens.
#
# Verified 2026-09-02 against three independent trackers that agreed on Terra
# and Luna after the 2026-07-30 cut (Luna -80%, Terra -20%). Sol is listed at
# its standard $5/$30; one tracker reports a promotional $4/$20 running through
# at least 2026-11-21, so treat Sol as an upper bound. Cached input bills at 10%
# of the input rate. Used for estimates only — the authority on what a run
# actually cost is the metered usage in `token_usage.py`.
MODEL_PRICING: Dict[str, Tuple[float, float, float]] = {
    "gpt-5.6-sol":   (5.00, 0.500, 30.00),
    "gpt-5.6-terra": (2.00, 0.200, 12.00),
    "gpt-5.6-luna":  (0.20, 0.020, 1.20),
    "gpt-5.5":       (5.00, 0.500, 30.00),
    "gpt-5.4-pro":   (30.00, 30.00, 180.00),
    "gpt-5.4":       (2.50, 0.250, 15.00),
    "gpt-5.4-mini":  (0.75, 0.075, 4.50),
    "gpt-5.4-nano":  (0.20, 0.020, 1.25),
    # DeepSeek quotes OFF-PEAK rates here; peak doubles them. Verified
    # 2026-09-02 against the official docs and one tracker, both agreeing on
    # the rates that took effect 2026-08-16 16:00 UTC. Resolved API ids, not
    # the catalog aliases — the alias map collapses instant/thinking onto the
    # same billed model.
    "deepseek-v4-pro":   (0.66, 0.022, 1.98),
    "deepseek-v4-flash": (0.22, 0.007, 0.66),
    "deepseek-v4-flash-vision-exp": (0.22, 0.007, 0.66),
}

# Models whose rates double during the provider's peak windows.
PEAK_PRICED_MODELS = frozenset(
    m for m in MODEL_PRICING if m.startswith("deepseek-")
)
PEAK_MULTIPLIER = 2.0
# UTC hour ranges, Monday-Friday. Weekends are off-peak throughout.
# Source: api-docs.deepseek.com, 2026-09-02.
PEAK_UTC_WINDOWS: Tuple[Tuple[int, int], ...] = ((1, 4), (6, 10))


def is_peak(at: Optional[datetime] = None) -> bool:
    """True when DeepSeek bills its doubled peak rate at this instant."""
    moment = at or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    if moment.weekday() >= 5:
        return False
    return any(start <= moment.hour < end for start, end in PEAK_UTC_WINDOWS)


_SNAPSHOT_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def normalize_model_id(model: str) -> str:
    """Strip a dated snapshot suffix so pricing resolves.

    Providers echo the resolved snapshot ("gpt-5.4-mini-2026-03-17") rather
    than the alias that was requested, and prices are published per family.
    """
    return _SNAPSHOT_SUFFIX.sub("", model or "")


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    at: Optional[datetime] = None,
) -> Optional[float]:
    """Dollar cost of one call, or None when the model has no listed price.

    ``cached_input_tokens`` is the cache-read portion of ``input_tokens``, as
    reported by TokenUsageTracker — not an addition to it.
    """
    name = normalize_model_id(model)
    price = MODEL_PRICING.get(name)
    if price is None:
        return None
    fresh = max(0, input_tokens - cached_input_tokens)
    input_rate, cached_rate, output_rate = price
    if name in PEAK_PRICED_MODELS and is_peak(at):
        input_rate *= PEAK_MULTIPLIER
        cached_rate *= PEAK_MULTIPLIER
        output_rate *= PEAK_MULTIPLIER
    return (
        fresh * input_rate + cached_input_tokens * cached_rate + output_tokens * output_rate
    ) / 1_000_000
