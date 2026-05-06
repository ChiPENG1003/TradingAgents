"""Validate structured backtest strategies emitted by Portfolio Manager."""

from __future__ import annotations

from typing import Any, Dict, Optional


_VALID_ACTIONS = {"BUY", "HOLD", "SELL"}
_SCHEMA_VERSION = "v3"


class StructuredStrategyError(ValueError):
    """Raised when a structured strategy cannot be validated."""


def extract_structured_strategy(
    structured_strategy: Any,
    ticker: Optional[str] = None,
    trade_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a normalized structured strategy dict.

    The Portfolio Manager is expected to populate `structured_strategy` via
    `with_structured_output`; this function intentionally does not parse fenced
    markdown or call a fallback LLM.
    """
    if structured_strategy is None:
        raise StructuredStrategyError("Portfolio Manager did not return structured_strategy.")

    if hasattr(structured_strategy, "model_dump"):
        data = structured_strategy.model_dump()
    elif isinstance(structured_strategy, dict):
        data = dict(structured_strategy)
    else:
        raise StructuredStrategyError(
            f"structured_strategy must be a dict or Pydantic model, got {type(structured_strategy).__name__}."
        )

    if ticker:
        data["ticker"] = ticker
    if trade_date:
        data["as_of_date"] = str(trade_date)

    _validate(data)
    return data


def _validate(data: Dict[str, Any]) -> None:
    version = data.get("schema_version")
    if version not in (None, _SCHEMA_VERSION):
        raise StructuredStrategyError(
            f"Unsupported schema_version {version!r}; expected {_SCHEMA_VERSION!r}."
        )
    data["schema_version"] = _SCHEMA_VERSION
    data.setdefault("entry", {"price": None, "size_pct": 0})
    data.setdefault("add_position", {"price": None, "size_pct": 0})
    legacy_take_profit = data.pop("take_profit", None) if "take_profit" in data and "reduce_position" in data else None
    if "take_profit" not in data:
        data["take_profit"] = data.pop("reduce_position", None) or legacy_take_profit or {"price": None, "size_pct": 0}
    else:
        data.pop("reduce_position", None)
    data.setdefault("reduce_stop", {"price": None, "size_pct": 0})
    data.setdefault("stop_loss", {"price": None})

    required = (
        "schema_version",
        "ticker",
        "as_of_date",
        "action",
        "entry",
        "add_position",
        "take_profit",
        "reduce_stop",
        "stop_loss",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise StructuredStrategyError(f"Missing required fields: {missing}")

    if data["action"] not in _VALID_ACTIONS:
        raise StructuredStrategyError(
            f"Invalid action {data['action']!r}; expected one of {_VALID_ACTIONS}"
        )
    if data["action"] == "SELL":
        data["entry"] = {"price": None, "size_pct": 0}
        data["add_position"] = {"price": None, "size_pct": 0}
        data["take_profit"] = {"price": None, "size_pct": 0}
        data["reduce_stop"] = {"price": None, "size_pct": 0}

    for label in ("entry", "add_position", "take_profit", "reduce_stop"):
        block = data.get(label) or {}
        size_pct = block.get("size_pct", 0) or 0
        try:
            size_pct = float(size_pct)
        except (TypeError, ValueError) as exc:
            raise StructuredStrategyError(f"{label}.size_pct must be numeric, got {block.get('size_pct')!r}") from exc
        if not (0.0 <= size_pct <= 100.0):
            raise StructuredStrategyError(f"{label}.size_pct out of range: {size_pct}")
        block["size_pct"] = size_pct
        data[label] = block

    for label in ("entry", "add_position", "take_profit", "reduce_stop", "stop_loss"):
        block = data.get(label) or {}
        price = block.get("price")
        if price is not None:
            try:
                block["price"] = float(price)
            except (TypeError, ValueError) as exc:
                raise StructuredStrategyError(
                    f"{label}.price must be numeric or null, got {price!r}"
                ) from exc
        data[label] = block
