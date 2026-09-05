"""Read-only live account snapshots. No quotes, research, or orders are sent here."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Number = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
Price = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
Symbol = Annotated[str, StringConstraints(strip_whitespace=True, to_upper=True, min_length=1, max_length=40)]
DEFAULT_PORTFOLIO_DIR = Path(__file__).resolve().parents[1] / "portfolio"


class Position(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: Symbol
    quantity: Number
    avg_buy_price: Price
    market_price: Price | None = None


class PortfolioSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total_portfolio: Price = Field(description="Current NAV: cash plus market value of ALL holdings.")
    holdings: list[Position]
    cash: Number | None = None
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")] = "USD"
    max_position_pct: Annotated[float, Field(strict=True, gt=0, le=100, allow_inf_nan=False)] = 100.0

    @model_validator(mode="after")
    def consistent_account(self):
        symbols = [p.ticker for p in self.holdings]
        if len(set(symbols)) != len(symbols):
            raise ValueError("Duplicate tickers: combine lots into one quantity and weighted average cost.")
        known_value = sum(p.quantity * p.market_price for p in self.holdings if p.market_price is not None)
        if not math.isfinite(known_value) or known_value + (self.cash or 0) > self.total_portfolio + 0.01:
            raise ValueError("Cash plus known market values exceeds total_portfolio.")
        if self.cash is not None and all(p.market_price is not None or p.quantity == 0 for p in self.holdings):
            if abs(known_value + self.cash - self.total_portfolio) > 0.01:
                raise ValueError("With all market prices supplied, total_portfolio must equal cash + holdings market value.")
        if any(not math.isfinite(p.quantity * p.avg_buy_price) for p in self.holdings):
            raise ValueError("Position cost value is too large.")
        return self

    def for_ticker(self, ticker: str) -> dict:
        ticker = ticker.strip().upper()
        target = next((p for p in self.holdings if p.ticker == ticker), None)
        positions = [p.model_dump() for p in self.holdings if p.quantity > 0]
        cash = self.cash
        if cash is None and all(p.market_price is not None or p.quantity == 0 for p in self.holdings):
            cash = max(0.0, self.total_portfolio - sum(p.quantity * (p.market_price or 0) for p in self.holdings))
        return {
            "nav": self.total_portfolio,
            "equity": self.total_portfolio,
            "quantity": target.quantity if target else 0.0,
            "avg_buy_price": target.avg_buy_price if target else None,
            "mark_price": target.market_price if target else None,
            "cash": cash,
            "currency": self.currency,
            "max_position_pct": self.max_position_pct,
            "portfolio_positions": positions,
            "portfolio_target": ticker,
        }


def load_live_portfolio(ticker: str, directory: str | Path | None = None, *, path: str | Path | None = None) -> dict | None:
    """Reload each call. Empty directory means manual entry; ambiguity/errors fail loudly."""
    if path is None:
        folder = Path(directory or os.getenv("TRADINGAGENTS_PORTFOLIO_DIR") or DEFAULT_PORTFOLIO_DIR).expanduser()
        if not folder.exists():
            return None
        if not folder.is_dir():
            raise ValueError(f"Portfolio directory is not a directory: {folder}")
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".json" and p.is_file())
        if not files:
            return None
        if len(files) != 1:
            raise ValueError(f"Keep exactly one portfolio JSON in {folder}; found {len(files)}.")
        path = files[0]
    path = Path(path).expanduser().resolve()
    try:
        snapshot = PortfolioSnapshot.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid portfolio snapshot {path}: {exc}") from exc
    result = snapshot.for_ticker(ticker)
    result["portfolio_source"] = str(path)
    return result


def allocation_context(holdings: dict, current_price: float | None = None) -> dict:
    """Compute account exposure without treating purchase cost as market value."""
    nav = float(holdings.get("nav") or holdings.get("equity") or 0)
    qty = float(holdings.get("quantity") or 0)
    mark = current_price or holdings.get("mark_price")
    target_value = qty * float(mark) if mark else (0.0 if qty == 0 else None)
    others = [p for p in holdings.get("portfolio_positions", []) if p["ticker"] != holdings.get("portfolio_target")]
    known_other = sum(p["quantity"] * p["market_price"] for p in others if p.get("market_price") is not None)
    unpriced = [p for p in others if p.get("market_price") is None]
    cash = holdings.get("cash")
    limit_pct = float(holdings.get("max_position_pct", 100))
    headroom = max(0.0, min(nav * limit_pct / 100, nav - known_other) - target_value) if nav > 0 and target_value is not None else 0.0
    # Unknown cash is never NAV minus purchase costs. The minimal JSON remains
    # useful for existing-position advice; funded buys require a cash snapshot.
    buy_budget = min(headroom, max(0.0, float(cash))) if cash is not None else 0.0
    return {
        "nav": nav, "target_market_value": target_value,
        "target_weight_pct": target_value / nav * 100 if nav > 0 and target_value is not None else None,
        "other_position_count": len(others), "other_known_market_value": known_other,
        "other_known_weight_pct": known_other / nav * 100 if nav > 0 else None,
        "other_unpriced_count": len(unpriced),
        "other_unpriced_cost_basis": sum(p["quantity"] * p["avg_buy_price"] for p in unpriced),
        "cash": cash, "max_position_pct": limit_pct, "buy_budget": buy_budget,
    }
