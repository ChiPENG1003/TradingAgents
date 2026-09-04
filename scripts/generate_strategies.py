"""Batch-generate backtest strategies for one ticker, non-interactively.

`cli/main.py` already generates strategies correctly, but only through an
interactive prompt flow. Driving `TradingAgentsGraph.propagate` directly is the
obvious alternative and the obvious mistake: it omits the three pieces of state
the CLI feeds forward between dates —

    holdings_info            the simulated position at the decision date
    trading_history_summary  realized PnL over a trailing window
    prior_pending_orders     orders from the previous strategy still unfilled

— and `policy_from_market_state` branches on `has_position`. Generated with an
empty holdings dict, every strategy takes the flat-book branch: it emits entry
sizing and never the add / take-profit / reduce-stop shapes an open position
calls for. The failure is silent; the files look complete.

This script runs the same co-simulation loop the CLI does, so a sweep over many
tickers produces the same strategies the interactive path would.

Usage:
    python -m scripts.generate_strategies --ticker JPM \\
        --dates 2024-01-02,2024-01-08,... \\
        [--weekly 2024-01-01:2024-12-31] [--ledger path.json]

Exactly one of --dates or --weekly is required.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from cli.main import _previous_trading_date, _simulate_backtest_holdings
from tradingagents.dataflows.stockstats_utils import load_ohlcv
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

DEFAULT_ANALYSTS = ["market", "social", "fundamentals"]


def weekly_dates(ticker: str, start: str, end: str, every_days: int = 7) -> list[str]:
    """Every Nth calendar day in the window, snapped forward to a trading day."""
    prices = load_ohlcv(ticker, end)
    days = pd.to_datetime(prices["Date"]).dt.normalize()
    days = pd.DatetimeIndex(
        sorted(set(days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]))
    )
    picked: list[str] = []
    cursor = pd.Timestamp(start)
    while cursor <= pd.Timestamp(end):
        later = days[days >= cursor]
        if len(later):
            picked.append(later[0].strftime("%Y-%m-%d"))
        cursor += pd.Timedelta(days=every_days)
    return sorted(set(picked))


def build_config(quick: Optional[str], deep: Optional[str]) -> dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    config.update(
        max_debate_rounds=1,
        max_risk_discuss_rounds=1,
        data_vendors={
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "yfinance",
            "news_data": "yfinance",
        },
    )
    if quick:
        config["quick_think_llm"] = quick
    if deep:
        config["deep_think_llm"] = deep
    return config


def generate(
    ticker: str,
    dates: list[str],
    *,
    analysts: list[str],
    config: dict[str, Any],
    ledger_path: Optional[Path],
    initial_capital: float,
) -> dict[str, Any]:
    """Generate each date in order, feeding the simulated book forward."""
    graph = TradingAgentsGraph(
        selected_analysts=analysts, debug=False, config=config, trading_mode="backtest"
    )
    strategy_dir = PROJECT_ROOT / "back_test" / "strategy" / ticker
    cadence = int(config.get("review_cadence_trading_days") or 5)

    ledger: dict[str, Any] = {"ticker": ticker, "runs": []}
    if ledger_path and ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    already = {r["date"] for r in ledger["runs"] if r.get("ok")}

    price_dates = load_ohlcv(ticker, dates[-1])["Date"]
    holdings: dict[str, Any] = {}
    history: dict[str, Any] = {}
    pending: list[dict[str, Any]] = []
    started = time.time()

    for i, date in enumerate(dates, 1):
        if date in already or (strategy_dir / f"{ticker}_{date}.json").exists():
            print(f"{ticker} [{i}/{len(dates)}] {date} SKIP", flush=True)
            continue
        step = time.time()
        try:
            graph.propagate(
                ticker,
                date,
                holdings_info=holdings,
                trading_mode="backtest",
                trading_history_summary=history,
                prior_pending_orders=pending,
            )
            totals = graph.token_usage.report()["totals"]
            record = {
                "date": date,
                "ok": True,
                "seconds": round(time.time() - step, 1),
                "cost_usd": totals["cost_usd"],
                "input": totals["input_tokens"],
                "output": totals["output_tokens"],
                "cached": totals["cache_read_tokens"],
                "holdings_quantity": float(holdings.get("quantity") or 0.0),
            }
            print(
                f"{ticker} [{i}/{len(dates)}] {date} OK {record['seconds']:.0f}s "
                f"${record['cost_usd'] or 0:.5f} held={record['holdings_quantity']:.0f}",
                flush=True,
            )
        except Exception as exc:  # one bad date must not abandon the rest
            record = {"date": date, "ok": False,
                      "error": f"{type(exc).__name__}: {exc}"[:200]}
            print(f"{ticker} [{i}/{len(dates)}] {date} FAILED {record['error']}", flush=True)
            traceback.print_exc(limit=3)

        # Advance the simulated book to the day before the next decision, which
        # is the state that decision would actually be taken with.
        if i < len(dates):
            as_of = _previous_trading_date(price_dates, dates[i])
            if as_of is None:
                holdings, history, pending = {"cash": initial_capital,
                                              "as_of_date": "pre-start"}, {}, []
            else:
                holdings, history, pending = _simulate_backtest_holdings(
                    ticker, dates[0], as_of,
                    initial_capital=initial_capital,
                    cadence_trading_days=cadence,
                    prior_strategy_date=date,
                )

        ledger["runs"] = [r for r in ledger["runs"] if r["date"] != date] + [record]
        ledger["runs"].sort(key=lambda r: r["date"])
        ok = [r for r in ledger["runs"] if r.get("ok")]
        ledger["summary"] = {
            "generated": len(ok),
            "failed": len(ledger["runs"]) - len(ok),
            "total_cost_usd": round(sum(r.get("cost_usd") or 0 for r in ok), 6),
            "elapsed_min": round((time.time() - started) / 60, 1),
            "dates_with_open_position": sum(
                1 for r in ok if (r.get("holdings_quantity") or 0) > 0
            ),
        }
        if ledger_path:
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            ledger_path.write_text(
                json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--dates", help="Comma-separated YYYY-MM-DD list.")
    parser.add_argument("--weekly", help="START:END — every 7th day, snapped to trading days.")
    parser.add_argument("--every-days", type=int, default=7,
                        help="Spacing for --weekly (default 7).")
    parser.add_argument("--analysts", default=",".join(DEFAULT_ANALYSTS),
                        help=f"Comma-separated (default {','.join(DEFAULT_ANALYSTS)}). "
                             "'news' needs FRED_API_KEY.")
    parser.add_argument("--quick-model", default=None)
    parser.add_argument("--deep-model", default=None)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--ledger", type=Path, default=None)
    args = parser.parse_args()

    if bool(args.dates) == bool(args.weekly):
        parser.error("pass exactly one of --dates or --weekly")
    if args.weekly:
        start, _, end = args.weekly.partition(":")
        dates = weekly_dates(args.ticker, start, end, args.every_days)
    else:
        dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    if not dates:
        parser.error("no dates resolved")

    ledger = generate(
        args.ticker, dates,
        analysts=[a.strip() for a in args.analysts.split(",") if a.strip()],
        config=build_config(args.quick_model, args.deep_model),
        ledger_path=args.ledger,
        initial_capital=args.initial_capital,
    )
    s = ledger["summary"]
    print(
        f"{args.ticker} DONE generated {s['generated']} failed {s['failed']} "
        f"${s['total_cost_usd']:.4f} {s['elapsed_min']:.0f}min "
        f"({s['dates_with_open_position']} dates decided while holding)",
        flush=True,
    )


if __name__ == "__main__":
    main()
