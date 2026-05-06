from __future__ import annotations

import argparse
import json
import math
import numbers
import os
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    if exc.name != "matplotlib":
        raise
    plt = None

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.stockstats_utils import load_ohlcv, yf_retry

from .engine import PROJECT_ROOT
from .metrics import summarize


RESULTS_DIR = PROJECT_ROOT / "back_test" / "results"
BENCHMARKS = ["^IXIC", "^GSPC"]


def _replace_nonfinite_numbers(value):
    if isinstance(value, dict):
        return {k: _replace_nonfinite_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_nonfinite_numbers(v) for v in value]
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric):
            return 0.0
    return value


def _benchmark_slug(benchmark: str) -> str:
    return benchmark.replace("^", "").lower()


def _load_benchmark_close(benchmark: str, start: str, end: str) -> pd.Series:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    bench_raw = load_ohlcv(benchmark, end)
    bench_raw = bench_raw[
        (bench_raw["Date"] >= start_ts)
        & (bench_raw["Date"] <= end_ts)
    ].set_index("Date").sort_index()

    if not bench_raw.empty:
        return bench_raw["Close"].rename(benchmark)

    fresh = yf_retry(lambda: yf.download(
        benchmark,
        start=start_ts.strftime("%Y-%m-%d"),
        end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        multi_level_index=False,
        progress=False,
        auto_adjust=True,
    ))

    if fresh.empty:
        return pd.Series(dtype=float, name=benchmark)

    fresh = fresh.reset_index()
    fresh["Date"] = pd.to_datetime(fresh["Date"], errors="coerce")
    fresh["Close"] = pd.to_numeric(fresh["Close"], errors="coerce")
    fresh = fresh.dropna(subset=["Date", "Close"]).set_index("Date").sort_index()

    return fresh["Close"].rename(benchmark)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare strategy equity curve to ^IXIC and ^GSPC in one plot."
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()

    results_path = RESULTS_DIR / f"{args.ticker}_{args.start}_{args.end}.json"

    if not results_path.exists():
        print(
            f"ERROR: Results file not found: {results_path}\n"
            f"Run `python -m back_test.run_backtest --ticker {args.ticker} "
            f"--start {args.start} --end {args.end}` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    strat_df = pd.DataFrame(data["equity_curve"])
    strat_df["date"] = pd.to_datetime(strat_df["date"])
    strat_df = strat_df.set_index("date").sort_index()
    strat_equity = strat_df["equity"].rename("strategy")

    buy_hold_label = f"{args.ticker}_buy_hold"
    ticker_buy_hold = _load_benchmark_close(
        args.ticker,
        args.start,
        args.end,
    ).rename(buy_hold_label)

    benchmark_series = [
        _load_benchmark_close(benchmark, args.start, args.end)
        for benchmark in BENCHMARKS
    ]
    all_benchmarks = [buy_hold_label] + BENCHMARKS

    aligned = pd.concat(
        [strat_equity, ticker_buy_hold] + benchmark_series,
        axis=1,
        join="inner",
    ).dropna()

    if aligned.empty:
        print(
            "ERROR: No overlapping trading dates between strategy and benchmarks.",
            file=sys.stderr,
        )
        sys.exit(1)

    strat_aligned = aligned["strategy"]

    strat_metrics = summarize(strat_aligned, data.get("trades"))

    benchmark_metrics = {}
    alpha = {}

    for benchmark in all_benchmarks:
        bench_aligned = aligned[benchmark]
        bench_metrics = summarize(bench_aligned)

        benchmark_metrics[benchmark] = bench_metrics
        alpha[benchmark] = {
            "alpha_total": (
                strat_metrics["total_return"]
                - bench_metrics["total_return"]
            ),
            "alpha_annualized": (
                strat_metrics["annualized_return"]
                - bench_metrics["annualized_return"]
            ),
        }

    comparison = {
        "ticker": args.ticker,
        "benchmarks": all_benchmarks,
        "start_date": args.start,
        "end_date": args.end,
        "strategy": strat_metrics,
        "benchmark_metrics": benchmark_metrics,
        "alpha": alpha,
    }

    comparison = _replace_nonfinite_numbers(comparison)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_slug = "_".join(_benchmark_slug(b) for b in all_benchmarks)

    metrics_path = (
        RESULTS_DIR
        / f"{args.ticker}_vs_{benchmark_slug}_{args.start}_{args.end}_metrics.json"
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, allow_nan=False)

    plot_path = (
        RESULTS_DIR
        / f"{args.ticker}_vs_{benchmark_slug}_{args.start}_{args.end}.png"
    )

    plot_written = False

    if plt is not None:
        fig, ax = plt.subplots(figsize=(12, 6))

        for column in aligned.columns:
            norm = aligned[column] / aligned[column].iloc[0] * 100.0

            if column == "strategy":
                label = f"{args.ticker} strategy"
                linewidth = 1.5
            elif column == buy_hold_label:
                label = f"{args.ticker} buy & hold"
                linewidth = 0.6
            else:
                label = column
                linewidth = 0.6

            ax.plot(
                norm.index,
                norm.values,
                label=label,
                linewidth=linewidth,
                alpha=0.9,
            )

        ax.set_title(
            f"{args.ticker} strategy vs buy & hold, ^IXIC, and ^GSPC "
            f"({args.start} → {args.end})"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized value (start = 100)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(plot_path, dpi=130)
        plt.close(fig)
        plot_written = True

    print(f"\n=== Strategy vs Benchmarks ({args.start} → {args.end}) ===")
    print(
        f"  Strategy total:      {comparison['strategy']['total_return']:>8.2%}   "
        f"annualized {comparison['strategy']['annualized_return']:>7.2%}   "
        f"sharpe {comparison['strategy']['sharpe_ratio']:.3f}"
    )

    for benchmark in all_benchmarks:
        bench_metrics = comparison["benchmark_metrics"][benchmark]
        alpha_metrics = comparison["alpha"][benchmark]
        label = f"{args.ticker} buy & hold" if benchmark == buy_hold_label else benchmark

        print(
            f"  {label:<19}total:{bench_metrics['total_return']:>9.2%}   "
            f"annualized {bench_metrics['annualized_return']:>7.2%}   "
            f"sharpe {bench_metrics['sharpe_ratio']:.3f}"
        )
        print(f"  Alpha vs {label:<19} total:      {alpha_metrics['alpha_total']:>8.2%}")
        print(f"  Alpha vs {label:<19} annualized: {alpha_metrics['alpha_annualized']:>8.2%}")

    if plot_written:
        print(f"\nPlot:    {plot_path}")
    else:
        print("\nPlot:    skipped (matplotlib is not installed in this Python environment)")

    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
