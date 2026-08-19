---
name: run-tradingagents
description: Build, run, and drive the TradingAgents framework. Use when asked to run TradingAgents, run a backtest, run the offline policy optimizer (Optuna), run its tests, or smoke-test the policy/backtest layer.
---

TradingAgents is a multi-agent LLM trading framework. It has two layers:
a **live LLM pipeline** (`main.py` / `cli/main.py`, needs API keys + network)
and an **offline policy/backtest layer** (replays cached strategy JSONs, no
LLM, no secrets). The offline layer is what PRs to the policy/optimizer
actually touch, and it is fully reproducible — so the driver targets it.

Drive everything through `.claude/skills/run-tradingagents/driver.sh`, a smoke
script that runs the backtest replay, the Optuna walk-forward optimizer, and
the test suite against committed cached data (`back_test/strategy/ORCL/`).

All paths below are relative to the repo root (where `main.py` lives).

## Prerequisites

A Python 3.12 venv already exists at `.venv/` with every dependency installed.
**Use it as-is** — `.venv/bin/python`. The driver auto-detects it.

To rebuild on a clean machine (Ubuntu):

```bash
sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# Install in two passes — the full langchain stack at once can OOM (see Gotchas):
.venv/bin/python -m pip install -e ".[optimize]"      # core + optuna
.venv/bin/python -m pip install python-dotenv pytest  # not declared in pyproject
```

`python-dotenv` is imported by `main.py`/`cli/main.py` but is missing from
`pyproject.toml`; install it explicitly. `pytest` is only needed for the full
test suite (the suite also runs under stdlib `unittest`).

## Run (agent path)

```bash
.claude/skills/run-tradingagents/driver.sh            # all three steps
.claude/skills/run-tradingagents/driver.sh backtest   # replay only
.claude/skills/run-tradingagents/driver.sh optimize   # Optuna only
.claude/skills/run-tradingagents/driver.sh test       # tests only
```

The driver `cd`s to the repo root, picks `.venv/bin/python`, and prints a
PASS/FAIL line per step. It exits non-zero on real failure but tolerates the
**2 known pre-existing test failures** (see Gotchas). Logs land in
`/tmp/ta_backtest.log`, `/tmp/ta_optimize.log`, `/tmp/ta_test.log`.

Expected tail of a full run:

```
   backtest PASS
   optimize PASS
   test PASS (<=2 known pre-existing failures)
ALL REQUESTED STEPS PASSED
```

### What each step runs (raw commands, all verified)

```bash
# 1. Backtest replay — main() prompts for a label via input(), so pipe one in.
#    Reads back_test/strategy/ORCL/*.json + cached OHLCV; writes a results JSON.
echo "smoke" | .venv/bin/python -m back_test.run_backtest \
    --ticker ORCL --start 2026-04-01 --end 2026-05-28

# 2. Optuna TPE walk-forward optimizer — NO LLM calls. Persists the study to
#    a SQLite RDB (back_test/results/optimization/optuna.db) for resumability.
.venv/bin/python -m back_test.optimize_policy \
    --ticker ORCL --start 2026-04-01 --end 2026-05-28 \
    --n-trials 5 --train-days 20 --test-days 10 --step-days 10

# 3. Test suite — 82 pass, 2 known fails (see Gotchas).
.venv/bin/python -m pytest tests/ -q
```

### Direct invocation (library path)

Most policy/optimizer PRs touch individual functions; import and call them
without the CLI:

```bash
.venv/bin/python -c "
import pandas as pd
from back_test.optimize_policy import build_walk_forward_folds, score_metrics
folds = build_walk_forward_folds(pd.Series(pd.date_range('2025-01-01', periods=60, freq='D')), train_days=20, test_days=10, step_days=10)
print('folds:', len(folds), '| first test_start:', folds[0].test_start)
print('score:', round(score_metrics({'total_return':0.1,'sharpe_ratio':1.0,'max_drawdown':-0.03,'n_trades':4}, {}), 4))
"
```

## Run (live LLM pipeline — secondary, needs keys + network)

One full multi-agent analysis. Makes real paid LLM calls and hits yfinance;
non-deterministic and slow. Requires a provider key in `.env` (e.g.
`OPENAI_API_KEY`). Not exercised by the driver.

```bash
.venv/bin/python main.py --ticker NVDA --date 2024-05-10 --trading-mode backtest
```

## Gotchas

- **`python main.py --help` crashes** with `ValueError: unsupported format
  character`. A policy-arg help string in `back_test/policy_config.py:373`
  contains a literal `%。` (`...表示 2%。`); argparse runs `%`-formatting on
  help text and chokes. Normal runs with real args are fine — only `--help`
  breaks. One-char fix: escape it to `%%。`.
- **2 pre-existing test failures**, present on clean HEAD (verified by stashing
  local edits), unrelated to current work:
  `test_policy_config.py::test_legacy_policy_args_still_parse_but_are_hidden_from_help`
  and `test_portfolio_state_manager.py::test_weak_uptrend_soft_volume_caps_pullback_add`.
  The driver tolerates ≤2 failures and fails on a 3rd.
- **`pip install -e ".[optimize]"` can OOM** (exit 137) if it builds the whole
  langchain stack in one pass on a memory-constrained box. Split into two
  `pip install` passes as shown in Prerequisites. The committed `.venv` already
  has everything — prefer reusing it.
- **`run_backtest` blocks on `input()`** for an output label. The driver pipes
  `"smoke"`; if you run it by hand, pipe a label or type one.
- **Offline replay still needs OHLCV.** `load_ohlcv` caches per symbol under
  `~/.tradingagents/cache/`; ORCL + `^GSPC`/`^IXIC` are already cached. A
  brand-new ticker triggers a one-time yfinance download (needs network).
- **`pytest` is not installed by default.** Without it the driver falls back to
  `unittest discover`, where 2 test modules (`test_deepseek_reasoning`,
  `test_safe_ticker_component`) ImportError on the missing `pytest` module —
  harmless, install pytest to run them.

## Troubleshooting

- `No module named pytest` → `.venv/bin/python -m pip install pytest`.
- `No price data found for <TICKER>` → no cached OHLCV and no network; use a
  cached ticker (ORCL) or run once with network to populate the cache.
- Optimizer instantly "resumes" with 0 new trials → a prior study of the same
  `{ticker}_{train}_{end}_s{seed}` name is complete in
  `back_test/results/optimization/optuna.db`; delete that file to start fresh.
