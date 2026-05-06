"""Backtest engine for TradingAgents weekly strategies.

Reads `back_test/strategy/{TICKER}/{TICKER}_{YYYY-MM-DD}.json` strategy files
emitted by the agent graph in backtest mode, replays them on historical daily
OHLCV using limit-order fill semantics, and produces performance metrics.
"""
