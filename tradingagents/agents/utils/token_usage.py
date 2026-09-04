"""Per-node LLM token accounting.

Every optimization in this repo so far had to be argued from modelled prompt
sizes because nothing recorded what the providers actually billed. This module
records it: a LangChain callback accumulates `usage_metadata` from each LLM
response and attributes it to whichever graph node is currently running.

Attribution works through a ContextVar that ``_timed_agent_node`` sets around
each node call, so no agent code needs to know the tracker exists.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from langchain_core.callbacks import BaseCallbackHandler

from tradingagents.llm_clients.model_catalog import estimate_cost

_current_node: ContextVar[str] = ContextVar("current_agent_node", default="unattributed")


@contextmanager
def attribute_to(node_name: str) -> Iterator[None]:
    """Attribute LLM usage recorded inside this block to ``node_name``."""
    token = _current_node.set(node_name)
    try:
        yield
    finally:
        _current_node.reset(token)


@dataclass
class NodeUsage:
    """Token totals for one graph node across every call it made."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    # Dollars, summed per call so a node that mixes models still adds up.
    # None when any call used a model with no listed price, because a partial
    # total is worse than an explicit "unknown".
    cost_usd: Optional[float] = 0.0
    models: set[str] = field(default_factory=set)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Share of input tokens served from cache."""
        return self.cache_read_tokens / self.input_tokens if self.input_tokens else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "models": sorted(self.models),
            "cost_usd": round(self.cost_usd, 6) if self.cost_usd is not None else None,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
        }


class TokenUsageTracker(BaseCallbackHandler):
    """Accumulate LLM token usage per graph node.

    Attach as a LangChain callback on the LLM clients. Providers that report
    cached prefixes populate ``input_token_details``; providers that do not
    simply leave those counters at zero, which is itself the signal that
    prompt caching is not reaching that call.
    """

    def __init__(self) -> None:
        self.by_node: dict[str, NodeUsage] = {}
        self.responses_without_usage = 0

    # ---------------------------------------------------------- callback API
    def on_llm_end(self, response, **kwargs: Any) -> None:
        usage = self._extract_usage(response)
        if usage is None:
            self.responses_without_usage += 1
            return
        node = self.by_node.setdefault(_current_node.get(), NodeUsage())
        node.calls += 1
        node.input_tokens += int(usage.get("input_tokens") or 0)
        node.output_tokens += int(usage.get("output_tokens") or 0)
        details = usage.get("input_token_details") or {}
        cache_read = int(details.get("cache_read") or 0)
        node.cache_read_tokens += cache_read
        node.cache_creation_tokens += int(details.get("cache_creation") or 0)

        model = self._extract_model(response)
        if model:
            node.models.add(model)
        call_cost = estimate_cost(
            model or "",
            int(usage.get("input_tokens") or 0),
            int(usage.get("output_tokens") or 0),
            cache_read,
        )
        if call_cost is None or node.cost_usd is None:
            node.cost_usd = None
        else:
            node.cost_usd += call_cost

    @staticmethod
    def _extract_model(response) -> Optional[str]:
        """Model id as the provider reported it, for pricing lookup."""
        raw = (getattr(response, "llm_output", None) or {}).get("model_name")
        if raw:
            return str(raw)
        for generation_list in getattr(response, "generations", None) or []:
            for generation in generation_list:
                metadata = getattr(
                    getattr(generation, "message", None), "response_metadata", None
                ) or {}
                name = metadata.get("model_name") or metadata.get("model")
                if name:
                    return str(name)
        return None

    @staticmethod
    def _extract_usage(response) -> Optional[dict[str, Any]]:
        """Pull usage_metadata out of an LLMResult, falling back to llm_output.

        Chat models attach it to the generated message; some providers only
        report totals in llm_output["token_usage"], which carries no cache
        breakdown but still gives input/output counts.
        """
        for generation_list in getattr(response, "generations", None) or []:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if usage:
                    return dict(usage)
        raw = (getattr(response, "llm_output", None) or {}).get("token_usage")
        if isinstance(raw, dict) and raw:
            return {
                "input_tokens": raw.get("prompt_tokens") or raw.get("input_tokens"),
                "output_tokens": raw.get("completion_tokens") or raw.get("output_tokens"),
            }
        return None

    # ------------------------------------------------------------- reporting
    @property
    def totals(self) -> NodeUsage:
        combined = NodeUsage()
        for usage in self.by_node.values():
            combined.calls += usage.calls
            combined.input_tokens += usage.input_tokens
            combined.output_tokens += usage.output_tokens
            combined.cache_read_tokens += usage.cache_read_tokens
            combined.cache_creation_tokens += usage.cache_creation_tokens
            combined.models |= usage.models
            if usage.cost_usd is None or combined.cost_usd is None:
                combined.cost_usd = None
            else:
                combined.cost_usd += usage.cost_usd
        return combined

    def report(self) -> dict[str, Any]:
        return {
            "by_node": {
                name: usage.as_dict()
                for name, usage in sorted(
                    self.by_node.items(), key=lambda kv: -kv[1].total_tokens
                )
            },
            "totals": self.totals.as_dict(),
            "responses_without_usage": self.responses_without_usage,
        }

    def render(self) -> str:
        """A fixed-width table for the run log."""
        rows = sorted(self.by_node.items(), key=lambda kv: -kv[1].total_tokens)
        if not rows:
            return "[token_usage] no LLM usage recorded"
        total = self.totals
        width = max(len(name) for name, _ in rows)
        lines = [
            f"{'node':{width}}  {'calls':>5} {'input':>9} {'output':>8} "
            f"{'cached':>9} {'hit%':>6} {'share%':>7} {'cost$':>9}",
            "-" * (width + 60),
        ]
        for name, usage in rows:
            share = usage.total_tokens / total.total_tokens if total.total_tokens else 0.0
            lines.append(
                f"{name:{width}}  {usage.calls:5d} {usage.input_tokens:9,d} "
                f"{usage.output_tokens:8,d} {usage.cache_read_tokens:9,d} "
                f"{usage.cache_hit_rate * 100:5.1f}% {share * 100:6.1f}% "
                + (f"{usage.cost_usd:9.5f}" if usage.cost_usd is not None else f"{'n/a':>9}")
            )
        lines.append("-" * (width + 60))
        lines.append(
            f"{'TOTAL':{width}}  {total.calls:5d} {total.input_tokens:9,d} "
            f"{total.output_tokens:8,d} {total.cache_read_tokens:9,d} "
            f"{total.cache_hit_rate * 100:5.1f}% {'':>7} "
            + (f"{total.cost_usd:9.5f}" if total.cost_usd is not None else f"{'n/a':>9}")
        )
        if self.responses_without_usage:
            lines.append(
                f"({self.responses_without_usage} responses reported no usage metadata)"
            )
        return "\n".join(lines)

    def write_report(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.report(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    def reset(self) -> None:
        self.by_node.clear()
        self.responses_without_usage = 0
