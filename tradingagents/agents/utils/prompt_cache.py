"""Provider-aware prompt-prefix caching.

Two provider families, two mechanisms:

* OpenAI-compatible endpoints cache automatically on the longest common prefix
  of the request, with no markers. Nothing is needed beyond keeping the static
  text at the FRONT of the prompt — which is what `split_cacheable` enforces by
  making the split explicit rather than accidental.
* Anthropic caches only up to an explicit ``cache_control`` breakpoint, so the
  static half has to be handed over as its own content block.

`build_cached_prompt` returns whichever shape the given LLM understands, so
call sites declare the static/variable boundary once and stay provider-neutral.
"""

from __future__ import annotations

from typing import Any, Union

from langchain_core.messages import HumanMessage

# Below this, Anthropic rejects a cache breakpoint and OpenAI never caches, so
# marking the prompt would cost a round trip's worth of overhead for nothing.
MIN_CACHEABLE_CHARS = 4096


def _is_anthropic(llm: Any) -> bool:
    """True when this LLM needs explicit cache_control breakpoints."""
    for klass in type(llm).__mro__:
        if klass.__name__ in {"ChatAnthropic", "AnthropicLLM"}:
            return True
        if klass.__module__.startswith("langchain_anthropic"):
            return True
    return False


def build_cached_prompt(
    llm: Any,
    static_prefix: str,
    variable_suffix: str,
) -> Union[str, list[HumanMessage]]:
    """Assemble a prompt whose static half is eligible for provider caching.

    Args:
        llm: The chat model the prompt will be sent to.
        static_prefix: Text identical across calls — instructions, ontology,
            schema. Must not contain per-run values, or the cache never hits.
        variable_suffix: Per-run text — ticker, date, anchors, reports.

    Returns:
        A plain string for providers that cache prefixes automatically, or a
        single HumanMessage carrying two content blocks with a cache breakpoint
        after the static one for Anthropic.
    """
    if not _is_anthropic(llm) or len(static_prefix) < MIN_CACHEABLE_CHARS:
        return static_prefix + variable_suffix
    return [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": static_prefix,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": variable_suffix},
            ]
        )
    ]
