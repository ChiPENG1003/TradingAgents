from typing import Optional

from .base_client import BaseLLMClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .google_client import GoogleClient


def create_llm_client(
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    **kwargs,
) -> BaseLLMClient:
    """Create an LLM client for the specified provider.

    Args:
        provider: LLM provider (openai, anthropic, google, xai, ollama, openrouter)
        model: Model name/identifier
        base_url: Optional base URL for API endpoint
        **kwargs: Additional provider-specific arguments
            - http_client: Custom httpx.Client for SSL proxy or certificate customization
            - http_async_client: Custom httpx.AsyncClient for async operations
            - timeout: Request timeout in seconds
            - max_retries: Maximum retry attempts
            - api_key: API key for the provider
            - callbacks: LangChain callbacks

    Returns:
        Configured BaseLLMClient instance

    Raises:
        ValueError: If provider is not supported
    """
    provider_lower = provider.lower()

    if provider_lower in ("openai", "ollama", "openrouter"):
        return OpenAIClient(model, base_url, provider=provider_lower, **kwargs)

    if provider_lower == "xai":
        return OpenAIClient(model, base_url, provider="xai", **kwargs)

    if provider_lower == "deepseek":
        # Resolve catalog aliases to underlying API model + thinking flag.
        # DeepSeek's real API ships only deepseek-v4-pro and deepseek-v4-flash.
        api_model, thinking = _resolve_deepseek_alias(model)
        kwargs["deepseek_thinking"] = thinking
        return OpenAIClient(api_model, base_url, provider="deepseek", **kwargs)

    if provider_lower == "anthropic":
        return AnthropicClient(model, base_url, **kwargs)

    if provider_lower == "google":
        return GoogleClient(model, base_url, **kwargs)

    raise ValueError(f"Unsupported LLM provider: {provider}")


_DEEPSEEK_ALIAS_MAP = {
    "deepseek-v4-pro":            ("deepseek-v4-pro",   True),
    "deepseek-v4-flash-thinking": ("deepseek-v4-flash", True),
    "deepseek-v4-flash-instant":  ("deepseek-v4-flash", False),
}


def _resolve_deepseek_alias(alias: str) -> tuple[str, bool]:
    """Map a catalog alias to (api_model, thinking_enabled).

    Unknown aliases are passed through untouched with thinking disabled,
    so users can still experiment with raw DeepSeek model IDs.
    """
    return _DEEPSEEK_ALIAS_MAP.get(alias, (alias, False))
