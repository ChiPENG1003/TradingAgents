import questionary
from typing import List, Optional, Tuple, Dict

from rich.console import Console

from cli.models import AnalystType
from tradingagents.llm_clients.model_catalog import get_model_tiers

console = Console()

TICKER_INPUT_EXAMPLES = "Examples: SPY, CNC.TO, 7203.T, 0700.HK"

ANALYST_ORDER = [
    ("Market Analyst", AnalystType.MARKET),
    ("Social Media Analyst", AnalystType.SOCIAL),
    ("News Analyst", AnalystType.NEWS),
    ("Fundamentals Analyst", AnalystType.FUNDAMENTALS),
]


def get_ticker() -> str:
    """Prompt the user to enter a ticker symbol."""
    ticker = questionary.text(
        f"Enter the exact ticker symbol to analyze ({TICKER_INPUT_EXAMPLES}):",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a valid ticker symbol.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not ticker:
        console.print("\n[red]No ticker symbol provided. Exiting...[/red]")
        exit(1)

    return normalize_ticker_symbol(ticker)


def normalize_ticker_symbol(ticker: str) -> str:
    """Normalize ticker input while preserving exchange suffixes."""
    return ticker.strip().upper()


def get_analysis_date() -> str:
    """Prompt the user to enter a date in YYYY-MM-DD format."""
    import re
    from datetime import datetime

    def validate_date(date_str: str) -> bool:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return False
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    date = questionary.text(
        "Enter the analysis date (YYYY-MM-DD):",
        validate=lambda x: validate_date(x.strip())
        or "Please enter a valid date in YYYY-MM-DD format.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not date:
        console.print("\n[red]No date provided. Exiting...[/red]")
        exit(1)

    return date.strip()


def select_analysts() -> List[AnalystType]:
    """Select analysts using an interactive checkbox."""
    choices = questionary.checkbox(
        "Select Your [Analysts Team]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in ANALYST_ORDER
        ],
        instruction="\n- Press Space to select/unselect analysts\n- Press 'a' to select/unselect all\n- Press Enter when done",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        console.print("\n[red]No analysts selected. Exiting...[/red]")
        exit(1)

    return choices


def select_research_depth() -> int:
    """Select research depth using an interactive selection."""

    # Define research depth options with their corresponding values
    DEPTH_OPTIONS = [
        ("Shallow - Quick research, few debate and strategy discussion rounds", 1),
        ("Medium - Middle ground, moderate debate rounds and strategy discussion", 3),
        ("Deep - Comprehensive research, in depth debate and strategy discussion", 5),
    ]

    choice = questionary.select(
        "Select Your [Research Depth]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in DEPTH_OPTIONS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:yellow noinherit"),
                ("highlighted", "fg:yellow noinherit"),
                ("pointer", "fg:yellow noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No research depth selected. Exiting...[/red]")
        exit(1)

    return choice


def _fetch_openrouter_models() -> List[Tuple[str, str]]:
    """Fetch available models from the OpenRouter API."""
    import requests
    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        return [(m.get("name") or m["id"], m["id"]) for m in models]
    except Exception as e:
        console.print(f"\n[yellow]Could not fetch OpenRouter models: {e}[/yellow]")
        return []


def select_openrouter_model() -> str:
    """Select an OpenRouter model from the newest available, or enter a custom ID."""
    models = _fetch_openrouter_models()

    choices = [questionary.Choice(name, value=mid) for name, mid in models[:5]]
    choices.append(questionary.Choice("Custom model ID", value="custom"))

    choice = questionary.select(
        "Select OpenRouter Model (latest available):",
        choices=choices,
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style([
            ("selected", "fg:magenta noinherit"),
            ("highlighted", "fg:magenta noinherit"),
            ("pointer", "fg:magenta noinherit"),
        ]),
    ).ask()

    if choice is None or choice == "custom":
        return questionary.text(
            "Enter OpenRouter model ID (e.g. google/gemma-4-26b-a4b-it):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a model ID.",
        ).ask().strip()

    return choice


def select_model_tier(provider: str) -> tuple[str, str]:
    """Select a model tier that sets both quick-thinking and deep-thinking models.

    Returns (quick_model, deep_model).
    OpenRouter: asks for one model ID used for both roles.
    """
    provider_lower = provider.lower()

    if provider_lower == "openrouter":
        model = select_openrouter_model()
        return model, model

    tiers = get_model_tiers(provider_lower)
    if not tiers:
        console.print(f"\n[yellow]No tier presets for {provider}. Enter model IDs manually.[/yellow]")
        quick = questionary.text("Quick-thinking model ID:").ask()
        deep = questionary.text("Deep-thinking model ID:").ask()
        if not quick or not deep:
            console.print("\n[red]No model selected. Exiting...[/red]")
            exit(1)
        return quick.strip(), deep.strip()

    choice = questionary.select(
        "Select Your [Model Tier]:",
        choices=[
            questionary.Choice(display, value=(quick, deep))
            for display, _, quick, deep in tiers
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No model tier selected. Exiting...[/red]")
        exit(1)

    return choice


_DEPTH_TO_EFFORT = {1: "low", 3: "medium", 5: "high"}


def derive_reasoning_effort(research_depth: int, provider: str) -> dict:
    """Derive provider-specific reasoning effort config from research depth.

    Shallow=1→low, Medium=3→medium, Deep=5→high.
    Google maps low→minimal, medium/high→high (binary API param).
    """
    effort = _DEPTH_TO_EFFORT.get(research_depth, "medium")
    provider_lower = provider.lower()
    if provider_lower == "openai":
        return {"openai_reasoning_effort": effort}
    if provider_lower == "anthropic":
        return {"anthropic_effort": effort}
    if provider_lower == "google":
        return {"google_thinking_level": "minimal" if effort == "low" else "high"}
    return {}

def select_llm_provider() -> tuple[str, str | None]:
    """Select the LLM provider and its API endpoint."""
    BASE_URLS = [
        ("OpenAI", "https://api.openai.com/v1"),
        ("Google", None),  # google-genai SDK manages its own endpoint
        ("Anthropic", "https://api.anthropic.com/"),
        ("DeepSeek", "https://api.deepseek.com/v1"),
        ("xAI", "https://api.x.ai/v1"),
        ("Openrouter", "https://openrouter.ai/api/v1"),
        ("Ollama", "http://localhost:11434/v1"),
    ]
    
    choice = questionary.select(
        "Select your LLM Provider:",
        choices=[
            questionary.Choice(display, value=(display, value))
            for display, value in BASE_URLS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()
    
    if choice is None:
        console.print("\n[red]no OpenAI backend selected. Exiting...[/red]")
        exit(1)
    
    display_name, url = choice
    print(f"You selected: {display_name}\tURL: {url}")

    return display_name, url



def select_trading_mode() -> str:
    """Select between live trading and backtest simulation."""
    choice = questionary.select(
        "Select Trading Mode:",
        choices=[
            questionary.Choice("Live - Real-time analysis for a single date", "live"),
            questionary.Choice("Backtest - Generate monthly strategies over a date range, then replay on historical OHLCV", "backtest"),
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()

    if choice is None:
        console.print("\n[red]No trading mode selected. Exiting...[/red]")
        exit(1)

    return choice


def select_backtest_range() -> Tuple[str, str]:
    """Prompt the user for backtest start/end dates (YYYY-MM-DD)."""
    import re
    from datetime import datetime

    def validate_date(date_str: str) -> bool:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return False
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    start = questionary.text(
        "Enter backtest START date (YYYY-MM-DD):",
        validate=lambda x: validate_date(x.strip())
        or "Please enter a valid date in YYYY-MM-DD format.",
    ).ask()
    end = questionary.text(
        "Enter backtest END date (YYYY-MM-DD):",
        validate=lambda x: validate_date(x.strip())
        or "Please enter a valid date in YYYY-MM-DD format.",
    ).ask()

    if not start or not end:
        console.print("\n[red]No backtest range provided. Exiting...[/red]")
        exit(1)

    start = start.strip()
    end = end.strip()
    if datetime.strptime(start, "%Y-%m-%d") >= datetime.strptime(end, "%Y-%m-%d"):
        console.print("\n[red]End date must be after start date. Exiting...[/red]")
        exit(1)

    return start, end


def select_review_cadence() -> int:
    """Ask how many trading days between strategy reviews. Default 5 (weekly)."""
    presets = [
        questionary.Choice("Daily — every 1 trading day", "1"),
        questionary.Choice("Every 2 trading days", "2"),
        questionary.Choice("Every 3 trading days", "3"),
        questionary.Choice("Weekly — every 5 trading days (default)", "5"),
        questionary.Choice("Bi-weekly — every 10 trading days", "10"),
        questionary.Choice("Monthly — every 21 trading days", "21"),
        questionary.Choice("Custom integer", "custom"),
    ]
    choice = questionary.select(
        "Select strategy review cadence (in trading days):",
        choices=presets,
        default=presets[3],
        instruction="\n- Higher cadence = fewer LLM calls / lower cost\n- Lower cadence = more reactive to market changes",
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()

    if choice is None:
        console.print("\n[red]No cadence selected. Exiting...[/red]")
        exit(1)

    if choice == "custom":
        raw = questionary.text(
            "Enter review cadence in trading days (1-60):",
            default="5",
            validate=lambda x: (
                x.strip().isdigit() and 1 <= int(x.strip()) <= 60
            ) or "Please enter an integer between 1 and 60.",
        ).ask()
        if not raw:
            console.print("\n[red]No cadence provided. Exiting...[/red]")
            exit(1)
        return int(raw.strip())

    return int(choice)


def ask_output_language() -> str:
    """Ask for report output language."""
    choice = questionary.select(
        "Select Output Language:",
        choices=[
            questionary.Choice("English (default)", "English"),
            questionary.Choice("Chinese (中文)", "Chinese"),
            questionary.Choice("Japanese (日本語)", "Japanese"),
            questionary.Choice("Korean (한국어)", "Korean"),
            questionary.Choice("Hindi (हिन्दी)", "Hindi"),
            questionary.Choice("Spanish (Español)", "Spanish"),
            questionary.Choice("Portuguese (Português)", "Portuguese"),
            questionary.Choice("French (Français)", "French"),
            questionary.Choice("German (Deutsch)", "German"),
            questionary.Choice("Arabic (العربية)", "Arabic"),
            questionary.Choice("Russian (Русский)", "Russian"),
            questionary.Choice("Custom language", "custom"),
        ],
        style=questionary.Style([
            ("selected", "fg:yellow noinherit"),
            ("highlighted", "fg:yellow noinherit"),
            ("pointer", "fg:yellow noinherit"),
        ]),
    ).ask()

    if choice == "custom":
        return questionary.text(
            "Enter language name (e.g. Turkish, Vietnamese, Thai, Indonesian):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a language name.",
        ).ask().strip()

    return choice
