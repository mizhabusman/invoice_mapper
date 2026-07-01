"""Central configuration, loaded once from the environment / .env file.

Every module imports settings from here so there is a single source of truth
for the API key, model name and tuning knobs. Nothing else in the codebase
reads os.environ directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env into the process environment. Safe to call at import time; if the
# file is missing, real environment variables are still honoured.
load_dotenv()

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_CONCURRENCY = 6
# Approx USD->INR rate for displaying cost in rupees. Override via .env to keep
# it current — exchange rates drift and this is only a cost estimate.
DEFAULT_USD_TO_INR = 88.0

# Published API pricing in USD per 1,000,000 tokens, as (input, output).
# Used only to show an estimated cost; update if Anthropic pricing changes.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
}


# Client-facing capability tiers -> model IDs. The client picks a tier
# (how much "power" the PDF needs); the underlying model name stays hidden.
MODEL_TIERS: dict[str, str] = {
    "low": "claude-haiku-4-5",     # fastest, cheapest — simple/clean invoices
    "medium": "claude-sonnet-4-6",  # balanced
    "high": "claude-opus-4-8",      # most accurate — complex/critical invoices
}
DEFAULT_TIER = "medium"  # balanced accuracy/cost; Low for simple, High for complex


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """Estimate USD cost for a token spend, or None if the model isn't priced.

    Prompt-cache pricing: cached reads bill at 0.1x the input rate, cache writes
    at 1.25x (5-minute TTL). `input_tokens` is the uncached remainder.
    """
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return None
    input_rate, output_rate = pricing
    return (
        (input_tokens / 1_000_000) * input_rate
        + (cache_read_tokens / 1_000_000) * input_rate * 0.1
        + (cache_write_tokens / 1_000_000) * input_rate * 1.25
        + (output_tokens / 1_000_000) * output_rate
    )


@dataclass(frozen=True)
class Settings:
    """Immutable view of the runtime configuration."""

    anthropic_api_key: str
    model: str
    max_concurrency: int
    usd_to_inr: float

    @property
    def is_api_key_present(self) -> bool:
        return bool(self.anthropic_api_key)


def _read_number(name: str, default, cast):
    """Read a numeric env var via `cast`, falling back to default if invalid."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def load_settings() -> Settings:
    """Build a Settings object from the current environment.

    Called lazily (not at import) so that importing the package never fails
    just because the key is absent — the UI can surface a friendly message
    instead of crashing.
    """
    return Settings(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip(),
        model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        max_concurrency=_read_number("MAX_CONCURRENCY", DEFAULT_CONCURRENCY, int),
        usd_to_inr=_read_number("USD_TO_INR", DEFAULT_USD_TO_INR, float),
    )
