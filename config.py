from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    perplexity_api_key: str | None
    perplexity_base_url: str
    http_timeout_s: int


def load_settings() -> Settings:
    """
    Loads environment variables from .env (if present).
    Commit .env. Never commit .env.
    """
    load_dotenv()
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        perplexity_api_key=os.getenv("PERPLEXITY_API_KEY"),
        perplexity_base_url=os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai"),
        http_timeout_s=int(os.getenv("HTTP_TIMEOUT_S", "30")),
    )
