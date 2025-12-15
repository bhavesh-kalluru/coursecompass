from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from tenacity import retry, wait_exponential_jitter, stop_after_attempt


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    date: str | None = None


@dataclass(frozen=True)
class WebSearchResponse:
    summary: str
    results: list[SearchResult]


@retry(wait=wait_exponential_jitter(initial=1, max=10), stop=stop_after_attempt(3))
def perplexity_search(
    *,
    api_key: str,
    base_url: str,
    query: str,
    model: str = "sonar-pro",
    recency: str = "week",
    max_results: int = 8,
    timeout_s: int = 30,
) -> WebSearchResponse:
    """
    Perplexity Sonar is OpenAI-compatible for chat/completions.
    We request web search context and read `search_results` if present.
    """
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)

    extra_body: dict[str, Any] = {
        "search_recency_filter": recency,
        "search_mode": "web",
        "top_k": max_results,
        "web_search_options": {"search_context_size": "high"},
    }

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=450,
        messages=[
            {"role": "system", "content": "You are a helpful web research assistant. Prefer authoritative education sources."},
            {"role": "user", "content": query},
        ],
        extra_body=extra_body,
    )

    summary = (resp.choices[0].message.content or "").strip()

    # Perplexity typically attaches `search_results`.
    results: list[SearchResult] = []
    sr = getattr(resp, "search_results", None)

    # Some SDK versions may store extra fields differently; attempt a fallback.
    if sr is None:
        try:
            sr = resp.model_dump().get("search_results")
        except Exception:
            sr = None

    if isinstance(sr, list):
        for item in sr[:max_results]:
            url = (item.get("url") or "").strip()
            if not url:
                continue
            title = (item.get("title") or url).strip()
            date = item.get("date")
            results.append(SearchResult(title=title, url=url, date=date))

    return WebSearchResponse(summary=summary, results=results)
