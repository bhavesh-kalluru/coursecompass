from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
import trafilatura
from openai import OpenAI
from tenacity import retry, wait_exponential_jitter, stop_after_attempt

from config import Settings
from ui import WeekBlock, ResourceCard
from web_search import perplexity_search


@dataclass(frozen=True)
class CourseCompassOptions:
    recency: str = "week"
    max_sources: int = 8
    deep_mode: bool = False

    perplexity_model: str = "sonar-pro"
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"


@dataclass(frozen=True)
class CourseCompassResult:
    goal: str
    level: str
    weeks: int
    hours_per_week: int
    format_pref: str

    plan: list[WeekBlock]
    resources: list[ResourceCard]
    sources: list[dict]
    evidence: list[dict]
    notes_markdown: str

    def export_markdown(self) -> str:
        lines = []
        lines.append("# CourseCompass Learning Plan\n")
        lines.append(f"**Goal:** {self.goal}")
        lines.append(f"**Level:** {self.level}")
        lines.append(f"**Duration:** {self.weeks} weeks")
        lines.append(f"**Time budget:** {self.hours_per_week} hrs/week")
        lines.append(f"**Preferred format:** {self.format_pref}")
        lines.append("\n---\n")
        if self.notes_markdown.strip():
            lines.append(self.notes_markdown.strip())
            lines.append("\n---\n")

        lines.append("## Week-by-week plan\n")
        for w in self.plan:
            lines.append(f"### Week {w.week}: {w.headline}")
            lines.append("**Outcomes**")
            lines.extend([f"- {o}" for o in w.outcomes])
            lines.append("\n**Tasks**")
            lines.extend([f"- {t}" for t in w.tasks])
            if w.mini_project:
                lines.append(f"\n**Mini-project:** {w.mini_project}")
            if w.checkpoint:
                lines.append(f"**Checkpoint:** {w.checkpoint}")
            lines.append("")

        lines.append("\n## Curated resources\n")
        for r in self.resources:
            tag_str = ", ".join(r.tags) if r.tags else ""
            meta = " • ".join([x for x in [r.provider, r.level, r.est_time] if x])
            lines.append(f"- [{r.title}]({r.url}){(' — ' + meta) if meta else ''}{(' — ' + tag_str) if tag_str else ''}")
            if r.why:
                lines.append(f"  - Why: {r.why}")

        lines.append("\n---\n## Sources\n")
        for i, s in enumerate(self.sources, start=1):
            title = s.get("title") or s.get("url")
            lines.append(f"- [{i}] {title} — {s.get('url')}")
        return "\n".join(lines).strip() + "\n"


def _clean_text(txt: str) -> str:
    txt = re.sub(r"\s+", " ", (txt or "")).strip()
    return txt


def _chunk_text(text: str, *, chunk_chars: int = 1800, overlap_chars: int = 250) -> list[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


@retry(wait=wait_exponential_jitter(initial=1, max=10), stop=stop_after_attempt(3))
def _extract_url_text(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return ""
    extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    return extracted or ""


@retry(wait=wait_exponential_jitter(initial=1, max=10), stop=stop_after_attempt(3))
def _embed(openai_client: OpenAI, model: str, texts: list[str]) -> np.ndarray:
    resp = openai_client.embeddings.create(model=model, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype=np.float32)


def _cosine_similarity(q: np.ndarray, docs: np.ndarray) -> np.ndarray:
    qn = q / (np.linalg.norm(q) + 1e-12)
    dn = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-12)
    return (dn @ qn).astype(np.float32)


def _extract_json(text: str) -> dict | None:
    """
    Best-effort JSON extraction.
    The model is instructed to output JSON only, but we guard anyway.
    """
    if not text:
        return None
    text = text.strip()

    # If it is pure JSON
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    # Otherwise find the first {...} block.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def run_coursecompass(
    *,
    goal: str,
    level: str,
    weeks: int,
    hours_per_week: int,
    format_pref: str,
    settings: Settings,
    options: CourseCompassOptions,
    status_callback: Callable[[str], None] | None = None,
) -> CourseCompassResult:
    def say(msg: str) -> None:
        if status_callback:
            status_callback(msg)

    if not settings.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")
    if not settings.perplexity_api_key:
        raise RuntimeError("Missing PERPLEXITY_API_KEY in environment.")

    # 1) Web retrieval (no URL paste)
    say("🔎 Retrieving best learning resources from the web (Perplexity)…")
    search_query = (
        f"Best up-to-date learning resources and roadmap for: {goal}. "
        f"Audience: {level}. Prefer {format_pref}. "
        "Include official docs, reputable courses, and high-quality tutorials."
    )

    ws = perplexity_search(
        api_key=settings.perplexity_api_key,
        base_url=settings.perplexity_base_url,
        query=search_query,
        model=options.perplexity_model,
        recency=options.recency,
        max_results=options.max_sources,
        timeout_s=settings.http_timeout_s,
    )

    sources = [{"title": r.title, "url": r.url, "date": r.date} for r in ws.results[: options.max_sources]]

    # 2) Fetch + extract
    say(f"📄 Fetching & extracting content from {len(sources)} sources…")
    all_chunks: list[str] = []
    chunk_meta: list[dict] = []
    for idx, s in enumerate(sources, start=1):
        raw = _extract_url_text(s["url"])
        txt = _clean_text(raw)[:14000]  # bound per-source text
        for c in _chunk_text(txt):
            all_chunks.append(c)
            chunk_meta.append({"source_id": idx, "title": s["title"], "url": s["url"], "chunk": c})

    evidence: list[dict] = []
    if not all_chunks:
        # Scrape blocked or empty; fall back to Perplexity summary only.
        plan = _fallback_plan(goal, level, weeks, hours_per_week, format_pref)
        notes = (
            "## Notes\n"
            "Some sites blocked text extraction. The plan below uses the web summary as a fallback.\n\n"
            f"### Web summary\n{ws.summary or '*No summary returned.*'}\n"
        )
        resources = _fallback_resources(goal, level)
        return CourseCompassResult(
            goal=goal,
            level=level,
            weeks=weeks,
            hours_per_week=hours_per_week,
            format_pref=format_pref,
            plan=plan,
            resources=resources,
            sources=sources,
            evidence=[],
            notes_markdown=notes,
        )

    # 3) Embed + retrieve top evidence
    say(f"🧠 Building semantic index (embeddings)… ({len(all_chunks)} chunks)")
    openai_client = OpenAI(api_key=settings.openai_api_key, timeout=settings.http_timeout_s)

    q_text = f"Learning roadmap and best resources for {goal} for a {level} learner."
    q_vec = _embed(openai_client, options.embedding_model, [q_text])[0]
    doc_vecs = _embed(openai_client, options.embedding_model, all_chunks)
    sims = _cosine_similarity(q_vec, doc_vecs)

    top_k = 10 if options.deep_mode else 6
    top_k = min(top_k, len(all_chunks))
    top_idx = np.argsort(-sims)[:top_k]

    context_blocks: list[str] = []
    for rank, i in enumerate(top_idx, start=1):
        meta = chunk_meta[int(i)]
        score = float(sims[int(i)])
        evidence.append(
            {
                "label": f"Evidence [{meta['source_id']}] score {score:.3f}",
                "url": meta["url"],
                "text": meta["chunk"],
            }
        )
        context_blocks.append(
            f"[{meta['source_id']}] {meta['title']}\nURL: {meta['url']}\nEVIDENCE:\n{meta['chunk']}\n"
        )

    # 4) Synthesis (structured JSON output)
    say("✨ Generating the plan (OpenAI)…")
    sections = {
        "plan": "Week-by-week plan",
        "resources": "Curated resources",
        "notes_markdown": "Short notes section (markdown)",
    }

    instructions = (
        "You are CourseCompass, an education-focused planner.\n"
        "Rules:\n"
        "1) Use ONLY the provided evidence blocks as ground truth for recommendations.\n"
        "2) When you recommend a resource, include a source_id citation like [1] or [2] in the 'why' field.\n"
        "3) If evidence is thin, say so in notes.\n"
        "4) Output MUST be valid JSON only. No extra text.\n"
    )

    desired_json_schema = {
        "plan": [
            {
                "week": 1,
                "headline": "string",
                "outcomes": ["string", "string"],
                "tasks": ["string", "string"],
                "mini_project": "string (optional)",
                "checkpoint": "string (optional)",
            }
        ],
        "resources": [
            {
                "title": "string",
                "url": "string",
                "provider": "string (optional)",
                "level": level,
                "est_time": "string (optional)",
                "why": "string (include citations like [1])",
                "tags": ["string", "string"],
            }
        ],
        "notes_markdown": "string markdown, concise",
    }

    prompt = (
        f"User goal: {goal}\n"
        f"User level: {level}\n"
        f"Duration: {weeks} weeks\n"
        f"Time budget: {hours_per_week} hrs/week\n"
        f"Preferred format: {format_pref}\n\n"
        "Create a realistic plan that fits the time budget.\n"
        "Include 5–10 resources max. Prefer high-quality, reputable sources.\n\n"
        f"Return JSON matching this schema example:\n{json.dumps(desired_json_schema, indent=2)}\n\n"
        "Evidence blocks:\n\n" + "\n---\n".join(context_blocks)
    )

    resp = openai_client.responses.create(
        model=options.openai_model,
        instructions=instructions,
        input=prompt,
    )
    out = (resp.output_text or "").strip()
    data = _extract_json(out)

    if not isinstance(data, dict):
        # Fallback: produce a reasonable plan from templates plus a notes section.
        plan = _fallback_plan(goal, level, weeks, hours_per_week, format_pref)
        resources = _fallback_resources(goal, level)
        notes = (
            "## Notes\n"
            "The model did not return valid JSON. Showing a safe fallback plan.\n"
        )
        return CourseCompassResult(
            goal=goal,
            level=level,
            weeks=weeks,
            hours_per_week=hours_per_week,
            format_pref=format_pref,
            plan=plan,
            resources=resources,
            sources=sources,
            evidence=evidence,
            notes_markdown=notes,
        )

    plan = []
    for item in (data.get("plan") or [])[:weeks]:
        try:
            plan.append(
                WeekBlock(
                    week=int(item.get("week")),
                    headline=str(item.get("headline", "")).strip(),
                    outcomes=[str(x).strip() for x in (item.get("outcomes") or [])][:6],
                    tasks=[str(x).strip() for x in (item.get("tasks") or [])][:8],
                    mini_project=str(item.get("mini_project", "")).strip(),
                    checkpoint=str(item.get("checkpoint", "")).strip(),
                )
            )
        except Exception:
            continue

    resources = []
    for r in (data.get("resources") or [])[:10]:
        url = str(r.get("url", "")).strip()
        title = str(r.get("title", "")).strip() or url
        if not url:
            continue
        resources.append(
            ResourceCard(
                title=title,
                url=url,
                provider=str(r.get("provider", "")).strip(),
                level=str(r.get("level", level)).strip(),
                est_time=str(r.get("est_time", "")).strip(),
                why=str(r.get("why", "")).strip(),
                tags=tuple([str(t).strip() for t in (r.get("tags") or []) if str(t).strip()][:8]),
            )
        )

    notes_md = str(data.get("notes_markdown", "")).strip()
    if not notes_md:
        notes_md = "## Notes\n- Plan generated from live web sources using Web‑RAG.\n"

    # Ensure we have exactly the requested number of weeks
    if len(plan) < weeks:
        plan = plan + _fallback_plan(goal, level, weeks, hours_per_week, format_pref)[len(plan):]

    return CourseCompassResult(
        goal=goal,
        level=level,
        weeks=weeks,
        hours_per_week=hours_per_week,
        format_pref=format_pref,
        plan=plan[:weeks],
        resources=resources,
        sources=sources,
        evidence=evidence,
        notes_markdown=notes_md,
    )


def _fallback_plan(goal: str, level: str, weeks: int, hours_per_week: int, format_pref: str) -> list[WeekBlock]:
    plan = []
    for w in range(1, weeks + 1):
        plan.append(
            WeekBlock(
                week=w,
                headline=f"{goal}: foundations → practice → feedback loop (Week {w})",
                outcomes=[
                    f"Build {level.lower()} understanding of core concepts",
                    "Practice with targeted exercises",
                    "Be able to explain key ideas clearly",
                ],
                tasks=[
                    f"{max(1, hours_per_week//2)}h: primary learning ({format_pref})",
                    f"{max(1, hours_per_week//3)}h: hands-on practice",
                    "30m: review + spaced repetition",
                    "30m: self-quiz + error log",
                ],
                mini_project="Mini project: small end-to-end exercise",
                checkpoint="Checkpoint: 10-question quiz + explain one topic from memory",
            )
        )
    return plan


def _fallback_resources(goal: str, level: str) -> list[ResourceCard]:
    # Kept generic—real resources come from Perplexity results.
    return [
        ResourceCard(
            title=f"Curated overview resources for {goal}",
            url="https://example.com",
            provider="(fallback)",
            level=level,
            est_time="~1 hour",
            why="Fallback card shown if structured resources could not be generated.",
            tags=("Fallback", "Overview"),
        )
    ]
