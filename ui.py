from __future__ import annotations

import streamlit as st
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ResourceCard:
    title: str
    url: str
    provider: str = ""
    level: str = ""
    est_time: str = ""
    why: str = ""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class WeekBlock:
    week: int
    headline: str
    outcomes: list[str]
    tasks: list[str]
    mini_project: str = ""
    checkpoint: str = ""


def inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 2.2rem; padding-bottom: 2.2rem; max-width: 1250px; }
          h1, h2, h3 { letter-spacing: -0.02em; }
          p { line-height: 1.5; }
          #MainMenu { visibility: hidden; }
          footer { visibility: hidden; }
          header { visibility: hidden; }

          .cc-hero {
            border-radius: 22px;
            padding: 18px 18px;
            border: 1px solid rgba(255,255,255,0.10);
            background:
              radial-gradient(1200px 450px at 10% -20%, rgba(110,231,255,0.18), transparent 55%),
              radial-gradient(900px 420px at 92% -10%, rgba(255,175,210,0.16), transparent 60%),
              rgba(255,255,255,0.03);
            box-shadow: 0 14px 40px rgba(0,0,0,0.22);
          }
          .cc-hero h1 { margin: 8px 0 0 0; font-size: 2.1rem; }
          .cc-hero p  { margin: 8px 0 0 0; opacity: 0.85; }

          .cc-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.14);
            background: rgba(255,255,255,0.04);
            font-size: 0.85rem;
            opacity: 0.92;
          }

          .cc-card {
            border-radius: 18px;
            padding: 14px 14px;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.03);
            box-shadow: 0 10px 26px rgba(0,0,0,0.18);
          }

          .cc-divider {
            height: 1px;
            background: rgba(255,255,255,0.10);
            margin: 12px 0 10px 0;
          }

          .cc-chip {
            display: inline-block;
            padding: 4px 9px;
            margin: 0 6px 6px 0;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.04);
            font-size: 0.80rem;
            opacity: 0.92;
          }

          .cc-muted { opacity: 0.78; }
          .cc-kbd {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            font-size: 0.86rem;
            padding: 2px 6px;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.05);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
        <div class="cc-hero">
          <div style="display:flex; flex-wrap:wrap; gap:10px; align-items:center; justify-content:space-between;">
            <div style="display:flex; gap:10px; flex-wrap:wrap;">
              <span class="cc-pill">🌐 Live Web Retrieval</span>
              <span class="cc-pill">🧠 Web‑RAG Evidence</span>
              <span class="cc-pill">🗺️ Week‑by‑Week Plan</span>
              <span class="cc-pill">⬇️ Export</span>
            </div>
            <div class="cc-pill">CourseCompass • Education</div>
          </div>
          <h1>CourseCompass</h1>
          <p>
            Turn any learning goal into a credible, structured plan using Perplexity web retrieval + OpenAI synthesis.
          </p>
          <div class="cc-muted" style="margin-top:10px;">
            Tip: Try <span class="cc-kbd">Learn LLM evaluation in 6 weeks</span> or <span class="cc-kbd">Intro to statistics</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")


def metric_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        with c:
            st.markdown(
                f"""
                <div class="cc-card">
                  <div class="cc-muted" style="font-size:0.85rem;">{label}</div>
                  <div style="font-size:1.4rem; font-weight:700; margin-top:4px;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_week_timeline(weeks: list[WeekBlock]) -> None:
    for w in weeks:
        st.markdown(
            f"""
            <div class="cc-card">
              <div class="cc-muted" style="font-size:0.85rem;">Week {w.week}</div>
              <div style="font-size:1.05rem; font-weight:700; margin-top:2px;">{w.headline}</div>
              <div class="cc-divider"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        colA, colB = st.columns([1, 1], gap="large")
        with colA:
            st.markdown("**Outcomes**")
            for o in w.outcomes:
                st.write(f"- {o}")
        with colB:
            st.markdown("**Tasks**")
            for t in w.tasks:
                st.write(f"- {t}")
        if w.mini_project:
            st.markdown(f"**Mini‑project:** {w.mini_project}")
        if w.checkpoint:
            st.markdown(f"**Checkpoint:** {w.checkpoint}")
        st.write("")


def render_resources(resources: Iterable[ResourceCard]) -> None:
    for r in resources:
        tag_html = "".join([f'<span class="cc-chip">{t}</span>' for t in r.tags])
        st.markdown(
            f"""
            <div class="cc-card">
              <div style="display:flex; justify-content:space-between; gap:12px;">
                <div style="min-width: 0;">
                  <div style="font-size:1.02rem; font-weight:700; margin-bottom:4px;">
                    <a href="{r.url}" target="_blank" style="text-decoration:none;">{r.title}</a>
                  </div>
                  <div class="cc-muted" style="font-size:0.86rem;">
                    {r.provider} {("• " + r.level) if r.level else ""} {("• " + r.est_time) if r.est_time else ""}
                  </div>
                </div>
              </div>
              <div style="margin-top:10px;">{tag_html}</div>
              <div class="cc-divider"></div>
              <div class="cc-muted" style="font-size:0.92rem;"><b>Why included:</b> {r.why}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")


def render_sources(sources: list[dict]) -> None:
    st.markdown("#### Sources used")
    if not sources:
        st.info("No sources yet.")
        return
    for s in sources:
        title = s.get("title") or s.get("url")
        url = s.get("url", "")
        date = s.get("date", "")
        st.markdown(f"- [{title}]({url}) {f'— {date}' if date else ''}")


def render_evidence(evidence: list[dict]) -> None:
    st.markdown("#### Evidence snippets (top retrieved chunks)")
    st.caption("These chunks are what the model is allowed to use as ground truth.")
    if not evidence:
        st.info("No evidence yet.")
        return
    for e in evidence:
        label = e.get("label", "Evidence")
        with st.expander(label, expanded=False):
            if e.get("url"):
                st.markdown(f"**URL:** {e['url']}")
            st.write(e.get("text", ""))
