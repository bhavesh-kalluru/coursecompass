from __future__ import annotations

import time
from dataclasses import asdict
import streamlit as st

from config import load_settings
from rag_pipeline import CourseCompassOptions, run_coursecompass
from ui import (
    inject_css,
    hero,
    metric_row,
    render_week_timeline,
    render_resources,
    render_sources,
    render_evidence,
)


def init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def main() -> None:
    st.set_page_config(
        page_title="CourseCompass — Web‑RAG Learning Path Builder",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    init_state()

    settings = load_settings()
    hero()

    # ---- Sidebar controls
    with st.sidebar:
        st.markdown("### Plan Controls")
        goal = st.text_input(
            "Learning goal / topic",
            placeholder="e.g., Learn LLM evaluation, Intro to statistics, Calculus derivatives…",
        )
        level = st.selectbox("Current level", ["Beginner", "Intermediate", "Advanced"], index=0)
        weeks = st.slider("Plan duration (weeks)", 1, 12, 6, 1)
        hours_per_week = st.slider("Hours per week", 2, 20, 6, 1)
        format_pref = st.selectbox("Preferred format", ["Mixed", "Articles", "Video", "Books/Docs"], index=0)

        st.divider()
        st.markdown("### Web‑RAG Settings")
        recency = st.selectbox("Recency", ["day", "week", "month"], index=1)
        max_sources = st.slider("Max sources", 3, 12, 8, 1)
        deep_mode = st.checkbox("Deep mode (more thorough, slower)", value=False)

        st.markdown("### Models")
        perplexity_model = st.selectbox("Perplexity model", ["sonar-pro", "sonar", "sonar-deep-research"], index=0)
        openai_model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-5", "gpt-5.2"], index=0)
        embedding_model = st.selectbox("Embedding model", ["text-embedding-3-small", "text-embedding-3-large"], index=0)

        st.divider()
        colA, colB = st.columns(2)
        with colA:
            run_btn = st.button("✨ Generate", use_container_width=True)
        with colB:
            clear_btn = st.button("🧹 Clear", use_container_width=True)

        if clear_btn:
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()

        st.caption("Tip: Keep sources lower for faster + cheaper runs.")

        # Key checks
        if not settings.openai_api_key:
            st.error("Missing OPENAI_API_KEY. Create a .env from .env.")
        if not settings.perplexity_api_key:
            st.error("Missing PERPLEXITY_API_KEY. Create a .env from .env.")

    # ---- Main layout
    left, right = st.columns([1.6, 1], gap="large")

    with left:
        metric_row(
            [
                ("Duration", f"{weeks} weeks"),
                ("Time budget", f"{hours_per_week} hrs/week"),
                ("Mode", "Deep" if deep_mode else "Fast"),
            ]
        )

        st.write("")
        tabs = st.tabs(["🗺️ Learning Plan", "📚 Curated Resources", "🔎 Sources & Evidence", "⬇️ Export"])

        if run_btn:
            if not goal.strip():
                st.warning("Enter a learning goal/topic first.")
                st.stop()

            options = CourseCompassOptions(
                recency=recency,
                max_sources=max_sources,
                deep_mode=deep_mode,
                perplexity_model=perplexity_model,
                openai_model=openai_model,
                embedding_model=embedding_model,
            )

            with st.status("Building your learning path…", expanded=True) as status:
                t0 = time.time()

                def say(msg: str) -> None:
                    status.write(msg)

                try:
                    result = run_coursecompass(
                        goal=goal.strip(),
                        level=level,
                        weeks=weeks,
                        hours_per_week=hours_per_week,
                        format_pref=format_pref,
                        settings=settings,
                        options=options,
                        status_callback=say,
                    )
                    st.session_state.last_result = result
                    st.session_state.history.append(
                        {"goal": goal.strip(), "weeks": weeks, "level": level, "ts": time.time()}
                    )

                    status.update(label=f"Done in {time.time()-t0:.1f}s", state="complete", expanded=False)

                except Exception as e:
                    status.update(label="Failed", state="error", expanded=True)
                    st.error(f"Error: {e}")

        last = st.session_state.last_result

        with tabs[0]:
            if last:
                render_week_timeline(last.plan)
            else:
                st.info("Use the sidebar to generate a plan.")

        with tabs[1]:
            if last:
                render_resources(last.resources)
            else:
                st.info("Generate a plan to see curated resources.")

        with tabs[2]:
            if last:
                render_sources(last.sources)
                st.write("")
                render_evidence(last.evidence)
            else:
                st.info("Generate a plan to see sources and evidence.")

        with tabs[3]:
            if last:
                st.download_button(
                    "⬇️ Download Markdown",
                    data=last.export_markdown().encode("utf-8"),
                    file_name="coursecompass_plan.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
                with st.expander("Preview", expanded=False):
                    st.markdown(last.export_markdown())
            else:
                st.info("Generate a plan to enable export.")

    with right:
        st.markdown("### Recent Plans")
        if st.session_state.history:
            for item in reversed(st.session_state.history[-10:]):
                st.markdown(f"- **{item['goal']}** · {item['level']} · {int(item['weeks'])}w")
        else:
            st.caption("No plans yet. Generate your first plan.")

        st.write("")
        st.markdown("### What makes this portfolio‑grade")
        st.markdown(
            "- Web retrieval (Perplexity) — no manual URL paste\n"
            "- Web‑RAG: extract → chunk → embed → top‑k evidence\n"
            "- Grounded generation with citations + evidence pane\n"
            "- Premium UI, export, Docker + Render ready\n"
        )


if __name__ == "__main__":
    main()
