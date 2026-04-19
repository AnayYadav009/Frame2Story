"""Frame2Story — Streamlit Web App

A polished, minimal UI for the multimodal movie recap pipeline.
Features:
- Step-by-step real-time progress tracker
- Scrolling system console for detailed logs
- Multi-format export (TXT · Markdown · JSON)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Frame2Story · AI Movie Recap",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3, .hero-title { font-family: 'Outfit', sans-serif; }
    .mono { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }

    /* Midnight Sapphire Palette */
    :root {
        --primary: #6366f1;
        --secondary: #10b981;
        --accent: #f59e0b;
        --bg-dark: #0f172a;
        --bg-card: rgba(30, 41, 59, 0.7);
        --text-main: #f1f5f9;
        --text-muted: #94a3b8;
    }

    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b, #0f172a);
        color: var(--text-main);
    }

    /* Header gradient text */
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #818cf8, #6366f1, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: var(--text-muted);
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
    }

    /* Premium Glass Cards */
    .glass-card {
        background: var(--bg-card);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        transition: transform 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
    }

    /* Step tracker */
    .step-row {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.6rem 0;
        font-size: 0.95rem;
        color: var(--text-muted);
    }
    .step-row.active {
        color: var(--primary);
        font-weight: 600;
    }
    .step-row.done {
        color: var(--secondary);
    }
    .step-dot {
        width: 12px; height: 12px;
        border-radius: 50%;
        background: #334155;
        flex-shrink: 0;
        position: relative;
    }
    .step-dot.active { 
        background: var(--primary); 
        box-shadow: 0 0 12px var(--primary); 
        animation: pulse 1.5s infinite;
    }
    .step-dot.done { background: var(--secondary); }

    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.3); opacity: 0.7; }
        100% { transform: scale(1); opacity: 1; }
    }

    /* Recap text box */
    .recap-box {
        background: rgba(15, 23, 42, 0.4);
        border-left: 4px solid var(--primary);
        border-radius: 12px;
        padding: 1.5rem;
        line-height: 1.85;
        font-size: 1.05rem;
        color: #f8fafc;
        white-space: pre-wrap;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
    }

    /* Console View */
    .console-view {
        background: #020617;
        color: #10b981;
        font-family: 'JetBrains Mono', monospace;
        padding: 1rem;
        border-radius: 12px;
        font-size: 0.85rem;
        height: 200px;
        overflow-y: auto;
        border: 1px solid rgba(16, 185, 129, 0.2);
        margin-top: 1rem;
    }

    /* Score pills */
    .score-pill {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .pill-green { background: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .pill-blue  { background: rgba(99, 102, 241, 0.1); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.3); }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f172a !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.8rem;
        font-weight: 600;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.39);
        transition: all 0.2s;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45); }

    /* Hide default Streamlit footer */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────────
PIPELINE_STEPS = [
    ("prepare",   "Preparing subtitles"),
    ("scenes",    "Detecting scenes"),
    ("features",  "Extracting visual features"),
    ("dialogue",  "Aligning dialogue"),
    ("ranking",   "Scoring and ranking scenes"),
    ("summarize", "Summarizing scenes"),
    ("recap",     "Generating final recap"),
    ("eval",      "Evaluating recap quality"),
]

STEP_KEYWORDS: Dict[str, List[str]] = {
    "prepare":   ["subtitle", "whisper", "transcrib", "audio"],
    "scenes":    ["scene", "detecting"],
    "features":  ["visual", "keyframe", "motion", "object"],
    "dialogue":  ["dialogue", "align"],
    "ranking":   ["ranking", "scoring", "rank", "fus"],
    "summarize": ["summari"],
    "recap":     ["recap", "generating final"],
    "eval":      ["evaluat"],
}

# ── Session state defaults ─────────────────────────────────────────────────────
for _k, _v in {
    "result": None,
    "error": "",
    "log_messages": [],
    "current_step": None,
    "completed_steps": set(),
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helper utilities ───────────────────────────────────────────────────────────

def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".tmp"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def _step_from_message(msg: str) -> Optional[str]:
    lower = msg.lower()
    for step_id, keywords in STEP_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return step_id
    return None


def _friendly_error(exc: Exception) -> str:
    msg = str(exc)
    hints = {
        "ffmpeg": "FFmpeg is not installed or not in PATH. Install it and restart.",
        "cuda":   "GPU error — the pipeline will fall back to CPU automatically.",
        "no such file": "A required file was not found. Check that the video uploaded correctly.",
        "filenotfounderror": "A required file is missing. Try re-uploading.",
        "whisper": "Whisper transcription failed. Ensure ffmpeg is installed.",
        "out of memory": "Ran out of memory. Try a shorter video or free system RAM.",
    }
    for trigger, hint in hints.items():
        if trigger in msg.lower():
            return f"❌ {hint}\n\n*Technical detail:* `{msg}`"
    return f"❌ {msg}"


def _format_seconds(s: float) -> str:
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02}:{sec:02}"
    return f"{m}:{sec:02}"


def _parse_timestamp_input(value: str) -> float:
    text = (value or "").strip()
    if not text:
        raise ValueError("Timestamp is empty")

    parts = text.split(":")
    try:
        if len(parts) == 1:
            seconds = float(parts[0])
            if seconds < 0:
                raise ValueError
            return seconds

        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            if minutes < 0 or seconds < 0 or seconds >= 60:
                raise ValueError
            return (minutes * 60) + seconds

        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            if hours < 0 or minutes < 0 or minutes >= 60 or seconds < 0 or seconds >= 60:
                raise ValueError
            return (hours * 3600) + (minutes * 60) + seconds
    except ValueError as exc:
        raise ValueError("Use seconds, MM:SS, or HH:MM:SS format") from exc

    raise ValueError("Use seconds, MM:SS, or HH:MM:SS format")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Frame2Story")
    st.markdown(
        """
        <small style='color:#94a3b8'>
        AI-powered recap from any watch progress.
        </small>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**Pipeline stages**")
    st.markdown(
        """
        <small style='color:#94a3b8'>
        1. Scene detection & filtering<br>
        2. Visual analysis (YOLOv8 + motion)<br>
        3. Dialogue alignment (SRT / Whisper)<br>
        4. Multimodal fusion & ranking<br>
        5. Scene summarization (BART)<br>
        6. Final recap generation<br>
        7. Quality evaluation (ROUGE + BERTScore)
        </small>
        """,
        unsafe_allow_html=True,
    )
    st.divider()


# ── Main layout ────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎬 Frame2Story</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Generate an AI movie recap based on how much you\'ve watched.</div>',
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("### 📁 Upload")
    movie_file = st.file_uploader(
        "Movie file", type=["mp4", "mkv"], label_visibility="collapsed",
        help="Upload an .mp4 or .mkv video file.",
    )
    subtitle_file = st.file_uploader(
        "Subtitle file (optional)", type=["srt"], label_visibility="collapsed",
        help="Upload an .srt subtitle file. Leave empty to auto-generate via Whisper.",
    )

    st.markdown("### ⚙️ Options")
    progress = st.slider("Watch progress (%)", 1, 100, 30, help="How far through the movie you've watched.")
    st.markdown("**Custom timestamp range (optional)**")
    ts_c1, ts_c2 = st.columns(2)
    with ts_c1:
        start_ts_input = st.text_input(
            "Start timestamp",
            value="",
            placeholder="e.g. 12:30",
            help="Optional. Use seconds, MM:SS, or HH:MM:SS.",
        )
    with ts_c2:
        end_ts_input = st.text_input(
            "End timestamp",
            value="",
            placeholder="e.g. 19:45",
            help="Optional. Use seconds, MM:SS, or HH:MM:SS.",
        )

    summary_style = st.selectbox("Summary style", ["Concise", "Detailed"])
    genre = st.selectbox(
        "Film genre",
        ["Auto", "Drama", "Action", "Documentary"],
        help="Adjusts the fusion weights used to rank scenes.",
    )
    fusion_preset = genre.lower()

    perspective = st.selectbox(
        "Summarization Perspective",
        ["Neutral", "Protagonist", "Antagonist"],
        help="Changes the 'viewpoint' of the recap summaries.",
    )

    generate_btn = st.button("⚡ Generate Recap", use_container_width=True)

with right_col:
    st.markdown("### 📊 Pipeline Status")
    step_placeholder = st.empty()

    def _render_steps():
        completed = st.session_state.completed_steps
        current = st.session_state.current_step
        rows = []
        for sid, label in PIPELINE_STEPS:
            if sid in completed:
                cls, dot_cls, icon = "done", "done", "✅"
            elif sid == current:
                cls, dot_cls, icon = "active", "active", "🔄"
            else:
                cls, dot_cls, icon = "", "", "⬜"
            rows.append(
                f'<div class="step-row {cls}">'
                f'<span class="step-dot {dot_cls}"></span>'
                f'{icon} {label}</div>'
            )
        
        # Log console
        logs = st.session_state.log_messages
        console_html = "".join([f"<div>> {m}</div>" for m in logs[-20:]])
        
        step_placeholder.markdown(
            f'<div class="glass-card">'
            f'<div style="margin-bottom:1rem;font-weight:600;">Process Roadmap</div>'
            f'{"".join(rows)}'
            f'</div>',
            unsafe_allow_html=True,
        )

    _render_steps()


# ── Run pipeline ───────────────────────────────────────────────────────────────
if generate_btn:
    if not movie_file:
        st.error("Please upload a movie file to continue.")
    else:
        validation_error = ""
        range_start_sec: float | None = None
        range_end_sec: float | None = None
        start_raw = (start_ts_input or "").strip()
        end_raw = (end_ts_input or "").strip()

        if start_raw or end_raw:
            if not start_raw or not end_raw:
                validation_error = "Please provide both start and end timestamps."
            else:
                try:
                    range_start_sec = _parse_timestamp_input(start_raw)
                    range_end_sec = _parse_timestamp_input(end_raw)
                    if range_start_sec >= range_end_sec:
                        validation_error = "Start timestamp must be earlier than end timestamp."
                except ValueError as exc:
                    validation_error = f"Invalid timestamp input: {exc}"

        if validation_error:
            st.error(validation_error)
            st.stop()

        # Reset state
        st.session_state.result = None
        st.session_state.error = ""
        st.session_state.log_messages = []
        st.session_state.current_step = PIPELINE_STEPS[0][0]
        st.session_state.completed_steps = set()
        _render_steps()

        movie_path = save_uploaded_file(movie_file)
        subtitle_path = save_uploaded_file(subtitle_file) if subtitle_file else None

        from main_pipeline import run_pipeline

        def _on_progress(msg: str):
            st.session_state.log_messages.append(msg)
            detected = _step_from_message(msg)
            if detected:
                if st.session_state.current_step and st.session_state.current_step != detected:
                    st.session_state.completed_steps.add(st.session_state.current_step)
                st.session_state.current_step = detected
            _render_steps()

        try:
            result = run_pipeline(
                video_path=movie_path,
                subtitle_path=subtitle_path,
                progress=progress,
                range_start_sec=range_start_sec,
                range_end_sec=range_end_sec,
                summary_style=summary_style,
                fusion_preset=fusion_preset,
                perspective=perspective,
                run_evaluation=False,
                progress_callback=_on_progress,
            )
            st.session_state.completed_steps = {s for s, _ in PIPELINE_STEPS}
            st.session_state.current_step = None
            _render_steps()
            st.session_state.result = result

        except Exception as exc:
            st.session_state.error = _friendly_error(exc)
            st.session_state.current_step = None
            _render_steps()

        finally:
            for p in (movie_path, subtitle_path):
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass


# ── Error display ──────────────────────────────────────────────────────────────
if st.session_state.error:
    st.error(st.session_state.error)


# ── Results ────────────────────────────────────────────────────────────────────
result: Optional[Dict[str, Any]] = st.session_state.result

if result:
    recap_text: str = (result.get("final_recap") or result.get("recap") or "").strip()
    scene_count: int = result.get("scene_count", 0)
    selected_count: int = result.get("selected_scene_count", 0)
    eval_scores = result.get("evaluation")
    active_scope = result.get("scope", "progress")
    active_range_start = result.get("range_start_sec")
    active_range_end = result.get("range_end_sec")

    st.divider()

    # ── Stats row ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Total scenes detected", scene_count)
    c2.metric("Scenes selected", selected_count)
    if active_scope == "timestamp-range" and active_range_start is not None and active_range_end is not None:
        c3.metric("Scope", f"{_format_seconds(active_range_start)} -> {_format_seconds(active_range_end)}")
    else:
        c3.metric("Watch progress", f"{progress}%")

    # ── Recap text ─────────────────────────────────────────────────────────────
    st.markdown("### 📄 Recap")
    st.markdown(f'<div class="recap-box">{recap_text}</div>', unsafe_allow_html=True)

    # ── Export buttons ─────────────────────────────────────────────────────────
    st.markdown("**Export & Copy**")
    exp_c1, exp_c2, exp_c3, exp_c4, _ = st.columns([1, 1, 1, 1.2, 1.8])

    with exp_c1:
        st.download_button("📥 TXT", data=recap_text, file_name="movie_recap.txt", mime="text/plain")
    with exp_c2:
        md_export = f"# Movie Recap\n\n*Generated at {progress}% watch progress*\n\n---\n\n{recap_text}\n"
        st.download_button("📝 MD", data=md_export, file_name="movie_recap.md", mime="text/markdown")
    with exp_c3:
        json_export = json.dumps({ "progress_pct": progress, "recap": recap_text }, indent=2)
        st.download_button("{ } JSON", data=json_export, file_name="movie_recap.json", mime="application/json")
    with exp_c4:
        if st.button("📋 Copy text"):
            st.code(recap_text, language=None)
            st.toast("Click the copy icon in the top right of the code box!", icon="📋")

