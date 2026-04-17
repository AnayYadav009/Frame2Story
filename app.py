"""Frame2Story — Streamlit Web App

A polished UI for the multimodal movie recap pipeline.
Features:
- Step-by-step real-time progress tracker
- Keyframe gallery for each selected scene (multi-frame)
- Scene rationale table (scores breakdown + visual score)
- Near-miss scenes expander
- ROUGE / BERTScore evaluation panel
- Multi-format export (TXT · Markdown · JSON)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
        color: #e2e8f0;
    }

    /* Header gradient text */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Glass cards */
    .glass-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(10px);
    }

    /* Step tracker */
    .step-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.45rem 0;
        font-size: 0.95rem;
        color: #94a3b8;
        transition: color 0.3s;
    }
    .step-row.active {
        color: #60a5fa;
        font-weight: 600;
    }
    .step-row.done {
        color: #34d399;
    }
    .step-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        background: #334155;
        flex-shrink: 0;
        transition: background 0.3s;
    }
    .step-dot.active { background: #60a5fa; box-shadow: 0 0 8px #60a5fa; }
    .step-dot.done   { background: #34d399; }

    /* Recap text box */
    .recap-box {
        background: rgba(255,255,255,0.04);
        border-left: 4px solid #a78bfa;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        line-height: 1.8;
        font-size: 1.02rem;
        color: #e2e8f0;
        white-space: pre-wrap;
    }

    /* Score pill */
    .score-pill {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 0.15rem;
    }
    .pill-green { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid #34d399; }
    .pill-blue  { background: rgba(96,165,250,0.15); color: #60a5fa; border: 1px solid #60a5fa; }
    .pill-purple{ background: rgba(167,139,250,0.15); color: #a78bfa; border: 1px solid #a78bfa; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.9) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed, #2563eb);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        font-size: 1rem;
        transition: opacity 0.2s, transform 0.1s;
    }
    .stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }
    .stDownloadButton > button {
        background: rgba(255,255,255,0.08);
        color: #e2e8f0;
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 8px;
    }

    /* Keyframe image border */
    .kf-img { border-radius: 10px; border: 1px solid rgba(255,255,255,0.12); }

    /* Hide default Streamlit header decoration */
    #MainMenu, footer { visibility: hidden; }
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


def _load_json_safe(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _keyframe_paths(scene_id: int, feat_map: Optional[Dict] = None, max_frames: int = 3, keyframes_dir: str = "data/keyframes") -> List[str]:
    """Return up to max_frames keyframe paths for a scene."""
    paths: List[str] = []

    if feat_map:
        feat = feat_map.get(scene_id) or feat_map.get(str(scene_id)) or {}
        stored = feat.get("keyframe_paths") or []
        for p in stored[:max_frames]:
            if os.path.exists(p):
                paths.append(p)

    if not paths:
        for i in range(1, max_frames + 1):
            for ext in (".jpg", ".png"):
                candidate = f"{keyframes_dir}/scene_{scene_id}_frame_{i}{ext}"
                if os.path.exists(candidate):
                    paths.append(candidate)
                    break

    return paths[:max_frames]


def _format_seconds(s: float) -> str:
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02}:{sec:02}"
    return f"{m}:{sec:02}"


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
    st.markdown(
        "<small style='color:#475569'>Results are cached — re-running the same video is fast.</small>",
        unsafe_allow_html=True,
    )


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
    summary_style = st.selectbox("Summary style", ["Concise", "Detailed"])
    genre = st.selectbox(
        "Film genre",
        ["Auto", "Drama", "Action", "Documentary"],
        help="Adjusts the fusion weights used to rank scenes.",
    )
    fusion_preset = genre.lower()

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
        step_placeholder.markdown(
            '<div class="glass-card">' + "".join(rows) + "</div>",
            unsafe_allow_html=True,
        )

    _render_steps()


# ── Run pipeline ───────────────────────────────────────────────────────────────
if generate_btn:
    if not movie_file:
        st.error("Please upload a movie file to continue.")
    else:
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
                summary_style=summary_style,
                fusion_preset=fusion_preset,
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

    st.divider()

    # ── Stats row ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Total scenes detected", scene_count)
    c2.metric("Scenes selected", selected_count)
    c3.metric("Watch progress", f"{progress}%")

    # ── Recap text ─────────────────────────────────────────────────────────────
    st.markdown("### 📄 Recap")
    st.markdown(f'<div class="recap-box">{recap_text}</div>', unsafe_allow_html=True)

    # ── Export buttons ─────────────────────────────────────────────────────────
    st.markdown("**Export**")
    exp_c1, exp_c2, exp_c3, _ = st.columns([1, 1, 1, 3])

    with exp_c1:
        st.download_button("📥 Plain text", data=recap_text, file_name="movie_recap.txt", mime="text/plain")
    with exp_c2:
        md_export = f"# Movie Recap\n\n*Generated at {progress}% watch progress*\n\n---\n\n{recap_text}\n"
        st.download_button("📝 Markdown", data=md_export, file_name="movie_recap.md", mime="text/markdown")
    with exp_c3:
        json_export = json.dumps({"progress_pct": progress, "style": summary_style, "recap": recap_text}, indent=2)
        st.download_button("{ } JSON", data=json_export, file_name="movie_recap.json", mime="application/json")

    # ── Evaluation scores ──────────────────────────────────────────────────────
    if eval_scores and not result.get("evaluation_error"):
        with st.expander("📈 Evaluation Scores (ROUGE + BERTScore)", expanded=False):
            rouge = eval_scores.get("rouge", {})
            bert = eval_scores.get("bert_score", {})
            st.markdown("**ROUGE**")
            r_c1, r_c2, r_c3, r_c4 = st.columns(4)
            r_c1.metric("ROUGE-1 F1",  f"{rouge.get('rouge1_f1', 0):.3f}")
            r_c2.metric("ROUGE-1 P",   f"{rouge.get('rouge1_precision', 0):.3f}")
            r_c3.metric("ROUGE-L F1",  f"{rouge.get('rougeL_f1', 0):.3f}")
            r_c4.metric("ROUGE-L P",   f"{rouge.get('rougeL_precision', 0):.3f}")
            st.markdown("**BERTScore**")
            b_c1, b_c2, b_c3 = st.columns(3)
            b_c1.metric("Precision", f"{bert.get('precision', 0):.3f}")
            b_c2.metric("Recall",    f"{bert.get('recall', 0):.3f}")
            b_c3.metric("F1",        f"{bert.get('f1', 0):.3f}")
    elif result.get("evaluation_error"):
        with st.expander("📈 Evaluation Scores", expanded=False):
            st.info("Evaluation skipped — no reference summary found.", icon="ℹ️")

    # ── Keyframe gallery ───────────────────────────────────────────────────────
    rationale_raw = _load_json_safe("data/intermediate/scene_rationale.json")
    summaries_raw = _load_json_safe("data/intermediate/scene_summaries.json") or {}

    selected_ids: List[int] = []
    if isinstance(rationale_raw, list):
        selected_ids = [r["scene_id"] for r in rationale_raw if r.get("selected")]

    if selected_ids:
        features_raw = _load_json_safe("data/intermediate/scene_features.json") or []
        feat_map: Dict = {}
        if isinstance(features_raw, list):
            feat_map = {f["scene_id"]: f for f in features_raw if isinstance(f, dict)}
        elif isinstance(features_raw, dict):
            feat_map = {int(k): v for k, v in features_raw.items()}

        with st.expander("🖼️ Keyframe Gallery", expanded=True):
            st.markdown("<small style='color:#94a3b8'>Up to 3 frames per selected scene.</small>", unsafe_allow_html=True)
            for sid in selected_ids:
                kfs = _keyframe_paths(sid, feat_map=feat_map)
                scene_summary = (summaries_raw.get(str(sid)) or "").strip()
                header = f"**Scene {sid}**" + (f" — {scene_summary[:90]}…" if scene_summary else "")
                st.markdown(header)
                if kfs:
                    img_cols = st.columns(len(kfs))
                    for col, kf_path in zip(img_cols, kfs):
                        col.image(kf_path, use_container_width=True)
                else:
                    st.markdown(f'<div class="glass-card" style="text-align:center;padding:1rem;">🎞️ <small>No keyframes saved for scene {sid}</small></div>', unsafe_allow_html=True)

    # ── Scene rationale table ──────────────────────────────────────────────────
    if isinstance(rationale_raw, list) and rationale_raw:
        with st.expander("🔍 Scene Rationale", expanded=False):
            st.markdown("<small style='color:#94a3b8'>Score breakdown for every detected scene.</small>", unsafe_allow_html=True)
            
            features_raw = _load_json_safe("data/intermediate/scene_features.json") or []
            feat_map_local: Dict = {}
            if isinstance(features_raw, list):
                feat_map_local = {f["scene_id"]: f for f in features_raw if isinstance(f, dict)}
            elif isinstance(features_raw, dict):
                feat_map_local = {int(k): v for k, v in features_raw.items()}

            table_rows = []
            for entry in rationale_raw:
                sid = entry.get("scene_id", "?")
                feat = feat_map_local.get(sid, {})
                table_rows.append({
                    "Scene": sid,
                    "Time": f"{_format_seconds(feat.get('start', 0))} → {_format_seconds(feat.get('end', 0))}",
                    "Selected": "✅" if entry.get("selected") else "—",
                    "Final": f"{entry.get('final_score', 0):.3f}",
                    "Dialogue": f"{entry.get('dialogue_score', 0):.3f}",
                    "Motion": f"{entry.get('motion_score', 0):.3f}",
                    "Objects": f"{entry.get('object_score', 0):.3f}",
                    "Visual": f"{entry.get('visual_score', 0):.3f}",
                })

            import pandas as pd
            df = pd.DataFrame(table_rows)
            st.dataframe(df, use_container_width=True, hide_index=True, column_config={
                "Scene": st.column_config.NumberColumn("Scene", width="small"),
                "Selected": st.column_config.TextColumn("✓", width="small"),
                "Final": st.column_config.NumberColumn("Final score", format="%.3f"),
                "Dialogue": st.column_config.NumberColumn("Dialogue", format="%.3f"),
                "Motion": st.column_config.NumberColumn("Motion", format="%.3f"),
                "Objects": st.column_config.NumberColumn("Objects", format="%.3f"),
                "Visual": st.column_config.NumberColumn("Visual", format="%.3f"),
            })

    # ── Near-miss scenes ───────────────────────────────────────────────────────
    if isinstance(rationale_raw, list) and rationale_raw:
        near_misses = [r for r in rationale_raw if not r.get("selected")]
        near_misses = sorted(near_misses, key=lambda x: x.get("final_score", 0), reverse=True)[:5]
        if near_misses:
            with st.expander("🎯 Near-miss Scenes", expanded=False):
                st.markdown("<small style='color:#94a3b8'>Top 5 non-selected scenes by final score.</small>", unsafe_allow_html=True)
                for r in near_misses:
                    st.markdown(f"**Scene {r['scene_id']}** — Score: `{r.get('final_score', 0):.3f}`")