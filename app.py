from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import json

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Frame2Story - Movie Recap",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
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
            return f"Error: {hint}\n\nTechnical details: {msg}"
    return f"Error: {msg}"


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


# ── CSS Styling ────────────────────────────────────────────────────────────────

def render_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Global Font & Text */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: var(--text-color);
        }

        /* Container centering & max-width */
        .block-container {
            max-width: 640px !important;
            padding-top: 3rem !important;
            padding-bottom: 3rem !important;
        }

        /* Headers & Typography */
        .app-title {
            font-size: 2.2rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
            color: var(--text-color);
        }

        .app-subtitle {
            font-size: 1rem;
            color: var(--text-color);
            opacity: 0.75;
            text-align: center;
            margin-bottom: 2.5rem;
            line-height: 1.5;
        }

        /* Flat bordered card using theme-agnostic transparency */
        .flat-card {
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            background-color: rgba(128, 128, 128, 0.05);
        }

        /* Customize only the dropzone section inside the uploader to prevent text visibility bugs */
        [data-testid="stFileUploader"] {
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
        }
        
        [data-testid="stFileUploader"] section {
            border: 1.5px dashed rgba(128, 128, 128, 0.3) !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            background-color: rgba(128, 128, 128, 0.03) !important;
            transition: border-color 0.2s ease !important;
        }
        
        [data-testid="stFileUploader"] section:hover {
            border-color: #6366f1 !important;
        }

        /* Pipeline Stepper component styles */
        .stepper-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 2rem 0;
            position: relative;
            padding: 0 10px;
        }

        .stepper-line {
            position: absolute;
            top: 16px;
            left: 24px;
            right: 24px;
            height: 2px;
            background-color: rgba(128, 128, 128, 0.2);
            z-index: 1;
        }

        .stepper-line-fill {
            position: absolute;
            top: 16px;
            left: 24px;
            height: 2px;
            background-color: #6366f1;
            z-index: 2;
            transition: width 0.3s ease;
        }

        .stepper-step {
            display: flex;
            flex-direction: column;
            align-items: center;
            z-index: 3;
            position: relative;
            width: 60px;
        }

        .step-circle {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background-color: rgba(128, 128, 128, 0.08);
            border: 2px solid rgba(128, 128, 128, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.85rem;
            color: var(--text-color);
            opacity: 0.8;
            transition: all 0.3s ease;
        }

        .step-circle.active {
            border-color: #6366f1;
            color: #6366f1;
            background-color: rgba(99, 102, 241, 0.1);
            opacity: 1.0;
        }

        .step-circle.done {
            border-color: #10b981;
            background-color: #10b981;
            color: #ffffff;
            opacity: 1.0;
        }

        .step-label {
            font-size: 0.7rem;
            color: var(--text-color);
            opacity: 0.7;
            margin-top: 0.5rem;
            font-weight: 500;
            text-align: center;
            white-space: nowrap;
        }

        .step-label.active {
            color: #6366f1;
            font-weight: 600;
            opacity: 1.0;
        }

        .step-label.done {
            color: #10b981;
            opacity: 1.0;
        }

        /* Scene Timeline Bar styles */
        .timeline-container {
            height: 12px;
            background-color: rgba(128, 128, 128, 0.1);
            border-radius: 6px;
            position: relative;
            overflow: hidden;
            margin: 1.5rem 0;
            border: 1px solid rgba(128, 128, 128, 0.15);
            width: 100%;
        }

        .timeline-segment {
            position: absolute;
            height: 100%;
            background-color: #6366f1;
            opacity: 0.85;
            transition: all 0.3s ease;
        }

        .timeline-segment.placeholder {
            width: 30%;
            left: 35%;
            animation: placeholder-slide 2s infinite ease-in-out;
            background: linear-gradient(90deg, rgba(128, 128, 128, 0.1) 0%, #6366f1 50%, rgba(128, 128, 128, 0.1) 100%);
        }

        @keyframes placeholder-slide {
            0% { left: -30%; }
            100% { left: 110%; }
        }

        /* Metric card styles */
        .metric-card {
            border: 1px solid rgba(128, 128, 128, 0.15);
            border-radius: 12px;
            padding: 1rem;
            background-color: rgba(128, 128, 128, 0.05);
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            height: 100%;
        }

        .metric-label {
            font-size: 0.75rem;
            font-weight: 500;
            color: var(--text-color);
            opacity: 0.7;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }

        .metric-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--text-color);
        }

        /* Badge component styles */
        .badge-container {
            display: flex;
            gap: 0.5rem;
            justify-content: flex-end;
            margin-bottom: 0.5rem;
        }

        .badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            font-size: 0.7rem;
            font-weight: 600;
            border-radius: 6px;
            background-color: rgba(128, 128, 128, 0.08);
            color: var(--text-color);
            border: 1px solid rgba(128, 128, 128, 0.15);
        }

        /* Technical Log Console view styles */
        .console-view {
            background: #0f172a;
            color: #38bdf8;
            font-family: 'JetBrains Mono', monospace;
            padding: 1rem;
            border-radius: 8px;
            font-size: 0.8rem;
            overflow-y: auto;
            border: 1px solid #334155;
            white-space: pre-wrap;
            line-height: 1.5;
            text-align: left;
        }

        /* Recap text text container */
        .recap-content {
            line-height: 1.7;
            font-size: 0.95rem;
            color: var(--text-color);
            white-space: pre-wrap;
            text-align: left;
        }

        /* Streamlit Button overrides */
        .stButton > button {
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
        }

        /* Hide footer */
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Stepper & Timeline Generators ──────────────────────────────────────────────

def get_stepper_html() -> str:
    completed = st.session_state.completed_steps
    current = st.session_state.current_step
    
    steps_keys = [s[0] for s in PIPELINE_STEPS]
    
    # Calculate line fill
    if current in steps_keys:
        active_idx = steps_keys.index(current)
        line_fill_pct = int((active_idx / (len(steps_keys) - 1)) * 100)
    elif len(completed) == len(steps_keys):
        line_fill_pct = 100
    else:
        line_fill_pct = 0
        
    steps_html = []
    short_labels = {
        "prepare": "Subtitles",
        "scenes": "Scenes",
        "features": "Features",
        "dialogue": "Dialogue",
        "ranking": "Ranking",
        "summarize": "Summarize",
        "recap": "Recap",
        "eval": "Eval"
    }
    
    for idx, (sid, label) in enumerate(PIPELINE_STEPS, start=1):
        if sid in completed:
            circle_cls = "done"
            label_cls = "done"
            icon = "✓"
        elif sid == current:
            circle_cls = "active"
            label_cls = "active"
            icon = "◷"
        else:
            circle_cls = ""
            label_cls = ""
            icon = str(idx)
            
        lbl = short_labels.get(sid, label.split()[0])
        
        steps_html.append(
            f'<div class="stepper-step">'
            f'  <div class="step-circle {circle_cls}">{icon}</div>'
            f'  <div class="step-label {label_cls}">{lbl}</div>'
            f'</div>'
        )
        
    html = f"""
    <div class="stepper-container">
        <div class="stepper-line"></div>
        <div class="stepper-line-fill" style="width: {line_fill_pct}%;"></div>
        {"".join(steps_html)}
    </div>
    """
    return html


def get_timeline_html() -> str:
    scenes_path = Path("data/intermediate/scenes.json")
    selected_path = Path("data/intermediate/selected_scenes.json")
    
    if not scenes_path.exists() or not selected_path.exists():
        return '<div class="timeline-container"><div class="timeline-segment placeholder"></div></div>'
        
    try:
        with open(scenes_path, "r") as f:
            all_scenes = json.load(f)
        with open(selected_path, "r") as f:
            selected_scenes = json.load(f)
            
        if not all_scenes or not selected_scenes:
            return '<div class="timeline-container"><div class="timeline-segment placeholder"></div></div>'
            
        total_duration = max(float(s.get("end", 0.0)) for s in all_scenes)
        if total_duration <= 0:
            return '<div class="timeline-container"><div class="timeline-segment placeholder"></div></div>'
            
        segments = []
        for s in selected_scenes:
            if isinstance(s, dict):
                start = float(s.get("start", 0.0))
                end = float(s.get("end", 0.0))
                scene_id = s.get("scene_id", "?")
            else:
                scene_id = int(s)
                matching = [x for x in all_scenes if int(x.get("scene_id", -1)) == scene_id]
                if not matching:
                    continue
                start = float(matching[0].get("start", 0.0))
                end = float(matching[0].get("end", 0.0))
                
            start_pct = min(100.0, max(0.0, (start / total_duration) * 100))
            width_pct = min(100.0 - start_pct, max(0.1, ((end - start) / total_duration) * 100))
            
            segments.append(
                f'<div class="timeline-segment" style="left: {start_pct:.2f}%; width: {width_pct:.2f}%;" '
                f'title="Scene {scene_id}: {int(start)}s-{int(end)}s"></div>'
            )
            
        return f'<div class="timeline-container">{"".join(segments)}</div>'
    except Exception:
        return '<div class="timeline-container"><div class="timeline-segment placeholder"></div></div>'


# ── Screen Renderers ───────────────────────────────────────────────────────────

main_container = st.empty()


def show_processing_screen(status_message=""):
    with main_container.container():
        st.markdown('<div class="app-title">Frame2Story</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-subtitle">Processing video file</div>', unsafe_allow_html=True)
        
        # 1. Pipeline Stepper
        st.markdown(get_stepper_html(), unsafe_allow_html=True)
        
        # 2. Status message line
        if status_message:
            st.markdown(f'<div style="text-align:center; font-size: 0.95rem; font-weight:500; color:#475569; margin: 1rem 0;">{status_message}</div>', unsafe_allow_html=True)
        else:
            current_label = dict(PIPELINE_STEPS).get(st.session_state.current_step, "Running...")
            st.markdown(f'<div style="text-align:center; font-size: 0.95rem; font-weight:500; color:#475569; margin: 1rem 0;">{current_label}</div>', unsafe_allow_html=True)
            
        # 3. Scene timeline bar
        st.markdown(get_timeline_html(), unsafe_allow_html=True)
        
        # 4. Technical log expander
        with st.expander("View technical log"):
            logs = st.session_state.log_messages
            console_html = "".join([f"<div>> {m}</div>" for m in logs])
            st.markdown(f'<div class="console-view" style="height: 200px;">{console_html}</div>', unsafe_allow_html=True)


def show_results_screen():
    result = st.session_state.result
    if not result:
        return
        
    recap_text = (result.get("translated_recap") or result.get("final_recap") or result.get("recap") or "").strip()
    scene_count = result.get("scene_count", 0)
    selected_count = result.get("selected_scene_count", 0)
    active_scope = result.get("scope", "progress")
    active_range_start = result.get("range_start_sec")
    active_range_end = result.get("range_end_sec")
    progress_pct = result.get("progress_percent", 30)
    genre_preset = result.get("detected_genre", "auto").title()
    lang = result.get("target_language", "English").title()

    with main_container.container():
        st.markdown('<div class="app-title">Frame2Story</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-subtitle">Recap generated successfully</div>', unsafe_allow_html=True)
        
        # 1. 3 Metric Cards row
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="metric-card">'
                f'  <div class="metric-label">Scenes detected</div>'
                f'  <div class="metric-value">{scene_count}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f'<div class="metric-card">'
                f'  <div class="metric-label">Scenes selected</div>'
                f'  <div class="metric-value">{selected_count}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with c3:
            if active_scope == "timestamp-range" and active_range_start is not None and active_range_end is not None:
                val = f"{_format_seconds(active_range_start)} - {_format_seconds(active_range_end)}"
                lbl = "Time range"
            else:
                val = f"{progress_pct}%"
                lbl = "Watch progress"
            st.markdown(
                f'<div class="metric-card">'
                f'  <div class="metric-label">{lbl}</div>'
                f'  <div class="metric-value" style="font-size: 1.1rem; line-height: 2.1; font-weight: 700;">{val}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            
        st.write("")
        
        # 2. Badges container (top right of recap)
        st.markdown(
            f'<div class="badge-container">'
            f'  <div class="badge">{lang}</div>'
            f'  <div class="badge">{genre_preset} preset</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # 3. Recap bordered card
        st.markdown(
            f'<div class="flat-card">'
            f'  <div class="recap-content">{recap_text}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # 4. Audio player (TTS) if available
        audio_path = result.get("audio_narration_path")
        if audio_path and os.path.exists(audio_path):
            st.audio(audio_path, format="audio/mp3")
            
        # 5. Timeline bar showing final scene selections
        st.markdown(get_timeline_html(), unsafe_allow_html=True)
        
        # 6. Export row (4 equal width buttons)
        exp_c1, exp_c2, exp_c3, exp_c4 = st.columns(4)
        with exp_c1:
            st.download_button("Download txt", data=recap_text, file_name="movie_recap.txt", mime="text/plain", use_container_width=True)
        with exp_c2:
            md_export = f"# Movie Recap\n\n*Generated at {progress_pct}% watch progress*\n\n---\n\n{recap_text}\n"
            st.download_button("Download md", data=md_export, file_name="movie_recap.md", mime="text/markdown", use_container_width=True)
        with exp_c3:
            json_export = json.dumps({ "progress_pct": progress_pct, "recap": recap_text }, indent=2)
            st.download_button("Download json", data=json_export, file_name="movie_recap.json", mime="application/json", use_container_width=True)
        with exp_c4:
            if st.button("Copy text", use_container_width=True):
                st.code(recap_text, language=None)
                st.toast("Click the copy icon in the top right of the code box!", icon="📋")
                
        st.write("")
        
        # 7. Start over button
        if st.button("Run another recap", use_container_width=True, type="primary"):
            st.session_state.result = None
            st.session_state.current_step = None
            st.session_state.completed_steps = set()
            st.session_state.log_messages = []
            st.rerun()


def show_upload_screen():
    with main_container.container():
        st.markdown('<div class="app-title">Frame2Story</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-subtitle">Generate a movie recap based on how much you have watched</div>', unsafe_allow_html=True)
        
        if st.session_state.error:
            st.error(st.session_state.error)
            if st.button("Dismiss error"):
                st.session_state.error = ""
                st.rerun()
        
        # 1. Primary drag-and-drop file uploader
        movie_file = st.file_uploader(
            "Drop your video file here",
            type=["mp4", "mkv"],
            help="Select or drag and drop an .mp4 or .mkv video file.",
        )
        
        # 2. Subtitle file uploader secondary row
        subtitle_file = st.file_uploader(
            "Optional — auto-generated with Whisper if skipped",
            type=["srt"],
            help="Select or drag and drop an .srt subtitle file.",
        )
        
        # 3. Always-visible Watch progress slider
        progress = st.slider("Watch progress", 1, 100, 30, help="How far through the movie you've watched.")
        
        # 4. Advanced settings expander
        with st.expander("Advanced settings"):
            adv_c1, adv_c2 = st.columns(2)
            with adv_c1:
                summary_style = st.selectbox("Summary style", ["Concise", "Detailed"])
                perspective = st.selectbox(
                    "Summarization perspective",
                    ["Neutral", "Protagonist", "Antagonist"],
                    help="Changes the viewpoint of the recap summaries.",
                )
            with adv_c2:
                genre = st.selectbox(
                    "Film genre",
                    ["Auto", "Drama", "Action", "Documentary"],
                    help="Adjusts the fusion weights used to rank scenes.",
                )
                target_language = st.selectbox(
                    "Translation language",
                    ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Hindi", "Bengali", "Chinese (Simplified)", "Japanese", "Korean"],
                    help="Translate the final recap text to the selected language.",
                )
                
            st.divider()
            enable_tts = st.toggle(
                "Enable audio narration (TTS)",
                value=False,
                help="Generate an audio reading of your recap using text-to-speech.",
            )
            
            st.divider()
            use_timestamp = st.checkbox("Use a specific time range instead of progress %")
            
            start_ts_input = ""
            end_ts_input = ""
            if use_timestamp:
                ts_c1, ts_c2 = st.columns(2)
                with ts_c1:
                    start_ts_input = st.text_input(
                        "Start timestamp",
                        value="",
                        placeholder="e.g. 12:30",
                        help="Use seconds, MM:SS, or HH:MM:SS format.",
                    )
                with ts_c2:
                    end_ts_input = st.text_input(
                        "End timestamp",
                        value="",
                        placeholder="e.g. 19:45",
                        help="Use seconds, MM:SS, or HH:MM:SS format.",
                    )
                    
        # 5. Full width Generate recap CTA button
        generate_btn = st.button("Generate recap", use_container_width=True, type="primary")
        
        # Handle click
        if generate_btn:
            if not movie_file:
                st.error("Please upload a movie file to continue.")
                st.stop()
                
            validation_error = ""
            range_start_sec = None
            range_end_sec = None
            
            if use_timestamp:
                start_raw = start_ts_input.strip()
                end_raw = end_ts_input.strip()
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
                
            # Reset state & trigger execution state
            st.session_state.result = None
            st.session_state.error = ""
            st.session_state.log_messages = []
            st.session_state.current_step = PIPELINE_STEPS[0][0]
            st.session_state.completed_steps = set()
            
            # Save files
            movie_path = save_uploaded_file(movie_file)
            subtitle_path = save_uploaded_file(subtitle_file) if subtitle_file else None
            
            # Render initial processing screen
            show_processing_screen("Preparing video file...")
            
            from main_pipeline import run_pipeline
            
            def _on_progress(msg: str):
                st.session_state.log_messages.append(msg)
                detected = _step_from_message(msg)
                if detected:
                    if st.session_state.current_step and st.session_state.current_step != detected:
                        st.session_state.completed_steps.add(st.session_state.current_step)
                    st.session_state.current_step = detected
                show_processing_screen(msg)
                
            try:
                result = run_pipeline(
                    video_path=movie_path,
                    subtitle_path=subtitle_path,
                    progress=progress,
                    range_start_sec=range_start_sec,
                    range_end_sec=range_end_sec,
                    summary_style=summary_style,
                    fusion_preset=genre.lower(),
                    perspective=perspective,
                    run_evaluation=False,
                    progress_callback=_on_progress,
                    target_language=target_language,
                    enable_tts=enable_tts,
                )
                st.session_state.completed_steps = {s for s, _ in PIPELINE_STEPS}
                st.session_state.current_step = None
                st.session_state.result = result
                st.rerun()
                
            except Exception as exc:
                st.session_state.error = _friendly_error(exc)
                st.session_state.current_step = None
                st.rerun()
                
            finally:
                for p in (movie_path, subtitle_path):
                    if p and os.path.exists(p):
                        try:
                            os.unlink(p)
                        except OSError:
                            pass


# ── Main Loop Execution ────────────────────────────────────────────────────────

render_custom_css()

if st.session_state.result:
    show_results_screen()
elif st.session_state.current_step:
    show_processing_screen()
else:
    show_upload_screen()
