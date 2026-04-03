import streamlit as st
import tempfile
import json
import os
from pathlib import Path

from main_pipeline import run_pipeline

st.title("🎬 AI Movie Recap System")

st.write("Generate a recap based on your watch progress.")

if "recap_text" not in st.session_state:
    st.session_state.recap_text = ""
if "recap_error" not in st.session_state:
    st.session_state.recap_error = ""
if "recap_progress" not in st.session_state:
    st.session_state.recap_progress = None
if "summary_style" not in st.session_state:
    st.session_state.summary_style = "Concise"

# Upload movie
movie_file = st.file_uploader("Upload Movie File", type=["mp4", "mkv"])

# Upload subtitle (optional)
subtitle_file = st.file_uploader("Upload Subtitle File (optional)", type=["srt"])

# Progress slider
progress = st.slider("Select Watch Progress (%)", 1, 100, 30)

# Optional summary style selector (reserved for future pipeline use)
summary_style = st.selectbox("Summary Style", ["Concise", "Detailed"])
st.session_state.summary_style = summary_style

# Button
generate = st.button("Generate Recap")

def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".tmp"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.read())
    temp_file.flush()
    temp_file.close()
    return temp_file.name

movie_path = None
subtitle_path = None

if movie_file:
    movie_path = save_uploaded_file(movie_file)

if subtitle_file:
    subtitle_path = save_uploaded_file(subtitle_file)
    
if generate:
    if not movie_file:
        st.session_state.recap_error = "Please upload a movie file."
    else:
        try:
            with st.spinner("🔄 Processing video, extracting scenes, analyzing visuals, and generating recap..."):
                recap = run_pipeline(
                    video_path=movie_path,
                    subtitle_path=subtitle_path,
                    progress=progress,
                    summary_style=summary_style,
                )
            st.session_state.recap_text = (recap or "").strip()
            st.session_state.recap_progress = progress
            st.session_state.recap_error = ""

        except Exception as e:
            st.session_state.recap_error = f"Error: {str(e)}"
        finally:
            for temp_path in (movie_path, subtitle_path):
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass

if st.session_state.recap_error:
    st.error(st.session_state.recap_error)
            
st.sidebar.title("About")

st.sidebar.write("""
AI-powered system that generates movie recaps using:
- Scene detection
- Visual understanding
- NLP summarization
""")

def format_scene_explanations(scene_features):
    explanations = []

    for scene in scene_features:
        scene_id = scene.get("scene_id")
        motion = scene.get("motion_level", "N/A")
        objects = scene.get("objects", [])
        importance = scene.get("importance", 0.0)

        explanation = {
            "scene_id": scene_id,
            "motion": motion,
            "objects": ", ".join(objects) if objects else "None",
            "importance": round(float(importance), 2),
        }

        explanations.append(explanation)

    return explanations


def load_scene_features_ui():
    candidate_paths = [
        "data/intermediate/scene_features.json",
        "outputs/scene_features.json",
        "output/scene_features.json",
        "data/scene_features.json",
    ]

    for path in candidate_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                return [
                    row
                    for row in payload.values()
                    if isinstance(row, dict) and "scene_id" in row
                ]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue

    return []


def get_importance_color(score):
    if score > 0.75:
        return "🟢 High"
    if score > 0.5:
        return "🟡 Medium"
    return "🔴 Low"


def infer_scene_type(motion, objects_text):
    motion_text = str(motion).upper()
    objects_lower = str(objects_text).lower()

    if any(token in objects_lower for token in ["gun", "knife", "weapon", "explosion"]):
        return "Conflict/High-stakes"
    if motion_text == "HIGH":
        return "Action"
    if motion_text == "MEDIUM":
        return "Transition"
    if objects_text == "None":
        return "Dialogue-focused"
    return "Setup/Context"


if st.session_state.recap_text:
    st.success("Recap Generated ✅")
    st.markdown("### 📄 Recap")
    st.write(st.session_state.recap_text)
    st.download_button(
        label="📥 Download Recap",
        data=st.session_state.recap_text,
        file_name="movie_recap.txt",
        mime="text/plain",
    )

    progress_to_show = (
        st.session_state.recap_progress
        if st.session_state.recap_progress is not None
        else progress
    )
    st.write(f"📊 Recap generated for **{progress_to_show}%** of the movie")

    st.subheader("🔍 Scene Explainability")

    scene_features = load_scene_features_ui()
    explanations = format_scene_explanations(scene_features)

    if not explanations:
        st.info("No scene features found yet. Generate a recap first to view explainability details.")
    else:
        for scene in explanations:
            with st.expander(f"Scene {scene['scene_id']}"):
                scene_type = infer_scene_type(scene["motion"], scene["objects"])
                st.write(f"**Motion:** {scene['motion']}")
                st.write(f"**Objects:** {scene['objects']}")
                st.write(f"**Importance Score:** {scene['importance']}")
                st.write(f"**Importance:** {get_importance_color(scene['importance'])} ({scene['importance']})")
                st.write(f"**Type:** {scene_type}")