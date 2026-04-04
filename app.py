import streamlit as st
import tempfile
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

genre = st.selectbox(
    "Film genre (affects scene ranking)",
    ["Auto", "Drama", "Action", "Documentary"],
)
fusion_preset = genre.lower()

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
        status = st.status("Running pipeline...", expanded=True)
        try:
            def update_progress(msg: str):
                status.write(msg)

            recap = run_pipeline(
                video_path=movie_path,
                subtitle_path=subtitle_path,
                progress=progress,
                summary_style=summary_style,
                fusion_preset=fusion_preset,
                progress_callback=update_progress,
            )
            status.update(label="Done", state="complete", expanded=False)
            st.session_state.recap_text = (recap or "").strip()
            st.session_state.recap_progress = progress
            st.session_state.recap_error = ""

        except Exception as e:
            status.update(label="Pipeline failed", state="error", expanded=True)
            status.write(str(e))
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