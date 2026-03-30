import streamlit as st
import tempfile
from pathlib import Path

from main_pipeline import run_pipeline

st.title("🎬 AI Movie Recap System")

st.write("Generate a recap based on your watch progress.")

if "recap_text" not in st.session_state:
    st.session_state.recap_text = ""
if "recap_error" not in st.session_state:
    st.session_state.recap_error = ""

# Upload movie
movie_file = st.file_uploader("Upload Movie File", type=["mp4", "mkv"])

# Upload subtitle (optional)
subtitle_file = st.file_uploader("Upload Subtitle File (optional)", type=["srt"])

# Progress slider
progress = st.slider("Select Watch Progress (%)", 1, 100, 30)

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
            with st.spinner("Processing..."):
                recap = run_pipeline(
                    video_path=movie_path,
                    subtitle_path=subtitle_path,
                    progress=progress,
                )
            st.session_state.recap_text = (recap or "").strip()
            st.session_state.recap_error = ""

        except Exception as e:
            st.session_state.recap_error = f"Error: {str(e)}"

if st.session_state.recap_error:
    st.error(st.session_state.recap_error)

if st.session_state.recap_text:
    st.success("Recap Generated ✅")
    st.subheader("📄 Recap")
    st.text_area("Generated recap", value=st.session_state.recap_text, height=260)
            
st.sidebar.title("About")

st.sidebar.write("""
AI-powered system that generates movie recaps using:
- Scene detection
- Visual understanding
- NLP summarization
""")