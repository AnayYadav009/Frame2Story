# Frame2Story

Frame2Story is a multimodal movie recap pipeline that generates a recap based on how much of a video has been watched.

Given a movie/video and watch progress, it:
- Detects scenes up to the watched timestamp.
- Extracts visual features (keyframes, motion, object detections).
- Aligns subtitle dialogue to scenes.
- Scores and ranks scenes using multimodal fusion.
- Summarizes selected scenes and produces a final recap.

The project includes both:
- A command-line pipeline (`main_pipeline.py`)
- A Streamlit app (`app.py`)

## Pipeline Overview

1. Scene Detection + Progress Filter
2. Visual Analysis
3. Dialogue Alignment + Dialogue Scoring
4. Multimodal Fusion + Scene Ranking
5. Scene Summarization
6. Final Recap Generation

## Requirements

- Python 3.10+
- FFmpeg in PATH (required when subtitles must be auto-generated from audio)
- Enough RAM/VRAM for model inference (BART + YOLO)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start (CLI)

Run the full pipeline:

```bash
python main_pipeline.py --video data/input/sample_video.mp4 --subtitle data/input/sample_himym.srt --progress 40 --output_dir outputs
```

Arguments:
- `--video`: path to input video file
- `--subtitle`: path to `.srt` subtitle file
- `--progress`: watch progress percentage (0-100)
- `--output_dir`: output root directory

If you do not have subtitles, pass an empty subtitle path and the pipeline will try to generate subtitles via Whisper:

```bash
python main_pipeline.py --video data/input/sample_video.mp4 --subtitle "" --progress 40 --output_dir outputs
```

## Streamlit App

Launch web app:

```bash
streamlit run app.py
```

Then upload:
- Movie file (`.mp4` or `.mkv`)
- Optional subtitle file (`.srt`)
- Select watch progress and generate recap

## Output Files

Typical outputs include:
- Intermediate data: `data/intermediate/*.json`
- Ranked scene IDs: `outputs/scenes/ranked_scene_ids.json`
- Scene summaries: `outputs/summaries/scene_summaries.json`
- Final recap text/json:
  - `outputs/final/final_recap.txt`
  - `outputs/final/final_recap.json`

## Notes

- `transformers` is pinned to `<5` for compatibility.
- `yolov8n.pt` can be provided locally or downloaded by Ultralytics when needed.
- First run may be slower due to model downloads and warm-up.

## Project Structure (High Level)

- `modules/scene`: scene detection and progress filtering
- `modules/visual`: keyframes, motion, object detection, visual scoring
- `modules/dialogue`: subtitle alignment and dialogue scoring
- `modules/summarization`: scene-level and final recap summarization
- `utils/`: shared helpers (I/O, audio extraction, fusion, ranking)
