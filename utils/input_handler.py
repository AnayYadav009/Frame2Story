import os
from typing import Callable, Optional

from utils.audio_extractor import extract_audio
from utils.speech_to_text import transcribe_audio

def subtitle_exists(subtitle_path: Optional[str]) -> bool:
    return subtitle_path is not None and os.path.exists(subtitle_path)

def get_subtitle(
    video_path: str,
    subtitle_path: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Return a valid .srt path, generating via Whisper if none is provided.

    Args:
        video_path: Path to the source video file.
        subtitle_path: Optional pre-existing .srt file.
        progress_callback: Status callable forwarded to Whisper transcription.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError("Video file not found")

    if subtitle_exists(subtitle_path):
        return subtitle_path  # type: ignore[return-value]

    if progress_callback:
        progress_callback("No subtitle file provided — generating via Whisper…")

    audio_path = extract_audio(video_path)
    if not os.path.exists(audio_path):
        raise RuntimeError("Audio extraction failed")

    generated_subtitle = transcribe_audio(
        audio_path,
        progress_callback=progress_callback,
    )
    if not generated_subtitle or not os.path.exists(generated_subtitle):
        raise RuntimeError("Subtitle generation failed: expected file not found")

    return generated_subtitle


if __name__ == "__main__":
    video_path = "data/sample_video.mp4"
    result = get_subtitle(video_path)
    print("Subtitle used:", result)
