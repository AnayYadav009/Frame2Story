import whisper
import os
from typing import Callable, Optional


def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"


def transcribe_audio(
    audio_path: str,
    output_srt: str = "data/generated_subtitles.srt",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Transcribe audio to SRT using OpenAI Whisper.

    Args:
        audio_path: Path to the audio file to transcribe.
        output_srt: Destination path for the generated .srt file.
        progress_callback: Optional callable that receives status strings,
            forwarded to the Streamlit UI (or any other caller).
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError("Audio file not found")

    if progress_callback:
        progress_callback("Loading Whisper model (base)…")

    model = whisper.load_model("base")

    if progress_callback:
        progress_callback("Transcribing audio with Whisper — this may take a few minutes…")

    result = model.transcribe(audio_path)

    os.makedirs(os.path.dirname(output_srt) if os.path.dirname(output_srt) else ".", exist_ok=True)
    with open(output_srt, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"]):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]

            f.write(f"{i + 1}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{text.strip()}\n\n")

    if progress_callback:
        progress_callback("Whisper transcription complete.")

    return output_srt


if __name__ == "__main__":
    audio_path = "data/audio.wav"
    srt_path = transcribe_audio(audio_path)
    print("Subtitles generated at:", srt_path)