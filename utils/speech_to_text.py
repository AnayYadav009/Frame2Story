import os
from typing import Callable, Optional
from faster_whisper import WhisperModel


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
    """Transcribe audio to SRT using faster-whisper.

    Args:
        audio_path: Path to the audio file to transcribe.
        output_srt: Destination path for the generated .srt file.
        progress_callback: Optional callable that receives status strings,
            forwarded to the Streamlit UI (or any other caller).
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError("Audio file not found")

    if progress_callback:
        progress_callback("Loading faster-whisper model (base)…")

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel("base", device=device, compute_type=compute_type)
    except Exception:
        model = WhisperModel("base", device="cpu", compute_type="int8")

    if progress_callback:
        progress_callback("Transcribing audio with faster-whisper — this is optimized and fast…")

    segments, info = model.transcribe(audio_path, beam_size=5)

    os.makedirs(os.path.dirname(output_srt) if os.path.dirname(output_srt) else ".", exist_ok=True)
    with open(output_srt, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments):
            start = segment.start
            end = segment.end
            text = segment.text

            f.write(f"{i + 1}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{text.strip()}\n\n")

    if progress_callback:
        progress_callback("faster-whisper transcription complete.")

    return output_srt


if __name__ == "__main__":
    audio_path = "data/audio.wav"
    srt_path = transcribe_audio(audio_path)
    print("Subtitles generated at:", srt_path)