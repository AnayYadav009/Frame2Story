import os
import subprocess

def extract_audio(video_path, output_path = "data/audio.wav"):
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-q:a",
        "0",
        "-map",
        "a",
        output_path,
        "-y",
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr_text = (exc.stderr or "").strip()
        raise RuntimeError(f"FFmpeg audio extraction failed: {stderr_text}") from exc
    
    if not os.path.exists(output_path):
        raise Exception("Audio file was not created")
    
    return output_path

if __name__ == "__main__":
    video_path = "data/sample_video.mp4"
    audio_path = extract_audio(video_path)
    print("Audio extracted to:", audio_path)