import os
from utils.audio_extractor import extract_audio
from utils.speech_to_text import transcribe_audio

def subtitle_exists(subtitle_path):
    return subtitle_path is not None and os.path.exists(subtitle_path)

def get_subtitle(video_path, subtitle_path=None):
    
    if not os.path.exists(video_path):
        raise FileNotFoundError("Video file not found")
    
    if subtitle_exists(subtitle_path):
        print("Using provided subtitle file")
        return subtitle_path
    print("No subtitle file exists. Generating using Whisper...")
    
    audio_path = extract_audio(video_path)
    if not os.path.exists(audio_path):
        raise Exception("Audio extraction failed")
    
    generated_subtitle = transcribe_audio(audio_path)
    if not generated_subtitle or not os.path.exists(generated_subtitle):
        raise Exception("Subtitle generation failed: expected file not found")

    return generated_subtitle

if __name__ == "__main__":
    video_path = "data/sample_video.mp4"

    # Case 1: With subtitle
    # subtitle_path = "data/sample_himym.srt"

    result = get_subtitle(video_path)

    print("Subtitle used:", result)