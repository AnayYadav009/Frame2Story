import whisper
import os

def format_time(seconds):

    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)

    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

def transcribe_audio(audio_path, output_srt = "data/generated_subtitles.srt"):
    if not os.path.exists(audio_path):
        raise FileNotFoundError("Audio file not found")
    
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    
    with open(output_srt, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"]):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            
            f.write(f"{i+1}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}")
            f.write(f"{text.strip()}\n\n")
            
    return output_srt

if __name__ == "__main__":
    audio_path = "data/audio.wav"
    srt_path = transcribe_audio(audio_path)
    print("Subtitles generated at:", srt_path)