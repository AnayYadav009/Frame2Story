import os

def extract_audio(video_path, output_path = "data/audio.wav"):
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    command = f"ffmpeg -i {video_path} -q:a 0 -map a {output_path} -y"
    
    result = os.system(command)
    
    if result != 0:
        raise Exception("FFmpeg audio extraction failed")
    
    if not os.path.exists(output_path):
        raise Exception("Audio file was not created")
    
    return output_path

if __name__ == "__main__":
    video_path = "data/sample_video.mp4"
    audio_path = extract_audio(video_path)
    print("Audio extracted to:", audio_path)