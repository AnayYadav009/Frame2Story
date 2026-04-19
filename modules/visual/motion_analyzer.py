import cv2
import numpy as np
import json
from utils.video_reader import get_frame_at_index, time_to_frame, read_video_properties

def compute_frame_difference(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray1, gray2)
    score = np.mean(diff)
    
    return score

def sample_frames(video_path, start_sec, end_sec, fps, sample_step_sec=1.0, reader=None):
    frames = []
    current_time = start_sec
    
    while current_time < end_sec:
        frame_idx = time_to_frame(current_time, fps)
        
        try:
            if reader:
                frame = reader.get_frame(frame_idx)
            else:
                frame = get_frame_at_index(video_path, frame_idx)
        except Exception:
            frame = None

        if frame is not None:
            frames.append(frame)
        current_time += sample_step_sec

    # Handle very short scenes by at least sampling the final timestamp once.
    if not frames and end_sec >= start_sec:
        try:
            last_idx = time_to_frame(end_sec, fps)
            if reader:
                frames.append(reader.get_frame(last_idx))
            else:
                frames.append(get_frame_at_index(video_path, last_idx))
        except Exception:
            pass
        
    return frames

def compute_scene_motion(video_path, scene, fps, sample_step_sec=1.0, reader=None):
    
    start = scene["start"]
    end = scene["end"]
    
    frames = sample_frames(video_path, start, end, fps, sample_step_sec=sample_step_sec, reader=reader)
    
    if len(frames) < 2:
        return 0
    
    diffs = []
    
    for i in range(len(frames) - 1):
        diff = compute_frame_difference(frames[i], frames[i+1])
        diffs.append(diff)
        
    return np.mean(diffs)


def classify_motion(score):
    if score < 10:
        return "LOW"
    elif score < 25:
        return "MEDIUM"
    else:
        return "HIGH"   
    
def analyze_scene_motion(video_path, scene, fps):
    
    score = compute_scene_motion(video_path, scene, fps)
    level = classify_motion(score)
    
    return {
        "scene_id": scene["scene_id"],
        "motion_score": score,
        "motion_level": level
    }
    
if __name__ == "__main__":
    video_path = "data/sample_video.mp4"

    with open("data/scenes.json", "r", encoding="utf-8") as f:
        scenes = json.load(f)

    info = read_video_properties(video_path)
    fps = info["fps"]

    for scene in scenes[:3]:
        result = analyze_scene_motion(video_path, scene, fps)
        print(result)