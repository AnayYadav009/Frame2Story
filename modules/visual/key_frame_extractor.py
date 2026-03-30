import json
import math
from utils.video_reader import get_frame_at_index, save_frame, read_video_properties

import os

def load_scenes(json_path):
    with open(json_path, "r") as f:
        return json.load(f)
    
def get_keyframe_indices(scene, fps, frame_count):
    
    start = scene["start"]
    end = scene["end"]

    # Treat scene boundaries as [start, end) so adjacent scenes do not share keyframes.
    start_idx = math.ceil(start * fps)
    end_idx = math.floor(end * fps) - 1

    max_idx = max(frame_count - 1, 0)
    start_idx = max(0, min(start_idx, max_idx))
    end_idx = max(0, min(end_idx, max_idx))

    # Very short scenes can collapse after boundary adjustment.
    if end_idx < start_idx:
        end_idx = start_idx

    mid_idx = (start_idx + end_idx) // 2
    return [start_idx, mid_idx, end_idx]


def get_scene_keyframes(video_path, scene, fps, frame_count):
    """Return the three keyframes (first/middle/last) and their indices for one scene."""
    frame_indices = get_keyframe_indices(scene, fps, frame_count)
    frames = [get_frame_at_index(video_path, frame_idx) for frame_idx in frame_indices]
    return frame_indices, frames
    
def extract_keyframes(video_path, scenes, fps, frame_count, output_dir="data/keyframes"):
    os.makedirs(output_dir, exist_ok=True)
    
    for scene in scenes:
        scene_id = scene["scene_id"]
        
        frame_indices, frames = get_scene_keyframes(video_path, scene, fps, frame_count)
        
        for i, frame in enumerate(frames):
            filename = f"{output_dir}/scene_{scene_id}_frame_{i+1}.jpg"

            save_frame(frame, filename)
            
    print("Keyframes extracted successfully")
    
if __name__ == "__main__":
    video_path = "data/sample_video.mp4"
    scene_path = "data/scenes.json"

    info = read_video_properties(video_path)
    scenes = load_scenes(scene_path)

    extract_keyframes(video_path, scenes, info["fps"], info["frame_count"])