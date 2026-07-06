import cv2
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


def is_flat_frame(frame, std_threshold=12.0, mean_threshold=15.0):
    """Checks if a frame is flat (low variance) or near-black (low mean)."""
    if frame is None:
        return True
    try:
        mean, std = cv2.meanStdDev(frame)
        return float(std[0][0]) < std_threshold or float(mean[0][0]) < mean_threshold
    except Exception:
        return False


def get_non_flat_frame(video_path, target_idx, start_bound, end_bound, reader=None):
    """Retrieve frame at target_idx, or search nearby offsets if target is flat/black."""
    if reader:
        frame = reader.get_frame(target_idx)
    else:
        try:
            frame = get_frame_at_index(video_path, target_idx)
        except Exception:
            frame = None

    if frame is not None and not is_flat_frame(frame):
        return target_idx, frame

    # Search window offsets in alternating directions, expanding outwards
    offsets = [1, -1, 3, -3, 5, -5, 10, -10, 15, -15]
    for offset in offsets:
        candidate_idx = target_idx + offset
        if start_bound <= candidate_idx <= end_bound:
            if reader:
                cand_frame = reader.get_frame(candidate_idx)
            else:
                try:
                    cand_frame = get_frame_at_index(video_path, candidate_idx)
                except Exception:
                    cand_frame = None
            if cand_frame is not None and not is_flat_frame(cand_frame):
                return candidate_idx, cand_frame

    return target_idx, frame


def get_scene_keyframes(video_path, scene, fps, frame_count, reader=None):
    """Return the three keyframes (first/middle/last) and their indices for one scene.
    
    If 'reader' is provided (VideoReader instance), it will be used for efficient 
    frame access. Uses contrast/variance checking to avoid flat or black frames.
    """
    default_indices = get_keyframe_indices(scene, fps, frame_count)
    
    start_bound = max(0, math.ceil(scene["start"] * fps))
    end_bound = min(frame_count - 1, math.floor(scene["end"] * fps) - 1)
    if end_bound < start_bound:
        end_bound = start_bound

    resolved_indices = []
    frames = []

    for idx in default_indices:
        res_idx, frame = get_non_flat_frame(
            video_path=video_path,
            target_idx=idx,
            start_bound=start_bound,
            end_bound=end_bound,
            reader=reader
        )
        resolved_indices.append(res_idx)
        frames.append(frame)

    return resolved_indices, frames

def extract_keyframes(video_path, scenes, fps, frame_count, output_dir="data/keyframes", reader=None):
    os.makedirs(output_dir, exist_ok=True)
    
    for scene in scenes:
        scene_id = scene["scene_id"]
        
        frame_indices, frames = get_scene_keyframes(video_path, scene, fps, frame_count, reader=reader)
        
        for i, frame in enumerate(frames):
            if frame is None:
                continue
            filename = f"{output_dir}/scene_{scene_id}_frame_{i+1}.jpg"
            save_frame(frame, filename)
            
    print("Keyframes extracted successfully")
    
if __name__ == "__main__":
    video_path = "data/sample_video.mp4"
    scene_path = "data/scenes.json"

    info = read_video_properties(video_path)
    scenes = load_scenes(scene_path)

    extract_keyframes(video_path, scenes, info["fps"], info["frame_count"])