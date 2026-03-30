import os

from modules.scene.scene_detector import detect_scenes
from utils.video_reader import read_video_properties


def get_progress_time(duration, progress_percentage):
    if duration < 0:
        raise ValueError("Duration must be non-negative")
    if progress_percentage < 0 or progress_percentage > 100:
        raise ValueError("Progress percentage must be between 0 and 100")
    return duration * (progress_percentage / 100)


def _is_valid_scene(scene):
    if not isinstance(scene, dict):
        return False
    if "scene_id" not in scene or "start" not in scene or "end" not in scene:
        return False
    if not isinstance(scene["scene_id"], int):
        return False
    if not isinstance(scene["start"], (int, float)) or not isinstance(scene["end"], (int, float)):
        return False
    if scene["start"] < 0 or scene["end"] < 0:
        return False
    if scene["start"] > scene["end"]:
        return False
    return True


def filter_scenes_by_progress(scenes, progress_time):
    if scenes is None:
        return []
    if progress_time < 0:
        raise ValueError("Progress time must be non-negative")

    filtered = []
    for scene in scenes:
        if not _is_valid_scene(scene):
            continue
        if scene["end"] <= progress_time:
            filtered.append(scene)

    return filtered


def get_filtered_scenes_for_progress(video_path, progress_percentage, threshold=40.0):
    if not video_path or not isinstance(video_path, str):
        raise ValueError("Video path must be a non-empty string")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_data = read_video_properties(video_path)
    duration = video_data.get("duration_seconds", video_data["duration"])
    progress_time = get_progress_time(duration, progress_percentage)

    scenes = detect_scenes(video_path, threshold=threshold)
    return filter_scenes_by_progress(scenes, progress_time)