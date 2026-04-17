import json

from scenedetect import SceneManager, VideoManager
from scenedetect.detectors import ContentDetector


def timecode_to_seconds(timecode):
    return timecode.get_seconds()


def extract_scene_data(scene_list):
    scenes = []
    for i, (start, end) in enumerate(scene_list, start=1):
        scene_data = {
            "scene_id": i,
            "start": round(timecode_to_seconds(start), 3),
            "end": round(timecode_to_seconds(end), 3),
        }
        scenes.append(scene_data)

    return scenes


def detect_scenes(video_path, threshold=40.0):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.start()
    try:
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
    finally:
        video_manager.release()

    return extract_scene_data(scene_list)


def save_scenes_to_json(scenes, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scenes, f, indent=4)


if __name__ == "__main__":
    video_path = "data/sample_video.mp4"
    output_path = "data/scenes.json"
    scenes = detect_scenes(video_path)
    save_scenes_to_json(scenes, output_path)
    print(f"Saved {len(scenes)} scenes to {output_path}")