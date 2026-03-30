import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence

from modules.visual.key_frame_extractor import load_scenes, get_scene_keyframes
from modules.visual.motion_analyzer import compute_scene_motion, classify_motion
from modules.visual.object_detector import detect_scene_objects
from modules.visual.visual_analyzer import compute_importance_from_features
from modules.scene.scene_detector import save_scenes_to_json
from modules.scene.scene_filter import get_filtered_scenes_for_progress
from utils.video_reader import read_video_properties, save_frame


Scene = Dict[str, Any]


def run_scene_pipeline(
    video_path: str,
    progress_percentage: float,
    output_path: str = "data/scenes.json",
    threshold: float = 40.0,
) -> List[Scene]:
    scenes = get_filtered_scenes_for_progress(
        video_path=video_path,
        progress_percentage=progress_percentage,
        threshold=threshold,
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_scenes_to_json(scenes, str(output_file))
    return scenes


def compute_scene_features(
    video_path: str,
    scenes_path: str,
    output_path: str = "output/scene_features.json",
    keyframes_dir: str = "data/keyframes",
    save_keyframes: bool = True,
    motion_sample_step_sec: float = 1.0,
    yolo_model_name: str = "yolov8n.pt",
    yolo_confidence: float = 0.25,
    relevant_objects: Optional[Sequence[str]] = None,
) -> List[Scene]:
    """Run Week 2 visual pipeline: keyframes -> motion -> objects per scene."""
    scenes = load_scenes(scenes_path)
    video_info = read_video_properties(video_path)

    fps = video_info["fps"]
    frame_count = video_info["frame_count"]

    if save_keyframes:
        Path(keyframes_dir).mkdir(parents=True, exist_ok=True)

    scene_analysis = []
    for scene in scenes:
        scene_id = scene["scene_id"]
        duration_seconds = max(0.0, float(scene["end"] - scene["start"]))

        keyframe_indices, keyframe_frames = get_scene_keyframes(video_path, scene, fps, frame_count)
        keyframe_paths = []

        if save_keyframes:
            for i, frame in enumerate(keyframe_frames, start=1):
                keyframe_path = Path(keyframes_dir) / f"scene_{scene_id}_frame_{i}.jpg"
                save_frame(frame, str(keyframe_path))
                keyframe_paths.append(str(keyframe_path))

        motion_score = float(compute_scene_motion(video_path,scene,fps,sample_step_sec=motion_sample_step_sec))
        motion_level = classify_motion(motion_score)
        
        motion_score_normalized = min(max(motion_score / 50.0, 0.0), 1.0)

        objects = detect_scene_objects(
            keyframe_frames,
            model_name=yolo_model_name,
            confidence=yolo_confidence,
            relevant_objects=relevant_objects,
        )

        scene_analysis.append(
            {
                "scene_id": scene_id,
                "start": scene["start"],
                "end": scene["end"],
                "duration_seconds": duration_seconds,
                "keyframe_indices": keyframe_indices,
                "keyframe_paths": keyframe_paths,
                "motion_score": motion_score,
                "motion_level": motion_level,
                "motion_score_normalized": motion_score_normalized,
                "objects": objects,
            }
        )

    max_duration = max((scene["duration_seconds"] for scene in scene_analysis), default=0.0)
    for scene in scene_analysis:
        scene["importance"] = compute_importance_from_features(
            motion_score=scene["motion_score_normalized"],
            motion_level=scene["motion_level"],
            objects=scene["objects"],
            duration=scene["duration_seconds"],
            max_duration=max_duration,
        )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scene_analysis, f, indent=2)

    return scene_analysis


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run scene pipelines")
    parser.add_argument("--mode", choices=["scene-filter", "week2"], default="week2", help="Pipeline mode")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", default="output/scene_features.json", help="Path to output JSON")

    # Scene filter pipeline (Week 1 / Day 6 style)
    parser.add_argument("--progress", type=float, help="Progress percentage (0-100)")
    parser.add_argument("--threshold", type=float, default=40.0, help="ContentDetector threshold")

    # Week 2 multimodal pipeline
    parser.add_argument("--scenes", default="data/scenes.json", help="Path to scenes JSON")
    parser.add_argument("--keyframes-dir", default="data/keyframes", help="Directory to store extracted keyframes")
    parser.add_argument("--no-save-keyframes", action="store_true", help="Do not write keyframes to disk")
    parser.add_argument("--motion-step", type=float, default=1.0, help="Seconds between sampled frames for motion")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLO model name/path")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--relevant-objects", nargs="*", default=None, help="Optional label allow-list")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.mode == "scene-filter":
        if args.progress is None:
            raise ValueError("--progress is required when --mode scene-filter")

        scenes = run_scene_pipeline(
            video_path=args.video,
            progress_percentage=args.progress,
            output_path=args.output,
            threshold=args.threshold,
        )
        print(f"Saved {len(scenes)} filtered scenes to {args.output}")
        return

    scene_analysis = compute_scene_features(
        video_path=args.video,
        scenes_path=args.scenes,
        output_path=args.output,
        keyframes_dir=args.keyframes_dir,
        save_keyframes=not args.no_save_keyframes,
        motion_sample_step_sec=args.motion_step,
        yolo_model_name=args.yolo_model,
        yolo_confidence=args.yolo_conf,
        relevant_objects=args.relevant_objects,
    )
    print(f"Saved Week 2 analysis for {len(scene_analysis)} scenes to {args.output}")


if __name__ == "__main__":
    main()
