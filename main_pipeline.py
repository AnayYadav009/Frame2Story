from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from modules.scene.scene_pipeline import run_scene_pipeline, compute_scene_features
from modules.dialogue.dialogue_aligner import load_subtitles, align_dialogue_to_scenes, save_scene_dialogues
from modules.dialogue.dialogue_analyzer import analyze_dialogues, save_dialogue_scores
from modules.summarization.scene_summarizer import summarize_all_scenes, save_scene_summaries
from modules.summarization.recap_generator import build_recap
from utils.fusion_engine import fusion_engine, save_fusion_output
from utils.input_handler import get_subtitle
from utils.scene_ranker import get_ranked_scenes, extract_scene_ids, save_selected_scenes


def run_full_pipeline(
    subtitle_path: str | None,
    video_path: str = "C:/Users/anayy/OneDrive/Desktop/Video Project 2.mp4",
    percent_progress: int = 70,
    scene_gap: float = 5.0,
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    # scene_gap is kept for API compatibility with existing callers.
    _ = scene_gap

    subtitle_path = get_subtitle(video_path, subtitle_path)

    intermediate_dir = Path("data/intermediate")
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    output_root = Path(output_dir)
    scenes_output_dir = output_root / "scenes"
    summaries_output_dir = output_root / "summaries"
    final_output_dir = output_root / "final"
    scenes_output_dir.mkdir(parents=True, exist_ok=True)
    summaries_output_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    scenes_path = intermediate_dir / "scenes.json"
    features_path = intermediate_dir / "scene_features.json"
    dialogues_path = intermediate_dir / "scene_dialogues.json"
    dialogue_scores_path = intermediate_dir / "dialogue_scores.json"
    summaries_path = intermediate_dir / "scene_summaries.json"
    fused_path = intermediate_dir / "fused_scores.json"
    selected_path = intermediate_dir / "selected_scenes.json"
    ranked_ids_path = intermediate_dir / "ranked_scene_ids.json"

    scenes = run_scene_pipeline(
        video_path=video_path,
        progress_percentage=float(percent_progress),
        output_path=str(scenes_path),
    )

    scene_features = compute_scene_features(
        video_path=video_path,
        scenes_path=str(scenes_path),
        output_path=str(features_path),
        keyframes_dir="data/keyframes",
        save_keyframes=True,
    )

    subs = load_subtitles(subtitle_path)
    scene_dialogues = align_dialogue_to_scenes(subs, scenes)
    save_scene_dialogues(scene_dialogues, str(dialogues_path))

    dialogue_scores = analyze_dialogues(scene_dialogues)
    save_dialogue_scores(dialogue_scores, str(dialogue_scores_path))

    # Level-3 multimodal fusion uses dialogue density + motion intensity + object activity.
    fused_scores = fusion_engine(scene_features, dialogue_scores, visual_data=scene_features)
    save_fusion_output(fused_scores, str(fused_path))

    ranked_scenes = get_ranked_scenes(fused_scores, threshold=0.3)
    ranked_scene_ids = extract_scene_ids(ranked_scenes)
    save_selected_scenes(ranked_scenes, str(selected_path))
    save_selected_scenes(ranked_scene_ids, str(ranked_ids_path))
    save_selected_scenes(ranked_scene_ids, str(scenes_output_dir / "ranked_scene_ids.json"))

    # Summarize selected scenes only, after ranking.
    selected_dialogues = {str(scene_id): scene_dialogues.get(str(scene_id), []) for scene_id in ranked_scene_ids}
    scene_summaries = summarize_all_scenes(selected_dialogues, scene_features=scene_features)
    save_scene_summaries(scene_summaries, str(summaries_path))
    save_scene_summaries(scene_summaries, str(summaries_output_dir / "scene_summaries.json"))

    final_recap = build_recap(ranked_scene_ids, scene_summaries, scene_features=scene_features)
    (final_output_dir / "final_recap.txt").write_text(final_recap, encoding="utf-8")
    with (final_output_dir / "final_recap.json").open("w", encoding="utf-8") as f:
        json.dump({"movie_recap": final_recap}, f, indent=2)

    return {
        "subtitle_path": subtitle_path,
        "scene_count": len(scenes),
        "selected_scene_count": len(ranked_scene_ids),
        "final_recap": final_recap,
    }


def run_pipeline(
    video_path: str,
    subtitle_path: str | None = None,
    progress: int = 70,
    output_dir: str = "outputs",
) -> str:
    """Backward-compatible wrapper used by the Streamlit app and legacy scripts."""
    result = run_full_pipeline(
        subtitle_path=subtitle_path,
        video_path=video_path,
        percent_progress=progress,
        output_dir=output_dir,
    )
    return result["final_recap"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run modular Frame2Story pipeline")
    parser.add_argument("--video", default="data/input/sample_video.mp4", help="Path to movie/video file")
    parser.add_argument("--subtitle", default="data/input/sample_himym.srt", help="Path to subtitle .srt file")
    parser.add_argument("--progress", type=int, default=40, help="Watch progress percentage")
    parser.add_argument("--output_dir", default="outputs", help="Final output directory")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_full_pipeline(
        subtitle_path=args.subtitle,
        video_path=args.video,
        percent_progress=args.progress,
        output_dir=args.output_dir,
    )
    print("\nFINAL RECAP:\n")
    print(result["final_recap"])


if __name__ == "__main__":
    main()
