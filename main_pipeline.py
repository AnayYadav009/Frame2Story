from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, Any

from modules.scene.scene_pipeline import run_scene_pipeline, compute_scene_features
from modules.dialogue.dialogue_aligner import (
    load_subtitles,
    align_dialogue_to_scenes,
    detect_subtitle_language,
    save_scene_dialogues,
)
from modules.dialogue.dialogue_analyzer import (
    analyze_dialogues,
    extract_scene_speakers,
    save_dialogue_scores,
    save_scene_speakers,
)
from modules.summarization.scene_summarizer import summarize_all_scenes, save_scene_summaries
from modules.summarization.recap_generator import build_recap
from modules.evaluation.eval import evaluate_recap
from utils.fusion_engine import PRESETS, fusion_engine, save_fusion_output
from utils.input_handler import get_subtitle
from utils.scene_ranker import get_ranked_scenes, extract_scene_ids, save_selected_scenes


def _read_text_if_exists(path: Path) -> str | None:
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            return text or None
    except OSError:
        return None
    return None


def _reference_from_scene_dialogues(scene_dialogues: Any, ranked_scene_ids: list[Any]) -> str | None:
    if not isinstance(scene_dialogues, dict):
        return None

    def _entry_text(entry: Any) -> str:
        if isinstance(entry, dict):
            line = entry.get("line", "")
            if isinstance(line, str):
                return line.strip()
            return str(line).strip()
        if isinstance(entry, str):
            return entry.strip()
        return str(entry).strip()

    lines: list[str] = []
    for scene_id in ranked_scene_ids:
        entries = scene_dialogues.get(str(scene_id), scene_dialogues.get(scene_id, []))
        if isinstance(entries, list):
            lines.extend(line for line in (_entry_text(entry) for entry in entries) if line)
        elif isinstance(entries, str) and entries.strip():
            lines.append(entries.strip())

    merged = " ".join(lines).strip()
    return merged or None


def _resolve_reference_text(scene_dialogues: Any, ranked_scene_ids: list[Any]) -> str | None:
    # Prefer explicit/manual references if present.
    candidates = [
        Path("data/reference_summary.txt"),
        Path("data/reference_recap.txt"),
        Path("outputs/reference_summary.txt"),
        Path("outputs/reference_recap.txt"),
    ]

    for candidate in candidates:
        text = _read_text_if_exists(candidate)
        if text:
            return text

    # Fallback reference: selected raw dialogue.
    return _reference_from_scene_dialogues(scene_dialogues, ranked_scene_ids)


def _load_json_if_exists(path: Path) -> Any | None:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def _video_hash(
    video_path: str,
    progress: int | float,
    summary_style: str,
    fusion_preset: str = "auto",
) -> str:
    h = hashlib.sha256()
    h.update(str(progress).encode("utf-8"))
    h.update(summary_style.encode("utf-8"))
    h.update(fusion_preset.encode("utf-8"))
    with open(video_path, "rb") as f:
        h.update(f.read(2 * 1024 * 1024))
    return h.hexdigest()[:16]


def _resolve_fusion_preset(preset: str) -> tuple[str, Any]:
    key = (preset or "auto").strip().lower()
    if key not in PRESETS:
        key = "auto"
    return key, PRESETS[key]


def _cache_valid(path: Path, cache_key: str) -> bool:
    marker = path.parent / ".cache_key"
    if not path.exists() or not marker.exists():
        return False

    try:
        return marker.read_text(encoding="utf-8").strip() == cache_key
    except OSError:
        return False


def _write_cache_key(path: Path, cache_key: str) -> None:
    marker = path.parent / ".cache_key"
    marker.write_text(cache_key, encoding="utf-8")


def _normalize_language(value: Any, fallback: str = "en") -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized:
            return normalized
    return fallback


def run_full_pipeline(
    subtitle_path: str | None,
    video_path: str | None = None,
    percent_progress: int = 70,
    scene_gap: float = 5.0,
    summary_style: str = "Concise",
    fusion_preset: str = "auto",
    output_dir: str = "outputs",
    progress_callback: Callable[[str], None] | None = None,
) -> Dict[str, Any]:
    # scene_gap is kept for API compatibility with existing callers.
    _ = scene_gap

    def notify(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    # If a video_path is provided, extract/compute everything starting from the video.
    # If no video is provided (e.g., user did not upload a file), attempt to run
    # the summarization stage using existing intermediate files in `data/intermediate`.
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
    language_path = intermediate_dir / "subtitle_language.json"
    speakers_path = intermediate_dir / "scene_speakers.json"
    dialogue_scores_path = intermediate_dir / "dialogue_scores.json"
    summaries_path = intermediate_dir / "scene_summaries.json"
    fused_path = intermediate_dir / "fused_scores.json"
    selected_path = intermediate_dir / "selected_scenes.json"
    ranked_ids_path = intermediate_dir / "ranked_scene_ids.json"

    resolved_fusion_preset, fusion_weights = _resolve_fusion_preset(fusion_preset)

    if video_path is not None:
        cache_key = _video_hash(video_path, percent_progress, summary_style, resolved_fusion_preset)

        notify("Preparing subtitles...")
        subtitle_path = get_subtitle(video_path, subtitle_path)

        notify("Detecting scenes...")
        scenes = None
        if _cache_valid(scenes_path, cache_key):
            cached_scenes = _load_json_if_exists(scenes_path)
            if isinstance(cached_scenes, list):
                scenes = cached_scenes
                notify("Detecting scenes... (cached)")

        if scenes is None:
            scenes = run_scene_pipeline(
                video_path=video_path,
                progress_percentage=float(percent_progress),
                output_path=str(scenes_path),
            )
            _write_cache_key(scenes_path, cache_key)

        notify("Extracting visual features...")
        scene_features = None
        if _cache_valid(features_path, cache_key):
            cached_features = _load_json_if_exists(features_path)
            if isinstance(cached_features, (list, dict)):
                scene_features = cached_features
                notify("Extracting visual features... (cached)")

        if scene_features is None:
            scene_features = compute_scene_features(
                video_path=video_path,
                scenes_path=str(scenes_path),
                output_path=str(features_path),
                keyframes_dir="data/keyframes",
                save_keyframes=True,
            )
            _write_cache_key(features_path, cache_key)

        notify("Aligning dialogue...")
        scene_dialogues = None
        detected_language = "en"
        if _cache_valid(dialogues_path, cache_key):
            cached_dialogues = _load_json_if_exists(dialogues_path)
            if isinstance(cached_dialogues, dict):
                scene_dialogues = cached_dialogues
                notify("Aligning dialogue... (cached)")

        if _cache_valid(language_path, cache_key):
            cached_language_payload = _load_json_if_exists(language_path)
            if isinstance(cached_language_payload, dict):
                detected_language = _normalize_language(cached_language_payload.get("language"))
            elif isinstance(cached_language_payload, str):
                detected_language = _normalize_language(cached_language_payload)

        if scene_dialogues is None:
            subs = load_subtitles(subtitle_path)
            scene_dialogues, detected_language = align_dialogue_to_scenes(subs, scenes)
            save_scene_dialogues(scene_dialogues, str(dialogues_path))
            _write_cache_key(dialogues_path, cache_key)
            with language_path.open("w", encoding="utf-8") as f:
                json.dump({"language": detected_language}, f, indent=2)
            _write_cache_key(language_path, cache_key)
        elif subtitle_path and detected_language == "en" and not language_path.exists():
            # Backfill language metadata for old caches created before language propagation.
            try:
                subs = load_subtitles(subtitle_path)
                detected_language = detect_subtitle_language(subs)
            except Exception:
                detected_language = "en"

            with language_path.open("w", encoding="utf-8") as f:
                json.dump({"language": detected_language}, f, indent=2)
            _write_cache_key(language_path, cache_key)

        scene_speakers = extract_scene_speakers(scene_dialogues)
        save_scene_speakers(scene_speakers, str(speakers_path))
        _write_cache_key(speakers_path, cache_key)

        notify("Scoring and ranking scenes...")
        ranked_scene_ids = None
        if _cache_valid(ranked_ids_path, cache_key):
            cached_ranked_ids = _load_json_if_exists(ranked_ids_path)
            if isinstance(cached_ranked_ids, list):
                try:
                    ranked_scene_ids = [
                        int(s["scene_id"]) if isinstance(s, dict) else int(s)
                        for s in cached_ranked_ids
                    ]
                    notify("Scoring and ranking scenes... (cached)")
                except (TypeError, ValueError, KeyError):
                    ranked_scene_ids = None

        if ranked_scene_ids is None:
            dialogue_scores = None
            if _cache_valid(dialogue_scores_path, cache_key):
                cached_dialogue_scores = _load_json_if_exists(dialogue_scores_path)
                if isinstance(cached_dialogue_scores, dict):
                    dialogue_scores = cached_dialogue_scores

            if dialogue_scores is None:
                dialogue_scores = analyze_dialogues(scene_dialogues)
                save_dialogue_scores(dialogue_scores, str(dialogue_scores_path))
                _write_cache_key(dialogue_scores_path, cache_key)

            fused_scores = None
            if _cache_valid(fused_path, cache_key):
                cached_fused_scores = _load_json_if_exists(fused_path)
                if isinstance(cached_fused_scores, list):
                    fused_scores = cached_fused_scores

            if fused_scores is None:
                # Level-3 multimodal fusion uses dialogue density + motion intensity + object activity.
                fused_scores = fusion_engine(
                    scene_features,
                    dialogue_scores,
                    visual_data=scene_features,
                    weights=fusion_weights,
                )
                save_fusion_output(fused_scores, str(fused_path))
                _write_cache_key(fused_path, cache_key)

            ranked_scenes = get_ranked_scenes(fused_scores, threshold=0.3)
            ranked_scene_ids = extract_scene_ids(ranked_scenes)
            save_selected_scenes(ranked_scenes, str(selected_path))
            save_selected_scenes(ranked_scene_ids, str(ranked_ids_path))
            _write_cache_key(selected_path, cache_key)
            _write_cache_key(ranked_ids_path, cache_key)

        # Keep output mirror updated even when ranking stage was served from cache.
        save_selected_scenes(ranked_scene_ids, str(scenes_output_dir / "ranked_scene_ids.json"))

        notify("Summarizing scenes...")
        scene_summaries = None
        if _cache_valid(summaries_path, cache_key):
            cached_summaries = _load_json_if_exists(summaries_path)
            if isinstance(cached_summaries, dict):
                scene_summaries = cached_summaries
                notify("Summarizing scenes... (cached)")

        if scene_summaries is None:
            selected_dialogues = {
                str(scene_id): scene_dialogues.get(str(scene_id), [])
                for scene_id in ranked_scene_ids
            }
            scene_summaries = summarize_all_scenes(
                selected_dialogues,
                scene_features=scene_features,
                summary_style=summary_style,
                language=detected_language,
            )
            save_scene_summaries(scene_summaries, str(summaries_path))
            _write_cache_key(summaries_path, cache_key)

        # Keep output mirror updated even when summarization stage was served from cache.
        save_scene_summaries(scene_summaries, str(summaries_output_dir / "scene_summaries.json"))
    else:
        # No video provided: load intermediate artifacts and run recap generation.
        notify("Loading intermediate artifacts...")
        scenes = _load_json_if_exists(scenes_path) or []
        scene_features = _load_json_if_exists(features_path) or {}
        scene_dialogues = _load_json_if_exists(dialogues_path) or {}
        # ranked_scene_ids and scene_summaries are required for summarization
        ranked_scene_ids = _load_json_if_exists(ranked_ids_path)
        scene_summaries = _load_json_if_exists(summaries_path)

        if not ranked_scene_ids or not scene_summaries:
            raise FileNotFoundError(
                "Missing required intermediate files (ranked_scene_ids or scene_summaries). "
                "Provide a video file to run full pipeline or place the intermediate files in data/intermediate."
            )

        # Normalize to expected types
        try:
            ranked_scene_ids = [int(x) for x in ranked_scene_ids]
        except Exception:
            ranked_scene_ids = ranked_scene_ids

    notify("Generating final recap...")
    final_recap = build_recap(
        ranked_scene_ids,
        scene_summaries,
        scene_features=scene_features,
        summary_style=summary_style,
    )
    (final_output_dir / "final_recap.txt").write_text(final_recap, encoding="utf-8")
    with (final_output_dir / "final_recap.json").open("w", encoding="utf-8") as f:
        json.dump({"movie_recap": final_recap}, f, indent=2)

    eval_scores = None
    eval_error = None
    reference_text = _resolve_reference_text(scene_dialogues, ranked_scene_ids)
    if reference_text:
        notify("Evaluating recap quality...")
        eval_output_path = output_root / "eval" / "scores.json"
        try:
            eval_scores = evaluate_recap(
                generated_recap=final_recap,
                reference_text=reference_text,
                output_path=str(eval_output_path),
            )
        except Exception as exc:
            eval_error = str(exc)

    return {
        "subtitle_path": subtitle_path,
        "scene_count": len(scenes),
        "selected_scene_count": len(ranked_scene_ids),
        "final_recap": final_recap,
        "evaluation": eval_scores,
        "evaluation_error": eval_error,
    }


def run_pipeline(
    video_path: str | None = None,
    subtitle_path: str | None = None,
    progress: int = 70,
    summary_style: str = "Concise",
    fusion_preset: str = "auto",
    output_dir: str = "outputs",
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    """Backward-compatible wrapper used by the Streamlit app and legacy scripts."""
    result = run_full_pipeline(
        subtitle_path=subtitle_path,
        video_path=video_path,
        percent_progress=progress,
        summary_style=summary_style,
        fusion_preset=fusion_preset,
        output_dir=output_dir,
        progress_callback=progress_callback,
    )
    return result["final_recap"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run modular Frame2Story pipeline")
    parser.add_argument("--video", default="data/input/sample_video.mp4", help="Path to movie/video file")
    parser.add_argument("--subtitle", default="data/input/sample_himym.srt", help="Path to subtitle .srt file")
    parser.add_argument("--progress", type=int, default=40, help="Watch progress percentage")
    parser.add_argument("--summary_style", choices=["Concise", "Detailed"], default="Concise", help="Recap summary style")
    parser.add_argument(
        "--fusion_preset",
        choices=["auto", "drama", "action", "documentary"],
        default="auto",
        help="Fusion weight preset for scene ranking",
    )
    parser.add_argument("--output_dir", default="outputs", help="Final output directory")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_full_pipeline(
        subtitle_path=args.subtitle,
        video_path=args.video,
        percent_progress=args.progress,
        summary_style=args.summary_style,
        fusion_preset=args.fusion_preset,
        output_dir=args.output_dir,
    )
    print("\nFINAL RECAP:\n")
    print(result["final_recap"])


if __name__ == "__main__":
    main()
