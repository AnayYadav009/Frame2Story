from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Dict, Any

from modules.scene.scene_pipeline import run_scene_pipeline, compute_scene_features
from modules.dialogue.dialogue_aligner import (
    load_subtitles,
    align_dialogue_to_scenes,
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
from modules.fusion.fusion_engine import PRESETS, fusion_engine, save_fusion_output, detect_genre_preset
from utils.input_handler import get_subtitle
from utils.scene_ranker import get_ranked_scenes, extract_scene_ids, save_selected_scenes

_RANKING_VERSION = "v5-robust-summaries"
_FUSION_VERSION = _RANKING_VERSION


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

    return _reference_from_scene_dialogues(scene_dialogues, ranked_scene_ids)


def _load_json_if_exists(path: Path) -> Any | None:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def _get_base_key(video_path: str) -> str:
    """Stable key for visual features (video content + version)."""
    h = hashlib.sha256()
    h.update(_RANKING_VERSION.encode("utf-8"))
    with open(video_path, "rb") as f:
        h.update(f.read(2 * 1024 * 1024))
    return h.hexdigest()[:16]


def _video_hash(
    video_path: str,
    progress: int | float,
    summary_style: str,
    fusion_preset: str = "auto",
    perspective: str = "Neutral",
) -> str:
    """Legacy hash for backward compatibility if needed. Now using split keys."""
    base = _get_base_key(video_path)
    h = hashlib.sha256(base.encode("utf-8"))
    h.update(str(progress).encode("utf-8"))
    h.update(summary_style.encode("utf-8"))
    h.update(fusion_preset.encode("utf-8"))
    h.update(perspective.encode("utf-8"))
    return h.hexdigest()[:16]


def _scope_key_fragment(
    percent_progress: int | float,
    range_start_sec: float | None,
    range_end_sec: float | None,
) -> str:
    if range_start_sec is not None and range_end_sec is not None:
        return f"range:{float(range_start_sec):.3f}-{float(range_end_sec):.3f}"
    return f"progress:{float(percent_progress):.3f}"


def _resolve_fusion_preset(preset: str) -> tuple[str, Any]:
    key = (preset or "auto").strip().lower()
    if key not in PRESETS:
        key = "auto"
    return key, PRESETS[key]


def _cache_marker_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".cache_key")


def _cache_valid(path: Path, cache_key: str) -> bool:
    marker = _cache_marker_path(path)
    if not path.exists() or not marker.exists():
        return False

    try:
        return marker.read_text(encoding="utf-8").strip() == cache_key
    except OSError:
        return False


def _write_cache_key(path: Path, cache_key: str) -> None:
    marker = _cache_marker_path(path)
    marker.write_text(cache_key, encoding="utf-8")


def _normalize_language(value: Any, fallback: str = "en") -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized:
            return normalized
    return fallback


def run_full_pipeline(
    subtitle_path: str | None = None,
    video_path: str | None = None,
    percent_progress: int | float = 70,
    range_start_sec: float | None = None,
    range_end_sec: float | None = None,
    summary_style: str = "Concise",
    fusion_preset: str = "auto",
    perspective: str = "Neutral",
    output_dir: str = "outputs",
    run_evaluation: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    custom_weights: Any | None = None,
) -> Dict[str, Any]:
    timings = {}
    
    def notify(message: str) -> None:
        if progress_callback:
            progress_callback(message)

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
    rationale_path = intermediate_dir / "scene_rationale.json"

    resolved_fusion_preset, fusion_weights = _resolve_fusion_preset(fusion_preset)

    if (range_start_sec is None) != (range_end_sec is None):
        raise ValueError("Both range_start_sec and range_end_sec must be provided together")

    if video_path is not None:
        # 1. Base Key: Video + Ranking Version (for visual features)
        base_key = _get_base_key(video_path)
        scope_fragment = _scope_key_fragment(percent_progress, range_start_sec, range_end_sec)
        
        scene_h = hashlib.sha256(base_key.encode("utf-8"))
        scene_h.update(scope_fragment.encode("utf-8"))
        scene_key = scene_h.hexdigest()[:16]
        
        # 2. Dialogue Key: scene_key + ... (for dialogue alignment/scores)
        # (Using scene_key as base ensures dialogue also shifts with progress)
        dialogue_h = hashlib.sha256(scene_key.encode("utf-8"))
        dialogue_key = dialogue_h.hexdigest()[:16]

        # 3. Fusion Key: Dialogue + Preset (for ranking)
        fusion_h = hashlib.sha256(dialogue_key.encode("utf-8"))
        fusion_h.update(resolved_fusion_preset.encode("utf-8"))
        fusion_key = fusion_h.hexdigest()[:16]

        # 4. Summary Key: Fusion + Style + Perspective (for final recap)
        summary_h = hashlib.sha256(fusion_key.encode("utf-8"))
        summary_h.update(summary_style.encode("utf-8"))
        summary_h.update(perspective.encode("utf-8"))
        summary_key = summary_h.hexdigest()[:16]

        # Optimize: Check if dialogue is already cached to avoid Whisper work
        if not subtitle_path and _cache_valid(dialogues_path, dialogue_key):
            notify("Using cached dialogue artifacts...")
        else:
            notify("Preparing subtitles...")
            t_start = time.perf_counter()
            subtitle_path = get_subtitle(video_path, subtitle_path, progress_callback=notify)
            timings["subtitles"] = time.perf_counter() - t_start

        notify("Detecting scenes...")
        scenes = None
        t_start = time.perf_counter()
        if _cache_valid(scenes_path, scene_key):
            cached_scenes = _load_json_if_exists(scenes_path)
            if isinstance(cached_scenes, list):
                scenes = cached_scenes
                notify("Detecting scenes... (cached)")

        if scenes is None:
            scenes = run_scene_pipeline(
                video_path=video_path,
                progress_percentage=float(percent_progress),
                start_time_sec=range_start_sec,
                end_time_sec=range_end_sec,
                output_path=str(scenes_path),
            )
            _write_cache_key(scenes_path, scene_key)
            timings["scene_detection"] = time.perf_counter() - t_start
        else:
            timings["scene_detection"] = 0.0

        notify("Extracting visual features...")
        scene_features = None
        t_start = time.perf_counter()
        if _cache_valid(features_path, scene_key):
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
                cleanup_keyframes=True,
            )
            _write_cache_key(features_path, scene_key)
            timings["visual_features"] = time.perf_counter() - t_start
        else:
            timings["visual_features"] = 0.0

        notify("Aligning dialogue...")
        scene_dialogues = None
        detected_language = "en"
        t_start = time.perf_counter()
        if _cache_valid(dialogues_path, dialogue_key):
            cached_dialogues = _load_json_if_exists(dialogues_path)
            if isinstance(cached_dialogues, dict):
                scene_dialogues = cached_dialogues
                notify("Aligning dialogue... (cached)")

        if _cache_valid(language_path, dialogue_key):
            cached_language_payload = _load_json_if_exists(language_path)
            if isinstance(cached_language_payload, dict):
                detected_language = _normalize_language(cached_language_payload.get("language"))
            elif isinstance(cached_language_payload, str):
                detected_language = _normalize_language(cached_language_payload)

        if scene_dialogues is None:
            subs = load_subtitles(subtitle_path)
            scene_dialogues, detected_language = align_dialogue_to_scenes(subs, scenes)
            save_scene_dialogues(scene_dialogues, str(dialogues_path))
            _write_cache_key(dialogues_path, dialogue_key)
            with language_path.open("w", encoding="utf-8") as f:
                json.dump({"language": detected_language}, f, indent=2)
            _write_cache_key(language_path, dialogue_key)
            timings["dialogue_alignment"] = time.perf_counter() - t_start
        else:
            timings["dialogue_alignment"] = 0.0

        scene_speakers = extract_scene_speakers(scene_dialogues)
        save_scene_speakers(scene_speakers, str(speakers_path))
        _write_cache_key(speakers_path, dialogue_key)

        notify("Scoring and ranking scenes...")
        ranked_scene_ids = None
        fused_scores = None
        t_start = time.perf_counter()
        if _cache_valid(ranked_ids_path, fusion_key):
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
            if _cache_valid(dialogue_scores_path, dialogue_key):
                cached_dialogue_scores = _load_json_if_exists(dialogue_scores_path)
                if isinstance(cached_dialogue_scores, dict):
                    dialogue_scores = cached_dialogue_scores

            if dialogue_scores is None:
                dialogue_scores = analyze_dialogues(scene_dialogues)
                save_dialogue_scores(dialogue_scores, str(dialogue_scores_path))
                _write_cache_key(dialogue_scores_path, dialogue_key)

            fused_scores = None
            if _cache_valid(fused_path, fusion_key):
                cached_fused_scores = _load_json_if_exists(fused_path)
                if isinstance(cached_fused_scores, list):
                    fused_scores = cached_fused_scores

            effective_preset = resolved_fusion_preset
            if fused_scores is None:
                if custom_weights is not None:
                    effective_weights = custom_weights
                    notify("Using custom fusion weights from Pro Mode")
                elif resolved_fusion_preset == "auto":
                    effective_preset = detect_genre_preset(scene_features, dialogue_scores)
                    notify(f"Inferred genre: {effective_preset.title()}")
                    _, effective_weights = _resolve_fusion_preset(effective_preset)
                else:
                    _, effective_weights = _resolve_fusion_preset(effective_preset)

                fused_scores = fusion_engine(
                    scene_features,
                    dialogue_scores,
                    visual_data=scene_features,
                    weights=effective_weights,
                )
                save_fusion_output(fused_scores, str(fused_path))
                _write_cache_key(fused_path, fusion_key)

            watched_duration_sec = max((s.get("end", 0) for s in scenes), default=0) if scenes else 0
            ranked_scenes = get_ranked_scenes(
                fused_scores,
                threshold=0.3,
                watched_duration_sec=watched_duration_sec,
            )
            ranked_scene_ids = extract_scene_ids(ranked_scenes)
            save_selected_scenes(ranked_scenes, str(selected_path))
            save_selected_scenes(ranked_scene_ids, str(ranked_ids_path))
            _write_cache_key(selected_path, fusion_key)
            _write_cache_key(ranked_ids_path, fusion_key)
            timings["ranking"] = time.perf_counter() - t_start
        else:
            timings["ranking"] = 0.0

        if not _cache_valid(rationale_path, fusion_key):
            fused_for_rationale = fused_scores or (_load_json_if_exists(fused_path) or [])
            if fused_for_rationale:
                selected_id_set = set(ranked_scene_ids)
                rationale = [
                    {
                        "scene_id": scene["scene_id"],
                        "selected": scene["scene_id"] in selected_id_set,
                        "final_score": scene["final"],
                        "dialogue_score": scene["dialogue_score"],
                        "motion_score": scene["motion_score"],
                        "object_score": scene["object_score"],
                        "visual_score": scene.get("visual_score", 0.0),
                    }
                    for scene in fused_for_rationale
                ]
                with rationale_path.open("w", encoding="utf-8") as f:
                    json.dump(rationale, f, indent=2)
                _write_cache_key(rationale_path, fusion_key)

        save_selected_scenes(ranked_scene_ids, str(scenes_output_dir / "ranked_scene_ids.json"))

        notify("Summarizing scenes...")
        scene_summaries = None
        t_start = time.perf_counter()
        if _cache_valid(summaries_path, summary_key):
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
                perspective=perspective,
            )
            save_scene_summaries(scene_summaries, str(summaries_path))
            _write_cache_key(summaries_path, summary_key)
            timings["scene_summarization"] = time.perf_counter() - t_start
        else:
            timings["scene_summarization"] = 0.0

        save_scene_summaries(scene_summaries, str(summaries_output_dir / "scene_summaries.json"))
    else:
        notify("Loading intermediate artifacts...")
        scenes = _load_json_if_exists(scenes_path) or []
        scene_features = _load_json_if_exists(features_path) or {}
        scene_dialogues = _load_json_if_exists(dialogues_path) or {}
        ranked_scene_ids = _load_json_if_exists(ranked_ids_path)
        scene_summaries = _load_json_if_exists(summaries_path)

        if not ranked_scene_ids or not scene_summaries:
            raise FileNotFoundError(
                "Missing required intermediate files (ranked_scene_ids or scene_summaries). "
                "Provide a video file to run full pipeline or place the intermediate files in data/intermediate."
            )

        try:
            ranked_scene_ids = [int(x) for x in ranked_scene_ids]
        except Exception:
            ranked_scene_ids = ranked_scene_ids

    notify("Generating final recap...")
    t_start = time.perf_counter()
    final_recap = build_recap(
        ranked_scene_ids,
        scene_summaries,
        scene_features=scene_features,
        summary_style=summary_style,
    )
    (final_output_dir / "final_recap.txt").write_text(final_recap, encoding="utf-8")
    with (final_output_dir / "final_recap.json").open("w", encoding="utf-8") as f:
        json.dump({"movie_recap": final_recap}, f, indent=2)
    timings["final_recap"] = time.perf_counter() - t_start

    eval_scores = None
    eval_error = None
    reference_text = _resolve_reference_text(scene_dialogues, ranked_scene_ids)
    
    if reference_text and run_evaluation:
        notify("Evaluating recap quality...")
        eval_output_path = output_root / "eval" / "scores.json"
        t_start = time.perf_counter()
        try:
            eval_scores = evaluate_recap(
                generated_recap=final_recap,
                reference_text=reference_text,
                output_path=str(eval_output_path),
            )
            timings["evaluation"] = time.perf_counter() - t_start
        except Exception as exc:
            eval_error = str(exc)

    return {
        "subtitle_path": subtitle_path,
        "scene_count": len(scenes),
        "selected_scene_count": len(ranked_scene_ids),
        "scope": "timestamp-range" if (range_start_sec is not None and range_end_sec is not None) else "progress",
        "range_start_sec": range_start_sec,
        "range_end_sec": range_end_sec,
        "progress_percent": percent_progress,
        "final_recap": final_recap,
        "reference_text": reference_text,
        "evaluation": eval_scores,
        "evaluation_error": eval_error,
        "timings": timings,
        "detected_genre": effective_preset if 'effective_preset' in locals() else resolved_fusion_preset,
    }


def run_pipeline(
    video_path: str | None = None,
    subtitle_path: str | None = None,
    progress: int = 70,
    range_start_sec: float | None = None,
    range_end_sec: float | None = None,
    summary_style: str = "Concise",
    fusion_preset: str = "auto",
    perspective: str = "Neutral",
    output_dir: str = "outputs",
    run_evaluation: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    custom_weights: Any | None = None,
) -> Dict[str, Any]:
    """Wrapper used by the Streamlit app.

    Returns the full result dict (same as run_full_pipeline) so the app
    can surface evaluation scores, scene counts, and other metadata.
    A ``recap`` convenience key mirrors ``final_recap`` for old callers.
    """
    result = run_full_pipeline(
        subtitle_path=subtitle_path,
        video_path=video_path,
        percent_progress=progress,
        range_start_sec=range_start_sec,
        range_end_sec=range_end_sec,
        summary_style=summary_style,
        fusion_preset=fusion_preset,
        perspective=perspective,
        output_dir=output_dir,
        run_evaluation=run_evaluation,
        progress_callback=progress_callback,
        custom_weights=custom_weights,
    )
    # Convenience alias so any legacy caller doing result["recap"] still works.
    result.setdefault("recap", result.get("final_recap", ""))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run modular Frame2Story pipeline")
    parser.add_argument("--video", default="data/input/sample_video.mp4", help="Path to movie/video file")
    parser.add_argument("--subtitle", default="data/input/sample_himym.srt", help="Path to subtitle .srt file")
    parser.add_argument("--progress", type=int, default=40, help="Watch progress percentage")
    parser.add_argument("--start_ts", type=float, default=None, help="Optional start timestamp (seconds)")
    parser.add_argument("--end_ts", type=float, default=None, help="Optional end timestamp (seconds)")
    parser.add_argument("--summary_style", choices=["Concise", "Detailed"], default="Concise", help="Recap summary style")
    parser.add_argument(
        "--fusion_preset",
        choices=["auto", "drama", "action", "documentary"],
        default="auto",
        help="Fusion weight preset for scene ranking",
    )
    parser.add_argument("--output_dir", default="outputs", help="Final output directory")
    parser.add_argument("--eval", action="store_true", help="Enable ROUGE/BERTScore evaluation")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_full_pipeline(
        subtitle_path=args.subtitle,
        video_path=args.video,
        percent_progress=args.progress,
        range_start_sec=args.start_ts,
        range_end_sec=args.end_ts,
        summary_style=args.summary_style,
        fusion_preset=args.fusion_preset,
        output_dir=args.output_dir,
        run_evaluation=args.eval,
    )
    print("\nFINAL RECAP:\n")
    print(result["final_recap"])


if __name__ == "__main__":
    main()