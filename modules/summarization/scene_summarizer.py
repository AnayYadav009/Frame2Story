import json
import re
from pathlib import Path
from typing import Any, Dict, List

from modules.summarization.extractive_summarizer import extractive_summary_from_text
from modules.summarization.abstractive_summarizer import (summarize_text_batch)

MIN_DIALOGUE_WORDS = 8

WHISPER_NOISE_PATTERNS = [
    r"\[.*?\]",           # [Music], [Laughter]
    r"\(.*?\)",           # (Background noise)
    r"♪.*?♪",             # Musical notes
    r"Thank you for watching",
    r"Please subscribe",
    r"Subtitles by",
]


def load_scene_dialogues(path):
    with open(path, "r") as f:
        return json.load(f)


def load_scene_features(path="output/scene_features.json"):
    if not Path(path).exists():
        return {}

    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return {str(scene["scene_id"]): scene for scene in data if isinstance(scene, dict) and "scene_id" in scene}
    if isinstance(data, dict):
        return {str(scene_id): scene for scene_id, scene in data.items() if isinstance(scene, dict)}
    return {}


MODEL_NAME = "philschmid/bart-large-cnn-samsum"


def combine_dialogue(dialogue_list):
    lines = []
    for entry in dialogue_list:
        if isinstance(entry, dict):
            line = entry.get("line", "")
            line = line.strip() if isinstance(line, str) else str(line).strip()
        else:
            line = str(entry).strip()

        if line:
            # Filter noise
            for pattern in WHISPER_NOISE_PATTERNS:
                line = re.sub(pattern, "", line, flags=re.IGNORECASE).strip()
            
            if line:
                lines.append(line)

    return " ".join(lines)


def format_as_sasum_dialogue(dialogue_list: List) -> str:
    lines = []
    for entry in dialogue_list:
        if isinstance(entry, dict):
            speaker = (entry.get("speaker") or "").strip()
            line = str(entry.get("line", "")).strip()
        else:
            speaker = ""
            line = str(entry).strip()

        if not line:
            continue

        if speaker:
            lines.append(f"{speaker}: {line}")
        else:
            lines.append(line)

    return "\n".join(lines)


def build_scene_prompt(text, speakers):
    if speakers:
        speaker_context = "Characters in this scene: " + ", ".join(speakers) + ". "
        return speaker_context + text
    return text


def build_scene_context(scene_id, feature_map):
    feature = feature_map.get(str(scene_id), {})
    parts = []

    start = feature.get("start")
    end = feature.get("end")
    if start is not None and end is not None:
        parts.append(f"Time {int(start)}s\u2013{int(end)}s")

    motion = feature.get("motion_level", "")
    if motion:
        parts.append(f"{motion.title()} motion")

    objects = [o for o in (feature.get("objects") or []) if o.lower() != "person"][:3]
    if objects:
        parts.append(f"Objects: {', '.join(objects)}")

    if not parts:
        return ""
    return "; ".join(parts)


def _build_fallback_summary(text: str, scene_id: str, feature_map: dict) -> str:
    feat = feature_map.get(str(scene_id), {})
    objects = feat.get("objects", [])
    motion = feat.get("motion_level", "low").lower()
    
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 5]
    if sentences:
        dialogue_part = sentences[0] + "."
    else:
        dialogue_part = ""

    interesting_objects = [o for o in objects if o.lower() != "person"][:2]
    if interesting_objects:
        obj_text = f"The scene features {', '.join(interesting_objects)}."
    else:
        obj_text = ""

    if motion == "high":
        motion_text = "There is intense action."
    else:
        motion_text = ""

    parts = [p for p in [dialogue_part, obj_text, motion_text] if p]
    if not parts:
        return text[:200].strip() + "..." if text else "A quiet scene."
    
    return " ".join(parts)


def summarize_scene(text, language="en", perspective="Neutral"):
    """Summarize a single scene's formatted dialogue text (Legacy API)."""
    base_language = (language or "en").strip().lower().split("-")[0]

    if base_language != "en":
        extractive = extractive_summary_from_text(text, language=language)
        return extractive or text[:300]

    # Re-apply perspective logic for single-scene calls
    persp = (perspective or "Neutral").strip().lower()
    if persp == "protagonist":
        input_text = f"Summary from the hero's perspective: {text}"
    elif persp == "antagonist":
        input_text = f"Summary from the villain's perspective: {text}"
    else:
        input_text = text

    results = summarize_text_batch([input_text], max_length=80, min_length=25)
    return results[0] if results else text[:300]


def trim_summary(summary, importance, max_sentences=3):
    sentences = [s.strip() for s in re.split(r"[.!?]+", summary) if s.strip()]
    if not sentences:
        return ""

    k = max(1, min(max_sentences, int(round(importance * max_sentences))))
    return ". ".join(sentences[:k]).strip() + "."


def _max_sentences_for_style(summary_style: str) -> int:
    style = (summary_style or "").strip().lower()
    if style == "concise":
        return 1
    return 3


def _normalize_scene_features(scene_features: Any) -> Dict[str, Dict[str, Any]]:
    if scene_features is None:
        current = load_scene_features("data/intermediate/scene_features.json")
        if current:
            return current
        return load_scene_features("outputs/scene_features.json")

    if isinstance(scene_features, list):
        return {
            str(scene["scene_id"]): scene
            for scene in scene_features
            if isinstance(scene, dict) and "scene_id" in scene
        }

    if isinstance(scene_features, dict):
        return {
            str(scene_id): scene
            for scene_id, scene in scene_features.items()
            if isinstance(scene, dict)
        }

    return {}


def summarize_all_scenes(
    scene_dialogues,
    scene_features=None,
    summary_style: str = "Detailed",
    language: str = "en",
    perspective: str = "Neutral",
):
    feature_map = _normalize_scene_features(scene_features)
    max_sentences = _max_sentences_for_style(summary_style)
    base_language = (language or "en").strip().lower().split("-")[0]

    scene_summaries = {}
    batch_ids = []
    batch_texts = []

    for scene_id, dialogue_list in scene_dialogues.items():
        if not dialogue_list:
            scene_summaries[scene_id] = ""
            continue

        raw_text = combine_dialogue(dialogue_list)

        # 1. Check for Fallback (Sparse dialogue)
        if len(raw_text.split()) < MIN_DIALOGUE_WORDS:
            summary = _build_fallback_summary(raw_text, scene_id, feature_map)
            importance = float(feature_map.get(str(scene_id), {}).get("importance", 0.5))
            scene_summaries[scene_id] = trim_summary(summary, importance, max_sentences=max_sentences)
            continue

        # 2. Check for Extractive (Non-English)
        if base_language != "en":
            summary = extractive_summary_from_text(raw_text, language=language) or raw_text[:300]
            importance = float(feature_map.get(str(scene_id), {}).get("importance", 0.5))
            scene_summaries[scene_id] = trim_summary(summary, importance, max_sentences=max_sentences)
            continue

        # 3. Queue for Abstractive Batch (English)
        formatted_text = format_as_sasum_dialogue(dialogue_list)
        persp = (perspective or "Neutral").strip().lower()
        if persp == "protagonist":
            input_text = f"Summary from the hero's perspective: {formatted_text}"
        elif persp == "antagonist":
            input_text = f"Summary from the villain's perspective: {formatted_text}"
        else:
            input_text = formatted_text
        
        batch_ids.append(scene_id)
        batch_texts.append(input_text)

    # Execute Batch Abstractive Summarization
    if batch_texts:
        summaries = summarize_text_batch(batch_texts, max_length=80, min_length=25)
        for scene_id, summary in zip(batch_ids, summaries):
            if not summary.strip():
                # Fallback if BART failed
                raw_text = combine_dialogue(scene_dialogues[scene_id])
                summary = _build_fallback_summary(raw_text, scene_id, feature_map)
            
            importance = float(feature_map.get(str(scene_id), {}).get("importance", 0.5))
            scene_summaries[scene_id] = trim_summary(summary, importance, max_sentences=max_sentences)

    return scene_summaries


def save_scene_summaries(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def main():
    dialogues_path = Path("data/scene_dialogues.json")
    if not dialogues_path.exists():
        raise FileNotFoundError("Missing required input: data/scene_dialogues.json. Run dialogue alignment first.")

    scene_dialogues = load_scene_dialogues(dialogues_path)
    scene_summaries = summarize_all_scenes(scene_dialogues)
    output_path = Path("data/scene_summaries.json")
    save_scene_summaries(scene_summaries, output_path)

    print(f"Model loaded: {MODEL_NAME}")
    print("Flow: SAMSum-formatted dialogue -> abstractive (English), extractive (non-English)")
    print(f"Scene summaries saved: {output_path}")
    print(f"Scenes summarized: {len(scene_summaries)}")


if __name__ == "__main__":
    main()