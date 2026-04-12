import json
import re
from pathlib import Path
from typing import Any, Dict

from modules.dialogue.dialogue_analyzer import get_scene_speakers
from modules.summarization.extractive_summarizer import extractive_summary_from_text
from modules.summarization.abstractive_summarizer import summarize_text as abstractive_summarize_text

MIN_DIALOGUE_WORDS = 25


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
            lines.append(line)

    return " ".join(lines)


def build_scene_prompt(text, speakers):
    """Prepend speaker list and a grounding instruction to dialogue text.

    The instruction prefix reduces BART's tendency to hallucinate details
    from its pre-training when the dialogue is ambiguous or contains names
    of entities the model recognises from training data.
    """
    prefix = "Summarize only what is said: "
    if speakers:
        speaker_context = "Characters: " + ", ".join(speakers) + ". "
        return prefix + speaker_context + text
    return prefix + text


def build_scene_context(scene_id, feature_map):
    """Build a compact visual-context string from scene features.

    For use in scene_rationale.json and Streamlit UI display only.
    Never pass this to any summarization model — BART paraphrases
    metadata tokens into natural language garbage such as
    "The scene is in time 207s" or "There is high motion going on scene".
    """
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


def _first_n_sentences(text, n=2):
    """Return the first n sentences of raw dialogue as a fallback summary."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return text[:200].strip()
    return ". ".join(sentences[:n]).strip() + "."


def summarize_scene(text, language="en"):
    """Summarize a single scene's dialogue text.

    Receives the fully-built prompt string (with instruction prefix and
    speaker context already prepended). The short-dialogue bypass lives in
    summarize_all_scenes and operates on raw dialogue text BEFORE the
    prompt is constructed — this function always receives text that has
    already passed the MIN_DIALOGUE_WORDS gate.
    """
    base_language = (language or "en").strip().lower().split("-")[0]

    if base_language != "en":
        extractive = extractive_summary_from_text(text, language=language)
        return extractive or text[:300]

    abstractive = abstractive_summarize_text(text, max_length=80, min_length=25)
    return abstractive or text[:300]


def trim_summary(summary, importance, max_sentences=3):
    sentences = [s.strip() for s in re.split(r"[.!?]+", summary) if s.strip()]
    if not sentences:
        return ""

    # Scale importance (0-1 -> 1-max_sentences)
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
):
    feature_map = _normalize_scene_features(scene_features)
    max_sentences = _max_sentences_for_style(summary_style)

    scene_summaries = {}

    for scene_id, dialogue_list in scene_dialogues.items():

        if not dialogue_list:
            scene_summaries[scene_id] = ""
            continue

        speakers = get_scene_speakers(dialogue_list if isinstance(dialogue_list, list) else [])
        text = combine_dialogue(dialogue_list)

        if len(text.split()) < MIN_DIALOGUE_WORDS:
            summary = _first_n_sentences(text, n=2)
            importance = float(feature_map.get(str(scene_id), {}).get("importance", 0.5))
            scene_summaries[scene_id] = trim_summary(summary, importance, max_sentences=max_sentences)
            continue

        prompt_text = build_scene_prompt(text, speakers)
        summary = summarize_scene(prompt_text, language=language)

        # Quality gate: only replace if the model returned nothing useful.
        if not summary.strip() and text:
            summary = _first_n_sentences(text, n=2)

        importance = float(feature_map.get(str(scene_id), {}).get("importance", 0.5))
        summary = trim_summary(summary, importance, max_sentences=max_sentences)

        scene_summaries[scene_id] = summary

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
    print("Flow: direct dialogue -> abstractive (English), extractive fallback (non-English)")
    print(f"Scene summaries saved: {output_path}")
    print(f"Scenes summarized: {len(scene_summaries)}")


if __name__ == "__main__":
    main()