import json
from pathlib import Path
import random
import re
from typing import Any, Dict
import torch
from modules.summarization.model_cache import get_model_components as get_cached_model_components

MODEL_NAME = "philschmid/bart-large-cnn-samsum"


def get_model_components():
    """Load tokenizer/config/model once and reuse across all summarization calls."""
    return get_cached_model_components(MODEL_NAME)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_ranked_scene_ids(path="data/ranked_scene_ids.json"):
    ranked_path = Path(path)
    if not ranked_path.exists():
        return []

    ranked_data = load_json(ranked_path)
    if not isinstance(ranked_data, list):
        return []

    normalized = []
    for item in ranked_data:
        if isinstance(item, dict) and "scene_id" in item:
            normalized.append(int(item["scene_id"]))
        elif isinstance(item, (int, str)):
            try:
                normalized.append(int(item))
            except ValueError:
                continue

    return normalized


def _normalize_scene_summaries(scene_summaries):
    summary_map = {}

    if isinstance(scene_summaries, dict):
        for scene_id, summary in scene_summaries.items():
            if isinstance(summary, list):
                summary_text = " ".join(str(part).strip() for part in summary if str(part).strip())
            else:
                summary_text = str(summary).strip()
            if summary_text:
                summary_map[str(scene_id)] = summary_text

    elif isinstance(scene_summaries, list):
        for item in scene_summaries:
            if not isinstance(item, dict):
                continue
            scene_id = item.get("scene_id")
            summary = item.get("summary", "")
            if isinstance(summary, list):
                summary_text = " ".join(str(part).strip() for part in summary if str(part).strip())
            else:
                summary_text = str(summary).strip()
            if scene_id is not None and summary_text:
                summary_map[str(scene_id)] = summary_text

    return summary_map


def _normalize_scene_features(scene_features: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(scene_features, list):
        return {
            str(item["scene_id"]): item
            for item in scene_features
            if isinstance(item, dict) and "scene_id" in item
        }

    if isinstance(scene_features, dict):
        return {
            str(scene_id): details
            for scene_id, details in scene_features.items()
            if isinstance(details, dict)
        }

    return {}


def _max_sentences_for_style(summary_style: str) -> int:
    style = (summary_style or "").strip().lower()
    if style == "concise":
        return 1
    return 3


def _trim_summary_to_sentences(summary: str, max_sentences: int) -> str:
    parts = [p.strip() for p in re.split(r"[.!?]+", str(summary)) if p and p.strip()]
    if not parts:
        return ""
    return ". ".join(parts[:max_sentences]).strip() + "."


def load_scene_features_with_fallback() -> Dict[str, Dict[str, Any]]:
    candidate_paths = [
        "data/intermediate/scene_features.json",
        "outputs/scene_features.json",
        "output/scene_features.json",
        "data/scene_features.json",
    ]

    for path in candidate_paths:
        feature_path = Path(path)
        if not feature_path.exists():
            continue
        try:
            return _normalize_scene_features(load_json(feature_path))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue

    return {}


def get_selected_summaries(ranked_scenes, scene_summaries):
    summary_map = _normalize_scene_summaries(scene_summaries)
    selected = []
    for scene_id in ranked_scenes:
        scene_id_str = str(scene_id)
        if scene_id_str in summary_map:
            selected.append(summary_map[scene_id_str])
    return selected


def restore_timeline_order(scene_ids):
    return sorted(scene_ids)


CONNECTORS = [
    "Meanwhile",
    "Later",
    "At the same time",
    "As a result",
    "Soon after",
    "In the next moment"
]


def combine_summaries(summaries):
    if not summaries:
        return ""

    combined = summaries[0].strip()

    for i in range(1, len(summaries)):
        connector = CONNECTORS[i % len(CONNECTORS)]
        next_summary = summaries[i].strip()
        if not combined.endswith((".", "!", "?")):
            combined += "."
        combined += f" {connector}, {next_summary}"

    return combined


def weighted_combine_summaries(scene_ids, scene_summaries, scene_features):
    combined = []

    for scene_id in scene_ids:
        summary = scene_summaries.get(str(scene_id), "")
        if not summary:
            continue

        importance = float(scene_features.get(str(scene_id), {}).get("importance", 0.5))

        if importance > 0.75:
            weight = 3
        elif importance > 0.5:
            weight = 2
        else:
            weight = 1

        combined.extend([summary] * weight)

    return combine_summaries(combined)


def chunk_text(text, chunk_size=400):
    if not text or not text.strip():
        return []

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks


def generate_final_recap(text, max_length=120, min_length=40):
    if not text or not text.strip():
        return ""

    tokenizer, _config, model, device = get_model_components()

    max_input_tokens = getattr(tokenizer, "model_max_length", 1024)
    if not isinstance(max_input_tokens, int) or max_input_tokens <= 0 or max_input_tokens > 100000:
        max_input_tokens = 1024

    encoded = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_input_tokens)
    truncated = tokenizer.decode(encoded.get("input_ids", []), skip_special_tokens=True)
    if not truncated:
        return ""

    inputs = tokenizer(
        truncated,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    if "input_ids" not in inputs or inputs["input_ids"].numel() == 0:
        return ""

    token_count = int(inputs["input_ids"].shape[-1])
    safe_max = max(10, min(max_length, token_count))
    safe_min = max(5, min(min_length, safe_max - 1))

    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_length=safe_max,
            min_length=safe_min,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
            do_sample=False,
        )

    if output_ids is None or output_ids.numel() == 0:
        return ""

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def hierarchical_summarization(text, max_length=200, min_length=60):
    """Summarize long text via chunk → summarize → merge → summarize.

    Each chunk is independently summarized at a shorter budget, then the
    chunk summaries are merged and passed through a final summarization pass
    at the caller-supplied budget.  This preserves detail from all parts of
    the input that a single-shot pass would compress away.

    Args:
        text: Combined scene summary text to condense.
        max_length: Token budget for the final merged summary.
        min_length: Minimum token length for the final merged summary.
    """
    if not text or not text.strip():
        return ""

    chunks = chunk_text(text)
    summaries = []

    for chunk in chunks:
        # Summarize each chunk at a tighter budget so intermediate outputs
        # remain digestible when merged for the final pass.
        summary = generate_final_recap(chunk, max_length=120, min_length=30)
        if summary:
            summaries.append(summary)

    if not summaries:
        return ""
    if len(summaries) == 1:
        # Single chunk: run one more pass to hit the requested budget
        return generate_final_recap(summaries[0], max_length=max_length, min_length=min_length)

    combined = " ".join(summaries)
    return generate_final_recap(combined, max_length=max_length, min_length=min_length)


def select_top_scenes(ranked_scenes, top_percent=0.3):
    k = max(1, int(len(ranked_scenes) * top_percent))
    return ranked_scenes[:k]


def build_recap(ranked_scenes, scene_summaries, scene_features=None, summary_style: str = "Detailed"):
    """Assemble a final recap from ranked scene summaries.

    Changes vs. previous version:
    - scene_features is now used: per-scene importance scales the sentence
      budget allocated to each scene in the combined input text.
    - select_top_scenes pre-filter removed: adaptive selection upstream
      already produces the right scene count; double-filtering discards
      scenes that were deliberately included.
    - Hierarchical summarization is used automatically when the combined
      input exceeds 500 words to avoid losing detail on long videos.
    """
    ordered = restore_timeline_order(ranked_scenes)
    summary_map = _normalize_scene_summaries(scene_summaries)
    feature_map = (
        _normalize_scene_features(scene_features)
        if scene_features is not None
        else load_scene_features_with_fallback()
    )

    style = (summary_style or "").strip().lower()
    max_sentences = _max_sentences_for_style(summary_style)

    ordered_texts = []
    for scene_id in ordered:
        key = str(scene_id)
        if key not in summary_map or not summary_map[key].strip():
            continue
        summary = summary_map[key].strip()

        # Scale sentence budget by importance: low-importance scenes get 1
        # sentence, high-importance scenes get up to max_sentences
        importance = float(feature_map.get(key, {}).get("importance", 0.5))
        k = max(1, round(importance * max_sentences))
        ordered_texts.append(_trim_summary_to_sentences(summary, k))

    if not ordered_texts:
        return ""

    combined_text = " ".join(ordered_texts)
    word_count = len(combined_text.split())

    if style == "concise":
        max_len, min_len = 140, 45
    else:
        max_len, min_len = 200, 60

    # Switch to hierarchical summarization for long combined inputs so that
    # detail from every scene reaches the final output.
    if word_count > 500:
        return hierarchical_summarization(combined_text, max_length=max_len, min_length=min_len)

    return generate_final_recap(combined_text, max_length=max_len, min_length=min_len)


def save_recap_outputs(recap_text):
    save_json({"movie_recap": recap_text}, "data/final_recap.json")
    Path("data/final_recap.txt").write_text(recap_text, encoding="utf-8")


def main():
    ranked_scenes = load_ranked_scene_ids("data/ranked_scene_ids.json")
    if not ranked_scenes:
        raise FileNotFoundError("Missing ranked scenes at data/ranked_scene_ids.json. Run scene_ranker.py first.")

    scene_summaries_path = Path("data/scene_summaries.json")
    if not scene_summaries_path.exists():
        scene_summaries_path = Path("output/scene_summaries.json")

    scene_summaries = load_json(scene_summaries_path)
    recap = build_recap(ranked_scenes, scene_summaries)
    save_recap_outputs(recap)

    print("\nFinal Recap:\n")
    print(recap)
    print("\nSaved recap to data/final_recap.json and data/final_recap.txt")


if __name__ == "__main__":
    main()