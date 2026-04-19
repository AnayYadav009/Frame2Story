import json
import math
import re
from collections import Counter


def load_scene_dialogues(path):
    with open(path,"r") as f:
        return json.load(f)


def _dialogue_line(entry):
    if isinstance(entry, dict):
        line = entry.get("line", "")
        return line.strip() if isinstance(line, str) else str(line).strip()
    return str(entry).strip()


def combine_dialogue(dialogue_list):
    return " ".join(line for line in (_dialogue_line(entry) for entry in dialogue_list) if line)


def get_scene_speakers(dialogue_list):
    counts = Counter(
        entry["speaker"]
        for entry in dialogue_list
        if isinstance(entry, dict) and entry.get("speaker")
    )
    return [speaker for speaker, _ in counts.most_common(3)]


def extract_scene_speakers(scene_dialogues):
    return {
        str(scene_id): get_scene_speakers(dialogues if isinstance(dialogues, list) else [])
        for scene_id, dialogues in scene_dialogues.items()
    }

def _split_sentences(text):
    parts = re.split(r"[.!?]+", text)
    return [part.strip() for part in parts if part and part.strip()]


def sentence_count_score(sentence_count):
    # More spoken exchanges usually indicate stronger dialogue presence.
    return min(sentence_count / 12.0, 1.0)


def average_sentence_length_score(avg_words_per_sentence):
    # Reward both punchy action-like pacing (~7 words) and longer dramatic lines (~20 words).
    action_like = math.exp(-((avg_words_per_sentence - 7.0) ** 2) / (2 * (4.0 ** 2)))
    drama_like = math.exp(-((avg_words_per_sentence - 20.0) ** 2) / (2 * (8.0 ** 2)))
    return max(action_like, drama_like)


def question_exclamation_density_score(text, sentence_count):
    punct_count = text.count("?") + text.count("!")
    density = punct_count / max(sentence_count, 1)
    # 0.6 punctuation marks per sentence is treated as high intensity.
    return min(density / 0.6, 1.0)

def compute_dialogue_score(text):
    sentences = _split_sentences(text)
    sentence_count = len(sentences)
    if sentence_count == 0:
        return 0.0

    words = [len(sentence.split()) for sentence in sentences]
    avg_words_per_sentence = sum(words) / max(sentence_count, 1)

    s_count_score = sentence_count_score(sentence_count)
    s_len_score = average_sentence_length_score(avg_words_per_sentence)
    qx_score = question_exclamation_density_score(text, sentence_count)

    # Weighted combination of genre-agnostic structural dialogue signals.
    final = 0.45 * s_count_score + 0.35 * s_len_score + 0.20 * qx_score

    return round(final, 3)

def analyze_dialogues(scene_dialogues):

    scores = {}

    for scene_id, dialogues in scene_dialogues.items():

        if not dialogues:
            scores[scene_id] = 0.0
            continue

        text = combine_dialogue(dialogues)
        score = compute_dialogue_score(text)
        scores[scene_id] = score

    return scores


def save_scene_speakers(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def save_dialogue_scores(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    scene_dialogues = load_scene_dialogues("data/scene_dialogues.json")

    scores = analyze_dialogues(scene_dialogues)
    speakers = extract_scene_speakers(scene_dialogues)

    save_dialogue_scores(scores, "data/dialogue_scores.json")
    save_scene_speakers(speakers, "data/scene_speakers.json")

    print("Dialogue analysis complete.")