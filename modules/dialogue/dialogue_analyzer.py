import json


def load_scene_dialogues(path):
    with open(path,"r") as f:
        return json.load(f)
    
def combine_dialogue(dialogues):
    return " ".join(dialogues)

def length_score(text):
    word_count = len(text.split())
    return min(word_count / 200, 1.0)

KEYWORDS = [
    "fight", "kill", "escape", "attack",
    "plan", "mission", "danger", "war",
    "police", "gun", "run"
]

def keyword_score(text):

    text_lower = text.lower()

    count = sum(1 for word in KEYWORDS if word in text_lower)

    return min(count / 5, 1.0)

def compute_dialogue_score(text):

    l_score = length_score(text)
    k_score = keyword_score(text)

    # weighted combination
    final = 0.6 * l_score + 0.4 * k_score

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

def save_dialogue_scores(data, path):

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    scene_dialogues = load_scene_dialogues("data/scene_dialogues.json")

    scores = analyze_dialogues(scene_dialogues)

    save_dialogue_scores(scores, "data/dialogue_scores.json")

    print("Dialogue analysis complete.")