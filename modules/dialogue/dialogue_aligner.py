import json
import re
import pysrt

from modules.summarization.extractive_summarizer import detect_language


SPEAKER_PATTERN = re.compile(r"^([A-Z][A-Z\s\-']{1,20}):\s*(.+)")


def load_scenes(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_subtitles(path):
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_error = None

    for encoding in encodings:
        try:
            return pysrt.open(path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error

    raise UnicodeDecodeError(
        "subtitle-decoder",
        b"",
        0,
        1,
        f"Could not decode subtitle file {path!r}. Last error: {last_error}",
    )


def time_to_seconds(t):
    return (
        t.hours * 3600
        + t.minutes * 60
        + t.seconds
        + t.milliseconds / 1000.0
    )


def extract_speaker(text):
    cleaned = text.replace("\n", " ").strip()
    match = SPEAKER_PATTERN.match(cleaned)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, cleaned


def clean_dialogue(text):
    _, line = extract_speaker(text)
    return line


def detect_subtitle_language(subs, sample_size=20):
    sample = " ".join(
        sub.text.replace("\n", " ").strip()
        for sub in list(subs)[:sample_size]
        if getattr(sub, "text", None)
    )
    return detect_language(sample)


def align_dialogue_to_scenes(subs, scenes):
    subs_list = list(subs)
    detected_language = detect_subtitle_language(subs_list)

    # scene_id -> list of structured subtitle entries
    scene_dialogues = {str(scene["scene_id"]): [] for scene in scenes}

    for sub in subs_list:
        sub_time = time_to_seconds(sub.start)

        for scene in scenes:
            # Half-open interval prevents boundary duplicates.
            if scene["start"] <= sub_time < scene["end"]:
                scene_id = str(scene["scene_id"])
                speaker, line = extract_speaker(sub.text)
                scene_dialogues[scene_id].append(
                    {
                        "speaker": speaker,
                        "line": line,
                    }
                )
                break

    return scene_dialogues, detected_language


def save_scene_dialogues(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def main():
    scenes = load_scenes("data/scenes.json")
    subs = load_subtitles("data/sample_himym.srt")
    scene_dialogues, language = align_dialogue_to_scenes(subs, scenes)
    save_scene_dialogues(scene_dialogues, "data/scene_dialogues.json")

    print("Alignment complete.")
    print("Total scenes:", len(scene_dialogues))
    print("Detected subtitle language:", language)


if __name__ == "__main__":
    main()