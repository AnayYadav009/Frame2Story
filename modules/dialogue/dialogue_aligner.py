import json
import pysrt


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


def clean_dialogue(text):
    return text.replace("\n", " ").strip()


def align_dialogue_to_scenes(subs, scenes):
    # scene_id -> list of subtitle lines
    scene_dialogues = {str(scene["scene_id"]): [] for scene in scenes}

    for sub in subs:
        sub_time = time_to_seconds(sub.start)

        for scene in scenes:
            # Half-open interval prevents boundary duplicates.
            if scene["start"] <= sub_time < scene["end"]:
                scene_id = str(scene["scene_id"])
                scene_dialogues[scene_id].append(clean_dialogue(sub.text))
                break

    return scene_dialogues


def save_scene_dialogues(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def main():
    scenes = load_scenes("data/scenes.json")
    subs = load_subtitles("data/sample_himym.srt")
    scene_dialogues = align_dialogue_to_scenes(subs, scenes)
    save_scene_dialogues(scene_dialogues, "data/scene_dialogues.json")

    print("Alignment complete.")
    print("Total scenes:", len(scene_dialogues))


if __name__ == "__main__":
    main()