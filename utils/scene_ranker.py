import json

def load_scene_scores(path):
    with open(path, "r") as f:
        return json.load(f)

def rank_scenes(scene_data):
    return sorted(
        scene_data,
        key=lambda x: x["final"],
        reverse=True
    )

def select_top_scenes(ranked_scenes, top_n=5):
    return ranked_scenes[:top_n]

def select_by_threshold(ranked_scenes, threshold=0.5):
    return [scene for scene in ranked_scenes if scene["final"] >= threshold]


def adaptive_select_scenes(ranked_scenes, watched_duration_sec=None, keep_fraction=0.40):
    """Select scenes using score quantile with duration-aware min/max caps.

    Rather than a fixed threshold, this computes a dynamic cut-off so that
    approximately the top `keep_fraction` of scenes by score survive, then
    enforces sensible count boundaries based on how much video was watched.

    Args:
        ranked_scenes: Scenes already sorted by score descending.
        watched_duration_sec: Seconds of video watched (drives min/max caps).
        keep_fraction: Fraction of scenes to retain (0.40 → top 40% survive).
                       The threshold is set at the (1-keep_fraction) percentile.
                       Example: keep_fraction=0.40, n=10 → cutoff_idx=6 →
                       threshold=scores[6] → top 4 scenes (40%) survive.
    """
    if not ranked_scenes:
        return []

    scores = sorted(s["final"] for s in ranked_scenes)
    # cutoff_idx points to the lowest-score entry that still makes the cut.
    cutoff_idx = max(0, int(len(scores) * (1 - keep_fraction)))
    dynamic_threshold = scores[cutoff_idx]

    if watched_duration_sec and watched_duration_sec > 0:
        # ~1 key scene per 5 minutes of watched content, min 4, max 20
        min_scenes = max(4, int(watched_duration_sec / 300))
        # Upper cap: ~1 scene per 90 seconds, bounded to [8, 20]
        max_scenes = min(20, max(8, int(watched_duration_sec / 90)))
    else:
        # Conservative defaults when duration is unknown (also keeps tests passing)
        min_scenes = 2
        max_scenes = 15

    selected = [s for s in ranked_scenes if s["final"] >= dynamic_threshold]

    # Pad from ranked list if quantile yields too few scenes
    if len(selected) < min_scenes:
        selected = ranked_scenes[:min_scenes]

    # Trim to cap if quantile is too permissive
    if len(selected) > max_scenes:
        selected = ranked_scenes[:max_scenes]

    return selected


def restore_timeline_order(selected_scenes):
    return sorted(selected_scenes, key=lambda x: x["scene_id"])


def get_ranked_scenes(scene_data, threshold=0.5, watched_duration_sec=None, adaptive=True):
    """Rank scenes and select the most important subset.

    Args:
        scene_data: List of scene dicts with a 'final' score field.
        threshold: Fallback fixed threshold (used only when adaptive=False).
        watched_duration_sec: Seconds watched; drives adaptive min/max caps.
        adaptive: When True (default), uses quantile-based adaptive selection
                  instead of the fixed threshold.
    """
    ranked = rank_scenes(scene_data)
    if adaptive:
        selected = adaptive_select_scenes(ranked, watched_duration_sec=watched_duration_sec)
    else:
        selected = select_by_threshold(ranked, threshold)
    ordered = restore_timeline_order(selected)
    return ordered


def save_selected_scenes(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def extract_scene_ids(selected_scenes):
    return [
        int(s["scene_id"]) if isinstance(s, dict) else int(s)
        for s in selected_scenes
    ]


def main():
    scene_data = load_scene_scores("data/fused_scores.json")

    selected_scenes = get_ranked_scenes(scene_data)
    selected_scene_ids = extract_scene_ids(selected_scenes)

    save_selected_scenes(selected_scenes, "data/selected_scenes.json")
    save_selected_scenes(selected_scene_ids, "data/ranked_scene_ids.json")

    print("Selected Scenes:")
    for scene in selected_scenes:
        print(scene)

    print("Saved selected scenes to data/selected_scenes.json")
    print("Saved ranked scene IDs to data/ranked_scene_ids.json")


if __name__ == "__main__":
    main()