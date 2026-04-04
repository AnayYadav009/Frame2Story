import json
from dataclasses import dataclass


@dataclass(frozen=True)
class FusionWeights:
    dialogue: float = 0.45
    motion: float = 0.35
    objects: float = 0.20

    def __post_init__(self):
        total = self.dialogue + self.motion + self.objects
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Fusion weights must sum to 1.0, got {total:.2f}")


PRESETS = {
    "auto": FusionWeights(0.45, 0.35, 0.20),
    "drama": FusionWeights(0.65, 0.20, 0.15),
    "action": FusionWeights(0.20, 0.55, 0.25),
    "documentary": FusionWeights(0.55, 0.25, 0.20),
}

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _normalize(value, max_value):
    if max_value <= 0:
        return 0.0
    return max(0.0, min(float(value) / float(max_value), 1.0))


def fusion_engine(
    scene_data,
    dialogue_data,
    visual_data=None,
    w_dialogue=0.45,
    w_motion=0.35,
    w_object=0.20,
    weights: FusionWeights | None = None,
):
    """Compute multimodal importance using dialogue + motion + object activity."""
    w = weights or FusionWeights(w_dialogue, w_motion, w_object)

    visual_rows = visual_data if visual_data is not None else scene_data
    visual_map = {int(row.get("scene_id", -1)): row for row in visual_rows if isinstance(row, dict)}

    max_motion = max((float(row.get("motion_score", 0.0)) for row in visual_map.values()), default=0.0)
    max_object_count = max((len(row.get("objects", []) or []) for row in visual_map.values()), default=0)

    fused_results = []

    for scene in scene_data:
        scene_id = int(scene.get("scene_id", -1))
        if scene_id < 0:
            continue

        visual_row = visual_map.get(scene_id, scene)
        dialogue_score = float(dialogue_data.get(str(scene_id), dialogue_data.get(scene_id, 0.0)))
        motion_raw = float(visual_row.get("motion_score", 0.0))
        object_count = len(visual_row.get("objects", []) or [])

        motion_score = _normalize(motion_raw, max_motion)
        object_score = _normalize(object_count, max_object_count)

        final_score = round(
            (w.dialogue * dialogue_score) + (w.motion * motion_score) + (w.objects * object_score),
            4,
        )

        fused_results.append({
            "scene_id": scene_id,
            "dialogue_score": round(dialogue_score, 4),
            "motion_score": round(motion_score, 4),
            "object_score": round(object_score, 4),
            "visual": float(visual_row.get("importance", 0.0)),
            "final": final_score
        })

    return fused_results


def fuse_scores(scene_features, dialogue_scores):
    # Backward-compatible API used by existing call sites.
    return fusion_engine(scene_features, dialogue_scores, visual_data=scene_features)

def save_fusion_output(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    scene_features = load_json("data/scene_features.json")
    dialogue_scores = load_json("data/dialogue_scores.json")

    fused = fuse_scores(scene_features, dialogue_scores)

    save_fusion_output(fused, "data/fused_scores.json")

    print("Fusion complete.")
    print("Scenes processed:", len(fused))