"""Multimodal fusion engine.

Computes a final scene-importance score by linearly combining four normalised
sub-scores: dialogue richness, motion intensity, object salience, and visual
importance (from the scene-feature pipeline).

FusionWeights now carries four fields (dialogue, motion, objects, visual).
All existing PRESETS have been updated so the four weights sum to 1.0.
The ``visual`` key in the output dict still carries the raw importance float for
backward compat; the new ``visual_score`` key carries the normalised value that
actually feeds the final score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class FusionWeights:
    dialogue: float = 0.48
    motion:   float = 0.20
    objects:  float = 0.10
    visual:   float = 0.22

    def __post_init__(self):
        total = self.dialogue + self.motion + self.objects + self.visual
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Fusion weights must sum to 1.0, got {total:.2f}")


PRESETS = {
    "auto":         FusionWeights(0.48, 0.20, 0.10, 0.22),
    "drama":        FusionWeights(0.62, 0.08, 0.08, 0.22),
    "action":       FusionWeights(0.15, 0.50, 0.20, 0.15),
    "documentary":  FusionWeights(0.52, 0.12, 0.08, 0.28),
}


def detect_genre_preset(scene_data: list[dict], dialogue_data: dict) -> str:
    """Infers the best genre preset from multimodal signals.
    
    Returns "action", "drama", or "auto" based on motion/dialogue distribution.
    """
    if not scene_data:
        return "auto"

    # Analyze motion intensity
    max_motion = max((float(s.get("motion_score", 0.0)) for s in scene_data), default=1.0)
    if max_motion <= 0:
        motion_ratio = 0.0
    else:
        high_motion_count = sum(1 for s in scene_data if float(s.get("motion_score", 0.0)) / max_motion > 0.7)
        motion_ratio = high_motion_count / len(scene_data)

    # Analyze dialogue richness
    dialogue_scores = [float(dialogue_data.get(str(s["scene_id"]), 0.0)) for s in scene_data]
    avg_dialogue = sum(dialogue_scores) / len(dialogue_scores) if dialogue_scores else 0.0

    # Prefer dialogue-led presets unless motion dominance is clearly strong.
    if avg_dialogue >= 0.45:
        return "drama"

    if motion_ratio > 0.35 and avg_dialogue < 0.40:
        return "action"

    return "auto"


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
    w_dialogue=0.40,
    w_motion=0.30,
    w_object=0.15,
    weights: FusionWeights | None = None,
):
    """Compute multimodal importance score per scene.

    Args:
        scene_data:     List of scene feature dicts (must have scene_id).
        dialogue_data:  Dict mapping scene_id → dialogue importance float.
        visual_data:    Optional separate list of visual feature dicts.  When
                        None, scene_data is used as the visual source.
        w_dialogue/w_motion/w_object: Legacy scalar weight overrides, only used
                        when ``weights`` is None.
        weights:        FusionWeights dataclass; takes priority over scalar args.

    Returns:
        List of dicts with keys:
            scene_id, dialogue_score, motion_score, object_score,
            visual_score, visual (raw importance), final.
    """
    # Build effective weights — legacy 3-arg callers default visual to 0.0
    if weights is None:
        w_vis = max(0.0, round(1.0 - w_dialogue - w_motion - w_object, 4))
        try:
            weights = FusionWeights(w_dialogue, w_motion, w_object, w_vis)
        except ValueError:
            weights = FusionWeights()  # safe fallback

    w = weights

    visual_rows = visual_data if visual_data is not None else scene_data
    visual_map = {
        int(row.get("scene_id", -1)): row
        for row in visual_rows
        if isinstance(row, dict)
    }

    max_motion = max(
        (float(row.get("motion_score", 0.0)) for row in visual_map.values()),
        default=0.0,
    )
    max_object_count = max(
        (len(row.get("objects", []) or []) for row in visual_map.values()),
        default=0,
    )
    max_importance = max(
        (float(row.get("importance", 0.0)) for row in visual_map.values()),
        default=0.0,
    )

    fused_results = []

    for scene in scene_data:
        scene_id = int(scene.get("scene_id", -1))
        if scene_id < 0:
            continue

        visual_row = visual_map.get(scene_id, scene)
        dialogue_score = float(
            dialogue_data.get(str(scene_id), dialogue_data.get(scene_id, 0.0))
        )
        motion_raw     = float(visual_row.get("motion_score", 0.0))
        object_count   = len(visual_row.get("objects", []) or [])
        importance_raw = float(visual_row.get("importance", 0.0))

        motion_score  = _normalize(motion_raw,     max_motion)
        object_score  = _normalize(object_count,   max_object_count)
        visual_score  = _normalize(importance_raw, max_importance)

        final_score = round(
            (w.dialogue * dialogue_score)
            + (w.motion  * motion_score)
            + (w.objects * object_score)
            + (w.visual  * visual_score),
            4,
        )

        fused_results.append({
            "scene_id":       scene_id,
            "dialogue_score": round(dialogue_score, 4),
            "motion_score":   round(motion_score,   4),
            "object_score":   round(object_score,   4),
            "visual_score":   round(visual_score,   4),   # new — normalised
            "visual":         importance_raw,              # kept for compat
            "final":          final_score,
        })

    return fused_results


def fuse_scores(scene_features, dialogue_scores):
    """Backward-compatible API used by existing call sites."""
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