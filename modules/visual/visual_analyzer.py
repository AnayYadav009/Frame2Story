
def motion_to_score(motion):
    """Map motion class to normalized score used by importance formula."""
    mapping = {"low": 0.2, "med": 0.5, "medium": 0.5, "high": 0.9}

    if motion is None:
        return 0.5
    return mapping.get(motion.lower(), 0.5)


def object_score(objects, relevant_objects=None):
    """Estimate normalized object salience from detected labels.
    
    If relevant_objects is provided, matched objects receive a custom weight boost.
    """
    weights = {
        "person": 0.5,
        "book": 0.3,
        "cell phone": 0.3,
        "cup": 0.28,
        "bottle": 0.24,
        "chair": 0.2,
        "couch": 0.22,
        "dining table": 0.22,
        "tv": 0.18,
        "car": 0.22,
        "vehicle": 0.22,
        "gun": 0.35,
        "weapon": 0.35,
        "knife": 0.3,
        "explosion": 0.25,
    }

    score = 0
    rel_set = {r.lower() for r in relevant_objects} if relevant_objects else set()

    for obj in objects:
        w = weights.get(obj.lower(), 0.12)
        if obj.lower() in rel_set:
            w += 0.25
        score += w

    return min(score, 1.0)


def normalize_duration(duration, max_duration):
    if max_duration == 0:
        return 0

    return duration / max_duration


def compute_importance_from_features(motion_score, motion_level, objects, duration, max_duration, relevant_objects=None):
    """
    Compute scene importance using true multimodal fusion.
    """

    motion_component = motion_score
    obj_component = object_score(objects, relevant_objects=relevant_objects)
    obj_component = min(max(obj_component, 0.0), 1.0)

    duration_component = normalize_duration(duration, max_duration)
    duration_component = min(max(duration_component, 0.0), 1.0)

    # Keep a small bias for explicit high-stakes cues without overwhelming
    # character/dialogue-heavy scenes.
    critical_objects = {"weapon", "gun", "knife", "explosion"}

    object_labels = {obj.lower() for obj in objects}
    critical_boost = 0.05 if any(obj in critical_objects for obj in object_labels) else 0.0

    importance = (
        0.3 * motion_component +
        0.35 * obj_component +
        0.35 * duration_component +
        critical_boost
    )

    # Clamp to 1.0
    importance = min(importance, 1.0)

    return round(importance, 3)


def compute_importance(scene, max_duration, relevant_objects=None):
    return compute_importance_from_features(
        motion_score=scene.get("motion_score_normalized", 0.5),
        motion_level=scene.get("motion", scene.get("motion_level", "MEDIUM")),
        objects=scene.get("objects", []),
        duration=scene.get("duration", scene.get("duration_seconds", 0)),
        max_duration=max_duration,
        relevant_objects=relevant_objects or scene.get("relevant_objects"),
    )