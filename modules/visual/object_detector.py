from typing import Iterable, Optional, Sequence, Set

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - import failure is runtime/environment specific
    YOLO = None


_MODEL = None


def get_model(model_name: str = "yolov8n.pt"):
    """Lazy-load YOLO model so importing this module stays lightweight."""
    global _MODEL
    if _MODEL is None:
        if YOLO is None:
            raise ImportError("ultralytics is not installed. Add it to requirements and install dependencies.")
        _MODEL = YOLO(model_name)
    return _MODEL


def detect_objects(image_or_frame, model_name: str = "yolov8n.pt", confidence: float = 0.25):
    """Run YOLOv8n and return unique object labels for one image/frame."""
    model = get_model(model_name=model_name)
    results = model(image_or_frame, conf=confidence, verbose=False)

    detections = set()
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detections.add(model.names[cls_id])

    return sorted(detections)


def _filter_relevant_objects(objects: Iterable[str], relevant_objects: Optional[Sequence[str]] = None) -> Set[str]:
    if not relevant_objects:
        return set(objects)

    keep = {item.lower() for item in relevant_objects}
    return {item for item in objects if item.lower() in keep}


def detect_scene_objects(
    images_or_frames,
    model_name: str = "yolov8n.pt",
    confidence: float = 0.25,
    relevant_objects: Optional[Sequence[str]] = None,
):
    """Aggregate object labels across multiple keyframes for one scene."""
    scene_objects = set()

    for sample in images_or_frames:
        objects = detect_objects(sample, model_name=model_name, confidence=confidence)
        scene_objects.update(objects)

    filtered = _filter_relevant_objects(scene_objects, relevant_objects=relevant_objects)
    return sorted(filtered)


if __name__ == "__main__":
    images = [
        "data/keyframes/scene_4_frame_1.jpg",
        "data/keyframes/scene_5_frame_2.jpg",
        "data/keyframes/scene_12_frame_3.jpg",
    ]
    print(detect_scene_objects(images))