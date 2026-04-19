from typing import Iterable, Optional, Sequence, Set, List, Any

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
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


def detect_objects_batch(
    images_or_frames: List[Any], 
    model_name: str = "yolov8n.pt", 
    confidence: float = 0.25,
    batch_size: int = 16
) -> List[List[str]]:
    """Run YOLOv8 on a list of images/frames and return labels for each.
    
    Uses internal batching for performance.
    """
    if not images_or_frames:
        return []

    model = get_model(model_name=model_name)
    
    # Process in chunks to avoid OOM on large videos if we pass thousands at once
    all_results = []
    for i in range(0, len(images_or_frames), batch_size):
        chunk = images_or_frames[i : i + batch_size]
        results = model(chunk, conf=confidence, verbose=False)
        all_results.extend(results)

    batch_detections = []
    for result in all_results:
        detections = set()
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detections.add(model.names[cls_id])
        batch_detections.append(sorted(detections))

    return batch_detections


def detect_objects(image_or_frame, model_name: str = "yolov8n.pt", confidence: float = 0.25):
    """Run YOLOv8n and return unique object labels for one image/frame."""
    results = detect_objects_batch([image_or_frame], model_name=model_name, confidence=confidence)
    return results[0] if results else []


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
    batch_results = detect_objects_batch(images_or_frames, model_name=model_name, confidence=confidence)
    
    scene_objects = set()
    for objects in batch_results:
        scene_objects.update(objects)

    filtered = _filter_relevant_objects(scene_objects, relevant_objects=relevant_objects)
    return sorted(filtered)