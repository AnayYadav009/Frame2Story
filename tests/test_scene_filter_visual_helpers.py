import pytest

from modules.scene import scene_detector as sd
from modules.scene import scene_filter as sf
from modules.visual import visual_analyzer as va


class _Timecode:
    def __init__(self, value):
        self._value = value

    def get_seconds(self):
        return self._value


def test_extract_scene_data_and_timecode_conversion():
    scene_list = [(_Timecode(0), _Timecode(5.9)), (_Timecode(6), _Timecode(11))]
    extracted = sd.extract_scene_data(scene_list)

    assert extracted == [
        {"scene_id": 1, "start": 0.0, "end": 5.9},
        {"scene_id": 2, "start": 6.0, "end": 11.0},
    ]


def test_scene_filter_progress_and_validation():
    assert sf.get_progress_time(100, 30) == 30

    with pytest.raises(ValueError):
        sf.get_progress_time(-1, 20)

    with pytest.raises(ValueError):
        sf.get_progress_time(100, 120)

    scenes = [
        {"scene_id": 1, "start": 0, "end": 9},
        {"scene_id": 2, "start": 10, "end": 20},
        {"scene_id": "bad", "start": 0, "end": 5},
    ]
    filtered = sf.filter_scenes_by_progress(scenes, progress_time=10)
    assert filtered == [{"scene_id": 1, "start": 0, "end": 9}]


def test_visual_analyzer_core_scoring_helpers():
    assert va.motion_to_score("HIGH") == 0.9
    assert va.motion_to_score("unknown") == 0.5

    assert va.object_score(["gun", "explosion", "person"]) == 1.0
    assert va.normalize_duration(4, 8) == 0.5
    assert va.normalize_duration(4, 0) == 0

    importance = va.compute_importance_from_features(
        motion_score=1.0,
        motion_level="HIGH",
        objects=["gun"],
        duration=20,
        max_duration=20,
    )
    assert 0.0 <= importance <= 1.0
