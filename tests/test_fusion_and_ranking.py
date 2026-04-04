import pytest

from utils import fusion_engine as fe
from utils import scene_ranker as sr


def test_fusion_engine_combines_modalities_with_normalization():
    scene_data = [
        {"scene_id": 1, "motion_score": 10, "objects": ["person"], "importance": 0.2},
        {"scene_id": 2, "motion_score": 20, "objects": ["gun", "car"], "importance": 0.9},
    ]
    dialogue_scores = {"1": 0.5, "2": 1.0}

    fused = fe.fusion_engine(scene_data, dialogue_scores, visual_data=scene_data)

    assert len(fused) == 2
    assert fused[0]["scene_id"] == 1
    assert fused[0]["final"] == pytest.approx(0.5, abs=1e-4)
    assert fused[1]["scene_id"] == 2
    assert fused[1]["final"] == pytest.approx(1.0, abs=1e-4)


def test_get_ranked_scenes_threshold_and_timeline_order():
    scene_data = [
        {"scene_id": 10, "final": 0.9},
        {"scene_id": 2, "final": 0.8},
        {"scene_id": 5, "final": 0.1},
    ]

    ranked = sr.get_ranked_scenes(scene_data, threshold=0.5)

    # After thresholding, restore_timeline_order sorts by scene_id ascending.
    assert [row["scene_id"] for row in ranked] == [2, 10]


def test_extract_scene_ids_supports_dict_and_int_inputs():
    ids = sr.extract_scene_ids([{"scene_id": "3"}, 5, {"scene_id": 7}])
    assert ids == [3, 5, 7]
