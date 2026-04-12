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


def test_fusion_weights_validation_and_presets():
    assert fe.PRESETS["action"].motion == pytest.approx(0.55)

    with pytest.raises(ValueError):
        fe.FusionWeights(dialogue=0.8, motion=0.2, objects=0.2)


def test_fusion_engine_preset_changes_scene_ranking_bias():
    scene_data = [
        {"scene_id": 1, "motion_score": 5, "objects": ["person"], "importance": 0.2},
        {"scene_id": 2, "motion_score": 20, "objects": ["car", "gun", "fire", "crowd"], "importance": 0.9},
    ]
    dialogue_scores = {"1": 1.0, "2": 0.2}

    drama = fe.fusion_engine(scene_data, dialogue_scores, visual_data=scene_data, weights=fe.PRESETS["drama"])
    action = fe.fusion_engine(scene_data, dialogue_scores, visual_data=scene_data, weights=fe.PRESETS["action"])

    drama_scores = {row["scene_id"]: row["final"] for row in drama}
    action_scores = {row["scene_id"]: row["final"] for row in action}

    assert drama_scores[1] > drama_scores[2]
    assert action_scores[2] > action_scores[1]


def test_get_ranked_scenes_threshold_and_timeline_order():
    scene_data = [
        {"scene_id": 10, "final": 0.9},
        {"scene_id": 2, "final": 0.8},
        {"scene_id": 5, "final": 0.1},
    ]

    ranked = sr.get_ranked_scenes(scene_data, threshold=0.5)

    assert [row["scene_id"] for row in ranked] == [2, 10]


def test_extract_scene_ids_supports_dict_and_int_inputs():
    ids = sr.extract_scene_ids([{"scene_id": "3"}, 5, {"scene_id": 7}])
    assert ids == [3, 5, 7]


def test_adaptive_selection_count_stays_within_caps():
    # 20 scenes, 600s watched:
    #   min = max(4, int(600/300)) = max(4, 2) = 4
    #   max = min(20, max(8, int(600/90))) = min(20, max(8, 6)) = 8
    scenes = [{"scene_id": i, "final": round(i / 20, 2)} for i in range(20)]
    result = sr.get_ranked_scenes(scenes, watched_duration_sec=600)
    assert 4 <= len(result) <= 8, f"Expected 4-8 scenes for 600s watched, got {len(result)}"


def test_adaptive_selection_respects_min_when_scores_are_low():
    # All scores are zero — quantile threshold = 0, so the min cap must kick in.
    scenes = [{"scene_id": i, "final": 0.0} for i in range(3)]
    result = sr.get_ranked_scenes(scenes, watched_duration_sec=600)
    # min_scenes = max(4, 600/300) = 4, but only 3 scenes exist → all 3 returned
    assert len(result) == 3


def test_adaptive_selection_trims_when_too_many_qualify():
    # 30 scenes all score 1.0 → all pass quantile → max cap trims to 20.
    scenes = [{"scene_id": i, "final": 1.0} for i in range(30)]
    result = sr.get_ranked_scenes(scenes, watched_duration_sec=600)
    assert len(result) <= 20, f"Expected at most 20 scenes, got {len(result)}"


def test_adaptive_selection_timeline_order_preserved():
    # Output should always be sorted by scene_id, not by score.
    scenes = [
        {"scene_id": 10, "final": 0.9},
        {"scene_id": 3, "final": 0.8},
        {"scene_id": 7, "final": 0.7},
    ]
    result = sr.get_ranked_scenes(scenes, watched_duration_sec=600)
    ids = [s["scene_id"] for s in result]
    assert ids == sorted(ids), f"Expected timeline order, got {ids}"