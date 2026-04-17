import pytest
import math
from utils import fusion_engine as fe
from utils import scene_ranker as sr


def test_fusion_engine_combines_modalities_with_visual():
    scene_data = [
        {"scene_id": 1, "motion_score": 10, "objects": ["person"], "importance": 0.2},
        {"scene_id": 2, "motion_score": 20, "objects": ["gun", "car"], "importance": 1.0},
    ]
    dialogue_scores = {"1": 0.5, "2": 1.0}

    # Using 'auto' weights: (0.40, 0.30, 0.15, 0.15)
    fused = fe.fusion_engine(scene_data, dialogue_scores, visual_data=scene_data)

    assert len(fused) == 2
    assert "visual_score" in fused[0]
    assert fused[1]["visual_score"] == 1.0
    
    # Scene 2 calculation:
    # dialogue (1.0 * 0.40) + motion (1.0 * 0.30) + objects (1.0 * 0.15) + visual (1.0 * 0.15) = 1.0
    assert fused[1]["final"] == pytest.approx(1.0, abs=1e-4)


def test_fusion_weights_validation_and_presets():
    # action: FusionWeights(0.15, 0.50, 0.20, 0.15)
    assert fe.PRESETS["action"].motion == pytest.approx(0.50)

    # Should raise if they don't sum to 1.0
    with pytest.raises(ValueError):
        fe.FusionWeights(dialogue=0.8, motion=0.2, objects=0.2, visual=0.2)


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


def test_adaptive_selection_new_bounds():
    # 100 scenes, 600s watched:
    # min_keep = max(8, ceil(100*0.10)=10, ceil(600/240)=3) = 10
    # max_keep = min(35, max(10, ceil(100*0.20)=20, ceil(600/90)=7, 12)) = min(35, 20) = 20
    scenes = [{"scene_id": i, "final": round(i / 100, 2)} for i in range(100)]
    result = sr.get_ranked_scenes(scenes, watched_duration_sec=600)
    assert 10 <= len(result) <= 20


def test_timeline_window_coverage_guarantee():
    # 80 scenes, but only 8 score high (one in each window)
    # Windows: [0-9], [10-19], ..., [70-79]
    # We put a high score at the start of each window, and 0 everywhere else.
    scenes = []
    for i in range(80):
        score = 1.0 if i % 10 == 0 else 0.0
        scenes.append({"scene_id": i, "final": score})
    
    # For 80 scenes and 0 duration:
    # min_keep = max(4, ceil(80 * 0.1)) = 8
    # max_keep = min(35, max(8, ceil(80 * 0.2)=16, 12)) = 16
    result = sr.get_ranked_scenes(scenes, watched_duration_sec=0)
    
    ids = [s["scene_id"] for s in result]
    # Threshold was 0.0, so it filled up to max_keep = 16
    assert len(ids) == 16
    # Ensure the 8 scenes that were the "best" in their windows are included
    expected_seeds = [0, 10, 20, 30, 40, 50, 60, 70]
    for seed in expected_seeds:
        assert seed in ids


def test_adaptive_selection_no_duplicates():
    # Ensure seeding + quantile + padding doesn't create duplicate scene IDs
    scenes = [{"scene_id": i, "final": 0.9} for i in range(20)]
    result = sr.get_ranked_scenes(scenes, watched_duration_sec=600)
    ids = [s["scene_id"] for s in result]
    assert len(ids) == len(set(ids))


def test_extract_scene_ids_supports_dict_and_int_inputs():
    ids = sr.extract_scene_ids([{"scene_id": "3"}, 5, {"scene_id": 7}])
    assert ids == [3, 5, 7]


def test_adaptive_selection_timeline_order_preserved():
    scenes = [
        {"scene_id": 10, "final": 0.9},
        {"scene_id": 3, "final": 0.8},
        {"scene_id": 7, "final": 0.7},
    ]
    # With 3 scenes, all will likely be selected by coverage or min_keep
    result = sr.get_ranked_scenes(scenes, watched_duration_sec=600)
    ids = [s["scene_id"] for s in result]
    assert ids == sorted(ids)