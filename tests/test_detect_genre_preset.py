from modules.fusion.fusion_engine import detect_genre_preset

def test_detect_genre_preset_all_dialogue():
    scene_data = [{"scene_id": 1, "motion_score": 0.1}]
    dialogue_data = {"1": 0.5}  # avg_dialogue >= 0.45
    assert detect_genre_preset(scene_data, dialogue_data) == "drama"

def test_detect_genre_preset_all_motion():
    # max_motion will be 1.0
    scene_data = [{"scene_id": 1, "motion_score": 1.0}, {"scene_id": 2, "motion_score": 0.8}]
    dialogue_data = {"1": 0.1, "2": 0.1} # avg_dialogue < 0.40, motion_ratio = 1.0 > 0.35
    assert detect_genre_preset(scene_data, dialogue_data) == "action"

def test_detect_genre_preset_empty_input():
    assert detect_genre_preset([], {}) == "auto"

def test_detect_genre_preset_balanced():
    scene_data = [{"scene_id": 1, "motion_score": 1.0}, {"scene_id": 2, "motion_score": 0.1}]
    dialogue_data = {"1": 0.42, "2": 0.42}  # avg 0.42 (not drama, maybe action?), but motion ratio = 0.5. Wait.
    # If avg_dialogue is 0.42, it's < 0.45 (not drama). But it's > 0.40, so it's NOT action either.
    # It should return "auto".
    assert detect_genre_preset(scene_data, dialogue_data) == "auto"
