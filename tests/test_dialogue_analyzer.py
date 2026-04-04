import pytest

from modules.dialogue import dialogue_analyzer as dan


def test_sentence_and_density_scores_are_bounded():
    assert dan.sentence_count_score(100) == 1.0
    assert dan.question_exclamation_density_score("A? B!", 2) == 1.0


def test_average_sentence_length_score_supports_short_and_long_styles():
    short_style = dan.average_sentence_length_score(7.0)
    long_style = dan.average_sentence_length_score(20.0)

    assert short_style > 0.9
    assert long_style > 0.9


def test_compute_dialogue_score_handles_empty_and_non_empty_text():
    assert dan.compute_dialogue_score("") == 0.0

    score = dan.compute_dialogue_score("Go now! Run! Why are we waiting?")
    assert 0.0 <= score <= 1.0


def test_analyze_dialogues_handles_empty_scene_dialogue_lists():
    scores = dan.analyze_dialogues({"1": ["Hello there."], "2": []})

    assert scores["2"] == 0.0
    assert 0.0 <= scores["1"] <= 1.0
