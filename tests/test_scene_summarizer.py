from modules.summarization import scene_summarizer as ss


def test_style_to_max_sentences_mapping():
    assert ss._max_sentences_for_style("Concise") == 1
    assert ss._max_sentences_for_style("Detailed") == 3
    assert ss._max_sentences_for_style("anything") == 3


def test_trim_summary_respects_importance_and_cap():
    summary = "One. Two. Three."

    assert ss.trim_summary(summary, importance=1.0, max_sentences=3) == "One. Two. Three."
    assert ss.trim_summary(summary, importance=0.33, max_sentences=3) == "One."


def test_summarize_all_scenes_applies_style_and_handles_empty(monkeypatch):
    monkeypatch.setattr(ss, "summarize_scene", lambda _text: "Alpha. Beta. Gamma.")

    scene_dialogues = {"1": ["hello"], "2": ["world"], "3": []}
    scene_features = {
        "1": {"importance": 1.0},
        "2": {"importance": 0.4},
        "3": {"importance": 0.9},
    }

    concise = ss.summarize_all_scenes(scene_dialogues, scene_features=scene_features, summary_style="Concise")
    assert concise["1"] == "Alpha."
    assert concise["2"] == "Alpha."
    assert concise["3"] == ""

    detailed = ss.summarize_all_scenes(scene_dialogues, scene_features=scene_features, summary_style="Detailed")
    assert detailed["1"] == "Alpha. Beta. Gamma."
    assert detailed["2"] == "Alpha."
