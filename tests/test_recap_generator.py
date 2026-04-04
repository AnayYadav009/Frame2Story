import json

from modules.summarization import recap_generator as rg


def test_load_ranked_scene_ids_normalizes_supported_values(tmp_path):
    path = tmp_path / "ranked_scene_ids.json"
    path.write_text(json.dumps([{"scene_id": "1"}, 2, "x", {"bad": 9}]), encoding="utf-8")

    assert rg.load_ranked_scene_ids(str(path)) == [1, 2]


def test_trim_summary_to_sentences():
    summary = "A1. A2. A3."
    assert rg._trim_summary_to_sentences(summary, 1) == "A1."
    assert rg._trim_summary_to_sentences(summary, 2) == "A1. A2."


def test_build_recap_applies_summary_style_cap(monkeypatch):
    monkeypatch.setattr(rg, "select_top_scenes", lambda ranked, top_percent=0.3: ranked)
    monkeypatch.setattr(rg, "hierarchical_summarization", lambda text: text)

    ranked_scenes = [1, 2]
    scene_summaries = {
        "1": "S1 first. S1 second.",
        "2": "S2 first. S2 second.",
    }
    scene_features = {
        "1": {"importance": 0.0},
        "2": {"importance": 0.0},
    }

    concise = rg.build_recap(ranked_scenes, scene_summaries, scene_features=scene_features, summary_style="Concise")
    assert "S1 first." in concise
    assert "S1 second." not in concise

    detailed = rg.build_recap(ranked_scenes, scene_summaries, scene_features=scene_features, summary_style="Detailed")
    assert "S1 second." in detailed


def test_combine_summaries_inserts_connectors():
    combined = rg.combine_summaries(["A", "B", "C"])
    assert "Meanwhile" in combined or "Later" in combined
