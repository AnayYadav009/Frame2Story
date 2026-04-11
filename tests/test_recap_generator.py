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


def test_build_recap_uses_one_shot_generation(monkeypatch):
    captured = {}

    monkeypatch.setattr(rg, "select_top_scenes", lambda ranked, top_percent=0.3: ranked)

    def _fake_generate(text, max_length=120, min_length=40):
        captured["text"] = text
        captured["max_length"] = max_length
        captured["min_length"] = min_length
        return "FINAL"

    monkeypatch.setattr(rg, "generate_final_recap", _fake_generate)

    ranked_scenes = [2, 1]
    scene_summaries = {
        "1": "S1 story.",
        "2": "S2 story.",
    }

    result = rg.build_recap(ranked_scenes, scene_summaries, summary_style="Detailed")

    assert result == "FINAL"
    assert captured["text"] == "S1 story. S2 story."
    assert captured["max_length"] == 200
    assert captured["min_length"] == 60


def test_build_recap_concise_uses_shorter_generation_budget(monkeypatch):
    captured = {}

    monkeypatch.setattr(rg, "select_top_scenes", lambda ranked, top_percent=0.3: ranked)

    def _fake_generate(text, max_length=120, min_length=40):
        captured["text"] = text
        captured["max_length"] = max_length
        captured["min_length"] = min_length
        return "FINAL"

    monkeypatch.setattr(rg, "generate_final_recap", _fake_generate)

    result = rg.build_recap([1], {"1": "Only scene."}, summary_style="Concise")

    assert result == "FINAL"
    assert captured["text"] == "Only scene."
    assert captured["max_length"] == 140
    assert captured["min_length"] == 45


def test_combine_summaries_inserts_connectors():
    combined = rg.combine_summaries(["A", "B", "C"])
    assert "Meanwhile" in combined or "Later" in combined
