from modules.summarization import scene_summarizer as ss


def test_style_to_max_sentences_mapping():
    assert ss._max_sentences_for_style("Concise") == 1
    assert ss._max_sentences_for_style("Detailed") == 3
    assert ss._max_sentences_for_style("anything") == 3


def test_trim_summary_respects_importance_and_cap():
    summary = "One. Two. Three."

    assert ss.trim_summary(summary, importance=1.0, max_sentences=3) == "One. Two. Three."
    assert ss.trim_summary(summary, importance=0.33, max_sentences=3) == "One."


def test_build_scene_prompt_includes_speakers_when_available():
    prompt = ss.build_scene_prompt("They argue.", ["TED", "MARSHALL"])
    assert prompt == "Characters in this scene: TED, MARSHALL. They argue."

    assert ss.build_scene_prompt("No names.", []) == "No names."


def test_summarize_all_scenes_applies_style_and_handles_empty(monkeypatch):
    monkeypatch.setattr(ss, "summarize_scene", lambda _text, language="en": "Alpha. Beta. Gamma.")

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


def test_summarize_all_scenes_passes_speaker_context_to_summarizer(monkeypatch):
    captured = []

    def _capture(text, language="en"):
        captured.append((text, language))
        return "Alpha."

    monkeypatch.setattr(ss, "summarize_scene", _capture)

    scene_dialogues = {
        "1": [
            {"speaker": "TED", "line": "You lied to me."},
            {"speaker": "MARSHALL", "line": "I had to."},
            {"speaker": "TED", "line": "No, you didn't."},
        ]
    }
    scene_features = {"1": {"importance": 1.0}}

    result = ss.summarize_all_scenes(
        scene_dialogues,
        scene_features=scene_features,
        summary_style="Detailed",
        language="es",
    )

    assert result["1"] == "Alpha."
    assert captured
    assert captured[0][0].startswith("Characters in this scene: TED, MARSHALL. ")
    assert captured[0][1] == "es"


def test_summarize_scene_non_english_skips_abstractive(monkeypatch):
    monkeypatch.setattr(ss, "extractive_summary_from_text", lambda _text, language=None: "Resumen corto")
    monkeypatch.setattr(
        ss,
        "abstractive_summarize_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Should not call abstractive")),
    )

    summary = ss.summarize_scene("Texto original.", language="es")
    assert summary == "Resumen corto"


def test_summarize_scene_english_skips_extractive(monkeypatch):
    monkeypatch.setattr(
        ss,
        "extractive_summary_from_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Should not call extractive for English")),
    )
    monkeypatch.setattr(ss, "abstractive_summarize_text", lambda text, **_kwargs: f"ABSTRACT::{text}")

    summary = ss.summarize_scene("Original dialogue.", language="en")
    assert summary == "ABSTRACT::Original dialogue."
