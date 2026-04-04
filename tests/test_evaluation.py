import json

import pytest

from modules.evaluation import eval as ev


def test_compute_rouge_scores_returns_expected_keys():
    scores = ev.compute_rouge_scores("A short generated recap.", "A short reference recap.")

    expected_keys = {
        "rouge1_precision",
        "rouge1_recall",
        "rouge1_f1",
        "rougeL_precision",
        "rougeL_recall",
        "rougeL_f1",
    }
    assert expected_keys.issubset(scores.keys())


def test_evaluate_recap_validates_inputs():
    with pytest.raises(ValueError):
        ev.evaluate_recap("", "reference")

    with pytest.raises(ValueError):
        ev.evaluate_recap("generated", "   ")


def test_evaluate_recap_writes_json_payload(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ev,
        "compute_bert_score",
        lambda _generated, _reference: {"precision": 0.1, "recall": 0.2, "f1": 0.15},
    )

    output_path = tmp_path / "scores.json"
    payload = ev.evaluate_recap("generated text", "reference text", output_path=str(output_path))

    assert output_path.exists()
    assert payload["bert_score"]["f1"] == 0.15

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["generated_length_words"] == 2
    assert loaded["reference_length_words"] == 2
