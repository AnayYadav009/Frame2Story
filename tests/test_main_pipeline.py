import json
from pathlib import Path

import pytest

import main_pipeline as mp


def test_video_hash_changes_with_style_and_progress(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"abc" * 100)

    h1 = mp._video_hash(str(video), 30, "Concise")
    h2 = mp._video_hash(str(video), 30, "Detailed")
    h3 = mp._video_hash(str(video), 40, "Concise")

    assert h1 != h2
    assert h1 != h3


def test_cache_marker_helpers(tmp_path):
    payload = tmp_path / "data" / "intermediate" / "scenes.json"
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_text("[]", encoding="utf-8")

    mp._write_cache_key(payload, "abc123")

    assert mp._cache_valid(payload, "abc123") is True
    assert mp._cache_valid(payload, "wrong") is False


def test_resolve_reference_prefers_manual_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    manual = tmp_path / "data" / "reference_summary.txt"
    manual.parent.mkdir(parents=True, exist_ok=True)
    manual.write_text("Manual reference", encoding="utf-8")

    ref = mp._resolve_reference_text({"1": ["fallback"]}, [1])
    assert ref == "Manual reference"


def test_run_pipeline_wrapper_returns_final_recap(monkeypatch):
    monkeypatch.setattr(
        mp,
        "run_full_pipeline",
        lambda **_kwargs: {"final_recap": "WRAPPED"},
    )

    result = mp.run_pipeline(video_path="v.mp4", subtitle_path=None)
    assert result == "WRAPPED"


def test_run_full_pipeline_no_video_uses_intermediate_files(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    intermediate = tmp_path / "data" / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    (intermediate / "ranked_scene_ids.json").write_text("[1]", encoding="utf-8")
    (intermediate / "scene_summaries.json").write_text('{"1": "A. B."}', encoding="utf-8")
    (intermediate / "scene_features.json").write_text('{"1": {"importance": 0.5}}', encoding="utf-8")
    (intermediate / "scene_dialogues.json").write_text('{"1": ["Reference line"]}', encoding="utf-8")

    monkeypatch.setattr(mp, "build_recap", lambda *args, **kwargs: "FINAL RECAP")
    monkeypatch.setattr(mp, "evaluate_recap", lambda *args, **kwargs: {"ok": True})

    logs = []
    result = mp.run_full_pipeline(
        subtitle_path=None,
        video_path=None,
        summary_style="Concise",
        output_dir="outputs_test",
        progress_callback=logs.append,
    )

    assert result["final_recap"] == "FINAL RECAP"
    assert result["evaluation"] == {"ok": True}
    assert "Loading intermediate artifacts..." in logs
    assert "Generating final recap..." in logs

    final_file = tmp_path / "outputs_test" / "final" / "final_recap.txt"
    assert final_file.exists()


def test_run_full_pipeline_no_video_raises_when_intermediate_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError):
        mp.run_full_pipeline(subtitle_path=None, video_path=None)
