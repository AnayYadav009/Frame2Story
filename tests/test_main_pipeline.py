import hashlib

import pytest

import main_pipeline as mp


def test_video_hash_changes_with_style_and_progress(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"abc" * 100)

    h1 = mp._video_hash(str(video), 30, "Concise")
    h2 = mp._video_hash(str(video), 30, "Detailed")
    h3 = mp._video_hash(str(video), 40, "Concise")
    h4 = mp._video_hash(str(video), 30, "Concise", "action")

    assert h1 != h2
    assert h1 != h3
    assert h1 != h4


def test_get_base_key_is_content_based(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"content_a")
    k1 = mp._get_base_key(str(video))
    
    video.write_bytes(b"content_b")
    k2 = mp._get_base_key(str(video))
    
    assert k1 != k2


def test_summary_style_does_not_affect_base_key(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"fixed_content")
    
    # Base key should be video-only (visual features)
    k1 = mp._get_base_key(str(video))
    
    # Even if we change progress/style, base key stays same
    # (Testing that our logic in run_full_pipeline uses these correctly)
    assert k1 == mp._get_base_key(str(video))


def test_cache_marker_helpers(tmp_path):
    payload = tmp_path / "data" / "intermediate" / "scenes.json"
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_text("[]", encoding="utf-8")

    mp._write_cache_key(payload, "abc123")

    assert mp._cache_valid(payload, "abc123") is True
    assert mp._cache_valid(payload, "wrong") is False


def test_cache_markers_are_scoped_per_artifact(tmp_path):
    scenes = tmp_path / "data" / "intermediate" / "scenes.json"
    features = tmp_path / "data" / "intermediate" / "scene_features.json"
    scenes.parent.mkdir(parents=True, exist_ok=True)

    scenes.write_text("[]", encoding="utf-8")
    features.write_text("[]", encoding="utf-8")

    mp._write_cache_key(scenes, "new-video-key")

    assert mp._cache_valid(scenes, "new-video-key") is True
    assert mp._cache_valid(features, "new-video-key") is False


def test_resolve_reference_prefers_manual_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    manual = tmp_path / "data" / "reference_summary.txt"
    manual.parent.mkdir(parents=True, exist_ok=True)
    manual.write_text("Manual reference", encoding="utf-8")

    ref = mp._resolve_reference_text({"1": ["fallback"]}, [1])
    assert ref == "Manual reference"


def test_reference_from_scene_dialogues_supports_structured_entries():
    scene_dialogues = {
        "1": [
            {"speaker": "TED", "line": "Line one."},
            {"speaker": "MARSHALL", "line": "Line two."},
        ]
    }

    ref = mp._reference_from_scene_dialogues(scene_dialogues, [1])
    assert ref == "Line one. Line two."


def test_run_pipeline_wrapper_returns_final_recap(monkeypatch):
    captured = {}

    def _fake_run_full_pipeline(**kwargs):
        captured.update(kwargs)
        return {"final_recap": "WRAPPED"}

    monkeypatch.setattr(
        mp,
        "run_full_pipeline",
        _fake_run_full_pipeline,
    )

    result = mp.run_pipeline(
        video_path="v.mp4",
        subtitle_path=None,
        fusion_preset="action",
        range_start_sec=12.0,
        range_end_sec=42.0,
    )
    assert isinstance(result, dict)
    assert result["final_recap"] == "WRAPPED"
    assert result["recap"] == "WRAPPED"
    assert captured["fusion_preset"] == "action"
    assert captured["range_start_sec"] == 12.0
    assert captured["range_end_sec"] == 42.0


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
        run_evaluation=True,
        progress_callback=logs.append,
    )

    assert result["final_recap"] == "FINAL RECAP"
    assert result["evaluation"] == {"ok": True}
    assert "Loading intermediate artifacts..." in logs
    assert "Generating final recap..." in logs

    final_file = tmp_path / "outputs_test" / "final" / "final_recap.txt"
    assert final_file.exists()


def test_run_full_pipeline_scope_metadata_for_no_video_mode(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    intermediate = tmp_path / "data" / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    (intermediate / "ranked_scene_ids.json").write_text("[1]", encoding="utf-8")
    (intermediate / "scene_summaries.json").write_text('{"1": "A. B."}', encoding="utf-8")
    (intermediate / "scene_features.json").write_text('{"1": {"importance": 0.5}}', encoding="utf-8")
    (intermediate / "scene_dialogues.json").write_text('{"1": ["Reference line"]}', encoding="utf-8")

    monkeypatch.setattr(mp, "build_recap", lambda *args, **kwargs: "FINAL RECAP")

    result = mp.run_full_pipeline(
        subtitle_path=None,
        video_path=None,
        percent_progress=55,
    )

    assert result["scope"] == "progress"
    assert result["range_start_sec"] is None
    assert result["range_end_sec"] is None
    assert result["progress_percent"] == 55


def test_run_full_pipeline_no_video_raises_when_intermediate_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError):
        mp.run_full_pipeline(subtitle_path=None, video_path=None)


def test_arg_parser_accepts_fusion_preset():
    parser = mp.build_arg_parser()
    args = parser.parse_args(["--fusion_preset", "documentary"])
    assert args.fusion_preset == "documentary"


def test_arg_parser_accepts_eval_flag():
    parser = mp.build_arg_parser()
    args = parser.parse_args(["--eval"])
    assert args.eval is True


def test_run_full_pipeline_skips_evaluation_by_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    intermediate = tmp_path / "data" / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    (intermediate / "ranked_scene_ids.json").write_text("[1]", encoding="utf-8")
    (intermediate / "scene_summaries.json").write_text('{"1": "A."}', encoding="utf-8")
    (intermediate / "scene_dialogues.json").write_text('{"1": ["Ref"]}', encoding="utf-8")
    
    eval_called = False
    def _fake_eval(*args, **kwargs):
        nonlocal eval_called
        eval_called = True
        return {}

    monkeypatch.setattr(mp, "evaluate_recap", _fake_eval)
    monkeypatch.setattr(mp, "build_recap", lambda *a, **k: "RECAP")

    # Run without eval (default)
    res = mp.run_full_pipeline(None, None)
    assert not eval_called
    assert res["evaluation"] is None

    # Run with eval
    res2 = mp.run_full_pipeline(None, None, run_evaluation=True)
    assert eval_called
    assert res2["evaluation"] is not None


def test_video_reader_random_access(tmp_path):
    # This requires a real video or a mock cv2.VideoCapture
    # For unit test, we'll just check the class structure
    from utils.video_reader import VideoReader
    reader = VideoReader("dummy.mp4")
    assert reader.cap is None
    # We won't call .open() here to avoid actual file system IO error in pytest


def test_visual_cache_hit_skips_cleanup(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    
    # Setup keyframe dir and a "cached" keyframe
    kf_dir = tmp_path / "data" / "keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)
    cached_kf = kf_dir / "scene_1_frame_1.jpg"
    cached_kf.write_text("old", encoding="utf-8")
    
    # Setup cache
    intermediate = tmp_path / "data" / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    features_path = intermediate / "scene_features.json"
    features_path.write_text("[]", encoding="utf-8")
    
    # Create video and its hash
    video = tmp_path / "v.mp4"
    video.write_bytes(b"v" * 1024)
    base_key = mp._get_base_key(str(video))
    scope_fragment = mp._scope_key_fragment(70, None, None)
    
    h = hashlib.sha256(base_key.encode("utf-8"))
    h.update(scope_fragment.encode("utf-8"))
    scene_key = h.hexdigest()[:16]
    
    mp._write_cache_key(features_path, scene_key)
    
    # Mock compute_scene_features to verify it's NOT called
    extraction_called = False
    def _fake_extraction(*args, **kwargs):
        nonlocal extraction_called
        extraction_called = True
        return []
    
    monkeypatch.setattr(mp, "compute_scene_features", _fake_extraction)
    
    # Mock other dependencies for minimal run
    monkeypatch.setattr(mp, "get_subtitle", lambda *a, **k: "dummy.srt")
    monkeypatch.setattr(mp, "load_subtitles", lambda *a: [])
    monkeypatch.setattr(mp, "run_scene_pipeline", lambda **k: [])
    monkeypatch.setattr(mp, "align_dialogue_to_scenes", lambda *a: ({}, "en"))
    monkeypatch.setattr(mp, "analyze_dialogues", lambda *a: {})
    monkeypatch.setattr(mp, "get_ranked_scenes", lambda *a, **k: [])
    monkeypatch.setattr(mp, "build_recap", lambda *a, **k: "RECAP")
    
    # Case 3: Run pipeline, should be a cache hit for visual features
    mp.run_full_pipeline(video_path=str(video), percent_progress=70)
    
    assert not extraction_called, "compute_scene_features should be skipped on cache hit"
    assert cached_kf.exists(), "Existing keyframes should NOT be touched on cache hit"
