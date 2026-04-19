import json
from pathlib import Path

from modules.scene import scene_pipeline as sp


def test_run_scene_pipeline_returns_filtered_scenes(monkeypatch, tmp_path):
    expected_scenes = [{"scene_id": 1, "start": 0, "end": 5}]

    monkeypatch.setattr(sp, "get_filtered_scenes_for_progress", lambda **_kwargs: expected_scenes)

    output_path = tmp_path / "scenes.json"
    result = sp.run_scene_pipeline("video.mp4", progress_percentage=25.0, output_path=str(output_path))

    assert result == expected_scenes
    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == expected_scenes


def test_run_scene_pipeline_supports_timestamp_range(monkeypatch, tmp_path):
    expected_scenes = [{"scene_id": 4, "start": 30, "end": 40}]

    monkeypatch.setattr(
        sp,
        "get_filtered_scenes_for_time_range",
        lambda **_kwargs: expected_scenes,
    )

    output_path = tmp_path / "scenes_range.json"
    result = sp.run_scene_pipeline(
        "video.mp4",
        progress_percentage=None,
        start_time_sec=30,
        end_time_sec=45,
        output_path=str(output_path),
    )

    assert result == expected_scenes
    assert output_path.exists()


def test_compute_scene_features_with_mocked_dependencies(monkeypatch, tmp_path):
    monkeypatch.setattr(sp, "load_scenes", lambda _path: [{"scene_id": 1, "start": 0, "end": 10}])
    monkeypatch.setattr(sp, "read_video_properties", lambda _path: {"fps": 30.0, "frame_count": 300})
    monkeypatch.setattr(sp, "get_scene_keyframes", lambda *_args, **_kwargs: ([10], ["frame"] ))
    monkeypatch.setattr(sp, "save_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sp, "compute_scene_motion", lambda *_args, **_kwargs: 20.0)
    monkeypatch.setattr(sp, "classify_motion", lambda _score: "MEDIUM")
    monkeypatch.setattr(sp, "detect_objects_batch", lambda frames, **_kwargs: [["person", "car"]] * len(frames))
    monkeypatch.setattr(sp, "compute_importance_from_features", lambda **_kwargs: 0.77)

    # Mock VideoReader to avoid actual cv2.VideoCapture call
    class MockReader:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get_frame(self, idx): return "frame"
    
    import utils.video_reader
    monkeypatch.setattr(utils.video_reader, "VideoReader", lambda *args: MockReader())

    output_path = tmp_path / "scene_features.json"
    result = sp.compute_scene_features(
        video_path="video.mp4",
        scenes_path="scenes.json",
        output_path=str(output_path),
        keyframes_dir=str(tmp_path / "keyframes"),
        save_keyframes=False,
    )

    assert len(result) == 1
    assert result[0]["scene_id"] == 1
    assert result[0]["motion_level"] == "MEDIUM"
    assert result[0]["importance"] == 0.77
    assert output_path.exists()


def test_compute_scene_features_cleans_stale_keyframes(monkeypatch, tmp_path):
    # Setup stale files
    kf_dir = tmp_path / "keyframes"
    kf_dir.mkdir()
    stale_kf = kf_dir / "scene_999_frame_1.jpg"
    stale_kf.write_text("stale", encoding="utf-8")
    other_file = kf_dir / "meta.txt"
    other_file.write_text("keep", encoding="utf-8")

    monkeypatch.setattr(sp, "load_scenes", lambda _p: [{"scene_id": 1, "start": 0, "end": 2}])
    monkeypatch.setattr(sp, "read_video_properties", lambda _p: {"fps": 1.0, "frame_count": 10})
    monkeypatch.setattr(sp, "get_scene_keyframes", lambda *_a, **_k: ([0], ["frame_data"]))
    monkeypatch.setattr(sp, "save_frame", lambda frame, path: Path(path).write_text(frame, encoding="utf-8"))
    monkeypatch.setattr(sp, "compute_scene_motion", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(sp, "classify_motion", lambda _s: "LOW")
    monkeypatch.setattr(sp, "detect_objects_batch", lambda frames, **_k: [[]] * len(frames))
    monkeypatch.setattr(sp, "compute_importance_from_features", lambda **_k: 0.1)

    import utils.video_reader
    class MockReader:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(utils.video_reader, "VideoReader", lambda *a: MockReader())

    # Case 1: Cleanup enabled
    sp.compute_scene_features(
        video_path="v.mp4",
        scenes_path="s.json",
        output_path=str(tmp_path / "f.json"),
        keyframes_dir=str(kf_dir),
        save_keyframes=True,
        cleanup_keyframes=True,
    )

    assert not stale_kf.exists(), "Stale keyframe should be deleted"
    assert other_file.exists(), "Unrelated files should be preserved"
    assert (kf_dir / "scene_1_frame_1.jpg").exists(), "New keyframe should exist"


def test_compute_scene_features_skips_cleanup_if_save_keyframes_false(monkeypatch, tmp_path):
    kf_dir = tmp_path / "keyframes"
    kf_dir.mkdir()
    stale_kf = kf_dir / "scene_999_frame_1.jpg"
    stale_kf.write_text("stale", encoding="utf-8")

    monkeypatch.setattr(sp, "load_scenes", lambda _p: [])
    monkeypatch.setattr(sp, "read_video_properties", lambda _p: {"fps": 1.0, "frame_count": 10})
    
    import utils.video_reader
    class MockReader:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(utils.video_reader, "VideoReader", lambda *a: MockReader())

    # Case 2: save_keyframes is False
    sp.compute_scene_features(
        video_path="v.mp4",
        scenes_path="s.json",
        output_path=str(tmp_path / "f.json"),
        keyframes_dir=str(kf_dir),
        save_keyframes=False,
        cleanup_keyframes=True,
    )

    assert stale_kf.exists(), "Stale keyframe should NOT be deleted if save_keyframes is False"
