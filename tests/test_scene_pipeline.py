import json

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


def test_compute_scene_features_with_mocked_dependencies(monkeypatch, tmp_path):
    monkeypatch.setattr(sp, "load_scenes", lambda _path: [{"scene_id": 1, "start": 0, "end": 10}])
    monkeypatch.setattr(sp, "read_video_properties", lambda _path: {"fps": 30.0, "frame_count": 300})
    monkeypatch.setattr(sp, "get_scene_keyframes", lambda *_args, **_kwargs: ([10], ["frame"] ))
    monkeypatch.setattr(sp, "save_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sp, "compute_scene_motion", lambda *_args, **_kwargs: 20.0)
    monkeypatch.setattr(sp, "classify_motion", lambda _score: "MEDIUM")
    monkeypatch.setattr(sp, "detect_scene_objects", lambda *_args, **_kwargs: ["person", "car"])
    monkeypatch.setattr(sp, "compute_importance_from_features", lambda **_kwargs: 0.77)

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
