import subprocess

import pytest

from utils import audio_extractor as ae
from utils import input_handler as ih


def test_extract_audio_requires_existing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        ae.extract_audio(str(tmp_path / "missing.mp4"), str(tmp_path / "audio.wav"))


def test_extract_audio_runs_ffmpeg_and_returns_output(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video-bytes")
    output = tmp_path / "out" / "audio.wav"

    observed = {}

    def _fake_run(command, check, capture_output, text):
        observed["command"] = command
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"audio-bytes")

    monkeypatch.setattr(ae.subprocess, "run", _fake_run)

    result = ae.extract_audio(str(video), str(output))

    assert result == str(output)
    assert observed["command"][0] == "ffmpeg"


def test_extract_audio_surfaces_ffmpeg_stderr(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video-bytes")

    def _fake_run(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, ["ffmpeg"], stderr="decode failure")

    monkeypatch.setattr(ae.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="decode failure"):
        ae.extract_audio(str(video), str(tmp_path / "audio.wav"))


def test_get_subtitle_uses_existing_subtitle(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    subtitle = tmp_path / "subtitle.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

    monkeypatch.setattr(ih, "extract_audio", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not run")))

    assert ih.get_subtitle(str(video), str(subtitle)) == str(subtitle)


def test_get_subtitle_generates_when_missing(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")

    audio = tmp_path / "audio.wav"
    generated = tmp_path / "generated.srt"

    def _fake_extract_audio(_video_path):
        audio.write_bytes(b"audio")
        return str(audio)

    def _fake_transcribe(_audio_path, progress_callback=None):
        generated.write_text("subtitle", encoding="utf-8")
        return str(generated)

    monkeypatch.setattr(ih, "extract_audio", _fake_extract_audio)
    monkeypatch.setattr(ih, "transcribe_audio", _fake_transcribe)

    assert ih.get_subtitle(str(video), None) == str(generated)
