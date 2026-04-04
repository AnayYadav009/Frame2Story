from types import SimpleNamespace

import pytest

from modules.dialogue import dialogue_aligner as da


def _time(hours: int, minutes: int, seconds: int, milliseconds: int):
    return SimpleNamespace(
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        milliseconds=milliseconds,
    )


def _subtitle(start_tuple, text: str):
    return SimpleNamespace(start=_time(*start_tuple), text=text)


def test_time_to_seconds_and_clean_dialogue():
    assert da.time_to_seconds(_time(0, 1, 2, 500)) == pytest.approx(62.5)
    assert da.clean_dialogue("  ROBIN:\nHi there  ") == "Hi there"


def test_extract_speaker_handles_tagged_and_untagged_lines():
    speaker, line = da.extract_speaker("TED: We need to talk")
    assert speaker == "TED"
    assert line == "We need to talk"

    speaker, line = da.extract_speaker("No speaker tag here")
    assert speaker is None
    assert line == "No speaker tag here"


def test_detect_subtitle_language_uses_first_lines(monkeypatch):
    seen = {}

    def _fake_detect(text):
        seen["text"] = text
        return "es"

    monkeypatch.setattr(da, "detect_language", _fake_detect)

    subs = [
        _subtitle((0, 0, 1, 0), "Hola"),
        _subtitle((0, 0, 2, 0), "Como estas"),
        _subtitle((0, 0, 3, 0), "Ignored"),
    ]

    language = da.detect_subtitle_language(subs, sample_size=2)

    assert language == "es"
    assert seen["text"] == "Hola Como estas"


def test_align_dialogue_to_scenes_half_open_boundary(monkeypatch):
    monkeypatch.setattr(da, "detect_language", lambda _text: "en")

    scenes = [
        {"scene_id": 1, "start": 0, "end": 10},
        {"scene_id": 2, "start": 10, "end": 20},
    ]
    subs = [
        _subtitle((0, 0, 5, 0), "  TED:\nhi\nthere  "),
        _subtitle((0, 0, 10, 0), "boundary line"),
        _subtitle((0, 0, 19, 0), "last"),
    ]

    aligned, language = da.align_dialogue_to_scenes(subs, scenes)

    assert language == "en"

    assert aligned == {
        "1": [{"speaker": "TED", "line": "hi there"}],
        "2": [
            {"speaker": None, "line": "boundary line"},
            {"speaker": None, "line": "last"},
        ],
    }


def test_load_subtitles_encoding_fallback(monkeypatch):
    calls = []

    def fake_open(_path, encoding):
        calls.append(encoding)
        if encoding != "latin-1":
            raise UnicodeDecodeError("decoder", b"x", 0, 1, "bad encoding")
        return ["ok"]

    monkeypatch.setattr(da.pysrt, "open", fake_open)

    result = da.load_subtitles("dummy.srt")

    assert result == ["ok"]
    assert calls == ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
