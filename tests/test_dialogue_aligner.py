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
    assert da.clean_dialogue("  ROBIN:\nHi there  ") == "ROBIN: Hi there"


def test_align_dialogue_to_scenes_half_open_boundary():
    scenes = [
        {"scene_id": 1, "start": 0, "end": 10},
        {"scene_id": 2, "start": 10, "end": 20},
    ]
    subs = [
        _subtitle((0, 0, 5, 0), "  hi\nthere  "),
        _subtitle((0, 0, 10, 0), "boundary line"),
        _subtitle((0, 0, 19, 0), "last"),
    ]

    aligned = da.align_dialogue_to_scenes(subs, scenes)

    assert aligned == {
        "1": ["hi there"],
        "2": ["boundary line", "last"],
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
