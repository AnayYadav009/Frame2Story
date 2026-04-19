
from modules.summarization import extractive_summarizer as es


def test_select_sentence_count_thresholds():
    assert es.select_sentence_count(10) == 2
    assert es.select_sentence_count(20) == 3
    assert es.select_sentence_count(60) == 5


def test_language_mapping_includes_indian_codes_and_default():
    assert es._to_sumy_language("hi") == "english"
    assert es._to_sumy_language("bn") == "english"
    assert es._to_sumy_language("te") == "english"
    assert es._to_sumy_language("es") == "spanish"
    assert es._to_sumy_language("unknown-code") == "english"


def test_detect_language_falls_back_when_detector_unavailable(monkeypatch):
    monkeypatch.setattr(es, "_HAS_LANGDETECT", False)
    assert es.detect_language("Hola, como estas?") == "en"
    assert es._detect_sumy_language("Hola, como estas?") == "english"


def test_detect_language_returns_detector_result(monkeypatch):
    monkeypatch.setattr(es, "_HAS_LANGDETECT", True)
    monkeypatch.setattr(es, "detect", lambda _text: "pt")

    assert es.detect_language("Ola mundo") == "pt"


def test_extractive_summary_uses_deterministic_fallback_when_ranker_fails(monkeypatch):
    class _BrokenSummarizer:
        def __call__(self, _document, _sentence_target):
            raise RuntimeError("forced failure")

    monkeypatch.setattr(es, "TextRankSummarizer", lambda: _BrokenSummarizer())

    text = "First sentence. Second sentence. Third sentence."
    summary = es.extractive_summary_from_text(text, language="en")

    # For <20 sentences, fallback keeps first 2.
    assert "First sentence." in summary
    assert "Second sentence." in summary
    assert "Third sentence." not in summary
