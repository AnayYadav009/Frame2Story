"""Lightweight extractive summarization utilities."""

from __future__ import annotations

import importlib
import re
from typing import List

from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer

try:
    _langdetect = importlib.import_module("langdetect")
    DetectorFactory = getattr(_langdetect, "DetectorFactory", None)
    LangDetectException = getattr(_langdetect, "LangDetectException", Exception)
    detect = getattr(_langdetect, "detect", None)

    if DetectorFactory is not None:
        DetectorFactory.seed = 0
    _HAS_LANGDETECT = callable(detect)
except Exception:
    LangDetectException = Exception
    detect = None
    _HAS_LANGDETECT = False

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_LANG_CODE_TO_SUMY = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "hi": "english",  # Hindi
    "bn": "english",  # Bengali
    "mr": "english",  # Marathi
    "te": "english",  # Telugu
    "ta": "english",  # Tamil
    "kn": "english",  # Kannada
    "ml": "english",  # Malayalam
}


def _split_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def select_sentence_count(sentence_count: int) -> int:
    if sentence_count < 20:
        return 2
    if sentence_count <= 50:
        return 3
    return 5


def _to_sumy_language(language: str | None) -> str:
    if not language:
        return "english"

    base = language.strip().lower().split("-")[0]
    return _LANG_CODE_TO_SUMY.get(base, "english")


def detect_language(text: str) -> str:
    if not _HAS_LANGDETECT or detect is None:
        return "en"

    try:
        detected = detect(text or "")
        if isinstance(detected, str) and detected.strip():
            return detected.strip().lower()
    except (LangDetectException, ValueError, TypeError):
        pass

    return "en"


def _detect_sumy_language(text: str) -> str:
    return _to_sumy_language(detect_language(text))


def _build_tokenizer(language: str) -> Tokenizer:
    try:
        return Tokenizer(language)
    except Exception:
        return Tokenizer("english")


def extractive_summary_from_text(text: str, language: str | None = None) -> str:
    """Summarize input text extractively with TextRank.

    If `language` is not provided, language is auto-detected from the text.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return ""

    sentence_target = min(select_sentence_count(len(sentences)), len(sentences))
    prepared_text = " ".join(sentences)

    sumy_language = _to_sumy_language(language) if language else _detect_sumy_language(prepared_text)
    parser = PlaintextParser.from_string(prepared_text, _build_tokenizer(sumy_language))
    summarizer = TextRankSummarizer()

    try:
        ranked = summarizer(parser.document, sentence_target)
        summary = " ".join(str(sentence).strip() for sentence in ranked if str(sentence).strip())
        if summary:
            return summary
    except Exception:
        pass

    return " ".join(sentences[:sentence_target])
