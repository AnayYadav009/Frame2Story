"""Lightweight extractive summarization utilities."""

from __future__ import annotations

import re
from typing import List

from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


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


def extractive_summary_from_text(text: str) -> str:
    """Summarize input text extractively with TextRank."""
    sentences = _split_sentences(text)
    if not sentences:
        return ""

    sentence_target = min(select_sentence_count(len(sentences)), len(sentences))
    parser = PlaintextParser.from_string(" ".join(sentences), Tokenizer("english"))
    summarizer = TextRankSummarizer()
    ranked = summarizer(parser.document, sentence_target)
    return " ".join(str(sentence).strip() for sentence in ranked if str(sentence).strip())
