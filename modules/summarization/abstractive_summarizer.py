"""Lightweight abstractive summarization utilities."""

from __future__ import annotations

from typing import Any, Tuple

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "facebook/bart-large-cnn"
_TOKENIZER = None
_CONFIG = None
_MODEL = None
_DEVICE = None


def get_model_components() -> Tuple[Any, Any, Any, str]:
    global _TOKENIZER, _CONFIG, _MODEL, _DEVICE
    if _TOKENIZER is None or _CONFIG is None or _MODEL is None or _DEVICE is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        _CONFIG = AutoConfig.from_pretrained(MODEL_NAME)
        _MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL.to(_DEVICE)
        _MODEL.eval()
    return _TOKENIZER, _CONFIG, _MODEL, _DEVICE


def summarize_text(text: str, max_length: int = 80, min_length: int = 25) -> str:
    """Summarize text abstractively with cached BART model."""
    if not text or not text.strip():
        return ""

    tokenizer, _config, model, device = get_model_components()

    max_input_tokens = getattr(tokenizer, "model_max_length", 1024)
    if not isinstance(max_input_tokens, int) or max_input_tokens <= 0 or max_input_tokens > 100000:
        max_input_tokens = 1024

    encoded = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_input_tokens)
    truncated = tokenizer.decode(encoded.get("input_ids", []), skip_special_tokens=True)
    if not truncated:
        return ""

    inputs = tokenizer(
        truncated,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    if "input_ids" not in inputs or inputs["input_ids"].numel() == 0:
        return ""

    token_count = int(inputs["input_ids"].shape[-1])
    safe_max = max(10, min(max_length, token_count))
    safe_min = max(5, min(min_length, safe_max - 1))

    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_length=safe_max,
            min_length=safe_min,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
            do_sample=False,
        )

    if output_ids is None or output_ids.numel() == 0:
        return ""

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
