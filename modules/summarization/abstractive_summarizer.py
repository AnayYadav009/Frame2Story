"""Lightweight abstractive summarization utilities with batch support."""

from __future__ import annotations

from typing import Any, Tuple, List

import torch
from modules.summarization.model_cache import get_model_components as get_cached_model_components

MODEL_NAME = "philschmid/bart-large-cnn-samsum"


def get_model_components() -> Tuple[Any, Any, Any, str]:
    return get_cached_model_components(MODEL_NAME)


def summarize_text_batch(
    texts: List[str], 
    max_length: int = 80, 
    min_length: int = 25,
    batch_size: int = 8
) -> List[str]:
    """Summarize multiple texts in batches for high throughput."""
    if not texts:
        return []

    cleaned_texts = []
    valid_indices = []
    for idx, t in enumerate(texts):
        if t and t.strip():
            cleaned_texts.append(t)
            valid_indices.append(idx)
    
    results = [""] * len(texts)
    if not cleaned_texts:
        return results

    tokenizer, _config, model, device = get_model_components()
    max_input_tokens = getattr(tokenizer, "model_max_length", 1024)
    if not isinstance(max_input_tokens, int) or max_input_tokens <= 0 or max_input_tokens > 100000:
        max_input_tokens = 1024

    all_processed_summaries = []

    for i in range(0, len(cleaned_texts), batch_size):
        chunk = cleaned_texts[i : i + batch_size]
        
        # Tokenize on CPU first, then move to device only if input is valid
        batch_inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_input_tokens,
        )

        if "input_ids" not in batch_inputs or batch_inputs["input_ids"].numel() == 0:
            all_processed_summaries.extend([""] * len(chunk))
            continue

        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                batch_inputs["input_ids"],
                attention_mask=batch_inputs.get("attention_mask"),
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True,
                do_sample=False,
            )

        if hasattr(tokenizer, "batch_decode"):
            summaries = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        else:
            # Fallback for simple mock tokenizers in tests
            summaries = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        
        all_processed_summaries.extend([s.strip() for s in summaries])

    # Map back to original indices
    for idx, summary in zip(valid_indices, all_processed_summaries):
        results[idx] = summary

    return results


def abstractive_summarize_text(text: str, max_length: int = 80, min_length: int = 25) -> str:
    """Summarize text abstractively (legacy API for backward compatibility)."""
    return summarize_text(text, max_length=max_length, min_length=min_length)


def summarize_text(text: str, max_length: int = 80, min_length: int = 25) -> str:
    """Summarize text abstractively with cached BART model."""
    if not text or not text.strip():
        return ""
    results = summarize_text_batch([text], max_length=max_length, min_length=min_length)
    return results[0] if results else ""
