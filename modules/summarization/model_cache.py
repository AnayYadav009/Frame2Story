"""Shared cache for transformer summarization models.

This module ensures each model name is loaded once per process and reused
across summarization components.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer


_MODEL_CACHE: Dict[str, Tuple[Any, Any, Any, str]] = {}


def get_model_components(model_name: str) -> Tuple[Any, Any, Any, str]:
    """Return cached (tokenizer, config, model, device) for a model name."""
    if model_name not in _MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        _MODEL_CACHE[model_name] = (tokenizer, config, model, device)

    return _MODEL_CACHE[model_name]