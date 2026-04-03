from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from bert_score import score as bert_score
from rouge_score import rouge_scorer


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def compute_rouge_scores(generated_recap: str, reference_text: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference_text, generated_recap)

    rouge1 = scores.get("rouge1")
    rouge_l = scores.get("rougeL")

    return {
        "rouge1_precision": _safe_float(getattr(rouge1, "precision", 0.0)),
        "rouge1_recall": _safe_float(getattr(rouge1, "recall", 0.0)),
        "rouge1_f1": _safe_float(getattr(rouge1, "fmeasure", 0.0)),
        "rougeL_precision": _safe_float(getattr(rouge_l, "precision", 0.0)),
        "rougeL_recall": _safe_float(getattr(rouge_l, "recall", 0.0)),
        "rougeL_f1": _safe_float(getattr(rouge_l, "fmeasure", 0.0)),
    }


def compute_bert_score(generated_recap: str, reference_text: str) -> Dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision, recall, f1 = bert_score(
        [generated_recap],
        [reference_text],
        lang="en",
        device=device,
        verbose=False,
    )

    return {
        "precision": _safe_float(precision.mean().item()),
        "recall": _safe_float(recall.mean().item()),
        "f1": _safe_float(f1.mean().item()),
    }


def save_eval_scores(scores: Dict[str, Any], output_path: str = "outputs/eval/scores.json") -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    return str(output)


def evaluate_recap(
    generated_recap: str,
    reference_text: str,
    output_path: str = "outputs/eval/scores.json",
) -> Dict[str, Any]:
    if not generated_recap or not generated_recap.strip():
        raise ValueError("generated_recap is empty")
    if not reference_text or not reference_text.strip():
        raise ValueError("reference_text is empty")

    generated = generated_recap.strip()
    reference = reference_text.strip()

    rouge = compute_rouge_scores(generated, reference)
    bert = compute_bert_score(generated, reference)

    payload: Dict[str, Any] = {
        "generated_length_words": len(generated.split()),
        "reference_length_words": len(reference.split()),
        "rouge": rouge,
        "bert_score": bert,
    }

    saved_to = save_eval_scores(payload, output_path=output_path)
    payload["saved_to"] = saved_to
    return payload
