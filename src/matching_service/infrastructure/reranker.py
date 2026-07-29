"""Optional cross-encoder reranker for deal↔listing pairs."""

from __future__ import annotations

import logging
import math
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from matching_service.infrastructure.embeddings import build_structured_texts

logger = logging.getLogger(__name__)
DEFAULT_RERANKER_MODEL = os.getenv(
    "MATCH_RERANKER_MODEL",
    "BAAI/bge-reranker-v2-m3",
)


@lru_cache(maxsize=1)
def _load_reranker(model_name: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    kwargs: dict[str, object] = {"trust_remote_code": True}
    if token:
        kwargs["token"] = token
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        **kwargs,
    )
    requested = os.getenv("MATCH_RERANKER_DEVICE")
    device = requested or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    if os.getenv("MATCH_TORCH_COMPILE", "0") == "1":
        model = torch.compile(
            model,
            mode=os.getenv("MATCH_TORCH_COMPILE_MODE", "reduce-overhead"),
        )
        logger.info("Enabled torch.compile for reranker model on %s", device)
    return tokenizer, model, device


def reranker_scores(
    queries: list[str],
    passages: list[str],
    model_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(queries) != len(passages):
        raise ValueError("queries and passages must contain the same number of rows")
    import torch

    selected_model = model_name or DEFAULT_RERANKER_MODEL
    tokenizer, model, device = _load_reranker(selected_model)
    batch_size = int(os.getenv("MATCH_RERANKER_BATCH_SIZE", "32"))
    raw_scores: list[float] = []
    for start in range(0, len(queries), batch_size):
        pairs = list(
            zip(
                queries[start : start + batch_size],
                passages[start : start + batch_size],
                strict=True,
            )
        )
        encoded = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits.view(-1)
        raw_scores.extend(float(value) for value in logits.cpu())
    raw = np.asarray(raw_scores, dtype=np.float32)
    probability = np.asarray([1.0 / (1.0 + math.exp(-float(value))) for value in raw])
    return raw, probability


def add_reranker_scores(
    frame: pd.DataFrame,
    model_name: str | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    out = frame.copy()
    selected_model = model_name or DEFAULT_RERANKER_MODEL
    try:
        raw, probability = reranker_scores(
            build_structured_texts(out, "deal"),
            build_structured_texts(out, "listing"),
            model_name=selected_model,
        )
        out["reranker_raw_score"] = raw
        out["reranker_score"] = probability
        out["reranker_model"] = selected_model
    except Exception as exc:
        if strict:
            raise
        logger.warning("Reranker inference failed for %s: %s", selected_model, exc)
        out["reranker_raw_score"] = np.nan
        out["reranker_score"] = np.nan
        out["reranker_error"] = str(exc)
    return out
