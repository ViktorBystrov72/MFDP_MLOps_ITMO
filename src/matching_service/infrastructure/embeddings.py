"""Semantic scoring for deal↔listing pairs using a configurable embedding model."""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
DEFAULT_MODEL = os.getenv(
    "MATCH_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
PRODUCTION_MODEL = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def _load_sentence_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    kwargs: dict[str, object] = {
        "device": os.getenv("MATCH_EMBEDDING_DEVICE") or None,
        "trust_remote_code": True,
    }
    if token:
        kwargs["token"] = token
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return SentenceTransformer(model_name, **kwargs)


@lru_cache(maxsize=1)
def _load_bge_model(model_name: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    kwargs: dict[str, object] = {"trust_remote_code": True}
    if token:
        kwargs["token"] = token
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    model = AutoModel.from_pretrained(model_name, **kwargs)
    requested = os.getenv("MATCH_EMBEDDING_DEVICE")
    if requested:
        device = requested
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model.to(device)
    model.eval()
    if os.getenv("MATCH_TORCH_COMPILE", "0") == "1":
        model = torch.compile(
            model,
            mode=os.getenv("MATCH_TORCH_COMPILE_MODE", "reduce-overhead"),
        )
        logger.info("Enabled torch.compile for embedding model on %s", device)
    return tokenizer, model, device


def _embed_bge_cls(texts: list[str], model_name: str) -> np.ndarray:
    import torch
    from torch.nn import functional

    tokenizer, model, device = _load_bge_model(model_name)
    batch_size = int(os.getenv("MATCH_EMBEDDING_BATCH_SIZE", "64"))
    max_length = int(os.getenv("MATCH_EMBEDDING_MAX_LENGTH", "8192"))
    batches: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = model(**encoded)
        normalized = functional.normalize(output.last_hidden_state[:, 0], p=2, dim=1)
        batches.append(normalized.cpu().numpy().astype(np.float32))
    return np.concatenate(batches, axis=0)


def embed_texts(texts: list[str], model_name: str | None = None) -> np.ndarray:
    selected_model = model_name or DEFAULT_MODEL
    cleaned = [str(t or "").strip() or " " for t in texts]
    backend = os.getenv("MATCH_EMBEDDING_BACKEND", "auto")
    if backend == "bge-cls" or (
        backend == "auto" and "bge-m3" in selected_model.lower()
    ):
        return _embed_bge_cls(cleaned, selected_model)
    model = _load_sentence_model(selected_model)
    vectors = model.encode(
        cleaned,
        batch_size=int(os.getenv("MATCH_EMBEDDING_BATCH_SIZE", "64")),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def cosine_pair_scores(
    left: list[str], right: list[str], model_name: str | None = None
) -> np.ndarray:
    """Return cosine similarity for aligned deal/listing text pairs."""
    if len(left) != len(right):
        raise ValueError("left and right must contain the same number of rows")
    uniq = list(
        dict.fromkeys([str(value or "").strip() or " " for value in [*left, *right]])
    )
    mat = embed_texts(uniq, model_name=model_name)
    idx = {t: i for i, t in enumerate(uniq)}
    li = np.array([idx[str(x or "").strip() or " "] for x in left])
    ri = np.array([idx[str(x or "").strip() or " "] for x in right])
    return np.sum(mat[li] * mat[ri], axis=1)


def _value(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series([""] * len(frame), index=frame.index, dtype="string")
    return frame[column].fillna("").astype(str)


def build_structured_texts(frame: pd.DataFrame, side: str) -> list[str]:
    if side == "deal":
        columns = [
            ("номер", "flat_number_deal"),
            ("номер_пд", "planned_premise_number"),
            ("этаж", "floor_deal"),
            ("подъезд", "entrance_deal"),
            ("позиция_на_этаже", "number_on_floor_deal"),
            ("площадь", "area_deal"),
            ("жилая_площадь_пд", "pd_living_area"),
            ("комнаты", "room_count_deal"),
            ("назначение_пд", "pd_purpose"),
            ("описание_объекта", "object_description_deal"),
            ("описание_локации", "location_description_deal"),
            ("договор", "realisation_contract"),
        ]
        prefix = "query: сделка"
    else:
        columns = [
            ("номер", "flat_number_exp"),
            ("этаж", "floor_exp"),
            ("подъезд", "entrance_exp"),
            ("позиция_на_этаже", "number_on_floor_exp"),
            ("площадь", "area_exp"),
            ("жилая_площадь", "living_area_exp"),
            ("комнаты", "room_count_exp"),
            ("источник", "source_name"),
            ("описание", "description_exp"),
        ]
        prefix = "passage: объявление"

    rows: list[str] = []
    values = {name: _value(frame, column) for name, column in columns}
    for position in range(len(frame)):
        parts = [prefix]
        for name, _ in columns:
            value = values[name].iloc[position].strip()
            if value:
                parts.append(f"{name}={value}")
        rows.append("; ".join(parts))
    return rows


def add_embedding_scores(
    df: pd.DataFrame,
    model_name: str | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    """Add a semantic score using descriptions and PD-aware structured context."""
    out = df.copy()
    left = build_structured_texts(out, "deal")
    right = build_structured_texts(out, "listing")
    selected_model = model_name or DEFAULT_MODEL
    try:
        out["emb_score"] = cosine_pair_scores(left, right, model_name=selected_model)
        out["embedding_model"] = selected_model
    except Exception as exc:
        if strict:
            raise
        logger.warning("Embedding inference failed for %s: %s", selected_model, exc)
        out["emb_score"] = np.nan
        out["emb_error"] = str(exc)
    return out
