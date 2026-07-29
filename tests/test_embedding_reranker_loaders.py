from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from matching_service.infrastructure import embeddings, reranker


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def test_sentence_and_bge_embedding_loaders(monkeypatch):
    embeddings._load_sentence_model.cache_clear()
    embeddings._load_bge_model.cache_clear()

    fake_st = MagicMock()
    fake_st.SentenceTransformer.return_value.encode.return_value = np.ones((2, 4))
    sys.modules["sentence_transformers"] = fake_st

    monkeypatch.setenv("MATCH_EMBEDDING_BACKEND", "sentence")
    monkeypatch.setenv("HF_TOKEN", "token")
    vectors = embeddings.embed_texts(["a", "b"], model_name="mini")
    assert vectors.shape == (2, 4)

    torch_mod = MagicMock()
    torch_mod.cuda.is_available.return_value = False
    torch_mod.backends.mps.is_available.return_value = False
    torch_mod.no_grad = lambda: _NoGrad()
    torch_mod.compile = lambda model, mode=None: model
    functional = MagicMock()
    functional.normalize.return_value.cpu.return_value.numpy.return_value = np.ones(
        (2, 4),
        dtype=np.float32,
    )
    torch_mod.nn.functional = functional
    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = torch_mod.nn
    sys.modules["torch.nn.functional"] = functional

    transformers = MagicMock()
    tokenizer = MagicMock()
    tokenizer.return_value.to.return_value = {"input_ids": object()}
    model = MagicMock()
    model.return_value = SimpleNamespace(
        last_hidden_state=np.zeros((2, 1, 4)),
    )
    hidden = MagicMock()
    hidden.__getitem__.return_value = object()
    model.return_value = SimpleNamespace(last_hidden_state=hidden)
    transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
    transformers.AutoModel.from_pretrained.return_value = model
    sys.modules["transformers"] = transformers

    monkeypatch.setenv("MATCH_EMBEDDING_BACKEND", "bge-cls")
    monkeypatch.setenv("MATCH_TORCH_COMPILE", "1")
    vectors_bge = embeddings._embed_bge_cls(["a", "b"], "BAAI/bge-m3")
    assert vectors_bge.shape[0] == 2

    monkeypatch.setattr(
        embeddings,
        "embed_texts",
        lambda texts, model_name=None: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    failed = embeddings.add_embedding_scores(pd.DataFrame({"flat_number_deal": ["1"]}))
    assert "emb_error" in failed.columns


def test_reranker_loader_and_scores(monkeypatch):
    reranker._load_reranker.cache_clear()
    torch_mod = MagicMock()
    torch_mod.cuda.is_available.return_value = False
    torch_mod.no_grad = lambda: _NoGrad()
    torch_mod.compile = lambda model, mode=None: model
    sys.modules["torch"] = torch_mod

    transformers = MagicMock()
    tokenizer = MagicMock()
    encoded = MagicMock()
    encoded.to.return_value = encoded
    tokenizer.return_value = encoded
    model = MagicMock()
    logits = MagicMock()
    logits.view.return_value.cpu.return_value = [0.0]
    model.return_value = SimpleNamespace(logits=logits)
    transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
    transformers.AutoModelForSequenceClassification.from_pretrained.return_value = model
    sys.modules["transformers"] = transformers

    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("MATCH_TORCH_COMPILE", "1")
    loaded = reranker._load_reranker("BAAI/bge-reranker-v2-m3")
    assert loaded[2] == "cpu"
    raw, prob = reranker.reranker_scores(["q"], ["p"], model_name="BAAI/bge-reranker-v2-m3")
    assert len(raw) == 1
    assert 0.0 <= float(prob[0]) <= 1.0

    monkeypatch.setattr(
        reranker,
        "reranker_scores",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    failed = reranker.add_reranker_scores(pd.DataFrame({"flat_number_deal": ["1"]}))
    assert "reranker_error" in failed.columns
