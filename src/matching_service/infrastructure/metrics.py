"""Prometheus-метрики сервиса матчинга."""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

MATCH_REQUESTS = Counter(
    "matching_requests_total",
    "Число batch-запросов матчинга",
    ["endpoint"],
)
MATCH_PAIRS = Counter(
    "matching_pairs_total",
    "Число обработанных пар",
    ["result"],  # match | review | no_match
)
MATCH_SCORE = Histogram(
    "matching_score",
    "Распределение ensemble/cb score",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)
MATCH_LATENCY = Histogram(
    "matching_latency_seconds",
    "Латентность batch-инференса",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
LLM_CALLS = Counter(
    "matching_llm_calls_total",
    "Вызовы LiteLLM для сложных кейсов",
    ["status"],
)
MODEL_LOADED = Gauge("matching_model_loaded", "1 если CatBoost загружен")


def metrics_response() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
