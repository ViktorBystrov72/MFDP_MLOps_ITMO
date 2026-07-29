# Matching Service — запуск

Сервис матчинга экспозиции и сделок (пилот Тюмень).

## Каскад

`rules → CatBoost → embeddings (BGE-M3) → reranker (BGE-reranker-v2-m3) → LiteLLM (ambiguous)`

## Модели

- **CatBoost** — обучается скриптом `scripts/train_baseline.py`, артефакт сохраняется в `artifacts/models/`.
- **Embeddings BGE-M3** и **reranker BGE-reranker-v2-m3** — подтягиваются с **HuggingFace** при первом вызове (`HF_TOKEN`), lazy-load в процессе API/worker.
- **LiteLLM** — внутренний OpenAI-совместимый шлюз (`LITELLM_BASE_URL`, `LITELLM_API_KEY`).

Env: `MATCH_EMBEDDING_MODEL`, `MATCH_RERANKER_MODEL`, `MATCH_USE_EMBEDDINGS`, `MATCH_USE_RERANKER`, `MATCH_USE_LLM`.

## Слои (DDD)

```
src/matching_service/
  domain/          # сущности и признаки
  application/     # dataset, train, cascade, llm_match
  infrastructure/  # Postgres, embeddings, lite_llm, metrics
  presentation/    # FastAPI + Streamlit
```

## Локальный запуск (dev)

```bash
source venv/bin/activate
export PYTHONPATH=src
uvicorn matching_service.presentation.api:app --reload --port 8000
streamlit run src/matching_service/presentation/ui_streamlit.py
```

## Docker Compose (полный стек)

```bash
docker compose up --build
```

| Сервис | URL |
|--------|-----|
| API | http://localhost:8000/docs |
| Metrics | http://localhost:8000/metrics |
| UI | http://localhost:8501 |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

Airflow (отдельный профиль):

```bash
docker compose --profile airflow up airflow
# UI :8081, DAG exposition_deal_matching
```

Batch cascade worker:

```bash
MATCH_USE_EMBEDDINGS=1 MATCH_USE_LLM=1 docker compose --profile batch up ml_worker
# или локально:
python scripts/run_cascade_batch.py --embeddings --llm --limit 100
```

## Env

- `LITELLM_API_KEY` / `LITELLM_BASE_URL` — внутренний шлюз (как в neolithic-airflow)
- `HF_TOKEN` — HuggingFace для embeddings
- `MATCH_USE_EMBEDDINGS` / `MATCH_USE_LLM` — флаги в worker/Airflow
- `MLFLOW_TRACKING_URI` — по умолчанию `http://mlflow:5000` в compose

## Данные и обучение

```bash
export PYTHONPATH=src
python scripts/build_tyumen_dataset.py
python scripts/train_baseline.py
```

В сдаваемых материалах: **продакшен-БД** и **серверы компании с GPU**.
