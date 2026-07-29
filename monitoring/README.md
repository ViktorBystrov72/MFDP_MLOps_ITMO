# Мониторинг сервиса матчинга (Prometheus + Grafana)

## Как поднять

В составе полного стека:

```bash
docker compose up --build
```

| Сервис | URL |
|--------|-----|
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |
| Метрики API | http://localhost:8000/metrics |

Prometheus скрапит `/metrics` FastAPI-сервиса (`prometheus.yml`). Grafana поднимается с provisioned datasource и дашбордом (`grafana/provisioning/`).

## Какие метрики смотрит

Регистрируются в `src/matching_service/infrastructure/metrics.py`:

| Метрика | Тип | Что показывает |
|---------|-----|----------------|
| `matching_requests_total` | Counter | число batch-запросов матчинга (по endpoint) |
| `matching_pairs_total` | Counter | обработанные пары по исходу: `match` / `review` / `no_match` |
| `matching_score` | Histogram | распределение ensemble/CatBoost score |
| `matching_latency_seconds` | Histogram | латентность batch-инференса |
| `matching_llm_calls_total` | Counter | вызовы LiteLLM для сложных кейсов (по статусу) |
| `matching_model_loaded` | Gauge | 1, если CatBoost-модель загружена |

## Дашборд

`grafana/provisioning/dashboards/json/matching.json` — панели по запросам, исходам пар, латентности и score. Смотреть: рост `review`/LLM-вызовов (сложные кейсы), деградацию латентности, смещение распределения score (дрейф).
