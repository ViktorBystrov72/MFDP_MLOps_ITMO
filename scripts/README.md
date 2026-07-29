# Скрипты пайплайна матчинга

Все запускаются из корня репозитория с `source venv/bin/activate` и `export PYTHONPATH=src`.

| Скрипт | Что делает |
|--------|------------|
| `build_tyumen_dataset.py` | Собирает датасет пар «экспозиция ↔ сделка» из продакшен-БД Neolithic (пилот Тюмень): positive из rule-сцепки, negative — случайные объявления того же корпуса. Сохраняет `artifacts/datasets/tyumen_pairs*.parquet`. |
| `train_baseline.py` | Обучает baseline-модели (rule / logistic / RF / CatBoost) на train/holdout, пишет метрики в `artifacts/metrics_baseline.txt`, логирует в MLflow. |
| `run_cascade_batch.py` | Batch-воркер каскада `rules → CatBoost → embeddings (BGE-M3) → LiteLLM` на parquet. Флаги `--embeddings`, `--llm`, `--limit`. Результат — `artifacts/cascade_batch.parquet`. |

Типовой порядок:

```bash
python scripts/build_tyumen_dataset.py   # 1. данные
python scripts/train_baseline.py         # 2. baseline
python scripts/run_cascade_batch.py --embeddings --llm --limit 100   # 3. каскад (опционально)
```
