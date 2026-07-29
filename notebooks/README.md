# Ноутбуки уроков (матчинг экспозиция ↔ сделки)

Все ноутбуки **выполнены** — выводы ячеек отображаются. Данные: пилот Тюмень из продакшен-БД Neolithic (`artifacts/datasets/tyumen_pairs*.parquet`).

| Ноутбук | Урок | Содержание |
|---------|------|------------|
| `lesson4_eda.ipynb` | 4 | EDA датасета: объём, баланс классов, пропуски, распределения признаков, выводы для моделирования. |
| `lesson6_baseline.ipynb` | 5–6 | Baseline: EDA + 4 модели (rule / logistic / RF / CatBoost), метрики Precision/Recall/F1/PR-AUC/ROC-AUC и сводный composite, MLflow. |
| `lesson7_improve.ipynb` | 7 | Улучшение модели: fuzzy-признак номера квартиры, калибровка порога по F1, очередь ручной проверки, разбор ошибок. |

Запуск: из папки `notebooks/` через Jupyter / `jupyter nbconvert --execute`, нужен `artifacts/datasets/*.parquet` (см. `scripts/README.md`).
