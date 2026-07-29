# DevOps: Docker / Compose / MLflow

Файлы окружения сервиса (не корень репо).

| Файл | Назначение |
|------|------------|
| `Dockerfile` | образ API / UI / worker |
| `Dockerfile.airflow` | образ Airflow |
| `docker-compose.yml` | стек matching (api, ui, mlflow, monitoring, profiles) |
| `ci/github-actions.yml` | зеркало пайплайна GitHub Actions |

MLflow SQLite (`mlflow.db`) живёт в Docker volume `mlflow-data`, не в репо.  
Локальный train без Compose: `artifacts/mlflow.db` (`scripts/train_baseline.py`).

`.dockerignore` — в **корне** репо (контекст Docker build).  
Рабочий CI: `.github/workflows/ci.yml` (зеркало — `devops/ci/github-actions.yml`).

Миграции: `alembic/alembic.ini`, команда `make migrate`.

## MinIO (file storage)

S3-сторадж для внешних датасетов / артефактов:

```bash
docker compose -f devops/docker-compose.yml up -d minio
docker compose -f devops/docker-compose.yml --profile storage up minio-init
```

API `:9000`, console `:9001`, бакеты `matching-datasets`, `matching-artifacts`.

## CI (GitHub Actions)

Пайплайн: `lint` → `test` + `migrate` (Postgres service), команды как в `Makefile`.  
Область lint — `src`, `scripts`, `tests`, `dags`, `alembic`.

## Запуск

Из **корня** репозитория:

```bash
docker compose -f devops/docker-compose.yml up --build
docker compose -f devops/docker-compose.yml --profile airflow up airflow
docker compose -f devops/docker-compose.yml --profile batch up ml_worker
docker compose -f devops/docker-compose.yml --profile models up -d location_matcher
```
