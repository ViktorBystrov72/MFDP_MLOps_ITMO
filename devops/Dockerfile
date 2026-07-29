FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
RUN python - <<'PY'
import subprocess
import sys
import tomllib

with open("pyproject.toml", "rb") as source:
    dependencies = tomllib.load(source)["project"]["dependencies"]
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *dependencies])
PY
COPY src ./src
COPY scripts ./scripts
COPY artifacts/models ./artifacts/models
RUN pip install --no-cache-dir --no-deps .
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "matching_service.presentation.api:app", "--host", "0.0.0.0", "--port", "8000"]
