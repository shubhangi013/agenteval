FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
ARG PIP_INDEX_URL=https://pypi.org/simple
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --index-url "$PIP_INDEX_URL" .
COPY goldens ./goldens
COPY scripts ./scripts

CMD ["uvicorn", "agenteval.api:app", "--host", "0.0.0.0", "--port", "8000"]
