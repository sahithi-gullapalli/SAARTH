FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

ARG EXTRAS="llm,rag,ui"
COPY pyproject.toml README.md ./
COPY saarth ./saarth
RUN pip install ".[${EXTRAS}]" "psycopg[binary]>=3.1" "pgvector>=0.3"

COPY ui ./ui
COPY data ./data

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
HEALTHCHECK CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "saarth.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
