FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[llm]"

COPY data/docs ./data/docs
RUN python -m docanswer.ingest

RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "docanswer.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
