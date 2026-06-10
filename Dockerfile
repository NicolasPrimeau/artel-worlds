FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.11 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

COPY automata/ automata/

ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --no-create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "automata.server:app", "--host", "0.0.0.0", "--port", "8000"]
