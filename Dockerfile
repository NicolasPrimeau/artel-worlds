FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.11 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

COPY automata/ automata/

ENV PATH="/app/.venv/bin:$PATH"

# Claude Code CLI (native binary) for the claude-sdk provider; no token in the image,
# inert unless AUTOMATA_LLM_PROVIDER=claude-sdk
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://claude.ai/install.sh | bash \
    && install -m 0755 /root/.local/bin/claude /usr/local/bin/claude \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

ENV DISABLE_AUTOUPDATER=1

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "automata.server:app", "--host", "0.0.0.0", "--port", "8000"]
