FROM ghcr.io/nicolasprimeau/artel:edge

USER root

COPY --from=caddy:2.8-alpine /usr/bin/caddy /usr/local/bin/caddy

WORKDIR /worlds

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

COPY llmrouter/ llmrouter/
COPY automata/ automata/
COPY phalanx/ phalanx/
COPY pitch/ pitch/
COPY verglas/ verglas/
COPY vibequest/ vibequest/
COPY watchtower/ watchtower/
COPY bundle/ bundle/

RUN chmod +x bundle/run.sh && chown -R artel /worlds

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

ENTRYPOINT ["/worlds/bundle/run.sh"]
CMD []
