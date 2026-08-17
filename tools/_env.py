from __future__ import annotations

# Credentials for the local dev tools, read from the repo's own .env.
#
# This used to reach into /home/nprimeau/projects/Artel/.env by absolute path and pick
# GROQ_API_KEY out of it by hand, which made these tools depend on an unrelated project
# living at one particular path on one particular machine. The keys live here now, and
# scripts/sync_env_secrets.py keeps them in AWS Secrets Manager (artel-worlds/env).

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load() -> dict[str, str]:
    if not ENV_FILE.exists():
        return {}
    values: dict[str, str] = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def require(name: str) -> str:
    """A credential the caller cannot run without.

    Raises rather than returning "" on purpose. A missing key here does not crash the
    tool it feeds — the provider just answers 401 and the run completes with an LLM that
    never said anything, which reads exactly like a real A/B result showing the model is
    worthless. Better to refuse to start than to publish a measurement of nothing.
    """
    value = os.environ.get(name) or load().get(name, "")
    if not value:
        raise SystemExit(
            f"{name} is not set. Put it in {ENV_FILE} "
            f"(uv run python scripts/sync_env_secrets.py --pull) or export it."
        )
    return value
