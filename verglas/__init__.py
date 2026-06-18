import os


def env(key: str, default: str = "") -> str:
    return os.environ.get(f"VERGLAS_{key}") or os.environ.get(f"ALIBI_{key}") or default
