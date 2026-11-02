import os


def env(key: str, default: str = "") -> str:
    return os.environ.get(f"VERGLAS_{key}") or default
