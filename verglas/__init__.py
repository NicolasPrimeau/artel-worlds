import os


def env(key: str, default: str = "") -> str:
    """A Verglas setting, under either prefix.

    The ALIBI_ arm is NOT dead legacy. The deployed app is still literally named
    artel-alibi and every secret on it is ALIBI_-prefixed (ALIBI_LLM_KEY,
    ALIBI_CEREBRAS_KEY, ALIBI_LLM2_KEY, ALIBI_AGENT_IDS, ALIBI_AGENT_KEYS) — there is
    not one VERGLAS_ secret set in production. Delete this fallback and Verglas loses
    its keys and its Artel identity on the next deploy, silently: llmrouter just gets
    empty strings and every provider answers 401.

    Renaming for real means `fly secrets set` the VERGLAS_ names on artel-alibi first
    (VERGLAS_ already wins here, so that step is safe on its own), confirming the app
    is healthy, and only then removing this arm and unsetting the old names. It cannot
    be done from the code side alone.
    """
    return os.environ.get(f"VERGLAS_{key}") or os.environ.get(f"ALIBI_{key}") or default
