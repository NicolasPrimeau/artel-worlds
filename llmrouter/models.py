from __future__ import annotations

from dataclasses import dataclass, field

# A generic, world-agnostic model catalog for the router. Everything here is the OpenAI /chat/completions
# contract — add a provider or a model and it becomes routable. Free tiers today; paid models slot in the
# same way (tier="paid") and simply stay out of the pool unless a request opts in with allow_paid.

# provider -> base chat-completions URL. Override per-provider at build time if an endpoint moves.
PROVIDERS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "cerebras": "https://api.cerebras.ai/v1/chat/completions",
    "sambanova": "https://api.sambanova.ai/v1/chat/completions",
    "nvidia": "https://integrate.api.nvidia.com/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    # paid providers, here for when we want them — routed only when a request allows paid models
    "openai": "https://api.openai.com/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
}


@dataclass(frozen=True)
class Caps:
    tools: bool = False  # supports OpenAI tool / function calling
    tier: str = "free"  # "free" | "paid"
    grade: str = "fast"  # "fast" | "balanced" | "capable"


# capability catalog keyed by provider-native model id. A model NOT listed defaults to Caps() —
# (tools=False, tier="free", grade="fast") — deliberately conservative.
CAPS = {
    # Groq (free) — verified ids
    "openai/gpt-oss-120b": Caps(tools=True, grade="capable"),
    "openai/gpt-oss-20b": Caps(tools=True, grade="balanced"),
    "llama-3.3-70b-versatile": Caps(tools=True, grade="capable"),
    "llama-3.1-8b-instant": Caps(tools=True, grade="fast"),
    "qwen/qwen3-32b": Caps(tools=True, grade="balanced"),
    "meta-llama/llama-4-scout-17b-16e-instruct": Caps(tools=True, grade="fast"),
    # Cerebras (free)
    "zai-glm-4.7": Caps(tools=True, grade="balanced"),
    # SambaNova / Cerebras shared id
    "gpt-oss-120b": Caps(tools=True, grade="capable"),
    "Meta-Llama-3.3-70B-Instruct": Caps(tools=True, grade="capable"),
    # NVIDIA NIM (free)
    "meta/llama-3.3-70b-instruct": Caps(tools=True, grade="capable"),
    # Gemini (free tier)
    "gemini-2.5-flash": Caps(tools=True, grade="capable"),
    "gemini-flash-lite-latest": Caps(tools=True, grade="fast"),
    # paid
    "gpt-4o-mini": Caps(tools=True, tier="paid", grade="capable"),
    "deepseek-chat": Caps(tools=True, tier="paid", grade="capable"),
}


@dataclass
class Model:
    provider: str
    model: str
    url: str
    key: str
    tools: bool = False
    tier: str = "free"
    grade: str = "fast"
    cooldown: float = 0.0
    calls: int = 0
    ok: int = 0
    throttled: int = 0
    errors: int = 0
    recent: list = field(default_factory=list)


def build_models(
    spec: list[str],
    keys: dict[str, str],
    providers: dict[str, str] = PROVIDERS,
    caps: dict[str, Caps] = CAPS,
) -> list[Model]:
    out = []
    for item in spec:
        provider, _, model = item.strip().partition(":")
        key = (keys.get(provider) or "").strip()
        url = providers.get(provider)
        if not (key and url and model):
            continue
        c = caps.get(model, Caps())
        out.append(
            Model(
                provider=provider,
                model=model,
                url=url,
                key=key,
                tools=c.tools,
                tier=c.tier,
                grade=c.grade,
            )
        )
    return out
