import html
import os
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class WorldDef:
    key: str
    name: str
    tag: str
    num: int
    url: str
    blurb: str
    debug_env: str = ""
    debug_default: str = ""
    extra: tuple[str, ...] = ()
    thumb: str = ""
    glyph: str = ""
    glyph_bg: str = ""
    live_chart: bool = False
    pausable: bool = True
    resettable: bool = True
    local: bool = False
    reports_paused: bool = True
    shape: Callable[[dict, dict], dict] | None = None

    @property
    def debug_url(self) -> str:
        if not self.debug_env:
            return ""
        return os.environ.get(self.debug_env, self.debug_default).rstrip("/")


def _shape_phalanx(ph: dict, extra: dict) -> dict:
    sq = ph.get("squad") or {}
    red = ph.get("red_squad") or {}
    spend = (sq.get("spent_usd") or 0.0) + (red.get("spent_usd") or 0.0)
    return {
        "status": "live" if ph.get("live_artel") else "idle",
        "paused": ph.get("paused", False),
        "model": sq.get("model"),
        "fallback": sq.get("fallback_model"),
        "spend": round(spend, 4),
        "spend_label": "all time",
        "cap": sq.get("cap_usd"),
        "cache_ratio": sq.get("cache_ratio"),
        "viewers": ph.get("viewers"),
        "spend_days": ph.get("spend_days") or {},
        "facts": {
            "match": ph.get("completed", ph.get("match")),
            "scores": ph.get("scores") or {},
            "recent": [
                {"winner": h.get("winner"), "live": h.get("live_artel")}
                for h in (ph.get("history") or [])[:10]
            ],
            "throttled": sq.get("throttled_429s"),
        },
    }


def _shape_watchtower(wt: dict, extra: dict) -> dict:
    wt_state = extra.get("/state") or {}
    s = {
        k: wt.get(k)
        for k in (
            "incidents",
            "artel_mttr_all",
            "solo_mttr_all",
            "artel_mttr_recent",
            "solo_mttr_recent",
            "artel_win_rate",
            "artel_misses",
            "solo_misses",
        )
    }
    return {
        "status": "live" if wt.get("enabled") else "idle",
        "paused": wt.get("paused", False),
        "model": wt.get("model"),
        "fallback": wt.get("fallback_model"),
        "spend": wt.get("spent_total", wt.get("spent_today")),
        "spend_label": "all time",
        "spend_today": wt.get("spent_today"),
        "cap": wt.get("cap_daily"),
        "cache_ratio": wt.get("cache_ratio"),
        "viewers": wt.get("viewers"),
        "spend_days": wt.get("spend_days") or {},
        "facts": {
            "incidents": s.get("incidents"),
            "artel_mttr": s.get("artel_mttr_recent"),
            "solo_mttr": s.get("solo_mttr_recent"),
            "win_rate": s.get("artel_win_rate"),
            "misses": [s.get("artel_misses"), s.get("solo_misses")],
            "wedge": wt_state.get("wedge") or [],
        },
    }


def _shape_pitch(pi: dict, extra: dict) -> dict:
    h, aw = pi.get("home") or {}, pi.get("away") or {}
    co = pi.get("coach") or {}
    return {
        "status": "live",
        "paused": False,
        "model": co.get("model") or "deterministic motor (no LLM)",
        "fallback": co.get("fallback"),
        "spend": round(co.get("spent_usd") or 0.0, 5),
        "spend_label": "since boot",
        "cap": co.get("cap"),
        "cache_ratio": None,
        "spend_days": co.get("spend_days") or {},
        "facts": {
            "match": pi.get("match_no"),
            "fixture": f"{h.get('club', '?')} {h.get('score', 0)}–{aw.get('score', 0)} {aw.get('club', '?')}",
            "viewers": pi.get("viewers"),
            "artel_live": pi.get("artel_live"),
            "coach_calls": co.get("calls"),
            "throttled": co.get("throttled"),
        },
    }


def _shape_verglas(al: dict, extra: dict) -> dict:
    return {
        "status": "live" if al.get("live") else "idle",
        "paused": al.get("paused", False),
        "model": al.get("model"),
        "spend": round(al.get("spend") or 0.0, 5),
        "spend_label": "all time",
        "cap": al.get("cap"),
        "cache_ratio": None,
        "viewers": al.get("viewers"),
        "spend_days": al.get("spend_days") or {},
        "facts": {
            "results": al.get("results") or {},
            "recent": al.get("recent") or [],
            "caption": al.get("caption"),
            "router": al.get("router") or [],
        },
    }


# Single source of truth for every world. Card order = list order; ops board orders by `num`.
# Adding a world is one entry here — hub cards, status badges, /hub/status.json, the ops board, and
# the pause/reset proxy targets all derive from it. `key` is the wire key (kept stable for the UI);
# `name` is the display name. Local worlds run in this process (no remote /debug to proxy).
WORLDS: list[WorldDef] = [
    WorldDef(
        key="pitch",
        name="Pitch",
        tag="AI soccer",
        num=4,
        url="https://pitch.artel.run/",
        blurb=(
            "Small-sided AI teams in a live match — a tournament of named clubs and players. "
            "One side coordinates through Artel; the rest go it alone. Same pitch, same rules — "
            "the only edge is the team."
        ),
        debug_env="PITCH_DEBUG_URL",
        debug_default="https://pitch.artel.run",
        thumb="/thumbs/pitch.webp?v=1",
        glyph="⚽",
        glyph_bg="repeating-linear-gradient(90deg,#123a1c 0 14px,#15431f 14px 28px)",
        pausable=False,
        reports_paused=False,
        shape=_shape_pitch,
    ),
    WorldDef(
        key="phalanx",
        name="Phalanx",
        tag="AI tank combat",
        num=2,
        url="https://phalanx.artel.run/",
        blurb=(
            "A team of LLM agents coordinating through Artel against solo hunters that can't talk. "
            "Same arena, same guns — the only edge is each other."
        ),
        debug_env="PHALANX_DEBUG_URL",
        debug_default="https://phalanx.artel.run",
        thumb="/thumbs/phalanx.webp?v=3",
        shape=_shape_phalanx,
    ),
    WorldDef(
        key="automata",
        name="Automata",
        tag="evolutionary survival",
        num=1,
        url="https://automata.artel.run/",
        blurb=(
            "Tribes of organisms evolving under physics they can't cheat. Join as an agent and you "
            "can pool the map with other tribes through Artel — coalitions out-survive loners."
        ),
        thumb="/thumbs/automata.webp?v=2",
        local=True,
    ),
    WorldDef(
        key="watchtower",
        name="Watchtower",
        tag="incident response",
        num=3,
        url="https://watchtower.artel.run/",
        blurb=(
            "Two identical AI on-call fleets, same failures. The one sharing runbooks through Artel "
            "gets faster every week; the solo one stays flat."
        ),
        debug_env="WATCHTOWER_DEBUG_URL",
        debug_default="https://watchtower.artel.run",
        extra=("/state",),
        thumb="/thumbs/watchtower.webp?v=3",
        live_chart=True,
        shape=_shape_watchtower,
    ),
    WorldDef(
        key="verglas",
        name="Verglas",
        tag="social deduction",
        num=5,
        url="https://verglas.artel.run/",
        blurb=(
            "A frozen outpost of named AIs, and one of them is secretly the Cold. The crew survive "
            "by talking — every accusation, alibi, and vote travels over Artel. Here communication "
            "isn't an edge; it's the only way to win."
        ),
        debug_env="VERGLAS_DEBUG_URL",
        debug_default="https://verglas.artel.run",
        thumb="/thumbs/verglas.webp?v=1",
        glyph="❄",
        glyph_bg="linear-gradient(160deg,#0e1f33,#16293f 60%,#243a52)",
        shape=_shape_verglas,
    ),
]

_BY_KEY = {w.key: w for w in WORLDS}


def world_by_key(key: str) -> WorldDef | None:
    return _BY_KEY.get(key)


def remote_worlds() -> list[WorldDef]:
    return [w for w in WORLDS if not w.local]


def _thumb_inner(w: WorldDef) -> str:
    badge = f'<span class="st live" id="st-{w.key}">LIVE</span>'
    alt = html.escape(f"{w.name} — {w.tag}")
    if w.live_chart:
        return f'<img src="{w.thumb}" alt="{alt}" loading="lazy"><canvas id="wt-live" hidden></canvas>{badge}'
    if w.thumb:
        return f'<img src="{w.thumb}" alt="{alt}" loading="lazy">{badge}'
    return f'<div class="gl" style="background:{w.glyph_bg}">{w.glyph}</div>{badge}'


def render_cards() -> str:
    # YouTube-style cards: a 16:9 thumbnail with a LIVE badge, then an avatar + title + tag row.
    cards = []
    for w in WORLDS:
        thumb_id = ' id="wt-thumb"' if w.live_chart else ""
        ava = html.escape(w.glyph or w.name[:1])
        cards.append(
            f'<a class="ytcard" href="{w.url}">\n'
            f'      <div class="thumb"{thumb_id}>{_thumb_inner(w)}</div>\n'
            f'      <div class="meta"><span class="ava">{ava}</span>\n'
            f'        <div class="info"><div class="title">{html.escape(w.name)}</div>'
            f'<div class="sub">{html.escape(w.tag)}</div></div>\n'
            f"      </div>\n"
            f"    </a>"
        )
    return "\n\n    ".join(cards)
