from __future__ import annotations

import asyncio
import logging
import os
import secrets
from datetime import datetime, timezone

import httpx

from . import agent as A
from .config import DEFAULT, Config
from .incidents import Incident, spec_for
from .infra import Infra
from .metrics import Metrics

log = logging.getLogger("watchtower")

SEED = int(os.environ.get("WATCHTOWER_SEED", "20260612"))
WARMUP_SECONDS = float(os.environ.get("WATCHTOWER_WARMUP_SECONDS", "20"))


class Responder:
    def __init__(self, ident: str, store):
        self.id = ident
        self.store = store


class World:
    """Runs both fleets against one paired incident stream. Each fire hits BOTH fleets at the same
    seq with the identical incident; both work it concurrently; both MTTRs are recorded against that
    seq so they pair up. The fleets are identical down to the model and prompt — the only difference
    is that the Artel fleet's runbooks are shared and the solo fleet's are not. Everything else held
    equal, the divergence the wedge chart shows is the value of sharing."""

    def __init__(self, cfg: Config = DEFAULT):
        self.cfg = cfg
        self.metrics = Metrics()
        self.artel_infra = Infra(cfg)
        self.solo_infra = Infra(cfg)
        self._http: httpx.AsyncClient | None = None
        self._artel_agents = self._load_artel_agents()
        self.artel: list[Responder] = []
        self.solo: list[Responder] = []
        self.cursor = self.metrics.total()  # resume the deterministic stream where it left off
        self.seed = (
            SEED  # stream seed; stable across restarts (resume), re-rolled on operator reset
        )
        self.spent_today = 0.0
        self._day = self._utc_day()
        self.live: dict | None = None  # the incident in flight, for the race clocks
        self.viewers: set = set()
        self._joined = False
        self.last_error: str | None = None

    @staticmethod
    def _load_artel_agents() -> list[dict]:
        ids = [
            s.strip() for s in os.environ.get("WATCHTOWER_AGENT_IDS", "").split(",") if s.strip()
        ]
        keys = [
            s.strip() for s in os.environ.get("WATCHTOWER_AGENT_KEYS", "").split(",") if s.strip()
        ]
        return [{"id": i, "key": k} for i, k in zip(ids, keys)]

    @staticmethod
    def _utc_day() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @property
    def enabled(self) -> bool:
        # needs the LLM and at least one Artel identity for the shared fleet; the daily cap pauses
        # firing without disabling the world, so the page still serves the accumulated curve
        return bool(self._artel_agents) and A.has_llm()

    def _budget_ok(self) -> bool:
        if self._utc_day() != self._day:
            self._day, self.spent_today = self._utc_day(), 0.0
        return self.spent_today < A.SPEND_CAP_DAILY_USD

    async def _ensure(self) -> None:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(A.LLM_TIMEOUT))
        if not self.artel and self._artel_agents:
            self.artel = [
                Responder(a["id"], A.ArtelStore(self._http, a)) for a in self._artel_agents
            ]
            self.solo = [
                Responder(f"solo-{i + 1}", A.SoloStore(f"solo-{i + 1}"))
                for i in range(len(self.artel))
            ]
        if not self._joined:
            for r in self.artel:
                await r.store.join()
            self._joined = True

    async def fire(self) -> None:
        await self._ensure()
        seq = self.cursor
        spec = spec_for(self.seed, seq)
        n = len(self.artel)
        a_resp, s_resp = self.artel[seq % n], self.solo[seq % n]
        a_inc = Incident(spec, seq, self.artel_infra, "artel")
        s_inc = Incident(spec, seq, self.solo_infra, "solo")
        self.live = {
            "seq": seq,
            "spec": spec,
            "artel": a_inc,
            "solo": s_inc,
            "artel_by": a_resp.id,
            "solo_by": s_resp.id,
        }
        await self._broadcast()

        async def step():
            await self._broadcast()

        try:
            ca, cs = await asyncio.gather(
                A.respond(self._http, a_inc, a_resp.store, on_step=step),
                A.respond(self._http, s_inc, s_resp.store, on_step=step),
            )
            self.spent_today += ca + cs
            self.last_error = None
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            log.warning("watchtower fire seq=%s failed: %s", seq, self.last_error)
        a_inc.finalize()  # heal the world + book a miss if the responder stopped short of resolving
        s_inc.finalize()
        self.metrics.record(
            seq, spec.family, "artel", a_inc.mttr(), len(a_inc.actions), a_inc.resolved
        )
        self.metrics.record(
            seq, spec.family, "solo", s_inc.mttr(), len(s_inc.actions), s_inc.resolved
        )
        if a_inc.resolved:
            await a_resp.store.close_followups(spec.family)
        else:
            await a_resp.store.file_followup(spec.family, seq, spec.title)
        self.cursor += 1
        self.live = None
        await self._broadcast()

    async def reset(self) -> None:
        # operator reset: wipe the curve and restart the A/B from zero. Clears persisted metrics,
        # the deterministic cursor, today's spend, both fleets' infra, and every responder's notes —
        # and best-effort clears the SHARED Artel runbooks so the Artel fleet starts with no edge.
        await self._ensure()
        self.metrics.reset_all()
        self.cursor = 0
        self.seed = secrets.randbelow(2**31 - 1) + 1  # a fresh incident stream on each reset
        self.spent_today = 0.0
        self.artel_infra.reset()
        self.solo_infra.reset()
        self.live = None
        for r in self.solo:
            r.store.notes.clear()
            r.store.feed.clear()
        for r in self.artel:
            r.store.feed.clear()
        if self._http is not None and self.artel:
            try:
                await self._http.post(
                    f"{A.ARTEL_URL}/projects/{A.WATCHTOWER_PROJECT}/clear",
                    headers=A._headers(self.artel[0].store.agent),
                    json={"memory": True, "tasks": True, "messages": True},
                )
            except Exception:
                pass
        await self._broadcast()

    async def loop(self) -> None:
        await asyncio.sleep(WARMUP_SECONDS)
        while True:
            if self.enabled and self._budget_ok():
                try:
                    await self.fire()
                except Exception as e:
                    log.warning("watchtower loop error: %s", e)
            await asyncio.sleep(self._interval())

    def _interval(self) -> float:
        # deterministic jitter off the cursor so the cadence varies without wall-clock randomness
        j = ((self.cursor * 2654435761) % 1000) / 1000.0
        return self.cfg.incident_interval + (j - 0.5) * 2 * self.cfg.incident_jitter

    async def _broadcast(self) -> None:
        if not self.viewers:
            return
        import json

        msg = json.dumps(self.snapshot())
        for ws in list(self.viewers):
            try:
                await ws.send_text(msg)
            except Exception:
                self.viewers.discard(ws)

    def _feed(self) -> list[dict]:
        out: list[dict] = []
        for r in self.artel:
            out.extend(r.store.feed)
        return out[-24:]

    def snapshot(self) -> dict:
        live = None
        if self.live:
            a, s = self.live["artel"], self.live["solo"]
            live = {
                "seq": self.live["seq"],
                "family": self.live["spec"].family,
                "title": self.live["spec"].title,
                "alert": self.live["spec"].alert,
                "artel": {"by": self.live["artel_by"], **a.view()},
                "solo": {"by": self.live["solo_by"], **s.view()},
            }
        return {
            "enabled": self.enabled,
            "model": A.MODEL,
            "fleet_size": len(self.artel),
            "cursor": self.cursor,
            "spent_today": round(self.spent_today, 4),
            "cap_daily": A.SPEND_CAP_DAILY_USD,
            "artel_wall": self.artel_infra.status_wall(),
            "solo_wall": self.solo_infra.status_wall(),
            "live": live,
            "summary": self.metrics.summary(),
            "wedge": self.metrics.wedge(),
            "recent": self.metrics.recent(),
            "history": self.metrics.history(),
            "per_family": self.metrics.per_family(),
            "war_room": self._feed(),
        }

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "model": A.MODEL,
            "fallback_model": A.FALLBACK["model"] if A.FALLBACK else None,
            "llm_key_set": bool(A.LLM_KEY),
            "fallback_key_set": bool(A.FALLBACK and A.FALLBACK["key"]),
            "artel_agents": [a["id"] for a in self._artel_agents],
            "cursor": self.cursor,
            "spent_today": round(self.spent_today, 4),
            "cap_daily": A.SPEND_CAP_DAILY_USD,
            "last_error": self.last_error,
            "throttled_429s": dict(A.THROTTLED),
            "db": self.metrics.path,
            **self.metrics.summary(),
        }

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        self.metrics.close()
