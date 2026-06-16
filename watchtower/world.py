from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from datetime import datetime, timezone

import httpx

from . import agent as A
from .config import DEFAULT, Config
from .incidents import Incident, cascade_root_spec, storm_for, symptom_spec
from .infra import Infra
from .metrics import Metrics

log = logging.getLogger("watchtower")

SEED = int(os.environ.get("WATCHTOWER_SEED", "20260612"))
WARMUP_SECONDS = float(os.environ.get("WATCHTOWER_WARMUP_SECONDS", "20"))
# Storm scheduling. Incidents now arrive in BURSTS, and only while someone is watching — brisk
# enough to be live theatre, with a wave that surges and recedes instead of a flat metronome.
# Cost is bounded three ways: ≤ fleet_size responders work at once (the backlog is free), firing
# only when viewers are present, and the hard daily spend cap as the backstop.
STORM_INTERVAL = float(
    os.environ.get("WATCHTOWER_STORM_INTERVAL", "40")
)  # real secs between storms
STORM_JITTER = float(os.environ.get("WATCHTOWER_STORM_JITTER", "12"))
IDLE_POLL = float(os.environ.get("WATCHTOWER_IDLE_POLL", "3"))  # cheap poll for a viewer when idle
STORM_WAVE = (1, 2, 2, 3, 5, 3, 2, 1)  # burst sizes cycled per storm — calm -> surge -> recede


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
        self.spent_total = 0.0
        self.spend_days: dict[str, float] = {}
        self._day = self._utc_day()
        self.storm: dict | None = None  # the burst in flight: both fleets' live + backlog incidents
        self.storm_no = 0  # counter driving the wave rhythm of burst sizes
        self.viewers: set = set()
        self._joined = False
        self.last_error: str | None = None
        self.paused = False  # operator toggle (ops page): halts firing/spend, page stays up

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
            self._restore_state()
        if not self._joined:
            for r in self.artel:
                await r.store.join()
            self._joined = True

    def _restore_state(self) -> None:
        # deploys must not wipe the experiment: the solo fleet's private notebooks live only in
        # this process (Artel's live on artel.run and already survive), and the spend counters
        # are the cost story. Both restore from the metrics volume; only /reset wipes them.
        try:
            raw = self.metrics.kv_get("solo_stores")
            if raw:
                saved = {s["agent_id"]: s for s in json.loads(raw)}
                for r in self.solo:
                    st = saved.get(r.store.agent_id)
                    if st:
                        r.store.notes = st.get("notes", [])
                        r.store.tasks = st.get("tasks", [])
                        r.store.last_shift = st.get("last_shift", "")
                        r.store.shift = int(st.get("shift", 0))
            raw = self.metrics.kv_get("spend")
            if raw:
                sp = json.loads(raw)
                self.spent_total = float(sp.get("total", 0.0))
                self.spend_days = {k: float(v) for k, v in (sp.get("days") or {}).items()}
                if sp.get("day") == self._utc_day():
                    self.spent_today = float(sp.get("today", 0.0))
            self.paused = self.metrics.kv_get("paused") == "1"
        except Exception as e:
            log.warning("watchtower state restore failed: %s", e)

    def set_paused(self, value: bool) -> None:
        self.paused = bool(value)
        try:
            self.metrics.kv_set("paused", "1" if self.paused else "0")
        except Exception as e:
            log.warning("watchtower pause persist failed: %s", e)

    def _persist_state(self) -> None:
        try:
            self.metrics.kv_set(
                "solo_stores",
                json.dumps(
                    [
                        {
                            "agent_id": r.store.agent_id,
                            "notes": r.store.notes,
                            "tasks": r.store.tasks,
                            "last_shift": r.store.last_shift,
                            "shift": r.store.shift,
                        }
                        for r in self.solo
                    ]
                ),
            )
            self.metrics.kv_set(
                "spend",
                json.dumps(
                    {
                        "total": round(self.spent_total, 6),
                        "today": round(self.spent_today, 6),
                        "day": self._day,
                        "days": self.spend_days,
                    }
                ),
            )
        except Exception as e:
            log.warning("watchtower state persist failed: %s", e)

    def _storm_size(self) -> int:
        # the wave: burst sizes cycle calm -> surge -> recede so load arrives in swells, not a
        # flat drip. Deterministic off the storm counter, no wall-clock randomness.
        return STORM_WAVE[self.storm_no % len(STORM_WAVE)]

    def _is_cascade(self) -> bool:
        # every other storm is a single-root CASCADE — one deep fault pages many symptom tickets,
        # and the fleet that shares the root (vs flailing on symptoms) drains it far cheaper. The
        # rest are independent bursts, for variety in the load.
        return self.storm_no % 2 == 1

    def _build_fleet(self, seq: int, infra, fleet: str, cascade: bool) -> list:
        if not cascade:
            specs = storm_for(self.seed, seq, self._storm_size())
            return [Incident(s, seq + i, infra, fleet) for i, s in enumerate(specs)]
        # cascade: one root fault, then a symptom ticket per dependent the propagation degraded
        root_spec = cascade_root_spec(self.seed, seq)
        root = Incident(root_spec, seq, infra, fleet)
        root_node = root_spec.fix[-1][1]
        syms = [
            n
            for n, st in infra.nodes.items()
            if n != root_node and st.status in ("degraded", "down")
        ][:3]
        tickets = [
            Incident(symptom_spec(root_spec, sym), seq + 1 + i, infra, fleet, cascade_root=root)
            for i, sym in enumerate(syms)
        ]
        return [root, *tickets]

    async def fire_storm(self) -> None:
        # Fire a STORM into both fleets at once — a burst of independent incidents, or a single-root
        # CASCADE. Each fleet's fleet_size responders pull from a shared backlog and work ONE
        # incident at a time, so concurrent LLM work is capped at fleet_size no matter the burst
        # size and queued incidents cost nothing. The backlog draining is the live theatre; on a
        # cascade, the wedge is the wasted actions a solo fleet burns flailing on symptoms.
        await self._ensure()
        seq = self.cursor
        cascade = self._is_cascade()
        a_incs = self._build_fleet(seq, self.artel_infra, "artel", cascade)
        s_incs = self._build_fleet(seq, self.solo_infra, "solo", cascade)
        k = len(a_incs)
        for inc in (*a_incs, *s_incs):
            inc.state = "pending"
            inc.by = None
        self.storm = {
            "seq": seq,
            "size": k,
            "cascade": cascade,
            "artel": a_incs,
            "solo": s_incs,
        }
        await self._broadcast()
        try:
            ca, cs = await asyncio.gather(
                self._work_fleet(self.artel, a_incs),
                self._work_fleet(self.solo, s_incs),
            )
            self.spent_today += ca + cs
            self.spent_total += ca + cs
            day = self._utc_day()
            self.spend_days[day] = round(self.spend_days.get(day, 0.0) + ca + cs, 6)
            for kk in sorted(self.spend_days)[:-30]:
                del self.spend_days[kk]
            self.last_error = None
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            log.warning("watchtower storm seq=%s failed: %s", seq, self.last_error)
        self.cursor += k  # k seqs consumed, resolved or not — the stream stays deterministic
        self.storm_no += 1
        self.storm = None
        self._persist_state()
        await self._broadcast()

    async def _work_fleet(self, responders: list, incidents: list) -> float:
        # one fleet's responders draining its backlog: a shared queue means no two responders work
        # the same incident, and at most fleet_size run concurrently. Returns the fleet's LLM cost.
        queue: asyncio.Queue = asyncio.Queue()
        for inc in incidents:
            queue.put_nowait(inc)
        total = 0.0

        async def step() -> None:
            await self._broadcast()

        async def worker(resp) -> None:
            nonlocal total
            while True:
                try:
                    inc = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                inc.state, inc.by = "active", resp.id
                await self._broadcast()
                task = await resp.store.open_incident(
                    inc.seq, inc.spec.title, inc.family, inc.spec.alert
                )
                try:
                    total += await A.respond(self._http, inc, resp.store, on_step=step)
                except Exception as e:
                    self.last_error = f"{type(e).__name__}: {e}"
                    log.warning("watchtower responder %s failed: %s", resp.id, self.last_error)
                inc.finalize()  # heal + book a miss if the responder stopped short
                inc.state = "resolved" if inc.resolved else "missed"
                self.metrics.record(
                    inc.seq, inc.family, inc.fleet, inc.mttr(), len(inc.actions), inc.resolved
                )
                miss_note = (
                    f"Closed UNRESOLVED after {len(inc.actions)} actions; MTTR booked at the cap. "
                    "Next responder on this family: crack it and record the runbook."
                )
                await resp.store.close_incident(task, inc.resolved, miss_note)
                if inc.resolved:
                    await resp.store.sweep_family(inc.family)
                outcome = (
                    f"resolved in {inc.mttr():.0f}s" if inc.resolved else "went UNRESOLVED (capped)"
                )
                await resp.store.save_handoff(
                    f"incident #{inc.seq} ({inc.family}): {outcome} after {len(inc.actions)} actions."
                )
                await self._broadcast()

        await asyncio.gather(*(worker(r) for r in responders))
        return total

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
        self.storm = None
        self.storm_no = 0
        for r in self.solo:
            r.store.notes.clear()
            r.store.tasks.clear()
            r.store.feed.clear()
            r.store.last_shift = ""
            r.store.shift = 0
        self.metrics.kv_delete("solo_stores")
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
            # VIEWER-GATED: storms (and the spend they cost) only fire while someone is watching.
            # Unwatched, the loop just polls cheaply for a viewer — the page still serves the
            # accumulated curve, but nothing burns. An in-flight storm always finishes.
            if self.enabled and not self.paused and self._budget_ok() and self.viewers:
                try:
                    await self.fire_storm()
                except Exception as e:
                    log.warning("watchtower loop error: %s", e)
                await asyncio.sleep(self._interval())
            else:
                await asyncio.sleep(IDLE_POLL)

    def _interval(self) -> float:
        # brisk, with deterministic jitter off the cursor so the cadence varies without wall-clock
        # randomness — storms arrive every ~STORM_INTERVAL seconds while watched
        j = ((self.cursor * 2654435761) % 1000) / 1000.0
        return STORM_INTERVAL + (j - 0.5) * 2 * STORM_JITTER

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

    @staticmethod
    def _inc_card(inc) -> dict:
        return {
            **inc.view(),
            "state": getattr(inc, "state", "pending"),
            "by": getattr(inc, "by", None),
        }

    @staticmethod
    def _fleet_board(incs: list) -> dict:
        cards = [World._inc_card(i) for i in incs]
        return {
            "incidents": cards,
            "pending": sum(1 for c in cards if c["state"] == "pending"),
            "active": sum(1 for c in cards if c["state"] == "active"),
            "open": sum(1 for c in cards if c["state"] in ("pending", "active")),
            "resolved": sum(1 for c in cards if c["state"] == "resolved"),
            "missed": sum(1 for c in cards if c["state"] == "missed"),
        }

    def snapshot(self) -> dict:
        storm = None
        if self.storm:
            storm = {
                "seq": self.storm["seq"],
                "size": self.storm["size"],
                "cascade": self.storm.get("cascade", False),
                "artel": self._fleet_board(self.storm["artel"]),
                "solo": self._fleet_board(self.storm["solo"]),
            }
        return {
            "enabled": self.enabled,
            "paused": self.paused,
            "model": A.MODEL,
            "fleet_size": len(self.artel),
            "cursor": self.cursor,
            "spent_today": round(self.spent_today, 4),
            "spent_total": round(self.spent_total, 4),
            "spend_days": dict(self.spend_days),
            "cap_daily": A.SPEND_CAP_DAILY_USD,
            "artel_wall": self.artel_infra.status_wall(),
            "solo_wall": self.solo_infra.status_wall(),
            "storm": storm,
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
            "paused": self.paused,
            "model": A.MODEL,
            "fallback_model": A.FALLBACK["model"] if A.FALLBACK else None,
            "llm_key_set": bool(A.LLM_KEY),
            "fallback_key_set": bool(A.FALLBACK and A.FALLBACK["key"]),
            "artel_agents": [a["id"] for a in self._artel_agents],
            "cursor": self.cursor,
            "spent_today": round(self.spent_today, 4),
            "spent_total": round(self.spent_total, 4),
            "spend_days": dict(self.spend_days),
            "cap_daily": A.SPEND_CAP_DAILY_USD,
            "cache_ratio": round(A.CACHE["cached_in"] / max(1, A.CACHE["input"]), 3),
            "cached_in": A.CACHE["cached_in"],
            "input_tok": A.CACHE["input"],
            "last_error": self.last_error,
            "throttled_429s": dict(A.THROTTLED),
            "db": self.metrics.path,
            **self.metrics.summary(),
        }

    async def aclose(self) -> None:
        if self.solo:
            self._persist_state()
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        self.metrics.close()
