from __future__ import annotations

import os
import sqlite3
import threading
import time

# Every resolved incident is one row per fleet, keyed by the shared incident seq so the two fleets'
# attempts at the SAME incident pair up by seq. Persisted to a Fly volume: the weeks-long MTTR wedge
# is the whole point, and it has to survive machine restarts and redeploys to mean anything.
DB_PATH = os.environ.get("WATCHTOWER_DB", "/data/watchtower.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seq INTEGER NOT NULL,
    family TEXT NOT NULL,
    fleet TEXT NOT NULL,
    mttr REAL NOT NULL,
    steps INTEGER NOT NULL,
    resolved INTEGER NOT NULL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_incidents_seq ON incidents(seq);
CREATE INDEX IF NOT EXISTS idx_incidents_fleet ON incidents(fleet);
"""


class Metrics:
    def __init__(self, path: str | None = None):
        path = path or os.environ.get("WATCHTOWER_DB", DB_PATH)
        self.path = path
        d = os.path.dirname(path)
        if d:
            try:
                os.makedirs(d, exist_ok=True)
            except OSError:
                self.path = path = os.path.basename(path)  # no volume (local/CI): fall back to cwd
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def record(self, seq, family, fleet, mttr, steps, resolved, ts=None):
        with self._lock:
            self._db.execute(
                "INSERT INTO incidents (seq, family, fleet, mttr, steps, resolved, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    seq,
                    family,
                    fleet,
                    float(mttr),
                    int(steps),
                    int(bool(resolved)),
                    ts or time.time(),
                ),
            )
            self._db.commit()

    def _rows(self, where="", args=()):
        with self._lock:
            return [
                dict(r)
                for r in self._db.execute(
                    f"SELECT seq, family, fleet, mttr, steps, resolved, ts FROM incidents {where}",
                    args,
                ).fetchall()
            ]

    def total(self) -> int:
        with self._lock:
            return self._db.execute(
                "SELECT COUNT(*) FROM incidents WHERE fleet='artel'"
            ).fetchone()[0]

    def wedge(self, bucket: int = 10) -> list[dict]:
        # the headline chart: mean MTTR per fleet across windows of `bucket` incidents in seq order.
        # Artel's line should bend down as runbooks accumulate; solo's should stay roughly level.
        rows = self._rows("ORDER BY seq")
        agg: dict[int, dict[str, list]] = {}
        for r in rows:
            b = r["seq"] // bucket
            agg.setdefault(b, {"artel": [], "solo": []})[r["fleet"]].append(r["mttr"])
        out = []
        for b in sorted(agg):
            a, s = agg[b]["artel"], agg[b]["solo"]
            out.append(
                {
                    "bucket": b,
                    "from_seq": b * bucket,
                    "artel_mttr": round(sum(a) / len(a), 1) if a else None,
                    "solo_mttr": round(sum(s) / len(s), 1) if s else None,
                    "n": max(len(a), len(s)),
                }
            )
        return out

    def summary(self) -> dict:
        rows = self._rows()
        by: dict[str, list] = {"artel": [], "solo": []}
        for r in rows:
            by.setdefault(r["fleet"], []).append(r)
        paired: dict[int, dict] = {}
        for r in rows:
            paired.setdefault(r["seq"], {})[r["fleet"]] = r["mttr"]
        complete = [p for p in paired.values() if "artel" in p and "solo" in p]
        wins = sum(1 for p in complete if p["artel"] < p["solo"])

        def avg(fleet, last=None):
            xs = [r["mttr"] for r in by.get(fleet, [])]
            xs = xs[-last:] if last else xs
            return round(sum(xs) / len(xs), 1) if xs else None

        return {
            "incidents": len(complete),
            "artel_mttr_all": avg("artel"),
            "solo_mttr_all": avg("solo"),
            "artel_mttr_recent": avg("artel", 20),
            "solo_mttr_recent": avg("solo", 20),
            "artel_win_rate": round(wins / len(complete), 3) if complete else None,
            "families_covered": len({r["family"] for r in by.get("artel", []) if r["resolved"]}),
        }

    def recent(self, n: int = 12) -> list[dict]:
        rows = self._rows("ORDER BY seq DESC LIMIT ?", (n * 2,))
        paired: dict[int, dict] = {}
        for r in rows:
            p = paired.setdefault(r["seq"], {"seq": r["seq"], "family": r["family"]})
            p[r["fleet"]] = round(r["mttr"], 1)
        return sorted(paired.values(), key=lambda p: p["seq"], reverse=True)[:n]

    def history(self, n: int = 120) -> list[dict]:
        # every paired head-to-head, oldest first — one bar per incident on the dashboard
        rows = self._rows("ORDER BY seq DESC LIMIT ?", (n * 2,))
        paired: dict[int, dict] = {}
        for r in rows:
            p = paired.setdefault(r["seq"], {"seq": r["seq"], "family": r["family"]})
            p[r["fleet"]] = round(r["mttr"], 1)
        out = [p for p in paired.values() if "artel" in p and "solo" in p]
        return sorted(out, key=lambda p: p["seq"])

    def per_family(self) -> list[dict]:
        rows = self._rows()
        fam: dict[str, dict[str, list]] = {}
        for r in rows:
            fam.setdefault(r["family"], {"artel": [], "solo": []})[r["fleet"]].append(r["mttr"])
        out = []
        for k in sorted(fam):
            a, s = fam[k]["artel"], fam[k]["solo"]
            out.append(
                {
                    "family": k,
                    "artel_mttr": round(sum(a) / len(a), 1) if a else None,
                    "solo_mttr": round(sum(s) / len(s), 1) if s else None,
                    "count": max(len(a), len(s)),
                }
            )
        return out

    def reset_all(self) -> None:
        # wipe the whole curve — used by the operator reset to restart the A/B from zero
        with self._lock:
            self._db.execute("DELETE FROM incidents")
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()
