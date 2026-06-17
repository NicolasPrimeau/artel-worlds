from __future__ import annotations

# Fair A/B with the REAL LLM as the strategic flank-reader, vs baseline. Identical to the heuristic
# hybrid in tools/ab.py — same every-tick game-management reflex, same flank-refresh cadence — so the
# ONLY difference is heuristic-read vs LLM-read. Isolates whether the LLM's read is worth anything.

import asyncio
import os

for _line in open("/home/nprimeau/projects/Artel/.env"):
    if _line.startswith("GROQ_API_KEY="):
        os.environ["PITCH_LLM_KEY"] = _line.split("=", 1)[1].strip()
os.environ["PITCH_MODEL"] = "openai/gpt-oss-120b"
os.environ.setdefault("PITCH_LLM_URL", "https://api.groq.com/openai/v1/chat/completions")

from pitch import bot, commander, llm  # noqa: E402
from pitch.engine import Pitch  # noqa: E402

CADENCE = 75


async def match(seed, coord):
    p = Pitch(seed=1000 + seed)
    p.setup(["x"] * 9, ["y"] * 9)
    cur = {"oy": None}

    def brain(pitch, pl):
        if pl.team == coord and cur["oy"] is not None:
            commit, low_block = commander.game_management(pitch, coord)  # reflex, every tick
            plan = commander.Plan(cur["oy"], commit, low_block, False)
            return commander.coordinated_decide(pitch, pl, plan)
        return bot.decide(pitch, pl)

    cur["oy"] = (await commander.author_plan_llm(p, coord)).overload_y
    while p.tick < p.cfg.match_ticks:
        if p.tick % CADENCE == 0:
            cur["oy"] = (
                await commander.author_plan_llm(p, coord)
            ).overload_y  # LLM re-reads the flank
        p.step(brain)
    opp = "away" if coord == "home" else "home"
    return p.score[coord], p.score[opp]


async def main():
    print("LLM:", llm.MODEL, "| enabled:", llm.enabled())
    w = ll = d = gf = ga = n = 0
    seeds = 4  # LLM calls are slow; small N (noisy) but enough to see if it clears ~52%
    for s in range(seeds):
        for coord in ("home", "away"):
            cf, ca = await match(s, coord)
            gf += cf
            ga += ca
            n += 1
            w += cf > ca
            ll += ca > cf
            d += cf == ca
            print(f"  seed {s} {coord}: {cf}-{ca}")
    wr = round(w / (w + ll) * 100) if (w + ll) else 0
    print(
        f"\nLLM hybrid vs baseline: {w}W {ll}L {d}D | win {wr}% | goals {gf / n:.2f} vs {ga / n:.2f}"
        f" (n={n}) | spent ${llm.SPEND['usd']:.4f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
