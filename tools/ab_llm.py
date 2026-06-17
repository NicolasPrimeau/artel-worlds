from __future__ import annotations

# A/B with the REAL LLM (Groq) authoring the directive, vs the baseline. Smaller N (LLM calls are
# slow) and noisier, but it tests the deployed thing: does an LLM-coached side actually beat baseline?

import asyncio
import os

for _line in open("/home/nprimeau/projects/Artel/.env"):
    if _line.startswith("GROQ_API_KEY="):
        os.environ["PITCH_LLM_KEY"] = _line.split("=", 1)[1].strip()
os.environ["PITCH_MODEL"] = "openai/gpt-oss-120b"
os.environ.setdefault("PITCH_LLM_URL", "https://api.groq.com/openai/v1/chat/completions")

from pitch import bot, commander, llm, plays  # noqa: E402
from pitch.engine import Pitch  # noqa: E402


async def match(seed, coord, every=250):
    p = Pitch(seed=1000 + seed)
    p.setup(["x"] * 9, ["y"] * 9)
    mgr = plays.PlayManager(coord)
    cur = {"plan": None}

    def brain(pitch, pl):
        plan = cur["plan"]
        if pl.team == coord and plan is not None:
            it = mgr.intent(pitch, pl) if plan.combos else None
            if it is not None:
                return it
            return commander.coordinated_decide(pitch, pl, plan)
        return bot.decide(pitch, pl)

    cur["plan"] = await commander.author_plan_llm(p, coord)  # LLM authors the opening directive
    calls = 1
    while p.tick < p.cfg.match_ticks:
        if p.tick % every == 0:
            cur["plan"] = await commander.author_plan_llm(p, coord)  # re-author live
            calls += 1
        if cur["plan"] and cur["plan"].combos:
            side = "left" if cur["plan"].overload_y < p.cfg.width / 2 else "right"
            mgr.call = {"combos": True, "channel": side}
            mgr.update(p)
        p.step(brain)
    opp = "away" if coord == "home" else "home"
    return p.score[coord], p.score[opp], calls


async def main():
    print("LLM enabled:", llm.enabled(), "| model:", llm.MODEL)
    if not llm.enabled():
        return
    w = ll = d = 0
    gf = ga = tc = 0
    seeds = 8
    for s in range(seeds):
        for coord in ("home", "away"):
            cf, ca, calls = await match(s, coord)
            gf += cf
            ga += ca
            tc += calls
            w += cf > ca
            ll += ca > cf
            d += cf == ca
            print(f"  seed {s} coord={coord}: {cf}-{ca}")
    n = seeds * 2
    wr = round(w / (w + ll) * 100) if (w + ll) else 0
    print(
        f"\nLLM-coached vs baseline: {w}W {ll}L {d}D | win {wr}% | goals {gf / n:.2f} vs {ga / n:.2f}"
        f" | {tc} LLM calls | spent ${llm.SPEND['usd']:.4f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
