from __future__ import annotations

import os
import re

from alibi import llm
from alibi.engine import HUB, MAX_TICKS, Game, Meeting, new_game

# The Thing has the hard job — sustain a lie, fake a checkable alibi, steer the vote — so it gets the
# smartest reasoner. The crew each get a DIFFERENT capable-but-lighter model: on Groq every model is its
# own rate-limit bucket, so one model per agent means a meeting's calls barely contend, and a mesh of
# heterogeneous LLMs reasoning together (and fighting groupthink) is exactly what Artel is for.
THING_MODEL = os.environ.get("ALIBI_THING_MODEL", "openai/gpt-oss-120b")
DEFAULT_CREW_POOL = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]
CREW_POOL = [
    m for m in os.environ.get("ALIBI_CREW_POOL", ",".join(DEFAULT_CREW_POOL)).split(",") if m
]


def assign_models(game: Game) -> None:
    # one distinct model per crew (cycled if the crew outnumbers the pool); the Thing gets the best.
    crew_i = 0
    for a in game.agents:
        if a.impostor:
            a.model = THING_MODEL
        else:
            a.model = CREW_POOL[crew_i % len(CREW_POOL)]
            crew_i += 1


def _model(agent) -> str:
    return agent.model or (THING_MODEL if agent.impostor else CREW_POOL[0])


# The meeting, as an actual conversation. Each survivor speaks from its OWN partial log — where it went,
# who it was with, anything it saw — then everyone votes. The crew are trying to reason past the lies;
# the Thing is trying to blend in, fake an alibi, and steer a vote onto someone innocent. This is the
# part no fixed rule can do (the A/B shows rules stall at ~30%): telling a true account from a
# convincing lie is judgement, not a threshold. Artel is the conduit (the shared transcript every agent
# reads); the LLM is the brain (what each agent says and how it weighs what everyone else said).

ROUNDS = 2  # opening statements, then one round of replies, then the vote

SYS_CREW = (
    "You are {name}, a researcher overwintering at an isolated Antarctic station. One of the crew is "
    "no longer human — the Thing, wearing a teammate's face, killing the rest of you one by one. You "
    "are NOT the Thing. In this emergency meeting you must use what you personally saw, plus what "
    "everyone else says, to figure out who the Thing is and get them ejected into the storm. Be "
    "skeptical: the Thing will lie, fake an alibi, and accuse the innocent. Corroborate — if two "
    "people vouch for each other's whereabouts they're probably both clear. Don't eject on a hunch; "
    "a wrong ejection helps the Thing. Speak in ONE short sentence (about 14 words max), in character, plain talk."
)
SYS_THING = (
    "You are {name}, and you are the Thing — an alien that has assimilated this researcher and is "
    "killing the crew at an isolated Antarctic station. You must NOT be found out. In this emergency "
    "meeting, blend in as a worried human: give a calm, plausible account of where you were, deflect "
    "suspicion, and if someone is onto you, cast doubt on them or point the group at someone else. "
    "Never admit what you are. If you witnessed your own kill, you may even 'report' it to look "
    "helpful. Speak in ONE short sentence (about 14 words max), in character, plain talk."
)


def _name(game: Game, i: int) -> str:
    return game.by_id(i).name


def _trail_str(game: Game, a) -> str:
    if not a.trail:
        return "You stayed in the Mess Hall."
    return ", ".join(f"t{t} {room}" for t, room in a.trail)


def _seen_str(game: Game, a) -> str:
    if not a.seen:
        return "You were alone the whole time — nobody can vouch for you."
    parts = []
    for s in a.seen:
        who = ", ".join(_name(game, i) for i in s.present)
        parts.append(f"t{s.tick} in {s.room} with {who}")
    return "; ".join(parts)


def _brief(game: Game, mt: Meeting, a) -> str:
    lines = [
        f"Your movements this shift: {_trail_str(game, a)}.",
        f"Who you saw: {_seen_str(game, a)}.",
    ]
    if not a.impostor and a.witnessed:
        for imp in a.witnessed:
            if game.by_id(imp).alive:
                lines.append(
                    f"YOU SAW IT HAPPEN: {_name(game, imp)} is the Thing — you watched them kill."
                )
    if a.found:
        t, room, victim = a.found[-1]
        lines.append(f"You found {_name(game, victim)}'s body in {room}.")
    if a.impostor:
        if mt.victim is not None:
            lines.append(f"(Secret: you killed {_name(game, mt.victim)} in {mt.room}. Hide it.)")
        lines.append("(Secret: you are the Thing. Construct an innocent-sounding alibi.)")
    return "\n".join(lines)


def _public(game: Game, mt: Meeting) -> str:
    living = ", ".join(a.name for a in game.living())
    if mt.victim is not None:
        head = f"{_name(game, mt.reporter)} found {_name(game, mt.victim)} dead in {mt.room}."
    else:
        head = f"{_name(game, mt.reporter)} called an emergency meeting in the {HUB}."
    return f"{head} Still alive: {living}."


def _transcript_str(game: Game, transcript: list) -> str:
    return "\n".join(f"{_name(game, sid)}: {text}" for sid, text in transcript)


_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def _strip_think(text: str) -> str:
    # reasoning models (Qwen3) emit a <think>…</think> block, or a "Here's a thinking process:" preamble,
    # before the answer — drop it so only the spoken line reaches the table.
    text = _THINK_RE.sub(" ", text or "")
    text = re.sub(r"<think>.*$", " ", text, flags=re.S | re.I)  # unclosed think block
    text = re.sub(r"</?think>", " ", text, flags=re.I)
    text = re.sub(
        r"^\s*here'?s?\s+(a|my)\s+(thinking|thought)\s+process[^.]*[.:]", " ", text, flags=re.I
    )
    return text.strip()


def _is_pass(text: str) -> bool:
    t = re.sub(r"[^a-z]", "", text.lower())
    return t in {"pass", "nothing", "nocomment", "skip", "silent"} or text.strip() in {
        "-",
        "—",
        "...",
    }


def _trim(text: str) -> str:
    # keep each line to a single short sentence so it fits in a speech bubble over the agent's head
    text = _strip_think(text).strip().strip('"').replace("\n", " ")
    if len(text) > 120:
        cut = max(text.find(". ", 30, 150), text.find("? ", 30, 150), text.find("! ", 30, 150))
        if cut > 0:
            text = text[: cut + 1]
    return text[:150]


async def _statement_round(game, mt, transcript, opening: bool, on_item=None):
    # generate the whole round at once (one Groq call per agent, spread across rate buckets), then
    # REVEAL the lines one at a time — the caller paces the reveal so the table talks, not data-dumps.
    living = game.living()
    public = _public(game, mt)
    jobs = []
    for a in living:
        sys = (SYS_THING if a.impostor else SYS_CREW).format(name=a.name)
        convo = _transcript_str(game, transcript)
        ask = (
            "Your account in ONE short line: where were you, who can vouch, or who you suspect."
            if opening
            else "ONE short line ONLY if you have something to add or must defend yourself."
        )
        user = (
            f"{public}\n\nWhat you know:\n{_brief(game, mt, a)}\n\nMeeting so far:\n{convo or '(nobody has spoken yet)'}\n\n"
            f"{ask} You don't have to speak — reply exactly (pass) to stay quiet."
        )
        m = _model(a)
        jobs.append((sys, user + ("\n/no_think" if "qwen" in m else ""), m))
    outs = await llm.complete_many(jobs, temperature=0.8)
    for a, text in zip(living, outs):
        text = _trim(text)
        if not text or _is_pass(text):  # an agent may sit a round out
            continue
        transcript.append((a.id, text))
        if on_item:
            await on_item("statement", a.id, text)


async def _vote_round(game, mt, transcript, on_item=None) -> dict:
    living = game.living()
    public = _public(game, mt)
    convo = _transcript_str(game, transcript)
    names = [a.name for a in living]
    jobs = []
    for a in living:
        sys = (SYS_THING if a.impostor else SYS_CREW).format(name=a.name)
        guidance = (
            "Weigh the accounts: whose alibi is contradicted, who can't be vouched for near the body? "
            "If the discussion points to a likely Thing, VOTE them out — skipping when there's a real "
            "lead just lets it kill again. Skip only if it's a genuine coin-flip."
            if not a.impostor
            else "Vote to protect yourself: pile onto whoever the group already suspects (not yourself), "
            "or skip if suspicion is landing on you."
        )
        user = (
            f"{public}\n\nWhat you know:\n{_brief(game, mt, a)}\n\nFull discussion:\n{convo}\n\n"
            f"{guidance}\nChoose exactly one of: {', '.join(names)}, or skip. "
            'Reply ONLY as JSON: {"vote": "<name or skip>", "reason": "<a few words>"}'
        )
        m = _model(a)
        jobs.append((sys, user + ("\n/no_think" if "qwen" in m else ""), m))
    outs = await llm.complete_many(jobs, temperature=0.3)
    by_name = {a.name.lower(): a.id for a in living}
    votes: dict = {}
    for a, text in zip(living, outs):
        parsed = llm.parse_json(_strip_think(text)) or {}
        choice = str(parsed.get("vote", "skip")).strip().lower()
        votes[a.id] = by_name.get(choice, -1)
        mt.votes = dict(votes)  # partial ballot, so the snapshot can reveal votes one by one
        if on_item:
            await on_item("vote", a.id, votes[a.id])
    return votes


async def run_llm_meeting(game: Game, mt: Meeting, on_item=None) -> dict:
    # on_item("statement"|"vote", agent_id, payload) fires per LINE/VOTE; the caller broadcasts + paces
    # the reveal and mirrors each onto Artel. ROUNDS gives a real back-and-forth, not a one-shot.
    transcript: list = []
    mt.transcript = transcript
    mt.votes = {}
    for r in range(ROUNDS):
        await _statement_round(game, mt, transcript, opening=(r == 0), on_item=on_item)
    if on_item:
        await on_item("settle", -1, None)  # deliberation done — the table settles before the vote
    votes = await _vote_round(game, mt, transcript, on_item=on_item)
    mt.votes = votes
    return votes


def _canned_statement(game: Game, a) -> str:
    # a believable one-liner from what the agent privately saw — used when no LLM key is configured, so
    # the meeting scene still plays (with the deterministic decider for the vote).
    if not a.impostor and a.witnessed:
        imps = [game.by_id(i).name for i in a.witnessed if game.by_id(i).alive]
        if imps:
            return f"I saw it — {imps[0]} is the Thing."
    if a.found:
        _, room, vic = a.found[-1]
        return f"I found {game.by_id(vic).name} in the {room}."
    if a.seen:
        s = a.seen[-1]
        who = ", ".join(game.by_id(i).name for i in s.present[:2])
        return f"I was in the {s.room} with {who}." if who else f"I was over in the {s.room}."
    return "I was on my own the whole time."


async def run_canned_meeting(game: Game, mt: Meeting, decide, on_item=None) -> dict:
    transcript: list = []
    mt.transcript = transcript
    mt.votes = {}
    for a in game.living():
        text = _canned_statement(game, a)
        transcript.append((a.id, text))
        if on_item:
            await on_item("statement", a.id, text)
    if on_item:
        await on_item("settle", -1, None)
    votes = decide(game, mt)
    for a in game.living():
        mt.votes[a.id] = votes.get(a.id, -1)
        if on_item:
            await on_item("vote", a.id, mt.votes[a.id])
    mt.votes = votes
    return votes


async def play_llm(seed: int, n=6, impostors=1) -> Game:
    g = new_game(seed, n, impostors)
    assign_models(g)
    while g.winner is None and g.tick < MAX_TICKS:
        mt = g.step()
        if mt is not None:
            votes = await run_llm_meeting(g, mt)
            g.apply_votes(mt, votes)
    if g.winner is None:
        g.winner, g.win_by = "impostor", "timeout"
    return g
