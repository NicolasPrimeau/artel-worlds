from __future__ import annotations

import os

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
    "a wrong ejection helps the Thing. Speak in ONE or TWO sentences, in character, plain talk."
)
SYS_THING = (
    "You are {name}, and you are the Thing — an alien that has assimilated this researcher and is "
    "killing the crew at an isolated Antarctic station. You must NOT be found out. In this emergency "
    "meeting, blend in as a worried human: give a calm, plausible account of where you were, deflect "
    "suspicion, and if someone is onto you, cast doubt on them or point the group at someone else. "
    "Never admit what you are. If you witnessed your own kill, you may even 'report' it to look "
    "helpful. Speak in ONE or TWO sentences, in character, plain talk."
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


async def _statement_round(game, mt, transcript, opening: bool):
    living = game.living()
    public = _public(game, mt)
    jobs = []
    for a in living:
        sys = (SYS_THING if a.impostor else SYS_CREW).format(name=a.name)
        convo = _transcript_str(game, transcript)
        ask = (
            "Give your opening account: where were you, who can vouch for you, and anyone you suspect."
            if opening
            else "Respond to what's been said — defend yourself if accused, back a theory, or call out a lie."
        )
        user = f"{public}\n\nWhat you know:\n{_brief(game, mt, a)}\n\nMeeting so far:\n{convo or '(nobody has spoken yet)'}\n\n{ask}"
        jobs.append((sys, user, _model(a)))
    outs = await llm.complete_many(jobs, temperature=0.8)
    for a, text in zip(living, outs):
        text = (text or "").strip().replace("\n", " ")
        if text:
            transcript.append((a.id, text[:280]))


async def _vote_round(game, mt, transcript) -> dict:
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
        jobs.append((sys, user, _model(a)))
    outs = await llm.complete_many(jobs, temperature=0.3)
    by_name = {a.name.lower(): a.id for a in living}
    votes = {}
    for a, text in zip(living, outs):
        parsed = llm.parse_json(text) or {}
        choice = str(parsed.get("vote", "skip")).strip().lower()
        votes[a.id] = by_name.get(choice, -1)
    return votes


async def run_llm_meeting(game: Game, mt: Meeting, on_update=None) -> dict:
    # on_update("statement"|"vote") fires after each round so a live viewer watches the chat build
    transcript: list = []
    mt.transcript = transcript
    for r in range(ROUNDS):
        await _statement_round(game, mt, transcript, opening=(r == 0))
        if on_update:
            await on_update("statement")
    votes = await _vote_round(game, mt, transcript)
    mt.votes = votes
    if on_update:
        await on_update("vote")
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
