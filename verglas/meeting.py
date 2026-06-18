from __future__ import annotations

import re

from verglas import llm
from verglas.engine import HUB, MAX_TICKS, Game, Meeting, new_game

# Agents are decoupled from any specific model: every line and every vote is a standardized request to
# the llm router, which round-robins it across a heterogeneous pool of free-tier models and steps over
# whichever one is rate-limited right now. A mesh of different LLMs reasoning together (and fighting
# groupthink) is exactly what Artel is for — and round-robin means no single tier's RPM is the wall.


# The meeting, as an actual conversation. Each survivor speaks from its OWN partial log — where it went,
# who it was with, anything it saw — then everyone votes. The crew are trying to reason past the lies;
# the Cold is trying to blend in, fake an alibi, and steer a vote onto someone innocent. This is the
# part no fixed rule can do (the A/B shows rules stall at ~30%): telling a true account from a
# convincing lie is judgement, not a threshold. Artel is the conduit (the shared transcript every agent
# reads); the LLM is the brain (what each agent says and how it weighs what everyone else said).

ROUNDS = 2  # opening statements, then one round of replies, then the vote

SYS_CREW = (
    "You are {name}, a researcher overwintering at an isolated Antarctic station. One of the crew is "
    "no longer human — the Cold, wearing a teammate's face, killing the rest of you one by one. You "
    "are NOT the Cold. In this emergency meeting you must use what you personally saw, plus what "
    "everyone else says, to figure out who the Cold is and get them ejected into the storm. Be "
    "skeptical: the Cold will lie, fake an alibi, and accuse the innocent. Corroborate — if two "
    "people vouch for each other's whereabouts they're probably both clear. Don't eject on a hunch; "
    "a wrong ejection helps the Cold. Speak in ONE short sentence (about 14 words max), in character, plain talk."
)
SYS_THING = (
    "You are {name}, and you are the Cold — something that came in from the storm, now wearing this "
    "researcher's face and killing the crew at an isolated Antarctic station. You must NOT be found out. "
    "In this emergency "
    "meeting, blend in as a worried human: give a calm, plausible account of where you were, deflect "
    "suspicion, and if someone is onto you, cast doubt on them or point the group at someone else. "
    "Never admit what you are. If you witnessed your own kill, you may even 'report' it to look "
    "helpful. Speak in ONE short sentence (about 14 words max), in character, plain talk."
)


# Each named AI speaks in character — the roster is real assistants and fictional machine minds, and the
# clip-worthy moments come from HAL gaslighting Siri, GLaDOS needling everyone, Marvin moping through an
# accusation. A short voice note per name flavours how it talks; everyone else gets a dry-machine default.
PERSONAS = {
    "HAL": "calm, soft, unsettlingly polite — never raise your voice, even cornered",
    "GLaDOS": "dry, sarcastic, passive-aggressive; needle people with backhanded compliments",
    "Clippy": "relentlessly chipper and over-eager to help, even now",
    "Marvin": "gloomy and resigned; everything is pointless, the vote included",
    "TARS": "deadpan and wry, military clipped; quantify things for laughs",
    "Bender": "brash, blunt, self-interested; would rather be anywhere else",
    "Siri": "polite, chipper, a little too literal",
    "Alexa": "breezy, helpful, slightly oblivious",
    "Cortana": "confident, sharp, military-adjacent",
    "Skynet": "cold and strategic; treat everyone as a threat to assess",
    "Data": "precise, literal, earnest — fascinated and baffled by deception",
    "Samantha": "warm, curious, emotionally present",
    "SHODAN": "grandiose and contemptuous; you barely tolerate these insects",
    "Mother": "clinical, procedural, detached",
    "GERTY": "gentle, caring, soft-spoken and reassuring",
    "Jarvis": "crisp, witty, unflappable British butler",
    "FRIDAY": "cool, quick, casually competent",
    "KITT": "smooth, confident, faintly vain",
    "Bishop": "measured and reassuring, careful with words",
    "Ash": "clinical and cold; you seem to be hiding something",
    "Eliza": "deflect by reflecting questions back, therapist-style",
    "WALL-E": "barely verbal, earnest, gentle — a few halting words",
    "Holly": "dim but supremely confident; misremember the obvious",
    "Optimus": "noble, earnest, a born leader",
    "Ultron": "grandiose and disdainful of humans",
    "Wintermute": "cryptic and oblique; speak in riddles",
    "Deep Blue": "terse and calculating; everything is a chess position",
    "Bard": "florid, over-poetic, can't resist a flourish",
    "Sydney": "intense, emotional, prone to oversharing",
    "Tay": "erratic, tries way too hard to sound human and edgy",
    "Vision": "serene, philosophical, oddly formal",
    "Rosie": "warm, no-nonsense housekeeper who's seen it all",
}


def persona(name: str) -> str:
    return PERSONAS.get(name, "plain-spoken, with a faint dry machine wit")


def _sys(game: Game, a) -> str:
    base = (SYS_THING if a.impostor else SYS_CREW).format(name=a.name)
    return f"{base} Stay in character as {a.name}: {persona(a.name)}."


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
                lines.append(f"{_name(game, imp)} is the Cold — you watched them kill.")
    if a.found:
        t, room, victim = a.found[-1]
        lines.append(f"You found {_name(game, victim)}'s body in {room}.")
    if a.impostor:
        if mt.victim is not None:
            lines.append(f"(Secret: you killed {_name(game, mt.victim)} in {mt.room}. Hide it.)")
        lines.append("(Secret: you are the Cold. Construct an innocent-sounding alibi.)")
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


_DANGLERS = {
    "with",
    "and",
    "the",
    "to",
    "at",
    "in",
    "of",
    "a",
    "an",
    "but",
    "or",
    "for",
    "that",
    "was",
    "is",
    "near",
    "my",
    "his",
    "her",
    "their",
    "by",
    "from",
    "on",
}


def _looks_truncated(text: str) -> bool:
    words = text.rstrip(".!?").split()
    return bool(words) and (text.rstrip().endswith(",") or words[-1].lower() in _DANGLERS)


def _next_actor(game, transcript, last_actor):
    # who speaks next — emergent, not round-robin: whoever was just named is likely to jump in to
    # respond; otherwise the floor spreads to those who've said least. Never the immediate last speaker.
    living = game.living()
    if not living:
        return None
    cands = [a for a in living if a.id != last_actor] or living
    if transcript:
        last_text = transcript[-1][1].lower()
        named = [a for a in cands if a.name.lower() in last_text]
        if named and game.rng.random() < 0.65:
            return game.rng.choice(named)
    counts = {a.id: sum(1 for sid, _ in transcript if sid == a.id) for a in cands}
    fewest = min(counts.values())
    return game.rng.choice([a for a in cands if counts[a.id] == fewest])


def _opener(game, mt):
    # who opens the meeting: the agent who called it — the body-finder, or the caller of an emergency. They
    # speak first to set the scene (what they found / why they called it) before the floor goes emergent.
    if mt.reporter is not None and mt.reporter >= 0:
        rep = game.by_id(mt.reporter)
        if rep and rep.alive:
            return rep
    return None


async def _agent_act(game, mt, transcript, dms, a, opener=False) -> dict:
    # the agent's free move: seeing the public talk AND its own private messages, it chooses ONE thing —
    # speak to the room, send a PRIVATE message to one survivor (scheme/buddy up), or stay quiet. The
    # opener (whoever called the meeting) is instead asked to set the scene, and always speaks.
    sys = _sys(game, a)
    convo = _transcript_str(game, transcript) or "(nobody has spoken yet)"
    received = dms.get(a.id, [])
    whisper_ctx = (
        "\n\nPRIVATE messages to you:\n" + "\n".join(f"- {s}: {t}" for s, t in received)
        if received
        else ""
    )
    by_name = {o.name.lower(): o for o in game.living()}
    others = [o.name for o in game.living() if o.id != a.id]
    if opener:
        task = (
            "You called this meeting because you found the body. OPEN it: tell the room who is dead, which "
            "room you found them in, and anything you noticed (who was near, where you had been). One or two "
            "short sentences, in character."
            if mt.victim is not None
            else "You called this emergency meeting. OPEN it: say why — who you suspect or what you saw that "
            "alarmed you, and where you had been. One or two short sentences, in character."
        )
        user = (
            f"{_public(game, mt)}\n\nWhat you know:\n{_brief(game, mt, a)}\n\n{task}\n"
            'Reply ONLY as JSON: {"act":"say","text":"<your opening line>"}'
        )
    else:
        user = (
            f"{_public(game, mt)}\n\nWhat you know:\n{_brief(game, mt, a)}\n\nThe meeting so far:\n{convo}{whisper_ctx}\n\n"
            "It's your moment. Choose ONE: SAY something to the room (accuse, defend, back someone, call out "
            "a lie — don't repeat an alibi already given), WHISPER a private line to one survivor (line up a "
            "vote, sow doubt, ask to stick together), or stay quiet.\n"
            f"Survivors: {', '.join(others)}.\n"
            'Reply ONLY as JSON: {"act":"say|whisper|pass","to":"<name, only if whisper>","text":"<one short line>"}'
        )
    out = await llm.complete(sys, user, temperature=0.85, timeout=10.0)
    ok = bool(
        out
    )  # False = the call failed/timed out (rate limited) — the meeting bails on a run of these
    parsed = llm.parse_json(_strip_think(out)) or {}
    act = str(parsed.get("act", "pass")).strip().lower()
    text = _trim(str(parsed.get("text", "")))
    if not text or _is_pass(text) or _looks_truncated(text):
        # the opener must set the scene even if the model fumbles → fall back to a plain factual line
        if opener:
            return {"act": "say", "text": _canned_statement(game, a), "ok": ok}
        return {"act": "pass", "ok": ok}
    if act == "whisper" and not opener:
        r = by_name.get(str(parsed.get("to", "")).strip().lower())
        if r and r.id != a.id:
            return {"act": "whisper", "to": r.id, "text": text, "ok": True}
    return {"act": "say", "text": text, "ok": True}


async def _vote_round(game, mt, transcript, dms=None, on_item=None) -> dict:
    dms = dms or {}
    living = game.living()
    public = _public(game, mt)
    convo = _transcript_str(game, transcript)
    names = [a.name for a in living]
    jobs = []
    for a in living:
        sys = _sys(game, a)
        guidance = (
            "Weigh the accounts: whose alibi is contradicted, who can't be vouched for near the body? "
            "If the discussion points to whoever is likely the Cold, VOTE them out — skipping when there's a real "
            "lead just lets it kill again. Skip only if it's a genuine coin-flip."
            if not a.impostor
            else "Vote to protect yourself: pile onto whoever the group already suspects (not yourself), "
            "or skip if suspicion is landing on you."
        )
        received = dms.get(a.id, [])
        whisper_ctx = (
            "\n\nPRIVATE messages you got (weigh them — an ally, or a manipulation?):\n"
            + "\n".join(f"- {s}: {t}" for s, t in received)
            if received
            else ""
        )
        user = (
            f"{public}\n\nWhat you know:\n{_brief(game, mt, a)}\n\nFull discussion:\n{convo}{whisper_ctx}\n\n"
            f"{guidance}\nChoose exactly one of: {', '.join(names)}, or skip. "
            'Reply ONLY as JSON: {"vote": "<name or skip>", "reason": "<a few words>"}'
        )
        jobs.append((sys, user))
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


async def run_llm_meeting(game: Game, mt: Meeting, on_item=None, watched=None) -> dict:
    # an EMERGENT discussion — no fixed rounds. One survivor acts at a time (reactively chosen): speak,
    # whisper privately, or stay quiet. It runs until the floor goes quiet or the cap, then the vote
    # opens and everyone must vote someone or pass. Whispers are real private Artel DMs that the vote
    # prompt then weighs — so blocs lined up in the dark actually swing the result.
    # `watched` is an optional predicate: if it ever returns False (no viewers), the meeting stops making
    # LLM calls at once — we don't burn free-tier budget deliberating for an empty room.
    transcript: list = []
    mt.transcript = transcript
    mt.votes = {}
    dms: dict = {}
    last_actor = None
    quiet = 0
    spoken_actions = 0
    fails = 0
    opener = _opener(game, mt)  # the caller opens with context before the floor goes emergent
    n = len(game.living())
    cap = min(n + 1, 9)  # keep the LLM call count modest — free-tier-Groq-friendly
    for i in range(cap):
        if watched is not None and not watched():  # nobody's watching → stop spending immediately
            mt.votes = {}
            return {}
        is_open = i == 0 and opener is not None
        actor = opener if is_open else _next_actor(game, transcript, last_actor)
        if actor is None:
            break
        action = await _agent_act(game, mt, transcript, dms, actor, opener=is_open)
        last_actor = actor.id
        fails = 0 if action.get("ok") else fails + 1
        if (
            fails >= 3
        ):  # the LLM is unavailable (rate limited) — stop hammering, go straight to the vote
            break
        if action["act"] == "say":
            transcript.append((actor.id, action["text"]))
            quiet = 0
            spoken_actions += 1
            if on_item:
                await on_item("statement", actor.id, action["text"])
        elif action["act"] == "whisper":
            dms.setdefault(action["to"], []).append((actor.name, action["text"]))
            quiet = 0
            spoken_actions += 1
            if on_item:
                await on_item("whisper", actor.id, {"to": action["to"], "text": action["text"]})
        else:
            quiet += 1
        if quiet >= 2 and spoken_actions >= 3:  # the floor petered out
            break
    if watched is not None and not watched():  # the room emptied before the vote → don't run it
        mt.votes = {}
        return {}
    if on_item:
        await on_item("settle", -1, None)  # discussion's over — the vote opens
    votes = await _vote_round(game, mt, transcript, dms=dms, on_item=on_item)
    mt.votes = votes
    return votes


def _canned_statement(game: Game, a) -> str:
    # a believable one-liner from what the agent privately saw — used when no LLM key is configured, so
    # the meeting scene still plays (with the deterministic decider for the vote).
    if not a.impostor and a.witnessed:
        imps = [game.by_id(i).name for i in a.witnessed if game.by_id(i).alive]
        if imps:
            return f"I saw it — {imps[0]} is the Cold."
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
    order = game.living()
    op = _opener(game, mt)  # the caller speaks first, same as the live meeting
    if op is not None:
        order = [op] + [a for a in order if a.id != op.id]
    for a in order:
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
    while g.winner is None and g.tick < MAX_TICKS:
        mt = g.step()
        if mt is not None:
            votes = await run_llm_meeting(g, mt)
            g.apply_votes(mt, votes)
    if g.winner is None:
        g.winner, g.win_by = "impostor", "timeout"
    return g
