from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .world import WorldMap, facing_from_delta, generate_world

# VibeQuest — a shared multiplayer DnD world where players collectively ARE the Dungeon Master.
# AI agents are the party. Players play cards that resolve the quest. Cards batch in time windows,
# resolve in random order, and the party reacts in character. One global instance, drop in anytime.
#
# Artel integration:
#   memories  → context fed to LLM before resolving each card (what the party remembers)
#   tasks     → quest steps (created at quest start, claimed/completed by AI agents)
#   messages  → in-the-moment agent coordination ("wizard to paladin: this is a terrible idea")

CARD_WINDOW = 5.0  # seconds between resolution checks

# ── Quest categories ───────────────────────────────────────────────────────────────────────────────
# A category fixes the world theme and provides a curated task + complication pool.
# The LLM constructs the narrative JIT as cards are played — only the starting situation is fixed.
# Future categories: grocery, commute, school, airport, etc.

QUEST_CATEGORIES: dict[str, dict] = {
    "office": {
        "theme": "office",
        "tasks": [
            {
                "title": "Q3 Expense Report",
                "hook": "Someone needs to submit the Q3 expense report before the finance system locks at 5pm.",
            },
            {
                "title": "Fix The Printer",
                "hook": "The printer on the third floor has stopped working. It needs to be running before the 2pm presentation.",
            },
            {
                "title": "The Missing Key Fob",
                "hook": "Someone's access key fob stopped working and they can't get into the building. Needs to be resolved before their shift ends.",
            },
            {
                "title": "Coffee Pod Situation",
                "hook": "Someone used the last coffee pod and didn't reorder. This needs to be addressed before the 9am stand-up.",
            },
            {
                "title": "Room Double-Booking",
                "hook": "The main conference room is double-booked for Thursday. Someone needs to sort this out before people start arriving.",
            },
            {
                "title": "The Unsigned NDA",
                "hook": "A contractor started today without signing the NDA. Legal needs it signed before end of day.",
            },
            {
                "title": "Laptop Recovery",
                "hook": "An ex-employee still has a company laptop. IT needs it back before the asset audit tomorrow.",
            },
            {
                "title": "The Fish Incident",
                "hook": "Someone microwaved fish in the break room and the smell is spreading to the open plan. This has to stop.",
            },
            {
                "title": "All-Hands Deck",
                "hook": "The all-hands presentation is in two hours and slides are still missing from three departments.",
            },
            {
                "title": "New Hire Setup",
                "hook": "A new hire started today with no desk, no computer, and no system access. Someone needs to fix this.",
            },
            {
                "title": "The IT Ticket",
                "hook": "An IT support ticket has been open for three weeks with no update. Someone needs to get it resolved before the end of the quarter.",
            },
            {
                "title": "Catering Mix-Up",
                "hook": "The catering order for tomorrow's client lunch was placed at the wrong branch. Someone needs to sort this out today.",
            },
            {
                "title": "Form 2309-B",
                "hook": "A form needs three signatures before it can be processed. Two of the signatories are in different buildings and one is not responding.",
            },
            {
                "title": "The Projector",
                "hook": "The projector in the small meeting room won't connect to any laptop. A client presentation starts in 20 minutes.",
            },
            {
                "title": "Parking Situation",
                "hook": "A VIP visitor is arriving in an hour and nobody arranged parking. The visitor is already on their way.",
            },
            {
                "title": "The Good Stapler",
                "hook": "Someone's good stapler has gone missing from their desk. They have asked for help recovering it. The regular staplers are not acceptable.",
            },
            {
                "title": "The Vending Machine",
                "hook": "The vending machine took someone's money and dispensed nothing. Three other people have also lost money. Someone needs to get to the bottom of this.",
            },
            {
                "title": "Birthday Logistics",
                "hook": "It is someone's birthday. A cake was ordered to the wrong address. The birthday person arrives in 45 minutes.",
            },
            {
                "title": "The Thermostat Dispute",
                "hook": "Two teams on the same floor are in a cold war over the office thermostat. Someone needs to mediate before it escalates to HR.",
            },
            {
                "title": "The Offboarding",
                "hook": "Someone is leaving at end of day and needs to be formally offboarded. Nobody owns this process and nothing has been started.",
            },
        ],
        "complications": [
            "Half the relevant people are in back-to-back meetings until 4pm.",
            "The system has been having intermittent outages all morning.",
            "The elevator to the relevant floor is out of service.",
            "Nobody seems to know who is actually responsible for this.",
            "There is a conflicting urgent priority from a different team.",
            "The person who normally handles this is out sick today.",
            "The deadline was moved up by two hours without notice.",
            "Someone already tried to fix this earlier and made it slightly worse.",
            "A new policy was announced this morning that changes how this should be handled.",
            "The approval chain requires someone who is currently unreachable.",
            "There is a mandatory compliance training that everyone must attend this afternoon.",
            "A vendor has been on hold for 40 minutes.",
            "The relevant documentation is in a shared drive that nobody has access to.",
            "Building security has flagged something unrelated but is now involved.",
            "There is loud construction noise on the relevant floor.",
        ],
    },
}

# ── Party archetypes ───────────────────────────────────────────────────────────────────────────────

NAMES = [
    "Dave",
    "Karen",
    "Steve",
    "Bethany",
    "Mike",
    "Brenda",
    "Todd",
    "Gary",
    "Ronathan",
    "Janet",
    "Chad",
    "Phyllis",
    "Kevin",
    "Tiffany",
    "Jim",
    "Barbara",
    "Doug",
    "Janice",
    "Tyler",
    "Pam",
    "Gerald",
    "Brad",
    "Linda",
    "Randy",
    "Jennifer",
    "Dwight",
    "Sandra",
    "Glen",
    "Brittany",
    "Barry",
    "Norbert",
    "Becky",
    "Dennis",
    "Mildred",
    "Scott",
    "Carol",
    "Kyle",
    "Susan",
    "Brent",
    "Donna",
    "Gareth",
    "Jeff",
    "Clive",
    "Nicole",
    "Derek",
    "Sheila",
    "Trevor",
    "Trish",
    "Phil",
    "Wanda",
    "Craig",
    "Denise",
    "Wayne",
    "Pamela",
    "Nigel",
    "Stacy",
    "Robert",
    "Cheryl",
    "Terry",
    "Deirdre",
]

PARTY_ARCHETYPES = [
    {
        "role": "worrier",
        "personality": "Deeply earnest. Treats every errand like it personally matters. Apologizes when things go wrong even if it was not their fault.",
    },
    {
        "role": "optimizer",
        "personality": "Slightly condescending. Believes every problem has an optimal solution that the others are ignoring. Keeps a notebook.",
    },
    {
        "role": "planner",
        "personality": "Overthinks everything. Has a plan but will not fully explain it. Sends very long follow-up emails.",
    },
    {
        "role": "scorekeeper",
        "personality": "Makes everything about themselves. Visibly keeping score. Will bring this up in a future one-on-one.",
    },
    {
        "role": "veteran",
        "personality": "Has seen worse. Keeps mentioning it. Calm to the point of being slightly alarming.",
    },
    {
        "role": "optimist",
        "personality": "Aggressively optimistic. Interprets every setback as a learning opportunity. Very hard to read.",
    },
]

# ── Card definitions ───────────────────────────────────────────────────────────────────────────────


class CardType(str, Enum):
    ACCUMULATE = "accumulate"
    ACTION = "action"
    CHAOS = "chaos"
    TWEAK = "tweak"


@dataclass
class CardDef:
    id: str
    name: str
    type: CardType
    description: str
    flavor: str
    weight: float = 1.0  # relative frequency in the hand


CARD_LIBRARY = [
    # ── Accumulate ────────────────────────────────────────────────────────────
    CardDef(
        "acc_encourage",
        "Quiet Confidence",
        CardType.ACCUMULATE,
        "Something small is going right. Someone noticed. Nobody said anything.",
        "Something is working.",
        1.5,
    ),
    CardDef(
        "acc_doubt",
        "Second Thoughts",
        CardType.ACCUMULATE,
        "A decision already made doesn't look as solid as it did earlier.",
        "Too late to reconsider.",
        1.5,
    ),
    CardDef(
        "acc_urgency",
        "Clock Ticking",
        CardType.ACCUMULATE,
        "The window is smaller than expected. Something just made that clear.",
        "Not much time.",
        1.2,
    ),
    CardDef(
        "acc_suspicion",
        "Someone Knows",
        CardType.ACCUMULATE,
        "A person here has been paying closer attention than anyone realized.",
        "They haven't said anything yet.",
        1.0,
    ),
    CardDef(
        "acc_confidence",
        "False Certainty",
        CardType.ACCUMULATE,
        "Confident. Wrong. Moving forward anyway.",
        "Reasonable. Wrong.",
        1.2,
    ),
    CardDef(
        "acc_sunk",
        "Too Far In",
        CardType.ACCUMULATE,
        "What's already been done can't be undone. That matters now.",
        "No going back.",
        1.0,
    ),
    CardDef(
        "acc_consensus",
        "Everyone Agrees",
        CardType.ACCUMULATE,
        "Unanimous agreement. This is unusual. It may not be a good sign.",
        "Suspicious unanimity.",
        1.1,
    ),
    CardDef(
        "acc_borrowed",
        "Window Closing",
        CardType.ACCUMULATE,
        "Something needed is about to become unavailable.",
        "Use it now.",
        1.0,
    ),
    CardDef(
        "acc_momentum",
        "Out of Hand",
        CardType.ACCUMULATE,
        "Someone else is now making decisions about this. They weren't asked.",
        "Rare.",
        0.2,
    ),
    # ── Action ────────────────────────────────────────────────────────────────
    CardDef(
        "act_npc",
        "Caught in the Act",
        CardType.ACTION,
        "The person here just saw something. They're still deciding what to do with it.",
        "They're processing.",
        1.0,
    ),
    CardDef(
        "act_obstacle",
        "The Required Step",
        CardType.ACTION,
        "There's a step that can't be skipped. Someone here controls it.",
        "There's a process.",
        1.0,
    ),
    CardDef(
        "act_shortcut",
        "The Back Way",
        CardType.ACTION,
        "There's a faster path. Nobody takes it. The reason isn't immediately clear.",
        "Why is this available.",
        0.8,
    ),
    CardDef(
        "act_revelation",
        "Late Intelligence",
        CardType.ACTION,
        "Information that should have arrived earlier is arriving now.",
        "Should have known sooner.",
        0.8,
    ),
    CardDef(
        "act_ally",
        "Unexpected Help",
        CardType.ACTION,
        "An offer to help. Genuine. Conditional on something.",
        "There's a catch.",
        0.7,
    ),
    CardDef(
        "act_misunderstanding",
        "Lost in Translation",
        CardType.ACTION,
        "Two people have been using the same words to mean different things.",
        "Both readings are plausible.",
        0.9,
    ),
    CardDef(
        "act_detour",
        "Wrong Door",
        CardType.ACTION,
        "The direct route is blocked. There's a mundane, specific reason.",
        "Classic.",
        0.8,
    ),
    CardDef(
        "act_advice",
        "The Expert",
        CardType.ACTION,
        "Someone here knows exactly what's happening. They're going to explain it.",
        "They're right. Unfortunately.",
        0.8,
    ),
    CardDef(
        "act_missing",
        "Not Where It Should Be",
        CardType.ACTION,
        "Something needed isn't where it was supposed to be.",
        "Somebody moved it.",
        0.8,
    ),
    CardDef(
        "act_confront",
        "Ask Directly",
        CardType.ACTION,
        "A direct question to the person here. They'll answer honestly.",
        "They told the truth.",
        1.1,
    ),
    CardDef(
        "act_loop_in",
        "Now They Know",
        CardType.ACTION,
        "The person here is being told what's actually going on.",
        "Can't un-tell someone.",
        0.9,
    ),
    CardDef(
        "act_precedent",
        "How It Was Done Before",
        CardType.ACTION,
        "Someone invokes precedent. The precedent doesn't quite apply.",
        "Procedurally sound. Wrong.",
        0.7,
    ),
    CardDef(
        "act_authority",
        "Someone Decides",
        CardType.ACTION,
        "A binding decision is made without being requested. It stands.",
        "Rare. Final.",
        0.2,
    ),
    # ── Chaos ─────────────────────────────────────────────────────────────────
    CardDef(
        "cha_wild",
        "Non Sequitur",
        CardType.CHAOS,
        "Someone announces something completely unrelated, then continues as if nothing happened.",
        "Context unknown.",
        0.6,
    ),
    CardDef(
        "cha_escalate",
        "Word Gets Out",
        CardType.CHAOS,
        "Someone who wasn't supposed to know now knows. They have opinions.",
        "They're already moving.",
        0.6,
    ),
    CardDef(
        "cha_mishap",
        "Equipment Failure",
        CardType.CHAOS,
        "Something essential stops working. Right now.",
        "Not my problem.",
        0.7,
    ),
    CardDef(
        "cha_witness",
        "Wrong Place, Wrong Time",
        CardType.CHAOS,
        "The wrong person has just seen the wrong thing.",
        "They will.",
        0.5,
    ),
    CardDef(
        "cha_arrival",
        "Unscheduled",
        CardType.CHAOS,
        "Someone arrives who wasn't expected. Their timing is terrible.",
        "Nobody invited them.",
        0.7,
    ),
    CardDef(
        "cha_containment",
        "Contained No Longer",
        CardType.CHAOS,
        "Something that was being quietly managed isn't quiet anymore.",
        "Everyone knows.",
        0.6,
    ),
    CardDef(
        "cha_cascade",
        "Collateral Damage",
        CardType.CHAOS,
        "Something done earlier has caused a problem somewhere else.",
        "Connected.",
        0.5,
    ),
    CardDef(
        "cha_category",
        "Wrong Team",
        CardType.CHAOS,
        "This has been routed to the wrong people. The right people are now asking questions.",
        "Rare. Structural.",
        0.2,
    ),
    # ── Tweak ─────────────────────────────────────────────────────────────────
    CardDef(
        "twk_scope",
        "Also That",
        CardType.TWEAK,
        "There's one more step. It was always going to be there.",
        "Adjacent. Unavoidable.",
        0.9,
    ),
    CardDef(
        "twk_reframe",
        "Different Angle",
        CardType.TWEAK,
        "The same situation, read differently. Both readings are defensible.",
        "Ambiguous.",
        0.8,
    ),
    CardDef(
        "twk_version",
        "Which Version",
        CardType.TWEAK,
        "Two versions of the same thing are both in circulation.",
        "Both are real.",
        0.8,
    ),
    CardDef(
        "twk_retroactive",
        "Always Priority One",
        CardType.TWEAK,
        "The urgency was just retroactively elevated. It was apparently always high.",
        "History revised.",
        0.7,
    ),
    CardDef(
        "twk_notes",
        "What Was Decided",
        CardType.TWEAK,
        "Two people remember the same decision differently. No record exists.",
        "Contested.",
        0.8,
    ),
    CardDef(
        "twk_info",
        "Always True",
        CardType.TWEAK,
        "Something that changes how this should be read has been true the whole time.",
        "Late arrival.",
        0.7,
    ),
    CardDef(
        "twk_pressure",
        "Updated Urgency",
        CardType.TWEAK,
        "One specific part of this just became more urgent.",
        "Upward only.",
        0.8,
    ),
    CardDef(
        "twk_recontextualize",
        "Still True. Different Now.",
        CardType.TWEAK,
        "Nothing changed. Everything's different.",
        "Rare. Everything shifts.",
        0.2,
    ),
]

CARD_BY_ID = {c.id: c for c in CARD_LIBRARY}

# ── Dice ──────────────────────────────────────────────────────────────────────────────────────────


class DiceResult(str, Enum):
    NAT_1 = "nat_1"  # critical failure
    LOW = "low"  # 2–6, things get worse
    MID = "mid"  # 7–14, partial success
    HIGH = "high"  # 15–19, clear success
    NAT_20 = "nat_20"  # critical success


def roll_d20(rng: random.Random) -> tuple[int, DiceResult]:
    value = rng.randint(1, 20)
    if value == 1:
        result = DiceResult.NAT_1
    elif value <= 6:
        result = DiceResult.LOW
    elif value <= 14:
        result = DiceResult.MID
    elif value <= 19:
        result = DiceResult.HIGH
    else:
        result = DiceResult.NAT_20
    return value, result


# ── Game state ────────────────────────────────────────────────────────────────────────────────────


@dataclass
class PartyMember:
    id: str
    name: str
    role: str
    personality: str
    hp: int = 20
    status: str = "ready"
    sprite: int = 1


@dataclass
class NPC:
    id: str
    name: str
    role: str
    personality: str
    sprite: int
    waypoint_idx: int
    behavior: str = "stationary"


@dataclass
class Prop:
    id: str
    label: str
    description: str
    waypoint_idx: int


@dataclass
class PlayedCard:
    id: str
    card_id: str
    player_id: str
    played_at: float = field(default_factory=time.time)
    target_npc_id: str | None = None


@dataclass
class CardResolution:
    card: PlayedCard
    card_def: CardDef
    dice_value: int
    dice_result: DiceResult
    narrative: str
    consequence: str


MIN_RESOLUTIONS = 3
MAX_RESOLUTIONS = 8
MAX_SCENE_ROUNDS = 4


REGISTERS = [
    "a tense heist thriller",
    "gothic horror",
    "a slow-burn romance",
    "hardboiled noir detective",
    "a deadpan nature documentary",
    "high courtroom drama",
    "Cold War spy espionage",
    "a sweeping disaster epic",
    "a corporate procedural",
    "an overwrought war film",
    "a cutthroat cooking competition",
    "a haunted-house ghost story",
    "a sports underdog story",
    "a prestige-TV crime saga",
]


@dataclass
class QuestState:
    id: str
    template_id: str
    title: str
    hook: str
    complication: str
    register: str = "a deadpan documentary"
    beats: list[str] = field(default_factory=list)
    result_history: list[str] = field(default_factory=list)
    objectives: list[str] = field(default_factory=list)
    npcs: list[NPC] = field(default_factory=list)
    resolution_count: int = 0
    momentum: int = 0
    tension: int = 0
    outcome: str | None = None
    scene_rounds: int = 0
    scene_beat_start: int = 0
    facts: list[str] = field(default_factory=list)
    pressure_pool: list[dict] = field(default_factory=list)
    props: list[Prop] = field(default_factory=list)


def apply_world_changes(quest: QuestState, changes: list[dict], rng: random.Random) -> list[dict]:
    side_effects: list[dict] = []
    for ch in changes:
        action = ch.get("action", "")
        if action == "add_prop":
            prop_id = str(ch.get("id") or f"prop_{len(quest.props)}")
            if not any(p.id == prop_id for p in quest.props):
                quest.props.append(
                    Prop(
                        id=prop_id,
                        label=str(ch.get("label", "ITEM"))[:24],
                        description=str(ch.get("description", "")),
                        waypoint_idx=int(ch.get("waypoint_idx", 0)),
                    )
                )
        elif action == "remove_prop":
            prop_id = str(ch.get("id", ""))
            quest.props = [p for p in quest.props if p.id != prop_id]
        elif action == "prop_update":
            prop_id = str(ch.get("id", ""))
            for p in quest.props:
                if p.id == prop_id:
                    if "label" in ch:
                        p.label = str(ch["label"])[:24]
                    if "description" in ch:
                        p.description = str(ch["description"])
                    if "waypoint_idx" in ch:
                        p.waypoint_idx = int(ch["waypoint_idx"])
                    break
        elif action == "add_npc":
            raw_name = str(ch.get("name", "unknown"))
            slug = "_".join(raw_name.lower().split())[:24]
            npc_id = f"npc_{slug}"
            if any(n.id == npc_id for n in quest.npcs):
                npc_id = f"npc_{slug}_{len(quest.npcs)}"
            quest.npcs.append(
                NPC(
                    id=npc_id,
                    name=str(ch.get("name", "Unknown")),
                    role=str(ch.get("role", "")),
                    personality=str(ch.get("personality", "")),
                    sprite=rng.randint(0, 5),
                    waypoint_idx=int(ch.get("waypoint_idx", 0)),
                    behavior=str(ch.get("behavior", "stationary")),
                )
            )
        elif action in ("move_npc", "npc_move"):
            npc_id = str(ch.get("npc_id", ""))
            new_wp = int(ch.get("waypoint_idx", 0))
            for npc in quest.npcs:
                if npc.id == npc_id:
                    npc.waypoint_idx = new_wp
                    break
        elif action == "remove_npc":
            npc_id = str(ch.get("npc_id", ""))
            quest.npcs = [n for n in quest.npcs if n.id != npc_id]
        elif action in ("npc_say", "schedule"):
            side_effects.append(ch)
    return side_effects


@dataclass
class WindowState:
    opened_at: float
    closes_at: float
    cards: list[PlayedCard] = field(default_factory=list)
    resolving: bool = False
    resolutions: list[CardResolution] = field(default_factory=list)


@dataclass
class GameState:
    run_id: str
    character: PartyMember
    quest: QuestState
    window: WindowState
    world: WorldMap | None = None
    lx: int = 0
    ly: int = 0
    facing: str = "up"
    rpos: int = 0
    target_idx: int = 0
    log: list[dict[str, Any]] = field(default_factory=list)
    tick: int = 0
    phase: str = "active"  # "intro" | "active" | "resolving" | "complete"
    viewers: int = 0

    def log_event(self, kind: str, text: str, data: dict | None = None) -> None:
        self.log.append({"t": time.time(), "kind": kind, "text": text, **(data or {})})
        if len(self.log) > 200:
            self.log = self.log[-200:]


def _make_character(rng: random.Random) -> PartyMember:
    arch = rng.choice(PARTY_ARCHETYPES)
    sprite = rng.randint(1, 10)
    return PartyMember(
        id=str(uuid.uuid4())[:8],
        name=rng.choice(NAMES),
        role=arch["role"],
        personality=arch["personality"],
        sprite=sprite,
    )


def make_quest(rng: random.Random) -> tuple[QuestState, str]:
    cat_name = rng.choice(list(QUEST_CATEGORIES.keys()))
    cat = QUEST_CATEGORIES[cat_name]
    task = rng.choice(cat["tasks"])
    complication = rng.choice(cat["complications"])
    quest_id = str(uuid.uuid4())[:8]
    quest = QuestState(
        id=quest_id,
        template_id=cat_name,
        title=task["title"],
        hook=task["hook"],
        complication=complication,
        register=rng.choice(REGISTERS),
    )
    return quest, cat["theme"]


def new_game(rng: random.Random | None = None, preset_quest: QuestState | None = None) -> GameState:
    rng = rng or random.Random()
    now = time.time()
    character = _make_character(rng)
    if preset_quest is not None:
        quest = preset_quest
        theme = QUEST_CATEGORIES.get(quest.template_id, {}).get("theme", "office")
    else:
        quest, theme = make_quest(rng)
    window = WindowState(opened_at=now, closes_at=now + CARD_WINDOW)
    run_id = str(uuid.uuid4())[:8]
    world = generate_world(rng, theme=theme, step_count=MAX_RESOLUTIONS)
    state = GameState(run_id=run_id, character=character, quest=quest, window=window, world=world)
    state.lx, state.ly = world.route[0]
    state.facing = "up"
    state.rpos = 0
    state.target_idx = 0
    state.log_event("quest_start", quest.hook)
    state.log_event("complication", quest.complication)
    return state


def sync_target(state: GameState) -> None:
    if state.world is None:
        return
    state.target_idx = min(state.quest.resolution_count + 1, len(state.world.waypoints) - 1)


def at_station(state: GameState) -> bool:
    if state.world is None or not state.world.wp_route_idx:
        return True
    boundary = state.world.wp_route_idx[min(state.target_idx, len(state.world.wp_route_idx) - 1)]
    return state.rpos >= boundary


TRAVEL_EVENTS = [
    "Are we there yet?",
    "I have a bad feeling about this corridor.",
    "Stay close. And stay quiet.",
    "Did anyone else hear that?",
    "I'm putting this on my expense report.",
    "We should've taken the elevator.",
    "Keep moving. Don't look back.",
    "This place wasn't on the map.",
    "My feet hurt. Quest-grade pain.",
    "If we make it back, I'm taking a long lunch.",
]


def maybe_travel_event(state: GameState, rng: random.Random) -> str | None:
    if at_station(state):
        return None
    if rng.random() > 0.10:
        return None
    return rng.choice(TRAVEL_EVENTS)


def step_party(state: GameState) -> bool:
    world = state.world
    if world is None or not world.route:
        return False
    allowed = world.wp_route_idx[min(state.target_idx, len(world.wp_route_idx) - 1)]
    if state.rpos >= allowed:
        return False
    nx, ny = world.route[state.rpos + 1]
    state.facing = facing_from_delta(nx - state.lx, ny - state.ly)
    state.lx, state.ly = nx, ny
    state.rpos += 1
    return True


def deal_hand(rng: random.Random, size: int = 5) -> list[CardDef]:
    pool = []
    for card in CARD_LIBRARY:
        pool.extend([card] * max(1, round(card.weight * 10)))
    seen_types: set[CardType] = set()
    hand: list[CardDef] = []
    attempts = 0
    while len(hand) < size and attempts < 100:
        card = rng.choice(pool)
        if card.type not in seen_types or len(hand) < 2:
            hand.append(card)
            seen_types.add(card.type)
        attempts += 1
    return hand


def advance_window(state: GameState, rng: random.Random) -> list[PlayedCard]:
    now = time.time()
    played = list(state.window.cards)
    rng.shuffle(played)
    state.window = WindowState(opened_at=now, closes_at=now + CARD_WINDOW)
    return played


# Dice mostly steer the LLM; crits are the real mechanical swings.
_DELTA = {
    DiceResult.NAT_1: -5,
    DiceResult.LOW: -1,
    DiceResult.MID: 1,
    DiceResult.HIGH: 1,
    DiceResult.NAT_20: 5,
}


def apply_card_effects(card_def: CardDef, dice: DiceResult, quest: QuestState) -> None:
    if card_def.type in (CardType.ACCUMULATE, CardType.ACTION):
        quest.momentum = max(-10, min(10, quest.momentum + _DELTA[dice]))
    elif card_def.type == CardType.CHAOS:
        quest.tension = min(10, quest.tension + (2 if dice == DiceResult.NAT_1 else 1))
    if dice == DiceResult.NAT_1:
        quest.tension = min(10, quest.tension + 1)


def classify_result(resolutions: list[CardResolution]) -> str:
    if not resolutions:
        return "uneventful"
    crit_hi = any(r.dice_result == DiceResult.NAT_20 for r in resolutions)
    crit_lo = any(r.dice_result == DiceResult.NAT_1 for r in resolutions)
    if crit_hi and crit_lo:
        return "chaotic"
    if crit_hi:
        return "breakthrough"
    if crit_lo:
        return "disaster"
    highs = sum(1 for r in resolutions if r.dice_result == DiceResult.HIGH)
    lows = sum(1 for r in resolutions if r.dice_result == DiceResult.LOW)
    net = highs - lows
    if net >= 1:
        return "triumph"
    if net <= -1:
        return "setback"
    return "mixed"


def apply_disaster(state: GameState, rng: random.Random) -> "PartyMember | None":
    ch = state.character
    if ch.status == "lost":
        return None
    ch.hp = max(0, ch.hp - 5)
    ch.status = "lost" if ch.hp <= 0 else "rattled"
    return ch
