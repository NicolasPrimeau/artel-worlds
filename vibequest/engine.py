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
    # Accumulate — build up pressure or momentum
    CardDef(
        "acc_encourage",
        "Words of Encouragement",
        CardType.ACCUMULATE,
        "The party feels slightly more capable than before.",
        "Stack these.",
        1.5,
    ),
    CardDef(
        "acc_doubt",
        "Creeping Doubt",
        CardType.ACCUMULATE,
        "Someone starts to question whether this was a good idea.",
        "Stack these.",
        1.5,
    ),
    CardDef(
        "acc_urgency",
        "Mounting Pressure",
        CardType.ACCUMULATE,
        "The deadline feels closer than it did a moment ago.",
        "Stack these.",
        1.2,
    ),
    CardDef(
        "acc_suspicion",
        "Suspicious Glances",
        CardType.ACCUMULATE,
        "Nobody is saying anything, but everyone is thinking something.",
        "Stack these.",
        1.0,
    ),
    # Action — something specific happens now
    CardDef(
        "act_npc",
        "Unexpected Witness",
        CardType.ACTION,
        "The person at this location has just witnessed your approach. They are forming an opinion.",
        "Causes a scene.",
        1.0,
    ),
    CardDef(
        "act_obstacle",
        "Bureaucratic Barrier",
        CardType.ACTION,
        "A form must be filled. A procedure must be followed.",
        "Unavoidable.",
        1.0,
    ),
    CardDef(
        "act_shortcut",
        "Suspicious Shortcut",
        CardType.ACTION,
        "There is a faster way. It is unclear why nobody uses it.",
        "Suspicious.",
        0.8,
    ),
    CardDef(
        "act_revelation",
        "Inconvenient Revelation",
        CardType.ACTION,
        "Something relevant to the quest comes to light at the worst possible moment.",
        "Always late.",
        0.8,
    ),
    CardDef(
        "act_ally",
        "Reluctant Ally",
        CardType.ACTION,
        "The person here offers to help. Their offer is specific and conditional on something not yet clear.",
        "Conditional.",
        0.7,
    ),
    # Chaos — wild magic surge energy
    CardDef(
        "cha_wild",
        "Wild Surge",
        CardType.CHAOS,
        "The person nearest to you stops mid-task, announces something unrelated, then continues as if nothing happened.",
        "Unpredictable.",
        0.6,
    ),
    CardDef(
        "cha_escalate",
        "Sudden Escalation",
        CardType.CHAOS,
        "The stakes just doubled. Someone in this building already knows. They are on their way to this location.",
        "Why.",
        0.6,
    ),
    CardDef(
        "cha_mishap",
        "Equipment Mishap",
        CardType.CHAOS,
        "The thing everyone here depends on has stopped working. The nearest person is explaining why it is not their problem.",
        "Classic.",
        0.7,
    ),
    CardDef(
        "cha_witness",
        "Very Bad Timing",
        CardType.CHAOS,
        "The wrong person has seen exactly the wrong thing at the wrong moment. They have not said anything yet. They will.",
        "Unavoidable in retrospect.",
        0.5,
    ),
    # Tweak — modify the current situation's context
    CardDef(
        "twk_weather",
        "Environmental Shift",
        CardType.TWEAK,
        "Conditions change in a way that makes everything slightly more difficult or absurd.",
        "Context shifts.",
        0.8,
    ),
    CardDef(
        "twk_reframe",
        "Alternative Interpretation",
        CardType.TWEAK,
        "The situation can be read differently. The party must decide which reading to act on.",
        "Ambiguous.",
        0.7,
    ),
    CardDef(
        "twk_stakes",
        "Revised Stakes",
        CardType.TWEAK,
        "It turns out what was at stake is slightly different than understood.",
        "Clarifying.",
        0.7,
    ),
    CardDef(
        "twk_tone",
        "Shift in Atmosphere",
        CardType.TWEAK,
        "The mood of the location changes. The quest proceeds differently because of it.",
        "Tonal.",
        0.6,
    ),
    # ── Accumulate (additional) ──────────────────────────────────────────────
    CardDef(
        "acc_confidence",
        "False Confidence",
        CardType.ACCUMULATE,
        "Someone is more certain than the situation warrants. Everyone finds this reassuring.",
        "Stack these.",
        1.2,
    ),
    CardDef(
        "acc_sunk",
        "Sunk Cost",
        CardType.ACCUMULATE,
        "Too much has already been invested to stop now. Nobody says this out loud.",
        "Stack these.",
        1.0,
    ),
    CardDef(
        "acc_consensus",
        "Group Consensus",
        CardType.ACCUMULATE,
        "Everyone agrees. Whether this is helpful remains to be seen.",
        "Stack these.",
        1.1,
    ),
    CardDef(
        "acc_borrowed",
        "Borrowed Time",
        CardType.ACCUMULATE,
        "The window for resolving this cleanly is narrowing. Everyone can feel it.",
        "Stack these.",
        1.0,
    ),
    CardDef(
        "acc_momentum",
        "Unstoppable Momentum",
        CardType.ACCUMULATE,
        "Events have taken on a life of their own. The party is no longer steering.",
        "Rare. Stack these.",
        0.2,
    ),
    # ── Action (additional) ──────────────────────────────────────────────────
    CardDef(
        "act_misunderstanding",
        "Critical Misunderstanding",
        CardType.ACTION,
        "Everyone means something different by the same word. This becomes clear at the worst moment.",
        "Avoidable in hindsight.",
        0.9,
    ),
    CardDef(
        "act_detour",
        "Unnecessary Detour",
        CardType.ACTION,
        "The direct path is unavailable for reasons nobody can fully explain.",
        "Standard.",
        0.8,
    ),
    CardDef(
        "act_advice",
        "Unsolicited Expertise",
        CardType.ACTION,
        "Someone not involved in this situation offers their analysis at considerable length.",
        "Thorough.",
        0.8,
    ),
    CardDef(
        "act_missing",
        "Missing Component",
        CardType.ACTION,
        "A key element is not where it should be. Nobody knows where it is.",
        "Classic.",
        0.8,
    ),
    CardDef(
        "act_precedent",
        "Appeals to Precedent",
        CardType.ACTION,
        "How this was done before becomes relevant. The record of how it was done before is incomplete.",
        "Procedural.",
        0.7,
    ),
    CardDef(
        "act_authority",
        "Sudden Authority",
        CardType.ACTION,
        "Someone with unclear jurisdiction makes a binding decision without being asked.",
        "Rare. Final.",
        0.2,
    ),
    # ── Chaos (additional) ───────────────────────────────────────────────────
    CardDef(
        "cha_proximity",
        "Proximity Effect",
        CardType.CHAOS,
        "Being near this situation has made it demonstrably worse. The person here is aware of your involvement.",
        "Move away.",
        0.6,
    ),
    CardDef(
        "cha_arrival",
        "Unexpected Arrival",
        CardType.CHAOS,
        "Someone not in any plan walks in and addresses the nearest person as if they are old colleagues. They are.",
        "Unscheduled.",
        0.7,
    ),
    CardDef(
        "cha_containment",
        "Containment Failure",
        CardType.CHAOS,
        "Whatever was being quietly managed is now being loudly managed. Everyone at this location is now involved.",
        "Spreading.",
        0.6,
    ),
    CardDef(
        "cha_cascade",
        "Cascade",
        CardType.CHAOS,
        "One thing going wrong has caused another thing to go wrong. The nearest person is watching both happen in real time.",
        "Inevitable.",
        0.5,
    ),
    CardDef(
        "cha_category",
        "Category Error",
        CardType.CHAOS,
        "This situation has been assigned to entirely the wrong person. The correct person is here now and has questions.",
        "Rare. Foundational.",
        0.2,
    ),
    # ── Tweak (additional) ───────────────────────────────────────────────────
    CardDef(
        "twk_pressure",
        "Time Pressure Update",
        CardType.TWEAK,
        "The urgency of the situation has been reassessed. The new assessment is higher.",
        "Upward only.",
        0.8,
    ),
    CardDef(
        "twk_scope",
        "Scope Creep",
        CardType.TWEAK,
        "The original task has quietly expanded to include several adjacent tasks nobody agreed to.",
        "Standard.",
        0.8,
    ),
    CardDef(
        "twk_info",
        "New Information",
        CardType.TWEAK,
        "A fact that changes how the situation should be read has come to light. It was always true.",
        "Late.",
        0.7,
    ),
    CardDef(
        "twk_perspective",
        "Perspective Shift",
        CardType.TWEAK,
        "Seen from a different angle, this situation looks entirely different. Both readings are valid.",
        "Reorienting.",
        0.7,
    ),
    CardDef(
        "twk_recontextualize",
        "Complete Recontextualization",
        CardType.TWEAK,
        "Everything known about the situation is technically still true, but now means something different.",
        "Rare. Everything changes.",
        0.2,
    ),
    CardDef(
        "act_confront",
        "Direct Inquiry",
        CardType.ACTION,
        "Address the person at this location directly. They must respond. Everyone nearby is listening.",
        "Unavoidable.",
        1.1,
    ),
    CardDef(
        "act_loop_in",
        "Loop In",
        CardType.ACTION,
        "Formally involve the person here in the situation. They now have context they cannot un-have.",
        "Irreversible.",
        0.9,
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


def apply_world_changes(quest: QuestState, changes: list[dict], rng: random.Random) -> None:
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
        elif action == "add_npc":
            npc_id = f"npc_dyn_{len(quest.npcs)}"
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
        elif action == "move_npc":
            npc_id = str(ch.get("npc_id", ""))
            new_wp = int(ch.get("waypoint_idx", 0))
            for npc in quest.npcs:
                if npc.id == npc_id:
                    npc.waypoint_idx = new_wp
                    break
        elif action == "remove_npc":
            npc_id = str(ch.get("npc_id", ""))
            quest.npcs = [n for n in quest.npcs if n.id != npc_id]


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
