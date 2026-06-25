from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .world import WorldMap, facing_from_delta, generate_world, pick_theme

# VibeQuest — a shared multiplayer DnD world where players collectively ARE the Dungeon Master.
# AI agents are the party. Players play cards that resolve the quest. Cards batch in time windows,
# resolve in random order, and the party reacts in character. One global instance, drop in anytime.
#
# Artel integration:
#   memories  → context fed to LLM before resolving each card (what the party remembers)
#   tasks     → quest steps (created at quest start, claimed/completed by AI agents)
#   messages  → in-the-moment agent coordination ("wizard to paladin: this is a terrible idea")

CARD_WINDOW = 30.0  # seconds per card window

# ── Quest templates ────────────────────────────────────────────────────────────────────────────────
# Each template is a recognizable situation treated as an epic DnD quest. The slots are filled
# randomly at quest start, producing infinite variety from finite archetypes.

ITEMS = [
    "the stapler",
    "Gerald's emotional support water bottle",
    "the Wi-Fi password",
    "a single AirPod",
    "the good scissors",
    "a parking validation stamp",
    "last quarter's TPS reports",
    "the aux cable",
    "someone's phone charger",
    "the spare key",
    "a very important sticky note",
    "the office plant (Frank)",
    "the last clean mug",
    "the conference room booking",
]

LOCATIONS = [
    "the enchanted forest (second floor, past the printer)",
    "accounting",
    "the dragon's second lair",
    "the break room microwave dimension",
    "a locked filing cabinet",
    "the parking garage level B",
    "the realm beyond the supply closet",
    "a suspiciously enthusiastic Slack channel",
    "the building's rooftop",
    "a forgotten shared drive folder",
    "the CEO's assistant's desk",
    "the haunted meeting room (the one with the broken AC)",
]

DEADLINES = [
    "the 3pm standup",
    "Dave notices it's missing",
    "the quarterly review",
    "end of business",
    "someone emails about it",
    "the all-hands",
    "the fire drill",
    "anyone else arrives Monday morning",
    "the auditors get here",
    "lunch",
]

COMPLICATIONS = [
    "the elevator is out of service",
    "there is a mandatory training happening in the way",
    "someone is already looking for the same thing",
    "the lights on that floor are motion-activated and very dramatic",
    "there is a very long queue",
    "a passive-aggressive note has been left at the scene",
    "a senior stakeholder is involved somehow",
    "the item may have been moved twice already",
    "it is unclear who is actually responsible for this",
    "building security has opinions",
]

QUEST_TEMPLATES = [
    {
        "id": "retrieve",
        "title": "The Retrieval",
        "hook": "The fellowship must retrieve {item} from {location} before {deadline}.",
        "complication": "{complication}.",
        "slots": {
            "item": ITEMS,
            "location": LOCATIONS,
            "deadline": DEADLINES,
            "complication": COMPLICATIONS,
        },
    },
    {
        "id": "escort",
        "title": "The Escort",
        "hook": "The party must safely escort {npc} from {location} to {destination} before {deadline}.",
        "complication": "{complication}.",
        "slots": {
            "npc": [
                "Gerald from IT",
                "the new intern",
                "a very important plant",
                "the visiting consultant",
                "someone's mother who stopped by",
                "the fire safety officer",
            ],
            "location": LOCATIONS,
            "destination": [
                "the exit",
                "the correct floor",
                "the parking lot",
                "the boardroom",
                "a taxi",
                "anywhere that isn't here",
            ],
            "deadline": DEADLINES,
            "complication": COMPLICATIONS,
        },
    },
    {
        "id": "negotiation",
        "title": "The Negotiation",
        "hook": "The fellowship must convince {npc} to {demand} before {deadline}.",
        "complication": "{complication}.",
        "slots": {
            "npc": [
                "the building manager",
                "a very stubborn vendor",
                "the IT department",
                "facilities",
                "HR",
                "the person who controls the thermostat",
                "whoever owns this calendar invite",
            ],
            "demand": [
                "approve the budget",
                "fix the printer",
                "move the meeting",
                "extend the deadline",
                "let the party through",
                "acknowledge the email",
                "read the document",
            ],
            "deadline": DEADLINES,
            "complication": COMPLICATIONS,
        },
    },
    {
        "id": "investigation",
        "title": "The Investigation",
        "hook": "Something has gone missing from {location}. The fellowship must determine what happened before {deadline}.",
        "complication": "{complication}.",
        "slots": {
            "location": LOCATIONS,
            "deadline": DEADLINES,
            "complication": COMPLICATIONS,
        },
    },
    {
        "id": "delivery",
        "title": "The Delivery",
        "hook": "The party must deliver {item} to {location} before {deadline}. It must arrive intact.",
        "complication": "{complication}.",
        "slots": {
            "item": ITEMS,
            "location": LOCATIONS,
            "deadline": DEADLINES,
            "complication": COMPLICATIONS,
        },
    },
]

# ── Party archetypes ───────────────────────────────────────────────────────────────────────────────

PARTY_ARCHETYPES = [
    {
        "role": "Paladin",
        "name_pool": ["Sir Reginald", "Dame Constance", "Brother Aldous", "Sister Beatrice"],
        "personality": "Deeply earnest. Treats every errand as a sacred oath. Apologizes when things go wrong even if it was not his fault.",
    },
    {
        "role": "Wizard",
        "name_pool": ["Archibald", "Millicent", "Professor Noonan", "Euphemia"],
        "personality": "Slightly condescending. Believes every problem has an optimal solution that the others are ignoring. Keeps a notebook.",
    },
    {
        "role": "Rogue",
        "name_pool": ["Sparrow", "Hex", "Desmond", "Clover"],
        "personality": "Overthinks everything. Has a plan but refuses to share it fully. Leaves cryptic notes.",
    },
    {
        "role": "Bard",
        "name_pool": ["Florian", "Celestine", "Percy", "Margaux"],
        "personality": "Makes everything about themselves. Currently working on a ballad about this exact situation.",
    },
    {
        "role": "Ranger",
        "name_pool": ["Scout", "Thorn", "Wren", "Fletcher"],
        "personality": "Has been on worse quests. Keeps saying so. Very calm, slightly unsettling.",
    },
    {
        "role": "Cleric",
        "name_pool": ["Theodora", "Brother Wick", "Seraphine", "Osmund"],
        "personality": "Optimistic to a fault. Interprets every catastrophe as part of a larger plan. Very comforting.",
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
        "Someone appears who absolutely should not be here.",
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
        "Someone offers to help. Their motives are unclear.",
        "Conditional.",
        0.7,
    ),
    # Chaos — wild magic surge energy
    CardDef(
        "cha_wild",
        "Wild Surge",
        CardType.CHAOS,
        "Something happens. It is not clear what caused it or what it means.",
        "Unpredictable.",
        0.6,
    ),
    CardDef(
        "cha_escalate",
        "Sudden Escalation",
        CardType.CHAOS,
        "The stakes, somehow, have just gotten higher.",
        "Why.",
        0.6,
    ),
    CardDef(
        "cha_mishap",
        "Equipment Mishap",
        CardType.CHAOS,
        "Something the party was relying on no longer works the way they expected.",
        "Classic.",
        0.7,
    ),
    CardDef(
        "cha_witness",
        "Very Bad Timing",
        CardType.CHAOS,
        "The wrong person sees exactly the wrong thing at exactly the wrong moment.",
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
class PlayedCard:
    id: str
    card_id: str
    player_id: str
    played_at: float = field(default_factory=time.time)


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


REGISTERS = [
    "a tense heist thriller",
    "gothic horror",
    "a slow-burn romance",
    "hardboiled noir detective",
    "a deadpan nature documentary",
    "high courtroom drama",
    "Cold War spy espionage",
    "a sweeping disaster epic",
    "a grim fairy tale",
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
    moments: list[str] = field(default_factory=list)
    resolution_count: int = 0
    momentum: int = 0
    tension: int = 0
    outcome: str | None = None


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
    party: list[PartyMember]
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


def _fill_template(template: dict, rng: random.Random) -> tuple[str, str]:
    slots = template["slots"]
    filled: dict[str, str] = {}
    for key, pool in slots.items():
        filled[key] = rng.choice(pool)
    hook = template["hook"].format(**filled)
    complication = template.get("complication", "").format(**filled)
    return hook, complication


def _make_party(rng: random.Random, size: int = 1) -> list[PartyMember]:
    archetypes = rng.sample(PARTY_ARCHETYPES, min(size, len(PARTY_ARCHETYPES)))
    sprites = rng.sample(range(1, 11), min(size, 10))
    members = []
    for i, arch in enumerate(archetypes):
        members.append(
            PartyMember(
                id=str(uuid.uuid4())[:8],
                name=rng.choice(arch["name_pool"]),
                role=arch["role"],
                personality=arch["personality"],
                sprite=sprites[i],
            )
        )
    return members


def _make_quest(rng: random.Random) -> QuestState:
    template = rng.choice(QUEST_TEMPLATES)
    hook, complication = _fill_template(template, rng)
    quest_id = str(uuid.uuid4())[:8]
    return QuestState(
        id=quest_id,
        template_id=template["id"],
        title=template["title"],
        hook=hook,
        complication=complication,
        register=rng.choice(REGISTERS),
    )


def new_game(rng: random.Random | None = None) -> GameState:
    rng = rng or random.Random()
    now = time.time()
    party = _make_party(rng)
    quest = _make_quest(rng)
    window = WindowState(opened_at=now, closes_at=now + CARD_WINDOW)
    run_id = str(uuid.uuid4())[:8]
    theme = pick_theme(quest.hook, quest.register)
    world = generate_world(rng, theme=theme, step_count=MAX_RESOLUTIONS)
    state = GameState(run_id=run_id, party=party, quest=quest, window=window, world=world)
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
    alive = [m for m in state.party if m.status != "lost"]
    if not alive:
        return None
    victim = rng.choice(alive)
    victim.hp = max(0, victim.hp - 5)
    victim.status = "lost" if victim.hp <= 0 else "rattled"
    return victim
