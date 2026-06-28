from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

MIN_RESOLUTIONS = 3
MAX_RESOLUTIONS = 7
MAX_SCENE_ROUNDS = 4
DEAL_INTERVAL = 22.0


class CardType(str, Enum):
    ACTION = "action"
    ACCUMULATE = "accumulate"
    CHAOS = "chaos"
    TWEAK = "tweak"


class DiceResult(str, Enum):
    HIT = "hit"
    MID = "mid"
    MISS = "miss"


class CardResolution(str, Enum):
    HIT = "hit"
    MID = "mid"
    MISS = "miss"


@dataclass
class CardDef:
    id: str
    name: str
    type: CardType
    description: str
    momentum_hit: int = 1
    momentum_mid: int = 0
    momentum_miss: int = -1


@dataclass
class PlayedCard:
    id: str
    card_id: str
    player_id: str
    dice: int
    result: CardResolution
    played_at: float


@dataclass
class CardWindow:
    opened_at: float
    closes_at: float
    cards: list[PlayedCard] = field(default_factory=list)
    resolving: bool = False


@dataclass
class PartyMember:
    id: str
    name: str
    cls: str
    personality: str
    hp: int = 10
    status: str = ""
    sprite: int = 1


@dataclass
class QuestState:
    title: str
    hook: str
    objectives: list[str]
    complication: str = ""
    beats: list[str] = field(default_factory=list)
    resolution_count: int = 0
    outcome: str = ""
    momentum: int = 0
    pressure_pool: list[dict] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    scene_beat_start: int = 0
    register: str = "a formal incident report"


@dataclass
class GameState:
    run_id: str
    phase: str
    tick: int
    party: list[PartyMember]
    quest: QuestState
    window: CardWindow
    log: list[str] = field(default_factory=list)

    def log_event(self, kind: str, text: str) -> None:
        self.log.append(f"[{kind}] {text}")
        if len(self.log) > 200:
            self.log = self.log[-200:]


SURREAL_ARC = [
    "a formal incident report",
    "a memo from the Compliance Department regarding recent irregularities",
    "an HR briefing calmly itemizing events that should not be possible",
    "the minutes of a committee meeting where the impossible is the third agenda item",
    "a straight-faced status update from inside a situation that has stopped obeying the applicable regulations",
]


def arc_register(resolution_count: int) -> str:
    idx = min(resolution_count, len(SURREAL_ARC) - 1)
    return SURREAL_ARC[idx]


def intensity(resolution_count: int) -> float:
    return min(1.0, resolution_count / max(MAX_RESOLUTIONS - 1, 1))


_CARDS: list[CardDef] = [
    CardDef(
        "action_encounter",
        "Roll for Encounter",
        CardType.ACTION,
        "A new obstacle, NPC, or complication appears in the path of progress.",
    ),
    CardDef(
        "action_reveal",
        "Reveal Information",
        CardType.ACTION,
        "Something important comes to light. The party learns a relevant fact.",
    ),
    CardDef(
        "action_deadline",
        "Time Pressure",
        CardType.ACTION,
        "A deadline is introduced or an existing one accelerates.",
    ),
    CardDef(
        "action_plot",
        "Plot Twist",
        CardType.ACTION,
        "An unexpected development recontextualises the current situation.",
    ),
    CardDef(
        "action_hazard",
        "Environmental Hazard",
        CardType.ACTION,
        "The location itself becomes a problem. Something structural or procedural intervenes.",
        momentum_miss=-2,
    ),
    CardDef(
        "action_authority",
        "Authority Intervenes",
        CardType.ACTION,
        "Someone with institutional power arrives and has opinions about the situation.",
    ),
    CardDef(
        "action_ally",
        "Unexpected Assistance",
        CardType.ACTION,
        "Help arrives from an unexpected quarter, though it may not be the right kind of help.",
        momentum_hit=2,
    ),
    CardDef(
        "acc_stakes",
        "Rising Stakes",
        CardType.ACCUMULATE,
        "Tension accumulates. The next resolution will matter more.",
        momentum_hit=0,
        momentum_mid=0,
        momentum_miss=-1,
    ),
    CardDef(
        "acc_crowd",
        "Crowd Gathers",
        CardType.ACCUMULATE,
        "Witnesses are accumulating. This is now a matter of public record.",
    ),
    CardDef(
        "acc_evidence",
        "Paper Trail",
        CardType.ACCUMULATE,
        "Documentation is being reviewed at a level above this floor.",
    ),
    CardDef(
        "acc_clock",
        "The Clock",
        CardType.ACCUMULATE,
        "A soft deadline is becoming a hard one. Someone has sent a calendar invite.",
        momentum_miss=-2,
    ),
    CardDef(
        "chaos_wild",
        "Wild Dice",
        CardType.CHAOS,
        "Anything could happen. The situation becomes genuinely unpredictable.",
        momentum_hit=3,
        momentum_miss=-3,
    ),
    CardDef(
        "chaos_guest",
        "Guest Appearance",
        CardType.CHAOS,
        "An entirely unexpected party enters. They have their own agenda.",
        momentum_hit=2,
        momentum_miss=-2,
    ),
    CardDef(
        "chaos_exception",
        "Rule Exception",
        CardType.CHAOS,
        "A rule that was supposed to apply apparently does not. This creates options.",
        momentum_mid=1,
    ),
    CardDef(
        "chaos_undo",
        "The Undo",
        CardType.CHAOS,
        "Something that was resolved has un-resolved itself. A prior outcome is reopened.",
        momentum_hit=1,
        momentum_miss=-2,
    ),
    CardDef(
        "tweak_amend",
        "Slight Amendment",
        CardType.TWEAK,
        "The situation is clarified. The clarification is, on balance, slightly worse.",
    ),
    CardDef(
        "tweak_clarify",
        "Clarification Required",
        CardType.TWEAK,
        "Additional information is needed before further progress can be made.",
        momentum_hit=0,
        momentum_mid=0,
        momentum_miss=0,
    ),
    CardDef(
        "tweak_redirect",
        "Redirect",
        CardType.TWEAK,
        "The correct solution is in a different direction than the current one.",
        momentum_hit=1,
        momentum_miss=-1,
    ),
    CardDef(
        "tweak_workaround",
        "Workaround",
        CardType.TWEAK,
        "Progress is technically possible via an undocumented route that nobody should use.",
        momentum_hit=2,
        momentum_mid=1,
        momentum_miss=0,
    ),
]

CARD_BY_ID = {c.id: c for c in _CARDS}

_QUEST_ARCHETYPES: list[dict[str, Any]] = [
    {
        "title": "The Stapler of Destiny",
        "hook": "A stapler belonging to Sandra from Accounts has been missing since April 2022. It has been located. It is inside the printer on level three. The printer is not broken. The stapler is simply inside it.",
        "objectives": [
            "Establish chain of custody for the stapler",
            "Extract the stapler from the printer without incident",
            "Return the stapler to Sandra from Accounts and obtain a signature",
        ],
    },
    {
        "title": "The Conference Room of Peril",
        "hook": "Conference Room B has been booked for the 10am all-hands. It has also been booked for a vendor presentation, a birthday lunch, and something called 'Recurring: TBD' since 2019. All four bookings are active.",
        "objectives": [
            "Determine which booking is legitimate",
            "Locate the parties responsible for the other three bookings",
            "Secure Conference Room B before 10am",
        ],
    },
    {
        "title": "The IT Ticket of the Ages",
        "hook": "IT ticket #4471 has been open since March 2022. It concerns a printer on the third floor. The ticket is marked 'assigned'. The assigned technician left the company in 2023. The printer is still printing.",
        "objectives": [
            "Locate a technician who will acknowledge the ticket",
            "Determine what the ticket is actually for",
            "Close ticket #4471 through legitimate means",
        ],
    },
    {
        "title": "The Offboarding",
        "hook": "Gerald from Accounting is leaving on Friday. His last day was three weeks ago. He still has a laptop, a keycard, and access to the payroll system. HR has sent him seven emails. Gerald has replied to all of them.",
        "objectives": [
            "Make contact with Gerald",
            "Retrieve the laptop and keycard",
            "Revoke Gerald's access to payroll without triggering any alerts",
        ],
    },
    {
        "title": "The Budget Approval",
        "hook": "An expense report for $147.50 requires three signatures. Two have been obtained. The third signature belongs to the VP of Operations, who is 'travelling' and has been 'travelling' since Q2.",
        "objectives": [
            "Determine the VP of Operations' actual location",
            "Obtain or approximate the third signature",
            "Submit the expense report before the fiscal year closes",
        ],
    },
    {
        "title": "The Free Lunch",
        "hook": "There is an unclaimed lunch in the break room fridge. It has been there since Tuesday. There is a note on it that says 'FOR GERALD'. Gerald no longer works here. The lunch is from a restaurant that closed in 2021.",
        "objectives": [
            "Determine the provenance of the lunch",
            "Establish whether the lunch is safe",
            "Resolve the lunch situation before Facilities sends another all-staff email about fridge hygiene",
        ],
    },
    {
        "title": "The Password Reset",
        "hook": "The password for the legacy reporting system must be reset. IT acknowledges the system exists. IT does not have admin access to it. The password was last reset in 2017 by someone whose account no longer exists but whose password reset emails are still arriving.",
        "objectives": [
            "Locate documentation for the legacy reporting system",
            "Identify someone with admin access",
            "Complete the password reset and update the password in the correct shared document",
        ],
    },
    {
        "title": "The New Hire",
        "hook": "A new associate has been attending meetings since Monday. She is very pleasant and appears to know where things are. She is not in any HR system, payroll system, or directory. She has a badge. Nobody knows who issued it.",
        "objectives": [
            "Establish the new associate's identity",
            "Determine how she received a building badge",
            "Complete her onboarding paperwork, retroactively",
        ],
    },
    {
        "title": "The Exit Interview",
        "hook": "HR must conduct an exit interview with Marcus from Product Strategy. Marcus left three months ago. He has continued to attend his recurring one-on-ones. His manager has not mentioned this.",
        "objectives": [
            "Schedule the exit interview with Marcus",
            "Conduct the interview according to the standard 14-point template",
            "Update Marcus's employment status to reflect that he no longer works here",
        ],
    },
    {
        "title": "The Awaited Reply",
        "hook": "An email was sent fourteen business days ago requesting approval for a minor infrastructure change. The recipient is listed as active in the directory. The email has been read. Three follow-up emails have also been read. Nobody has replied.",
        "objectives": [
            "Locate the email recipient",
            "Determine why they have not replied",
            "Obtain either approval or a reason to proceed without it",
        ],
    },
    {
        "title": "The Shipment",
        "hook": "A package addressed to this floor has been delivered to the correct floor of the wrong building. The wrong building is across the street. The package is marked FRAGILE. The tracking shows it was signed for by 'O. Hargrove'. There is no O. Hargrove in any directory.",
        "objectives": [
            "Locate the package",
            "Identify O. Hargrove or establish their non-existence",
            "Return the package to this floor without incident",
        ],
    },
    {
        "title": "The Holiday Party Venue",
        "hook": "The holiday party venue must be booked by end of week. All dietary restrictions must be accommodated. The list of dietary restrictions includes three contradictions, one restriction that is a philosophical position, and one that appears to be a brand name.",
        "objectives": [
            "Reconcile the dietary restriction list",
            "Identify a venue that can accommodate the reconciled list",
            "Book the venue before someone else in the building books it first",
        ],
    },
    {
        "title": "The Broken Chair",
        "hook": "The chair in the northeast corner of the fourth floor has had a wobbly wheel since 2020. Six facilities tickets have been submitted. All six are marked 'completed'. The wheel is still wobbly. The chair now has a Post-it note on it that says 'DO NOT SIT' which has been there long enough to yellow.",
        "objectives": [
            "Submit a seventh facilities ticket with sufficient documentation",
            "Escalate the ticket to a tier that will acknowledge the chair",
            "Resolve the chair situation and remove the Post-it note",
        ],
    },
    {
        "title": "The Parking Validation",
        "hook": "Parking validation stamps are available from the front desk on the seventh floor. The seventh floor was decommissioned in 2021 and is now a storage area. The validation stamps are still there. The elevator no longer stops at seven.",
        "objectives": [
            "Access the seventh floor",
            "Locate the validation stamps",
            "Return to the lobby and validate the parking without anyone asking too many questions",
        ],
    },
    {
        "title": "The Fire Drill",
        "hook": "A mandatory fire drill is scheduled for 2pm. The fire drill coordinator retired in 2019. The role was not backfilled. The drill is in the system. Nobody knows who scheduled it or how to cancel it.",
        "objectives": [
            "Identify the correct fire drill protocol",
            "Coordinate the evacuation of all staff, including those who are on calls",
            "Complete the headcount and return staff to the building within the allotted window",
        ],
    },
    {
        "title": "The Agenda",
        "hook": "A meeting agenda must be distributed before the meeting begins. The meeting began twelve minutes ago. There is no agreed agenda. Three people have already sent their own agendas. The agendas do not agree on what the meeting is about.",
        "objectives": [
            "Reconcile the three competing agendas",
            "Distribute the agreed agenda to all attendees",
            "Ensure the meeting reaches at least one actionable conclusion",
        ],
    },
    {
        "title": "The NDA",
        "hook": "A vendor has been cc'd on internal email threads since Q1. They have never signed an NDA. Legal has been informed. Legal sent a template NDA. The vendor replied asking which version of the template this was. Nobody has replied.",
        "objectives": [
            "Determine which version of the NDA template this is",
            "Respond to the vendor's reply",
            "Obtain a signed NDA before the vendor is cc'd on anything else",
        ],
    },
    {
        "title": "The Emergency Contact",
        "hook": "HR requires all employees to update their emergency contacts. The system was last updated in 2008. Several emergency contacts are now also employees. Two of the emergency contacts are listed as each other's emergency contacts.",
        "objectives": [
            "Audit the emergency contact system for circular references",
            "Reach out to staff with unresolvable emergency contact situations",
            "Update the system before the HR deadline, which was yesterday",
        ],
    },
    {
        "title": "The Recurring Meeting",
        "hook": "A recurring meeting with no agenda and no clear owner has been on everyone's calendar since 2021. It has a 97% attendance rate. When asked, nobody knows who started it or what it is for. Cancelling it requires the organiser's account, which belongs to someone who left in 2021.",
        "objectives": [
            "Identify the original organiser or locate their credentials",
            "Determine whether the meeting serves any current purpose",
            "Cancel or repurpose the meeting and send a clear explanation to attendees",
        ],
    },
    {
        "title": "The Reference Letter",
        "hook": "A reference letter must be written for a former colleague. The colleague left under circumstances that are described, in HR's file, only as 'transition'. The reference letter must be positive. The manager who should write it does not remember the colleague.",
        "objectives": [
            "Locate documentation about the colleague's work",
            "Draft a reference letter that is accurate and positive",
            "Have the letter signed by the manager before the colleague's deadline",
        ],
    },
]

_CLASSES = ["Bard", "Fighter", "Rogue", "Wizard", "Ranger", "Paladin", "Cleric", "Druid"]

_FIRST_NAMES = [
    "Theodore",
    "Constance",
    "Reginald",
    "Beatrice",
    "Mortimer",
    "Millicent",
    "Archibald",
    "Prudence",
    "Cornelius",
    "Lavinia",
    "Algernon",
    "Ottoline",
    "Percival",
    "Josephine",
    "Ferdinand",
    "Rosalind",
    "Bartholomew",
    "Mildred",
]

_LAST_NAMES = [
    "Plimpton",
    "Fothergill",
    "Ashworth",
    "Pendleton",
    "Grimsby",
    "Wentworth",
    "Cholmondeley",
    "Featherstone",
    "Winterbottom",
    "Ramsbottom",
    "Scattergood",
    "Cruickshank",
    "Fairweather",
    "Dunbar-Hartley",
    "Sinclair-Booth",
]

_TITLES = [
    "IT Liaison",
    "Facilities Coordinator",
    "Process Analyst",
    "Compliance Officer",
    "Project Associate",
    "Records Manager",
    "Systems Auditor",
    "Budget Coordinator",
    "Transition Specialist",
    "Continuity Planner",
    "Documentation Lead",
    "Change Management Advisor",
    "Resource Allocation Analyst",
]

_PERSONALITY_TRAITS = [
    "has strong opinions about the correct format for email subject lines",
    "always cc's their manager on everything, including replies to themselves",
    "has never taken a sick day and mentions this frequently",
    "submits all expenses exactly fourteen days after the deadline",
    "has a laminated copy of the org chart from 2019",
    "keeps a printed copy of every email they have ever sent",
    "refers to the employee handbook by page number",
    "has attended every fire drill since 2008 and knows all the muster point coordinates",
    "uses the phrase 'as per my last email' without apparent irony",
    "is still waiting for a response to an email they sent in Q3",
    "has a second monitor exclusively for tracking tickets",
    "knows every printer on every floor by name",
    "is working on a spreadsheet to replace the current spreadsheet",
    "has opinions about the break room fridge that are documented and on file",
    "has read the visitor policy in full, more than once",
    "uses the word 'bandwidth' as a unit of measurement",
    "has submitted three formal suggestions to the suggestion box, none acknowledged",
    "always logs off at exactly 5pm and has done so since 2012",
]


def _pick(pool: list, rng: random.Random) -> Any:
    return rng.choice(pool)


def _gen_party(rng: random.Random) -> list[PartyMember]:
    classes = rng.sample(_CLASSES, 4)
    used_names: set[str] = set()
    members = []
    for i, cls in enumerate(classes):
        while True:
            name = f"{_pick(_FIRST_NAMES, rng)} {_pick(_LAST_NAMES, rng)}"
            if name not in used_names:
                used_names.add(name)
                break
        title = _pick(_TITLES, rng)
        trait = _pick(_PERSONALITY_TRAITS, rng)
        personality = f"{cls} / {title}. {name} {trait}."
        members.append(
            PartyMember(
                id=str(uuid.uuid4())[:8],
                name=name,
                cls=cls,
                personality=personality,
                hp=10,
                sprite=i + 1,
            )
        )
    return members


def _pick_quest(rng: random.Random) -> QuestState:
    arch = rng.choice(_QUEST_ARCHETYPES)
    return QuestState(
        title=arch["title"],
        hook=arch["hook"],
        objectives=list(arch["objectives"]),
        register=SURREAL_ARC[0],
    )


def new_game(rng: random.Random, now: float) -> GameState:
    return GameState(
        run_id=str(uuid.uuid4())[:8],
        phase="idle",
        tick=0,
        party=_gen_party(rng),
        quest=_pick_quest(rng),
        window=CardWindow(opened_at=now, closes_at=now + DEAL_INTERVAL),
    )


def roll_d20(rng: random.Random) -> int:
    return rng.randint(1, 20)


def classify_result(dice: int) -> tuple[CardResolution, DiceResult]:
    if dice == 20:
        return CardResolution.HIT, DiceResult.HIT
    if dice == 1:
        return CardResolution.MISS, DiceResult.MISS
    if dice >= 12:
        return CardResolution.HIT, DiceResult.HIT
    if dice >= 6:
        return CardResolution.MID, DiceResult.MID
    return CardResolution.MISS, DiceResult.MISS


def deal_hand(rng: random.Random, n: int = 5) -> list[str]:
    weights = {
        CardType.ACTION: 4,
        CardType.ACCUMULATE: 2,
        CardType.CHAOS: 1,
        CardType.TWEAK: 2,
    }
    pool = []
    for card in _CARDS:
        pool.extend([card.id] * weights[card.type])
    return rng.choices(pool, k=n)


def apply_card_effects(card: CardDef, result: CardResolution, quest: QuestState) -> int:
    delta_map = {
        CardResolution.HIT: card.momentum_hit,
        CardResolution.MID: card.momentum_mid,
        CardResolution.MISS: card.momentum_miss,
    }
    delta = delta_map[result]
    quest.momentum = max(-10, min(10, quest.momentum + delta))
    return delta


def advance_window(state: GameState, rng: random.Random) -> list[PlayedCard]:
    played = list(state.window.cards)
    rng.shuffle(played)
    return played


def sync_register(state: GameState) -> None:
    state.quest.register = arc_register(state.quest.resolution_count)
