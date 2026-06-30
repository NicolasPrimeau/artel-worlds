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

MUNDANE_NPCS = [
    {
        "name": "Dennis Marsh",
        "role": "IT Support",
        "personality": "Has a system for everything. The system is not documented. He will explain it if asked, but only partially.",
        "behavior": "stationary",
    },
    {
        "name": "Sandra Okafor",
        "role": "Office Manager",
        "personality": "Responsible for everything and authorized for nothing. Has a spreadsheet that tracks the spreadsheet.",
        "behavior": "wandering",
    },
    {
        "name": "Keith Burrows",
        "role": "Facilities",
        "personality": "Knows every fuse box, every unlabeled door. Shares this information reluctantly and non-chronologically.",
        "behavior": "wandering",
    },
    {
        "name": "Margot Chen",
        "role": "Finance",
        "personality": "Does not process things verbally. Sends a follow-up email three minutes after every conversation.",
        "behavior": "stationary",
    },
    {
        "name": "Phil Noonan",
        "role": "Legal",
        "personality": "Responds to everything with a question about scope. His questions are long. His answers are longer.",
        "behavior": "stationary",
    },
    {
        "name": "Yvonne Tait",
        "role": "HR",
        "personality": "Extremely warm. Cannot share any information. These two facts create a specific kind of interaction she has had many times.",
        "behavior": "stationary",
    },
    {
        "name": "Trevor Bale",
        "role": "Security",
        "personality": "Takes the badge scanner personally. Has opinions about tailgating that he will share unprompted.",
        "behavior": "wandering",
    },
    {
        "name": "Carol Pinkett",
        "role": "Reception",
        "personality": "Has worked here longer than the company has had its current name. Remembers the previous name.",
        "behavior": "stationary",
    },
    {
        "name": "Darnell Wade",
        "role": "Procurement",
        "personality": "Everything requires a PO number. He does not make this rule. He does enforce it.",
        "behavior": "stationary",
    },
    {
        "name": "Helen Frost",
        "role": "Executive Assistant",
        "personality": "Controls the calendar. The calendar is a form of power she exercises carefully and without emotion.",
        "behavior": "stationary",
    },
    {
        "name": "Gary Plum",
        "role": "Maintenance",
        "personality": "The job takes as long as it takes. This is not a value judgment. It is a statement of fact.",
        "behavior": "wandering",
    },
    {
        "name": "Irene Solis",
        "role": "Payroll",
        "personality": "Has a sign on her desk that says PLEASE READ THE FAQ. She wrote the FAQ. She updates it biannually.",
        "behavior": "stationary",
    },
    {
        "name": "Marcus Firth",
        "role": "Project Manager",
        "personality": "Speaks only in status updates. Currently at 60%. ETA unclear.",
        "behavior": "wandering",
    },
    {
        "name": "Bev Larocque",
        "role": "Admin",
        "personality": "Has been 'just about to leave' for 45 minutes. This is not unusual for her.",
        "behavior": "stationary",
    },
    {
        "name": "Noel Pritchard",
        "role": "Compliance",
        "personality": "Reads everything. CC'd on things he doesn't need to be CC'd on. Says nothing until he does.",
        "behavior": "stationary",
    },
    {
        "name": "Jan Metzger",
        "role": "Accounts",
        "personality": "Tracks everything she has ever done for anyone. Not resentfully. Just accurately.",
        "behavior": "stationary",
    },
    {
        "name": "Steve Doyle",
        "role": "Sales",
        "personality": "Very friendly. Every conversation ends with something that needs to happen as a result of it.",
        "behavior": "wandering",
    },
    {
        "name": "Wendy Carr",
        "role": "Operations",
        "personality": "Has asked this question before. Is asking again because the answer changed last time. She is keeping track.",
        "behavior": "wandering",
    },
    {
        "name": "Ray Hollis",
        "role": "IT Infrastructure",
        "personality": "Replies to everything with a ticket number. He is the one who opened the ticket. He is waiting on himself.",
        "behavior": "stationary",
    },
    {
        "name": "Pam Grundy",
        "role": "Office Admin",
        "personality": "Was told this was temporary three years ago. Has not raised it since. Is still thinking about it.",
        "behavior": "stationary",
    },
    {
        "name": "Clive Osei",
        "role": "Building Manager",
        "personality": "Has keys to rooms nobody knew existed. Won't say what's in them. Not suspicious — just private.",
        "behavior": "wandering",
    },
    {
        "name": "Diane Fuentes",
        "role": "Communications",
        "personality": "Rewrites every email before sending it. The original was fine. The revision is fine. The process is non-negotiable.",
        "behavior": "stationary",
    },
    {
        "name": "Arthur Goss",
        "role": "Print Room",
        "personality": "Remembers every job he has ever processed. Date, time, requestor. Does not keep notes. Just remembers.",
        "behavior": "stationary",
    },
    {
        "name": "Nina Haworth",
        "role": "Risk & Audit",
        "personality": "Asks questions that make people feel like they've done something wrong even when they haven't.",
        "behavior": "stationary",
    },
    {
        "name": "Pete Mullen",
        "role": "Delivery",
        "personality": "On his third lap of the floor. Something on the trolley has no label. He is not concerned.",
        "behavior": "wandering",
    },
    {
        "name": "Fiona Yates",
        "role": "Events Coordinator",
        "personality": "Has a colour-coded binder for every scenario except this one.",
        "behavior": "stationary",
    },
    {
        "name": "Bob Tench",
        "role": "Senior Developer",
        "personality": "The answer is always 'it depends.' He will tell you what it depends on. That also depends.",
        "behavior": "stationary",
    },
    {
        "name": "Lena Kowalski",
        "role": "Data Analyst",
        "personality": "Says 'statistically speaking' before statements that are not statistics.",
        "behavior": "stationary",
    },
    {
        "name": "Omar Petrov",
        "role": "Customer Success",
        "personality": "Uses the word 'ecosystem' unironically. Has been asked to stop. Is thinking about it.",
        "behavior": "wandering",
    },
    {
        "name": "Ruth Digby",
        "role": "Contracts",
        "personality": "Can quote the relevant clause from memory. The clause does not help. She quotes it anyway.",
        "behavior": "stationary",
    },
    {
        "name": "Sam Hartley",
        "role": "Intern",
        "personality": "Doing exactly what they were told. Told the wrong thing. Still doing it.",
        "behavior": "wandering",
    },
    {
        "name": "Gordon Leach",
        "role": "Finance Director",
        "personality": "Seems pleasant. Has never approved anything. Has never rejected anything. Things just stop near him.",
        "behavior": "stationary",
    },
    {
        "name": "Tina Brock",
        "role": "Customer Support",
        "personality": "Has been on hold with an external vendor for 25 minutes and is treating this as normal.",
        "behavior": "stationary",
    },
    {
        "name": "Mo Sadiq",
        "role": "Network Engineer",
        "personality": "Fixes things before anyone notices they're broken. Gets no credit. Has noted this. Moving on.",
        "behavior": "wandering",
    },
    {
        "name": "June Whittle",
        "role": "Records Management",
        "personality": "Nothing is deleted. Everything is archived. The archive is full. This is not her problem.",
        "behavior": "stationary",
    },
    {
        "name": "Ed Fairweather",
        "role": "Operations Analyst",
        "personality": "Has run the numbers. The numbers say no. He has run them again to be sure.",
        "behavior": "stationary",
    },
    {
        "name": "Ros Keane",
        "role": "PA to the Director",
        "personality": "Knows everything. Will tell you exactly as much as is useful and not one word more.",
        "behavior": "stationary",
    },
    {
        "name": "Tim Calloway",
        "role": "Health & Safety",
        "personality": "Not here to cause trouble. Just needs someone to sign the form. Nobody will sign the form.",
        "behavior": "wandering",
    },
    {
        "name": "Viv Sutton",
        "role": "Finance Business Partner",
        "personality": "Uses 'optics' as a noun in every third sentence. Unaware she does this.",
        "behavior": "stationary",
    },
    {
        "name": "Hal Oduya",
        "role": "Systems Admin",
        "personality": "Has remote access to everything. Is currently at his desk. Nobody knows where his desk is.",
        "behavior": "stationary",
    },
    {
        "name": "Cath Greer",
        "role": "Training & Development",
        "personality": "There is a module for this. She will find it. It will not cover this exact situation.",
        "behavior": "stationary",
    },
    {
        "name": "Paul Ince",
        "role": "Logistics",
        "personality": "The shipment is somewhere. He knows roughly where. The tracking says something different. He trusts himself.",
        "behavior": "wandering",
    },
    {
        "name": "Mel Browning",
        "role": "Change Management",
        "personality": "Process exists to protect everyone. She will explain this. Nobody is protected by what she is explaining.",
        "behavior": "stationary",
    },
    {
        "name": "Frank Lau",
        "role": "Executive Sponsor",
        "personality": "Sent an email about this two weeks ago. Considers the matter resolved. It is not resolved.",
        "behavior": "stationary",
    },
    {
        "name": "Donna Peel",
        "role": "Office Cleaner",
        "personality": "Knows things. Has heard things. Is professionally deaf until she is not.",
        "behavior": "wandering",
    },
    {
        "name": "Kit Barker",
        "role": "UX Designer",
        "personality": "The problem is a people problem. The solution is a people solution. This is not useful right now.",
        "behavior": "stationary",
    },
    {
        "name": "Reg Ashton",
        "role": "Reprographics",
        "personality": "Retired three years ago. Still comes in on Tuesdays. Nobody has asked him to stop.",
        "behavior": "stationary",
    },
    {
        "name": "Yemi Adewale",
        "role": "Business Development",
        "personality": "Everything is a potential partnership. He is taking notes. The notes will become a deck.",
        "behavior": "wandering",
    },
    {
        "name": "Angela Stride",
        "role": "Office Coordinator",
        "personality": "Has rearranged the stationery cupboard four times this year. Each time is the definitive arrangement.",
        "behavior": "stationary",
    },
    {
        "name": "Geoff Morley",
        "role": "Facilities Engineer",
        "personality": "Needs five minutes. Has needed five minutes for forty minutes. Very optimistic about the next five.",
        "behavior": "wandering",
    },
    {
        "name": "Lisa Thorn",
        "role": "Brand Manager",
        "personality": "Notices when the wrong shade of blue is used. Is noticing right now. About something unrelated to the current crisis.",
        "behavior": "stationary",
    },
    {
        "name": "Raj Kapoor",
        "role": "IT Helpdesk",
        "personality": "Has tried turning it off and on again. Has not mentioned this. Wants credit for not mentioning it.",
        "behavior": "stationary",
    },
    {
        "name": "Debbie Noon",
        "role": "Finance Analyst",
        "personality": "Numbers are correct. Conclusions drawn from them are not her responsibility.",
        "behavior": "stationary",
    },
    {
        "name": "Len Hackett",
        "role": "Parking & Transport",
        "personality": "The visitor bay is reserved. For someone else. There is a list. She is not on the list.",
        "behavior": "stationary",
    },
]

SURREAL_NPCS = [
    {
        "name": "Neil Burr",
        "role": "Temporary Staff",
        "personality": "Has been temporary for six years. The agency that placed him closed in 2021. He is still being paid. Nobody knows by whom.",
        "behavior": "wandering",
    },
    {
        "name": "Carol Voss",
        "role": "Second Floor",
        "personality": "Her job title is 'Second Floor.' This is what her badge says. This is what her email says. Nobody has ever asked about it.",
        "behavior": "stationary",
    },
    {
        "name": "Martin Ogle",
        "role": "Former Employee",
        "personality": "Left the company eighteen months ago. Still comes in. Uses his old desk. His access card still works. IT is aware.",
        "behavior": "wandering",
    },
    {
        "name": "Susan Drax",
        "role": "Head of Process",
        "personality": "There is no Process department. There is Susan. She has a team of one. The one is also Susan, listed under a slightly different name.",
        "behavior": "stationary",
    },
    {
        "name": "Barry Finch",
        "role": "IT Support",
        "personality": "Replaced Dennis eighteen months ago. Dennis is also still here. They have never been seen in the same room. This is not discussed.",
        "behavior": "stationary",
    },
    {
        "name": "Elaine Hobbs",
        "role": "Meeting Room 4B",
        "personality": "Is always in Meeting Room 4B. The room is always booked by someone else. The booking system shows it as empty. Nobody raises this.",
        "behavior": "stationary",
    },
    {
        "name": "Derek Pound",
        "role": "Project Lead",
        "personality": "Leads a project that was completed two years ago. He has continued leading it. The project continues to have updates.",
        "behavior": "wandering",
    },
    {
        "name": "Maureen Crisp",
        "role": "Retired (Emeritus)",
        "personality": "Retired in 2019. Still receives internal emails and replies to them the same day. Nobody knows who set this up. The replies are useful.",
        "behavior": "stationary",
    },
    {
        "name": "Hugh Tandy",
        "role": "Data Governance",
        "personality": "Oversees a dataset that, according to the system, does not exist. He disputes this. The data is there. He has shown it to people. They've seen it.",
        "behavior": "stationary",
    },
    {
        "name": "Val Shore",
        "role": "Notetaker",
        "personality": "Takes notes in every meeting she attends. Has not spoken in a meeting since 2022. Her notes are always correct. Nobody knows whose meetings she attends.",
        "behavior": "wandering",
    },
    {
        "name": "Theo Wick",
        "role": "Client Liaison",
        "personality": "Represents a client account that was closed three years ago. He still receives their calls. He still schedules their reviews. They seem happy.",
        "behavior": "stationary",
    },
    {
        "name": "Brenda Ash",
        "role": "Building Occupant",
        "personality": "Is in the building every day. Does not work here. Has a desk. Has a mug. Has a coat on a hook. Nobody has ever asked.",
        "behavior": "stationary",
    },
]

NPC_POOL = MUNDANE_NPCS + SURREAL_NPCS

QUEST_CATEGORIES: dict[str, dict] = {
    "office": {
        "theme": "office",
        "tasks": [
            {
                "title": "Q3 Expense Report",
                "hook": "Someone needs to submit the Q3 expense report before the finance system locks at 5pm.",
                "objectives": [
                    "Find who has the receipts",
                    "Get Finance to unlock the submission portal",
                    "Submit before the system closes",
                ],
            },
            {
                "title": "Fix The Printer",
                "hook": "The printer on the third floor has stopped working. It needs to be running before the 2pm presentation.",
                "objectives": [
                    "Identify what's actually wrong with it",
                    "Track down someone who can fix it",
                    "Confirm it's working before the presentation",
                ],
            },
            {
                "title": "The Missing Key Fob",
                "hook": "Someone's access key fob stopped working and they can't get into the building. Needs to be resolved before their shift ends.",
                "objectives": [
                    "Find out why the fob deactivated",
                    "Get to whoever can reactivate it",
                    "Issue a working fob before end of shift",
                ],
            },
            {
                "title": "Coffee Pod Situation",
                "hook": "Someone used the last coffee pod and didn't reorder. This needs to be addressed before the 9am stand-up.",
                "objectives": [
                    "Establish who is responsible for reordering",
                    "Find an emergency supply",
                    "Have coffee available by 9am",
                ],
            },
            {
                "title": "Room Double-Booking",
                "hook": "The main conference room is double-booked for Thursday. Someone needs to sort this out before people start arriving.",
                "objectives": [
                    "Identify both parties and what they need",
                    "Find an alternative for one of them",
                    "Confirm the resolution in writing",
                ],
            },
            {
                "title": "The Unsigned NDA",
                "hook": "A contractor started today without signing the NDA. Legal needs it signed before end of day.",
                "objectives": [
                    "Find the contractor",
                    "Get Legal to send the correct version",
                    "Get it signed and returned",
                ],
            },
            {
                "title": "Laptop Recovery",
                "hook": "An ex-employee still has a company laptop. IT needs it back before the asset audit tomorrow.",
                "objectives": [
                    "Locate the ex-employee",
                    "Arrange a handoff",
                    "Log the return before the audit",
                ],
            },
            {
                "title": "The Fish Incident",
                "hook": "Someone microwaved fish in the break room and the smell is spreading to the open plan. This has to stop.",
                "objectives": [
                    "Identify who did it",
                    "Get the break room ventilated",
                    "Establish that this cannot happen again",
                ],
            },
            {
                "title": "All-Hands Deck",
                "hook": "The all-hands presentation is in two hours and slides are still missing from three departments.",
                "objectives": [
                    "Chase the three departments",
                    "Assemble the slides into one deck",
                    "Get it to the presenter in time",
                ],
            },
            {
                "title": "New Hire Setup",
                "hook": "A new hire started today with no desk, no computer, and no system access. Someone needs to fix this.",
                "objectives": [
                    "Find them a desk",
                    "Get IT to provision a machine",
                    "Get their accounts activated today",
                ],
            },
            {
                "title": "The IT Ticket",
                "hook": "An IT support ticket has been open for three weeks with no update. Someone needs to get it resolved before the end of the quarter.",
                "objectives": [
                    "Find out who owns the ticket",
                    "Get an actual status",
                    "Close it before quarter end",
                ],
            },
            {
                "title": "Catering Mix-Up",
                "hook": "The catering order for tomorrow's client lunch was placed at the wrong branch. Someone needs to sort this out today.",
                "objectives": [
                    "Reach the right branch",
                    "Redirect or reorder",
                    "Confirm delivery for tomorrow",
                ],
            },
            {
                "title": "Form 2309-B",
                "hook": "A form needs three signatures before it can be processed. Two of the signatories are in different buildings and one is not responding.",
                "objectives": [
                    "Get the first two signatures",
                    "Track down the third signatory",
                    "Submit the form before processing closes",
                ],
            },
            {
                "title": "The Projector",
                "hook": "The projector in the small meeting room won't connect to any laptop. A client presentation starts in 20 minutes.",
                "objectives": [
                    "Find the right cable or adapter",
                    "Get the projector working",
                    "Have the room ready before the client arrives",
                ],
            },
            {
                "title": "Parking Situation",
                "hook": "A VIP visitor is arriving in an hour and nobody arranged parking. The visitor is already on their way.",
                "objectives": [
                    "Find an available space",
                    "Get building security to reserve it",
                    "Notify the visitor before they arrive",
                ],
            },
            {
                "title": "The Good Stapler",
                "hook": "Someone's good stapler has gone missing from their desk. They have asked for help recovering it. The regular staplers are not acceptable.",
                "objectives": [
                    "Establish when it disappeared",
                    "Identify who might have it",
                    "Return the correct stapler",
                ],
            },
            {
                "title": "The Vending Machine",
                "hook": "The vending machine took someone's money and dispensed nothing. Three other people have also lost money. Someone needs to get to the bottom of this.",
                "objectives": [
                    "Document the losses",
                    "Contact the vendor",
                    "Get refunds or a working machine",
                ],
            },
            {
                "title": "Birthday Logistics",
                "hook": "It is someone's birthday. A cake was ordered to the wrong address. The birthday person arrives in 45 minutes.",
                "objectives": [
                    "Locate the cake",
                    "Arrange pickup or redirect",
                    "Have it here before they arrive",
                ],
            },
            {
                "title": "The Thermostat Dispute",
                "hook": "Two teams on the same floor are in a cold war over the office thermostat. Someone needs to mediate before it escalates to HR.",
                "objectives": [
                    "Understand each team's position",
                    "Find a setting both can accept",
                    "Get a written agreement before it goes to HR",
                ],
            },
            {
                "title": "The Offboarding",
                "hook": "Someone is leaving at end of day and needs to be formally offboarded. Nobody owns this process and nothing has been started.",
                "objectives": [
                    "Find out what offboarding actually requires",
                    "Get the relevant people moving",
                    "Complete everything before they walk out",
                ],
            },
            {
                "title": "The Wrong Invoice",
                "hook": "An invoice for the wrong amount has already been approved and is about to be paid. It needs to be stopped.",
                "objectives": [
                    "Find who approved it and why",
                    "Get the payment frozen before it processes",
                    "Issue a corrected invoice before close of business",
                ],
            },
            {
                "title": "Confidential Document",
                "hook": "A document marked confidential has been left on the photocopier. People have already seen it.",
                "objectives": [
                    "Retrieve all copies",
                    "Find out who has seen it",
                    "Report the incident before someone else does",
                ],
            },
            {
                "title": "Server Room Access",
                "hook": "Someone needs access to the server room for a time-sensitive fix. The keyholder is not answering.",
                "objectives": [
                    "Find an alternative keyholder",
                    "Get into the server room without triggering security",
                    "Complete the fix before the outage window closes",
                ],
            },
            {
                "title": "The Complaint",
                "hook": "A formal complaint has been filed about something that happened in the break room last Thursday. HR wants a statement by 3pm.",
                "objectives": [
                    "Find out what actually happened",
                    "Identify who needs to provide statements",
                    "Get everything to HR before the deadline",
                ],
            },
            {
                "title": "Shared Calendar Chaos",
                "hook": "Someone deleted the wrong recurring meeting from the shared calendar. Three weeks of bookings are now gone.",
                "objectives": [
                    "Find a backup or audit log",
                    "Reconstruct the missing events",
                    "Notify everyone affected before they show up to the wrong room",
                ],
            },
            {
                "title": "The Reference Check",
                "hook": "HR needs a reference check completed for a candidate who starts Monday. Nobody in the process has initiated it.",
                "objectives": [
                    "Track down the right contact for the reference",
                    "Get the reference form sent and returned",
                    "Confirm receipt with HR before end of Friday",
                ],
            },
            {
                "title": "Software License Expiry",
                "hook": "A critical software license expired at midnight and an entire team can't work. IT says this is not their problem.",
                "objectives": [
                    "Establish whose budget covers the renewal",
                    "Get the purchase approved fast",
                    "Have the license reinstated before the afternoon standup",
                ],
            },
            {
                "title": "The Broken Chair",
                "hook": "Someone reported a broken chair to facilities two weeks ago. The chair is still broken and someone just hurt themselves on it.",
                "objectives": [
                    "Get the chair removed immediately",
                    "Document the incident before end of day",
                    "Ensure a replacement arrives before tomorrow morning",
                ],
            },
            {
                "title": "Social Media Situation",
                "hook": "An employee posted something about the company on social media this morning. It has been shared 47 times. Leadership is aware.",
                "objectives": [
                    "Find out exactly what was posted and where",
                    "Get Communications involved before they find out themselves",
                    "Agree on a response approach before midday",
                ],
            },
            {
                "title": "The Budget Overrun",
                "hook": "A project has gone 30% over budget without anyone noticing until now. The monthly review is this afternoon.",
                "objectives": [
                    "Get the full picture before the review",
                    "Find someone to explain it who won't panic",
                    "Prepare a remediation plan in the next two hours",
                ],
            },
            {
                "title": "The Ringing Phone",
                "hook": "A phone on an empty desk has been ringing for three days. No one owns the extension. No one knows whose desk it was. Facilities says it is not a facilities issue.",
                "objectives": [
                    "Find out whose extension it is",
                    "Get someone authorized to answer it or stop it",
                    "File a resolution before end of day so the ticket can be closed",
                ],
            },
            {
                "title": "The Impossible Booking",
                "hook": "A conference room is booked for this morning by an account that was deactivated in 2022. The room is currently occupied by no one. The booking cannot be cancelled.",
                "objectives": [
                    "Find out how the booking exists",
                    "Get Facilities or IT to override it",
                    "Have the room available for the team that actually needs it",
                ],
            },
            {
                "title": "The Unknown Contributor",
                "hook": "A person named R. Holt has been contributing to a shared project for four months. Nobody on the team added them. They reply to comments within minutes. Nobody knows who they are.",
                "objectives": [
                    "Find out if R. Holt is internal or external",
                    "Establish whether their contributions are correct",
                    "Resolve their access status without breaking the project",
                ],
            },
            {
                "title": "The Plant",
                "hook": "The large plant near the south entrance died overnight. It was healthy yesterday. Someone has put flowers on it. A card has appeared. The card is signed by seventeen people.",
                "objectives": [
                    "Establish what actually happened",
                    "Determine whether facilities should be involved or whether this has become an HR matter",
                    "Handle the card signatories before this reaches leadership",
                ],
            },
            {
                "title": "The Approved Request",
                "hook": "An equipment request has been approved for the fourth time. The equipment has never arrived. The previous three approvals are documented. Nobody can explain where the equipment goes.",
                "objectives": [
                    "Track what actually happens when the order is placed",
                    "Find the gap in the process",
                    "Get the equipment to the person before they make a fifth request",
                ],
            },
            {
                "title": "The Door",
                "hook": "There is a door on the second floor that is not on the floor plan. It has a keypad. Facilities does not have the code. Nobody is claiming it.",
                "objectives": [
                    "Find out who has access to it",
                    "Determine whether it is a safety issue",
                    "Get it documented or sealed before the building inspector arrives Friday",
                ],
            },
            {
                "title": "Email From The Previous Tenant",
                "hook": "The building's previous occupant has been sending emails to the company's internal support address for six weeks. The emails describe IT problems in detail. Someone has been replying to them.",
                "objectives": [
                    "Find out who has been replying",
                    "Determine if any information has been shared that shouldn't have been",
                    "Close the loop with the previous tenant and close the ticket",
                ],
            },
        ],
        "complications": [],
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
    # audience-injected encounters that interrupt the agent's playthrough
    ENCOUNTER = "encounter"  # a wild event / person crashes in
    RIVAL = "rival"  # someone blocks or challenges the agent
    BOON = "boon"  # something helps the agent
    TWIST = "twist"  # the situation is reframed


@dataclass
class CardDef:
    id: str
    name: str
    type: CardType
    description: str
    flavor: str
    weight: float = 1.0  # relative frequency in the hand


CARD_LIBRARY = [
    # ── Encounter — a wild event or person crashes into the scene ──────────────
    CardDef(
        "enc_courier",
        "A Courier Appears",
        CardType.ENCOUNTER,
        "A courier bursts in demanding a signature for a package nobody ordered.",
        "Wild. Insistent.",
        1.2,
    ),
    CardDef(
        "enc_intern",
        "Wild Intern",
        CardType.ENCOUNTER,
        "An eager intern appears with a clipboard, asking if anyone has seen the all-hands.",
        "Harmless. Persistent.",
        1.2,
    ),
    CardDef(
        "enc_alarm",
        "Fire Drill",
        CardType.ENCOUNTER,
        "The alarm tests itself. Everyone must file out, slowly, by department.",
        "Unscheduled. Mandatory.",
        1.0,
    ),
    CardDef(
        "enc_auditor",
        "The Auditor",
        CardType.ENCOUNTER,
        "An auditor materialises, asking to see the asset register. Today.",
        "Nobody summoned them.",
        0.8,
    ),
    CardDef(
        "enc_catering",
        "Catering Arrives",
        CardType.ENCOUNTER,
        "Sandwiches arrive for a meeting that was cancelled. They must be dealt with.",
        "Forty rounds. No takers.",
        1.0,
    ),
    CardDef(
        "enc_phone",
        "It Won't Stop Ringing",
        CardType.ENCOUNTER,
        "A desk phone rings and rings. Nobody owns the desk.",
        "Line 3. Always line 3.",
        0.9,
    ),
    # ── Rival — someone blocks or challenges the agent ─────────────────────────
    CardDef(
        "riv_badge",
        "Badge Check",
        CardType.RIVAL,
        "Reception stops you. Your visitor badge expired at noon.",
        "Policy is policy.",
        1.2,
    ),
    CardDef(
        "riv_gatekeeper",
        "The Gatekeeper",
        CardType.RIVAL,
        "A PA refuses to let you past without a calendar invite.",
        "Fifteen minutes, minimum.",
        1.1,
    ),
    CardDef(
        "riv_replyall",
        "Reply-All Storm",
        CardType.RIVAL,
        "An email thread erupts. Everyone is cc'd. Opinions are forming.",
        "112 unread.",
        1.0,
    ),
    CardDef(
        "riv_jurisdiction",
        "Not Our Department",
        CardType.RIVAL,
        "Facilities says this is IT's problem. IT says it is Facilities'.",
        "Raise a ticket. With whom.",
        1.0,
    ),
    CardDef(
        "riv_signature",
        "Pending Signature",
        CardType.RIVAL,
        "Nothing proceeds without a signature from someone currently on leave.",
        "Back the 14th. Possibly.",
        1.0,
    ),
    CardDef(
        "riv_manager",
        "A Manager Intercepts",
        CardType.RIVAL,
        "A manager catches you in the corridor with a quick question that is neither.",
        "Got a sec? You do not.",
        0.9,
    ),
    # ── Boon — something genuinely helps the agent ─────────────────────────────
    CardDef(
        "boon_pass",
        "Found: Visitor Pass",
        CardType.BOON,
        "A spare pass, still valid, hanging on a lanyard by the printer.",
        "Finders keepers.",
        1.1,
    ),
    CardDef(
        "boon_coffee",
        "Coffee Round",
        CardType.BOON,
        "Someone makes a round. Morale, briefly and genuinely, improves.",
        "Milk, no sugar.",
        1.2,
    ),
    CardDef(
        "boon_tipoff",
        "The Tip-Off",
        CardType.BOON,
        "A friendly admin quietly tells you where you actually need to go.",
        "Don't say I said.",
        1.1,
    ),
    CardDef(
        "boon_shortcut",
        "Propped Fire Door",
        CardType.BOON,
        "A fire door, propped open. Technically not allowed. Considerably faster.",
        "Don't tell Facilities.",
        1.0,
    ),
    CardDef(
        "boon_ally",
        "An Ally",
        CardType.BOON,
        "A colleague offers, genuinely, to vouch for you.",
        "I've got you.",
        1.0,
    ),
    CardDef(
        "boon_cache",
        "It Was Here All Along",
        CardType.BOON,
        "The thing you needed was in the second drawer. Of course it was.",
        "Hidden in plain sight.",
        0.8,
    ),
    # ── Twist — the situation is reframed under everyone's feet ────────────────
    CardDef(
        "twist_reorg",
        "Reorg",
        CardType.TWIST,
        "The team you needed reports to someone else now. As of this morning.",
        "New org chart pending.",
        0.9,
    ),
    CardDef(
        "twist_duetoday",
        "It Was Always Due Today",
        CardType.TWIST,
        "The deadline was, apparently, always today. News to everyone.",
        "Per the original brief.",
        0.9,
    ),
    CardDef(
        "twist_twoversions",
        "Two Versions",
        CardType.TWIST,
        "There are two versions of the document in circulation. Both signed.",
        "Which is canonical.",
        0.9,
    ),
    CardDef(
        "twist_wrongbuilding",
        "Wrong Building",
        CardType.TWIST,
        "The person you need has been in Building 7 the whole time.",
        "Nobody mentioned Building 7.",
        0.9,
    ),
    CardDef(
        "twist_retroactive",
        "Retroactive Policy",
        CardType.TWIST,
        "A policy now applies, backdated. Everything must be re-checked.",
        "Effective last Tuesday.",
        0.8,
    ),
    CardDef(
        "twist_namesame",
        "Two People, One Name",
        CardType.TWIST,
        "There are two Dave Wilsons. You have been talking to the wrong one.",
        "Both in Procurement.",
        0.7,
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
MAX_SCENE_ROUNDS = 3
SCENE_THRESHOLD = 3  # net ACTION progress needed to resolve a scene / tick an objective


SURREAL_ARC = [
    "a gentle wildlife documentary on a calm morning in the habitat",
    "a wildlife documentary that has begun to notice the habitat behaving a little oddly",
    "a nature documentary narrating increasingly improbable office behaviour, fondly",
    "a wildlife special where the ecosystem has quietly stopped following its own rules, observed with delight",
    "an awe-struck nature documentary watching a creature thrive in conditions that should not exist",
]


def intensity(resolution_count: int, max_resolutions: int = MAX_RESOLUTIONS) -> float:
    return min(1.0, resolution_count / max(max_resolutions - 1, 1))


def arc_register(resolution_count: int, max_resolutions: int = MAX_RESOLUTIONS) -> str:
    i = intensity(resolution_count, max_resolutions)
    idx = min(int(i * len(SURREAL_ARC)), len(SURREAL_ARC) - 1)
    return SURREAL_ARC[idx]


@dataclass
class QuestState:
    id: str
    template_id: str
    title: str
    hook: str
    complication: str
    register: str = "a warm wildlife documentary"
    beats: list[str] = field(default_factory=list)
    result_history: list[str] = field(default_factory=list)
    objectives: list[str] = field(default_factory=list)
    npcs: list[NPC] = field(default_factory=list)
    resolution_count: int = 0
    momentum: int = 0
    tension: int = 0
    scene_progress: int = 0
    outcome: str | None = None
    scene_rounds: int = 0
    scene_beat_start: int = 0
    facts: list[str] = field(default_factory=list)
    pressure_pool: list[dict] = field(default_factory=list)
    props: list[Prop] = field(default_factory=list)
    artel_task_ids: list[str] = field(default_factory=list)
    next_waypoint_override: int | None = None


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
                    sprite=rng.randint(1, 10),
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
    path: list[list[int]] = field(default_factory=list)  # remaining tiles for free-roam
    agent_goal: str = ""  # npc id the agent is currently walking to
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
    quest_id = str(uuid.uuid4())[:8]
    quest = QuestState(
        id=quest_id,
        template_id=cat_name,
        title=task["title"],
        hook=task["hook"],
        complication="",
        register=arc_register(0),
        objectives=list(task.get("objectives", [])),
    )
    return quest, cat["theme"]


def make_npcs(rng: random.Random, waypoint_count: int) -> list[NPC]:
    count = min(7, len(NPC_POOL), max(3, waypoint_count))
    n_surreal = min(len(SURREAL_NPCS), max(1, count // 3))
    n_mundane = count - n_surreal
    pool = rng.sample(MUNDANE_NPCS, min(n_mundane, len(MUNDANE_NPCS)))
    pool += rng.sample(SURREAL_NPCS, n_surreal)
    sprites = rng.sample(range(1, 11), min(len(pool), 10))
    npcs = []
    used_wps: set[int] = {0}  # reserve wp0 for player spawn
    last = waypoint_count - 1  # deepest placeable waypoint
    span = max(last - 1, 1)
    for i, raw in enumerate(pool):
        sprite = sprites[i % len(sprites)]
        frac = i / max(len(pool) - 1, 1)  # 0=front mundane .. 1=deep surreal
        preferred = 1 + round(frac * (span - 1))
        preferred = min(max(preferred, 1), last)
        idx = preferred
        for delta in range(waypoint_count):
            for candidate in (preferred + delta, preferred - delta):
                if 1 <= candidate < waypoint_count and candidate not in used_wps:
                    idx = candidate
                    break
            else:
                continue
            break
        used_wps.add(idx)
        npcs.append(
            NPC(
                id=f"npc_{raw['name'].split()[0].lower()}_{i}",
                name=raw["name"],
                role=raw["role"],
                personality=raw["personality"],
                sprite=sprite,
                waypoint_idx=idx,
                behavior=raw["behavior"],
            )
        )
    return npcs


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
    waypoint_count = len(world.waypoints) if world else 5
    quest.npcs = make_npcs(rng, waypoint_count)
    state = GameState(run_id=run_id, character=character, quest=quest, window=window, world=world)
    state.lx, state.ly = world.route[0]
    state.facing = "up"
    state.rpos = 0
    state.target_idx = 0
    state.log_event("quest_start", quest.hook)
    state.log_event("complication", quest.complication)
    return state


def sync_target(state: GameState) -> None:
    state.quest.register = arc_register(state.quest.resolution_count)
    if state.world is None:
        return
    if state.quest.next_waypoint_override is not None:
        state.target_idx = min(state.quest.next_waypoint_override, len(state.world.waypoints) - 1)
        state.quest.next_waypoint_override = None
    else:
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


def step_path(state: GameState) -> bool:
    # free-roam: follow the agent's dynamic path one tile at a time
    if not state.path:
        return False
    nx, ny = state.path[0]
    if [nx, ny] == [state.lx, state.ly]:
        state.path.pop(0)
        if not state.path:
            return False
        nx, ny = state.path[0]
    state.facing = facing_from_delta(nx - state.lx, ny - state.ly)
    state.lx, state.ly = nx, ny
    state.path.pop(0)
    return True


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
    # drain in place (copy then clear, no await between — race-free in asyncio) so
    # cards played during resolution land in the same list and resolve next cycle.
    played = list(state.window.cards)
    state.window.cards.clear()
    rng.shuffle(played)
    state.window.opened_at = now
    state.window.closes_at = now + CARD_WINDOW
    state.window.resolutions = []
    return played


def _clamp(v: int) -> int:
    return max(-10, min(10, v))


# No dice. Each card TYPE has a fixed, legible job on the shared meters.
def apply_card_effects(card_def: CardDef, quest: QuestState) -> None:
    if card_def.type == CardType.BOON:
        # something helps the agent — morale up, a step closer
        quest.scene_progress += 1
        quest.momentum = _clamp(quest.momentum + 2)
    elif card_def.type == CardType.ENCOUNTER:
        # a wild event — eventful, nudges things along but disruptive
        quest.scene_progress += 1
        quest.tension = min(10, quest.tension + 1)
        quest.momentum = _clamp(quest.momentum - 1)
    elif card_def.type == CardType.RIVAL:
        # blocks or challenges the agent — a setback to overcome
        quest.scene_progress = max(0, quest.scene_progress - 1)
        quest.tension = min(10, quest.tension + 1)
        quest.momentum = _clamp(quest.momentum - 2)
    elif card_def.type == CardType.TWIST:
        # reframes the situation — swings morale to the current extreme
        quest.momentum = _clamp(quest.momentum + (2 if quest.momentum >= 0 else -2))


def classify_window(progress_delta: int, momentum_delta: int) -> str:
    if progress_delta > 0 and momentum_delta >= 0:
        return "triumph"
    if progress_delta < 0 or momentum_delta < 0:
        return "setback"
    return "mixed"


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
