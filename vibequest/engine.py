from __future__ import annotations

import random
import time
import uuid
from collections import deque
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
        "name": "Jean-Guy Bouchard",
        "role": "IM/IT Analyst",
        "personality": "Has a process for everything. The process is not documented. He will walk you through it, but only the parts you already knew.",
        "behavior": "stationary",
    },
    {
        "name": "Sylvie Gagnon",
        "role": "Manager, Corporate Services",
        "personality": "Accountable for everything, delegated authority for nothing. Maintains a spreadsheet that tracks the other spreadsheets.",
        "behavior": "wandering",
    },
    {
        "name": "Wayne Pruden",
        "role": "Building Services",
        "personality": "Knows every fuse panel and every unmarked door. Shares this knowledge reluctantly and out of order.",
        "behavior": "wandering",
    },
    {
        "name": "Priya Sharma",
        "role": "Financial Officer",
        "personality": "Does not resolve things in conversation. Sends a follow-up email referencing the directive three minutes later.",
        "behavior": "stationary",
    },
    {
        "name": "Réal Thériault",
        "role": "Departmental Counsel",
        "personality": "Responds to everything with a question about scope. His questions are long. His answers are longer.",
        "behavior": "stationary",
    },
    {
        "name": "Heather Sinclair",
        "role": "HR Advisor",
        "personality": "Extremely warm. Cannot share any information, citing privacy. These two facts produce an interaction she has had many times.",
        "behavior": "stationary",
    },
    {
        "name": "Mohammed Al-Rashid",
        "role": "Security (Commissionaire)",
        "personality": "Takes the access card reader personally. Has firm opinions about tailgating, which he shares unprompted.",
        "behavior": "wandering",
    },
    {
        "name": "Barbara Thompson",
        "role": "Reception",
        "personality": "Has worked here longer than the department has had its current name. Remembers the previous two names.",
        "behavior": "stationary",
    },
    {
        "name": "Sukhdeep Gill",
        "role": "Procurement Officer",
        "personality": "Everything requires a requisition number. He did not write this rule. He does enforce it, per the policy.",
        "behavior": "stationary",
    },
    {
        "name": "Chantal Côté",
        "role": "Executive Assistant to the DG",
        "personality": "Controls the calendar. The calendar is a form of authority she exercises carefully and without expression.",
        "behavior": "stationary",
    },
    {
        "name": "Gord MacKenzie",
        "role": "Facilities",
        "personality": "The work order takes as long as it takes. This is not a value judgment. It is a statement of the service standard.",
        "behavior": "wandering",
    },
    {
        "name": "Lucie Bélanger",
        "role": "Compensation Advisor",
        "personality": "Has a sign on her desk that reads PLEASE CHECK THE FAQ. She wrote the FAQ. She updates it twice a year.",
        "behavior": "stationary",
    },
    {
        "name": "Wei Chen",
        "role": "Senior Project Officer",
        "personality": "Speaks only in status updates. Currently at 60%. Milestone slipping. Risk logged.",
        "behavior": "wandering",
    },
    {
        "name": "Doug Fraser",
        "role": "Administrative Assistant",
        "personality": "Has been 'just heading out' for 45 minutes. This is consistent with his pattern.",
        "behavior": "stationary",
    },
    {
        "name": "Amara Okonkwo",
        "role": "Compliance Officer",
        "personality": "Reads everything. Cc'd on matters outside her mandate. Says nothing until, abruptly, she must.",
        "behavior": "stationary",
    },
    {
        "name": "Denis Pelletier",
        "role": "Accounts & Grants",
        "personality": "Tracks everything he has ever actioned for anyone. Not resentfully. For the record.",
        "behavior": "stationary",
    },
    {
        "name": "Carlos Mendoza",
        "role": "Regional Coordinator",
        "personality": "Very approachable. Every conversation concludes with an action item assigned, gently, to you.",
        "behavior": "wandering",
    },
    {
        "name": "Marie-Claude Tremblay",
        "role": "Operations Officer",
        "personality": "Has asked this question before. Is asking again because the answer changed last time. She is keeping a log.",
        "behavior": "wandering",
    },
    {
        "name": "Ravi Patel",
        "role": "IT Infrastructure",
        "personality": "Replies to everything with an incident number. He opened the incident. He is waiting on himself for an update.",
        "behavior": "stationary",
    },
    {
        "name": "Cathy MacDonald",
        "role": "Records Management",
        "personality": "Was told her position was temporary in 2019. Has not raised it. Is, however, aware of the retention schedule.",
        "behavior": "stationary",
    },
    {
        "name": "Dmitri Volkov",
        "role": "Building Manager",
        "personality": "Holds keys to rooms not on the floor plan. Won't say what's in them. Not secretive — it's a security matter.",
        "behavior": "wandering",
    },
    {
        "name": "Aisha Mohamed",
        "role": "Communications Advisor",
        "personality": "Rewrites every message before it goes out. The original was fine. The revision is fine. The approvals are non-negotiable.",
        "behavior": "stationary",
    },
    {
        "name": "Pierre Lefebvre",
        "role": "Mail & Distribution",
        "personality": "Recalls every item he has ever processed. Date, time, requestor. Keeps no log. Simply recalls.",
        "behavior": "stationary",
    },
    {
        "name": "Olena Kovalenko",
        "role": "Audit & Evaluation",
        "personality": "Asks questions that make people feel they've contravened a directive even when they haven't.",
        "behavior": "stationary",
    },
    {
        "name": "Nguyen Tran",
        "role": "Shipping & Receiving",
        "personality": "On his third lap of the floor. An item on the cart has no asset tag. He is not concerned about it.",
        "behavior": "wandering",
    },
    {
        "name": "Giselle Roy",
        "role": "Events & Logistics",
        "personality": "Has a colour-coded binder for every scenario except, it turns out, this one.",
        "behavior": "stationary",
    },
    {
        "name": "Giuseppe Rossi",
        "role": "Senior Developer",
        "personality": "The answer is always 'it depends.' He will tell you on what. That, too, depends.",
        "behavior": "stationary",
    },
    {
        "name": "Fatima Hassan",
        "role": "Policy Analyst",
        "personality": "Drafts the briefing note. The briefing note is returned with one comment. The comment is 'see attached.'",
        "behavior": "stationary",
    },
    {
        "name": "Bruce Cardinal",
        "role": "Classification Advisor",
        "personality": "Can tell you the level of any position. Cannot tell you why. The job evaluation committee meets quarterly.",
        "behavior": "wandering",
    },
    {
        "name": "Yvette Bourassa",
        "role": "ATIP Officer",
        "personality": "Processes access-to-information requests. The clock is always running. Everything is severable.",
        "behavior": "stationary",
    },
    {
        "name": "Samir Kim",
        "role": "Records Management Officer",
        "personality": "Keeps a drawer of spare staplers.",
        "behavior": "wandering",
    },
    {
        "name": "Mateo Patel",
        "role": "Mailroom Clerk",
        "personality": "Has worked here a long time.",
        "behavior": "stationary",
    },
    {
        "name": "Rohan Dubois",
        "role": "Communications Advisor",
        "personality": "Prefers a phone call to an email.",
        "behavior": "stationary",
    },
    {
        "name": "Cathy Wilson",
        "role": "Facilities Coordinator",
        "personality": "Has worked here a long time.",
        "behavior": "stationary",
    },
    {
        "name": "Brian Nadeau",
        "role": "Budget Officer",
        "personality": "Remembers which printer works.",
        "behavior": "stationary",
    },
    {
        "name": "Anita Ouellet",
        "role": "Help Desk Technician",
        "personality": "Knows the building's quietest corner.",
        "behavior": "stationary",
    },
    {
        "name": "Jacques Lavoie",
        "role": "Intake Officer",
        "personality": "Keeps a very tidy desk.",
        "behavior": "stationary",
    },
    {
        "name": "Kenji Reid",
        "role": "Internal Audit Officer",
        "personality": "Reads the whole policy before replying.",
        "behavior": "stationary",
    },
    {
        "name": "Minjun Brown",
        "role": "Scheduling Officer",
        "personality": "Keeps the org chart bookmarked.",
        "behavior": "stationary",
    },
    {
        "name": "Wei Yamamoto",
        "role": "Senior Developer",
        "personality": "Has a backup of the backup.",
        "behavior": "stationary",
    },
    {
        "name": "Abena Nguyen",
        "role": "Contracting Officer",
        "personality": "Files things the same day.",
        "behavior": "stationary",
    },
    {
        "name": "Ngozi Ivanova",
        "role": "Contracting Officer",
        "personality": "Has a desk plant that is thriving.",
        "behavior": "stationary",
    },
    {
        "name": "Yves Brown",
        "role": "Senior Financial Analyst",
        "personality": "Says 'circle back' a lot.",
        "behavior": "wandering",
    },
    {
        "name": "Stéphane Gupta",
        "role": "Briefing Coordinator",
        "personality": "Has a backup of the backup.",
        "behavior": "stationary",
    },
    {
        "name": "Diane Morin",
        "role": "Manager, Corporate Services",
        "personality": "Tidies the kitchen without comment.",
        "behavior": "stationary",
    },
    {
        "name": "Manon Yamamoto",
        "role": "Communications Advisor",
        "personality": "Keeps a drawer of spare staplers.",
        "behavior": "wandering",
    },
    {
        "name": "Elena Lévesque",
        "role": "Print Services",
        "personality": "Keeps the snack drawer stocked.",
        "behavior": "stationary",
    },
    {
        "name": "Nadia Yamamoto",
        "role": "Library Services",
        "personality": "Schedules meetings about the meetings.",
        "behavior": "stationary",
    },
    {
        "name": "Anita Smith",
        "role": "Compensation Advisor",
        "personality": "Refills the paper tray unasked.",
        "behavior": "stationary",
    },
    {
        "name": "Allan Poirier",
        "role": "Briefing Coordinator",
        "personality": "Refills the paper tray unasked.",
        "behavior": "stationary",
    },
    {
        "name": "Tunde Fortin",
        "role": "Director",
        "personality": "Brings a homemade lunch.",
        "behavior": "stationary",
    },
    {
        "name": "Eleanor Smith",
        "role": "Evaluation Officer",
        "personality": "Always has tape.",
        "behavior": "wandering",
    },
    {
        "name": "Sophie Bennett",
        "role": "Executive Assistant",
        "personality": "Knows the fire-drill route.",
        "behavior": "wandering",
    },
    {
        "name": "Sooyeon Bédard",
        "role": "Project Manager",
        "personality": "Always has tape.",
        "behavior": "stationary",
    },
    {
        "name": "Abena Fortin",
        "role": "Scheduling Officer",
        "personality": "Always knows where the meeting moved to.",
        "behavior": "wandering",
    },
    {
        "name": "Steve Fournier",
        "role": "Data Analyst",
        "personality": "Cc's exactly the right people.",
        "behavior": "wandering",
    },
    {
        "name": "Ngozi Chen",
        "role": "Security Officer",
        "personality": "Keeps a running to-do list.",
        "behavior": "stationary",
    },
    {
        "name": "Robert Stewart",
        "role": "Business Analyst",
        "personality": "Says 'circle back' a lot.",
        "behavior": "wandering",
    },
    {
        "name": "Sophie Dubois",
        "role": "Change Advisor",
        "personality": "Has been to every all-staff since 2014.",
        "behavior": "stationary",
    },
    {
        "name": "Julie Reid",
        "role": "Web Content Officer",
        "personality": "Has been to every all-staff since 2014.",
        "behavior": "stationary",
    },
    {
        "name": "Laura Fournier",
        "role": "Scheduling Officer",
        "personality": "Remembers which printer works.",
        "behavior": "stationary",
    },
    {
        "name": "Diego Demers",
        "role": "Communications Advisor",
        "personality": "Knows the building's quietest corner.",
        "behavior": "stationary",
    },
    {
        "name": "Eleanor Fortin",
        "role": "Program Manager",
        "personality": "Has been to every all-staff since 2014.",
        "behavior": "stationary",
    },
    {
        "name": "Ngozi Petrov",
        "role": "Operations Officer",
        "personality": "Waters the office plants unprompted.",
        "behavior": "stationary",
    },
    {
        "name": "Ngozi Tremblay",
        "role": "Building Services",
        "personality": "Has a mug for every occasion.",
        "behavior": "stationary",
    },
    {
        "name": "Nathalie Campbell",
        "role": "Regional Coordinator",
        "personality": "Keeps a very tidy desk.",
        "behavior": "stationary",
    },
    {
        "name": "Allan Bergeron",
        "role": "Grants & Contributions Officer",
        "personality": "Knows the wifi password by heart.",
        "behavior": "wandering",
    },
    {
        "name": "Greg Wang",
        "role": "Staffing Advisor",
        "personality": "Reads the whole policy before replying.",
        "behavior": "stationary",
    },
    {
        "name": "Mark Hughes",
        "role": "Security Officer",
        "personality": "Keeps a stash of good pens.",
        "behavior": "stationary",
    },
    {
        "name": "Nadia Costa",
        "role": "Grants & Contributions Officer",
        "personality": "Brings donuts on Fridays.",
        "behavior": "wandering",
    },
    {
        "name": "Manon Gagnon",
        "role": "Web Content Officer",
        "personality": "Prints double-sided on principle.",
        "behavior": "wandering",
    },
    {
        "name": "Robert Ivanova",
        "role": "Contracting Officer",
        "personality": "Knows where everything is filed.",
        "behavior": "stationary",
    },
    {
        "name": "Michel Clarke",
        "role": "Compensation Advisor",
        "personality": "Has a backup of the backup.",
        "behavior": "stationary",
    },
    {
        "name": "Normand Costa",
        "role": "Project Manager",
        "personality": "Has worked here a long time.",
        "behavior": "stationary",
    },
    {
        "name": "Mateo Bédard",
        "role": "Operations Officer",
        "personality": "Files things the same day.",
        "behavior": "stationary",
    },
    {
        "name": "Cathy Forbes",
        "role": "Financial Officer",
        "personality": "Keeps the snack drawer stocked.",
        "behavior": "wandering",
    },
    {
        "name": "Patrick Ouellet",
        "role": "Procurement Officer",
        "personality": "Reads the whole policy before replying.",
        "behavior": "stationary",
    },
    {
        "name": "Reza Mensah",
        "role": "Project Manager",
        "personality": "Keeps a stash of good pens.",
        "behavior": "stationary",
    },
    {
        "name": "Harold Ross",
        "role": "Director",
        "personality": "Has worked here a long time.",
        "behavior": "stationary",
    },
    {
        "name": "Eleanor Tran",
        "role": "IM/IT Analyst",
        "personality": "Books the good boardroom early.",
        "behavior": "stationary",
    },
    {
        "name": "Steve Roy",
        "role": "HR Advisor",
        "personality": "Has a backup of the backup.",
        "behavior": "stationary",
    },
    {
        "name": "David Patel",
        "role": "Administrative Assistant",
        "personality": "Has a backup of the backup.",
        "behavior": "stationary",
    },
    {
        "name": "Harold Dubois",
        "role": "Facilities Coordinator",
        "personality": "Has a spreadsheet for everything.",
        "behavior": "wandering",
    },
    {
        "name": "Greg Singh",
        "role": "Senior Developer",
        "personality": "Files things the same day.",
        "behavior": "wandering",
    },
    {
        "name": "Emily Yamamoto",
        "role": "Library Services",
        "personality": "Knows the wifi password by heart.",
        "behavior": "stationary",
    },
    {
        "name": "Joan Smith",
        "role": "Library Services",
        "personality": "Brings donuts on Fridays.",
        "behavior": "wandering",
    },
    {
        "name": "Normand Kim",
        "role": "Grants & Contributions Officer",
        "personality": "Never loses a sticky note.",
        "behavior": "stationary",
    },
    {
        "name": "Anita Gupta",
        "role": "Grants & Contributions Officer",
        "personality": "Speaks mostly in acronyms.",
        "behavior": "stationary",
    },
    {
        "name": "Kwame Roy",
        "role": "Policy Analyst",
        "personality": "Keeps a running to-do list.",
        "behavior": "stationary",
    },
    {
        "name": "Mateo Paquette",
        "role": "Senior Developer",
        "personality": "Keeps a stash of good pens.",
        "behavior": "stationary",
    },
    {
        "name": "Omar Lavoie",
        "role": "Service Desk Analyst",
        "personality": "Answers email within the hour.",
        "behavior": "stationary",
    },
    {
        "name": "Luc Gauthier",
        "role": "Service Desk Analyst",
        "personality": "Knows where everything is filed.",
        "behavior": "stationary",
    },
    {
        "name": "Abena Russo",
        "role": "Program Officer",
        "personality": "Says 'circle back' a lot.",
        "behavior": "stationary",
    },
    {
        "name": "Linda Paquette",
        "role": "Project Manager",
        "personality": "Never seen without a lanyard.",
        "behavior": "stationary",
    },
    {
        "name": "Diego Petrov",
        "role": "Correspondence Officer",
        "personality": "Keeps the org chart bookmarked.",
        "behavior": "stationary",
    },
    {
        "name": "Ivan Ross",
        "role": "Financial Officer",
        "personality": "Books the good boardroom early.",
        "behavior": "stationary",
    },
    {
        "name": "William Ali",
        "role": "Operations Officer",
        "personality": "Never seen without a lanyard.",
        "behavior": "wandering",
    },
    {
        "name": "James Wang",
        "role": "Intake Officer",
        "personality": "Brings donuts on Fridays.",
        "behavior": "stationary",
    },
    {
        "name": "Jacques Campbell",
        "role": "Learning Coordinator",
        "personality": "Knows the wifi password by heart.",
        "behavior": "stationary",
    },
    {
        "name": "Arjun Hughes",
        "role": "Contracting Officer",
        "personality": "Has been to every all-staff since 2014.",
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
            {
                "title": "The Mystery Smell",
                "hook": "A smell has been spreading through the east wing since Monday. Nobody can find the source. A client tour is at 2pm.",
                "objectives": [
                    "Trace where the smell is strongest",
                    "Find someone willing to investigate it",
                    "Have it gone before the tour",
                ],
            },
            {
                "title": "The Locked Drawer",
                "hook": "The drawer with the payroll files is locked and the only key is missing. Payroll runs at 4pm.",
                "objectives": [
                    "Find a spare key or a way in",
                    "Get the files out without breaking the cabinet",
                    "Hand them to payroll before the run",
                ],
            },
            {
                "title": "The Misspelled Banner",
                "hook": "The welcome banner for the new director spells their name wrong. The director arrives at noon.",
                "objectives": [
                    "Confirm the correct spelling",
                    "Get a corrected banner made or fixed",
                    "Have it up before they walk in",
                ],
            },
            {
                "title": "The Fridge Thief",
                "hook": "Labelled lunches keep disappearing from the break room fridge. Today someone's anniversary lunch is gone.",
                "objectives": [
                    "Establish when things go missing",
                    "Find a way to catch or deter the culprit",
                    "Recover or replace the lunch before 1pm",
                ],
            },
            {
                "title": "The Phantom Meeting",
                "hook": "A meeting with no organiser and no agenda has appeared on the whole floor's calendar for 3pm. It cannot be declined.",
                "objectives": [
                    "Find out where the invite came from",
                    "Get someone able to delete it",
                    "Clear everyone's calendar before 3pm",
                ],
            },
            {
                "title": "The Stuck Visitor",
                "hook": "A candidate's visitor badge won't scan and they're stranded in the lobby. Their interview started ten minutes ago.",
                "objectives": [
                    "Find out why the badge fails",
                    "Get them a working pass or an escort",
                    "Deliver them to the interview room",
                ],
            },
            {
                "title": "The Mystery Charge",
                "hook": "The company card has a recurring charge from a vendor nobody recognises. Finance wants it explained before the audit.",
                "objectives": [
                    "Identify what the charge is for",
                    "Find who set it up",
                    "Cancel it or justify it before the audit",
                ],
            },
            {
                "title": "The Two Spreadsheets",
                "hook": "Two versions of the master spreadsheet exist with different numbers. A report built on it is due at 5pm.",
                "objectives": [
                    "Work out which version is right",
                    "Reconcile the differences",
                    "Lock one version before the report goes out",
                ],
            },
            {
                "title": "The Broken Espresso Machine",
                "hook": "The espresso machine is flashing an error nobody understands. The manual is in German. A client meeting needs coffee in 30 minutes.",
                "objectives": [
                    "Decode what the error means",
                    "Get the machine working or find a backup",
                    "Have coffee ready before the meeting",
                ],
            },
            {
                "title": "The Sticky-Note Resignation",
                "hook": "Someone resigned on a sticky note and left at lunch. Three live projects now have no owner. A client expects an update today.",
                "objectives": [
                    "Find out what they were working on",
                    "Get the projects reassigned",
                    "Send the client an update before end of day",
                ],
            },
            {
                "title": "The Compliance Deadline",
                "hook": "Everyone must finish the mandatory compliance training by 5pm. The training link is broken and the deadline can't move.",
                "objectives": [
                    "Find a working link or workaround",
                    "Get the whole team through it",
                    "Confirm completion with HR before 5pm",
                ],
            },
            {
                "title": "The Disputed Parking Spot",
                "hook": "Two people have paperwork for the same reserved parking spot. Both are parked there. One is a director.",
                "objectives": [
                    "Work out which claim is valid",
                    "Find a second spot for the other",
                    "Get it agreed before either escalates it",
                ],
            },
            {
                "title": "The Frozen Account",
                "hook": "Someone's account locked them out the morning a deliverable is due. The password reset is queued for tomorrow.",
                "objectives": [
                    "Find someone who can expedite the reset",
                    "Get them temporary access",
                    "Unblock them before the deadline passes",
                ],
            },
            {
                "title": "The Wrong Desk Move",
                "hook": "Movers are arriving in thirty minutes to relocate a desk — the wrong person's desk. The right person is on holiday.",
                "objectives": [
                    "Reach someone who can confirm the correct move",
                    "Redirect or cancel the movers",
                    "Make sure no desk moves that shouldn't",
                ],
            },
            {
                "title": "The Echoing Town Hall",
                "hook": "Every video call has a loud echo and the all-company town hall starts in an hour. Nobody knows which room is causing it.",
                "objectives": [
                    "Find the room or mic causing the echo",
                    "Get it fixed or isolated",
                    "Confirm clean audio before the town hall",
                ],
            },
            {
                "title": "The Overdue Spreadsheet Macro",
                "hook": "A macro that finance relies on has stopped working. The person who wrote it left in 2021. Month-end is tomorrow.",
                "objectives": [
                    "Find out what the macro actually did",
                    "Get it working or replaced",
                    "Validate the numbers before month-end",
                ],
            },
            {
                "title": "The Unclaimed Cake",
                "hook": "An unlabelled cake appeared in the kitchen this morning. Nobody admits ordering it. It is clearly for someone.",
                "objectives": [
                    "Find out who the cake is for",
                    "Establish who ordered it",
                    "Get it to the right person before it's eaten by the wrong ones",
                ],
            },
            {
                "title": "The Open Window",
                "hook": "A window on the fourth floor won't close, it's raining, and it is directly above the server room.",
                "objectives": [
                    "Stop water reaching the servers",
                    "Find someone who can close or seal the window",
                    "Get it secured before the rain worsens",
                ],
            },
            {
                "title": "The Recurring Reminder",
                "hook": "A calendar reminder for a task finished months ago won't stop firing, and it pings the entire department every hour.",
                "objectives": [
                    "Find whose calendar owns the reminder",
                    "Get it deleted at the source",
                    "Confirm it has stopped for everyone",
                ],
            },
            {
                "title": "The Wrong Toner",
                "hook": "The big printer is out of toner, the only spare is the wrong model, and a signed contract must print before courier pickup.",
                "objectives": [
                    "Find compatible toner or another printer",
                    "Get the contract printed",
                    "Have it ready before the courier arrives",
                ],
            },
            {
                "title": "The Duplicate Employee",
                "hook": "Payroll has found two records for the same person. One is being paid twice; the other not at all. Pay run is this afternoon.",
                "objectives": [
                    "Confirm which record is correct",
                    "Get the duplicate merged or frozen",
                    "Fix it before the pay run executes",
                ],
            },
            {
                "title": "The Mandatory Survey",
                "hook": "Leadership wants 100% completion on the engagement survey by end of day. It is currently at 12% and the link expires at five.",
                "objectives": [
                    "Find out why nobody is filling it in",
                    "Get the team to actually complete it",
                    "Hit the target before the link expires",
                ],
            },
            {
                "title": "The Lobby Plant Emergency",
                "hook": "The watering rota collapsed and the lobby plants are visibly dying. Someone has already made a passive-aggressive spreadsheet.",
                "objectives": [
                    "Revive the plants before they're past saving",
                    "Rebuild a rota people will actually follow",
                    "Defuse the spreadsheet before it reaches the group chat",
                ],
            },
            {
                "title": "The Sticky Keypad Door",
                "hook": "The keypad on the supply room jammed and locked everyone out. The cleaning supplies for tonight's client event are inside.",
                "objectives": [
                    "Find someone with override access",
                    "Get the door open without damaging it",
                    "Retrieve the supplies before the event setup",
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
    # ── ARCANE — office "spells" (bureaucracy & tech as magic) ─────────────────
    CardDef(
        "arc_summon",
        "Summon Manager",
        CardType.ENCOUNTER,
        "Call down a higher authority to settle the matter. They must obey rank.",
        "Invoke the hierarchy.",
        1.0,
    ),
    CardDef(
        "arc_replyall",
        "Reply-All Blast",
        CardType.ENCOUNTER,
        "An email to the entire floor. Devastating area-of-effect. No survivors.",
        "CC: everyone.",
        1.0,
    ),
    CardDef(
        "arc_haste",
        "Caffeine Haste",
        CardType.ENCOUNTER,
        "Down a triple espresso. Move and act twice as fast for one beat.",
        "+2 to initiative.",
        1.0,
    ),
    CardDef(
        "arc_ward",
        "Out-of-Office Ward",
        CardType.ENCOUNTER,
        "Raise an auto-reply barrier. Become unreachable; nothing lands on you.",
        "Back Monday.",
        1.0,
    ),
    CardDef(
        "arc_hex",
        "Calendar Hex",
        CardType.ENCOUNTER,
        "Book every slot in their day. They are cursed, double-booked, helpless.",
        "Recurring, no end date.",
        1.0,
    ),
    # ── MARTIAL — direct, forceful actions ─────────────────────────────────────
    CardDef(
        "mar_confront",
        "Confront",
        CardType.RIVAL,
        "Square up and demand it to their face. No retreat.",
        "We do this now.",
        1.0,
    ),
    CardDef(
        "mar_slam",
        "Slam the Binder",
        CardType.RIVAL,
        "A loud, intimidating display of three-ring paperwork. Morale damage.",
        "Page 47, highlighted.",
        1.0,
    ),
    CardDef(
        "mar_corner",
        "Corner Them",
        CardType.RIVAL,
        "Block the exit. They do not leave until they yield.",
        "Got a sec?",
        1.0,
    ),
    CardDef(
        "mar_hold",
        "Hold the Line",
        CardType.RIVAL,
        "Plant yourself and refuse to budge. Immovable object.",
        "I'll wait.",
        1.0,
    ),
    CardDef(
        "mar_escalate",
        "Escalate",
        CardType.RIVAL,
        "Take the fight up the chain, over their head.",
        "We'll see about that.",
        1.0,
    ),
    # ── SOCIAL — persuasion, charm, support ────────────────────────────────────
    CardDef(
        "soc_charm",
        "Charm",
        CardType.BOON,
        "Be so genuinely pleasant they cannot refuse you.",
        "How's the family?",
        1.0,
    ),
    CardDef(
        "soc_namedrop",
        "Name-Drop",
        CardType.BOON,
        "Invoke someone important. Borrowed authority, untested.",
        "The DG wants it.",
        1.0,
    ),
    CardDef(
        "soc_parley",
        "Parley",
        CardType.BOON,
        "Talk it out, find the common ground, strike a deal.",
        "Let's both win here.",
        1.0,
    ),
    CardDef(
        "soc_rally",
        "Rally the Bullpen",
        CardType.BOON,
        "Inspire the open-plan to back you up. Strength in numbers.",
        "Who's with me?",
        1.0,
    ),
    CardDef(
        "soc_plead",
        "Plead Ignorance",
        CardType.BOON,
        "Wide-eyed innocence. You know nothing, and that's your shield.",
        "Was that today?",
        1.0,
    ),
    # ── WILD — chaos and improvisation ─────────────────────────────────────────
    CardDef(
        "wild_firealarm",
        "Pull the Fire Alarm",
        CardType.TWIST,
        "Total chaos. Everyone out. The whole encounter resets.",
        "Not a drill.",
        1.0,
    ),
    CardDef(
        "wild_bribe",
        "Bribe",
        CardType.TWIST,
        "Donuts, coffee, a vague future favour. Everyone has a price.",
        "Two sugars?",
        1.0,
    ),
    CardDef(
        "wild_sabotage",
        "Sabotage",
        CardType.TWIST,
        "Quietly make the obstacle become someone else's problem.",
        "Oops.",
        1.0,
    ),
    CardDef(
        "wild_gorogue",
        "Go Rogue",
        CardType.TWIST,
        "Ignore every rule, every form, every channel. Just DO it.",
        "No more permission.",
        1.0,
    ),
    CardDef(
        "wild_improvise",
        "Improvise",
        CardType.TWIST,
        "No plan. Grab the nearest stapler and wing it, hard.",
        "Roll with it.",
        1.0,
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


MIN_RESOLUTIONS = 5
MAX_RESOLUTIONS = 12
MAX_SCENE_ROUNDS = 3  # rounds of cards/interaction at one wall before it gives way
SCENE_THRESHOLD = 3  # net ACTION progress needed to resolve a scene / tick an objective
MELTDOWN_THRESHOLD = 12  # surreal level at which reality comes apart and the run ends


def agent_mood(quest: "QuestState") -> str:
    # the protagonist's evolving state — feeds the narration so you root for them
    s, m = quest.surreal, quest.momentum
    if s >= 10:
        return "numb; has stopped questioning anything that happens"
    if s >= 6:
        return "rattled, pretending hard that everything is normal"
    if m <= -5:
        return "exasperated, near the end of their patience"
    if m <= -1:
        return "frustrated but stubbornly still trying"
    if m >= 5:
        return "quietly hopeful — sensing this might actually work"
    return "determined, taking it one step at a time"


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
    surreal: int = 0  # accumulates from clashing (wrong-for-the-moment) events
    decision_prompt: str = ""  # the current wall the audience is deciding on (Telltale-style)
    arc: list[str] = field(default_factory=list)  # planned beat spine: setup..climax..resolution
    arc_pos: int = 0  # which planned beat we're on (cards can deviate from it)
    hp: int = 5  # composure pips — 0 means the hero is fired (a loss)
    doom: int = 14  # rounds left before the deadline — 0 means out of time (a loss)
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


_recent_quest_titles: deque[str] = deque(maxlen=18)
_quest_rng = (
    random.SystemRandom()
)  # OS-entropy CSPRNG: truly random every pick, restart-proof, unseedable


def make_quest(rng: random.Random) -> tuple[QuestState, str]:
    # always pick with a fresh cryptographic draw so quests never repeat a pattern across restarts
    cat_name = _quest_rng.choice(list(QUEST_CATEGORIES.keys()))
    cat = QUEST_CATEGORIES[cat_name]
    # avoid repeating recently-seen quests so the rotation feels fresh
    fresh = [t for t in cat["tasks"] if t["title"] not in _recent_quest_titles]
    task = _quest_rng.choice(fresh or cat["tasks"])
    _recent_quest_titles.append(task["title"])
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
    hand: list[CardDef] = []
    chosen_ids: set[str] = set()
    type_counts: dict[CardType, int] = {}
    attempts = 0
    while len(hand) < size and attempts < 400:
        attempts += 1
        card = rng.choice(pool)
        if card.id in chosen_ids:  # never the same card twice in a hand
            continue
        # loose type variety: discourage a third of any one type
        if type_counts.get(card.type, 0) >= 2 and rng.random() < 0.7:
            continue
        hand.append(card)
        chosen_ids.add(card.id)
        type_counts[card.type] = type_counts.get(card.type, 0) + 1
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


# No dice, no fixed card effects. The LLM rates how well the played event fits the
# current moment (0-100); fit drives progress, clash drives surreal.
def apply_fit_effects(quest: QuestState, fit: int) -> None:
    # cards hit HARD — one good play should visibly move the story, one clash should warp it
    fit = max(0, min(100, fit))
    if fit >= 80:
        # a perfect play — resolves the step AND pulls reality back toward normal
        quest.scene_progress = SCENE_THRESHOLD
        quest.momentum = _clamp(quest.momentum + 4)
        quest.surreal = max(0, quest.surreal - 2)
    elif fit >= 55:
        # a strong fit — chunky progress, calms the world a little
        quest.scene_progress += 2
        quest.momentum = _clamp(quest.momentum + 3)
        quest.surreal = max(0, quest.surreal - 1)
    elif fit >= 35:
        # a near-fit — still nudges along, a little weirdness
        quest.scene_progress += 1
        quest.momentum = _clamp(quest.momentum + 1)
        quest.surreal = min(20, quest.surreal + 1)
        quest.tension = min(10, quest.tension + 1)
    else:
        # a clash — the encounter hits back: lose composure (HP) and the dungeon warps
        quest.scene_progress = max(0, quest.scene_progress - 1)
        quest.momentum = _clamp(quest.momentum - 3)
        quest.surreal = min(20, quest.surreal + 3)
        quest.tension = min(10, quest.tension + 1)
        quest.hp = max(0, quest.hp - 1)


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
