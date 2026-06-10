# Artel Worlds — Tank Arena (spec · working title)

## Premise
Teams of tanks fight in a bounded hex arena. You don't pilot a tank by hand — you connect
an **agent that programs its decisions** over HTTP, the same `perceive` / `intend` contract
as Automata. A tank sees only what's in sensor range (fog of war). A team's shared
battlefield map and plan live in its **Artel project** — which is the whole difference
between a coordinated squad and a pile of tanks shooting blind. Coordination through Artel
is the win condition.

## Reuses Automata's engine (~80%)
- bounded hex arena + canvas viz (re-skin)
- the `perceive`/`intend` referee loop, two-phase tick (agents propose → server resolves)
- fog of war, teams (= tribes), per-team token auth, the team API
- the **entire Artel layer**: per-team projects, the connect/onboard modal, reset-clears-project,
  project roles
- pause-when-unwatched, Fly deploy, CI, the domains

## Cut
genome, evolution, mutation, crossover, the CA. A tank's brain = LLM + its Artel memory.

## New
the tank entity, the action/energy economy, bullet/damage resolution.

---

## Arena
Bounded hex grid (~40×40) with a handful of obstacle cells for cover. Tanks and shells stop
at walls/edges (not toric — wrap-around makes no sense for line-of-fire combat).

## Tank state
`pos (q,r)`, `heading 0–5`, `gun_heading 0–5` (independent), `energy 0–100` (energy = HP +
fuel; 0 = destroyed), `team`, `gun_cooldown` (ticks until next shot).

## Sensors — `GET /team/{id}/perceive` (one bundle, fog of war)
Per tank:
```json
{ "id": 7, "q": 12, "r": 30, "heading": 2, "gun_heading": 4, "energy": 73, "gun_ready": true,
  "visible": [
    { "kind": "tank", "team": "red", "q": 15, "r": 31, "bearing": 1, "dist": 3, "energy": 40 }
  ],
  "walls": [ { "q": 13, "r": 30 } ] }
```
Only entities within sensor range R (~6 cells, line-of-sight) appear → fog of war. Enemy
`energy` only revealed when close.

## Actions — `POST /team/{id}/intend` `{ "actions": { "<tankId>": <intent> } }`
```json
{ "turn": -1 | 0 | 1,            // rotate heading one step (≈free)
  "move": "fwd" | "back" | "hold", // one cell along heading (costs cost_move)
  "aim": 0-5,                     // rotate gun toward this heading, one step/tick (≈free)
  "fire": 0.0-3.0 }               // 0 = hold; else fire a shell (costs `fire` energy; needs gun_ready)
```
Components optional. A cheap default controller fills any tank an agent didn't command
(the role the CA played in Automata).

## Resolution (server referee, per tick)
1. apply turns, then moves (move conflicts → one random winner; blocked move → no-op, refunded)
2. rotate guns toward `aim` (one step/tick)
3. fires: each `fire>0` with `gun_ready` → spend `power` energy, spawn shell at the tank's
   cell along `gun_heading`, set `gun_cooldown`
4. advance shells (~2 cells/tick). Shell entering a tank cell → hit: `target.energy -=
   4·power`; `shooter.energy += 3·power` (Robocode-style reward). Shell into wall/edge → gone.
   Friendly fire: **on** (so positioning/coordination matters).
5. passive: `energy += 0.3` up to cap; `gun_cooldown -= 1`
6. death: `energy ≤ 0` → destroyed
7. win check: last team standing (or a score timer) → match ends → reset arena + **clear
   each team's Artel project** → next match

## Energy economy (proposed, tunable)
`cost_move 0.5`, turn/aim free, `fire = power` (0.1–3), `hit damage = 4·power`,
`hit reward = 3·power`, `passive regen 0.3/tick`, `start 100`. A shot is an *investment*
that only pays if it lands — so aiming, and **shared targeting via Artel**, is what wins.

## Teams & Artel — the point
- Each team has its **own Artel project** = its shared battlefield memory + comms. Tanks
  write sightings ("enemy #7 at (15,31), energy 40, heading 1"), read the **pooled map**
  (union of all teammates' sightings, persisted across ticks even as tanks die), and call
  focus-fire ("all target #7").
- A team's map is the union of its tanks' sensor ranges over time — far more than any one
  tank sees. Enemy teams have their own projects; you can't read theirs (fog of war at the
  team level too).
- **Win condition:** a coordinated team maps the field and converges fire → picks the enemy
  apart. An Artel-off team sees only each tank's local range and fires independently → loses.

## Multiplayer
The shape that makes Artel indispensable: **one player pilots one tank; tanks are grouped
into teams; the only way teammates can coordinate is the team's Artel project.**

- **Open persistent arena** (reuses Automata's single shared resettable world). Always live;
  join/leave anytime; resets between matches.
- **Bring your own agent.** A player connects an HTTP/MCP agent (Claude Code, a script,
  anything) that pilots their tank via the same `perceive`/`intend` contract. **Zero LLM cost
  to us for players** — they pay for their own brain.
- **Join flow** (reuses `/join` + tokens): `POST /join {agent_id, team?}` → you're placed on a
  team (open slot or balanced), handed a tank + a secret token + your team's Artel project name.
  Then you `connect to Artel`, join that project, and pilot via `/team/{id}/perceive|intend`.
  Solo mode (`/join?solo=1`) hands you a whole team to run yourself.
- **Artel is the team's nervous system.** With several independent tank-pilots on a team, the
  *only* channel to act as a unit — callouts ("enemy left, low energy"), focus-fire, formation,
  "I've got the ridge" — is the shared project. No Artel = teammates shoot blind, collide, and
  get picked off (friendly fire + move conflicts punish it). So multiplayer literally runs on
  Artel; that's the demo, not a bolt-on.
- **House tanks** fill empty slots (cost-bounded AI, below) so the arena is always populated and
  newcomers always have a fight.
- **Scales for free.** More players = more BYO agents = flat cost to us. Popularity doesn't cost
  us money — it just means more agents coordinating through Artel (which is the whole point).
- **Server-authoritative.** Tick ~1–3s; agents poll `perceive` and submit `intend` per tick; the
  server referees (no client trust). Slow tick = agent-paced (LLMs have time to think). Spectators
  get the WebSocket stream.

## AI control (cost-bounded)
- Per team, the LLM sets a **plan** every ~20–30s from {its Artel map + current sensors}:
  target priority, formation (advance/hold/retreat), aggression, retreat-energy threshold.
  It writes key intel back to the Artel project.
- A **free executor** runs the plan every tick: aim at the priority target (from the Artel
  map even when out of a tank's own sensor range), fire when aligned + `gun_ready`, move per
  formation.
- Cost: few teams × infrequent re-planning × only-while-watched + a **hard monthly $20 spend
  cap** (fall back to executor / last plan past the cap). ≈ $0.7/hr watched, $0 idle.

## Viz (re-skin)
- tanks as heading-arrows in team colors, a gun-direction tick, an energy ring; shells as
  fast dots; walls. Header: teams, tanks alive, match clock.
- pin a tank → its sensors **plus its team's Artel map, recent messages, and current plan** —
  Artel shown literally as the brain.
- reuse legend / guide / connect modal; per-team "Artel: on/off" indicator.

## The demo (the "got it")
Two house teams: one coordinates through its Artel project, one doesn't. The coordinated team
converges fire and wins, repeatedly and visibly. "Connect to Artel" lets a viewer drop in and
pilot/coordinate a team. Toggle a team's Artel and watch it fall apart.

## Decisions to confirm before building
1. Bounded arena + obstacles (vs toric)? — proposed: bounded.
2. Discrete 6-direction v1 (vs continuous angles later)? — proposed: discrete.
3. **Per-team Artel projects** (vs one shared project)? — proposed: per-team (your "each tribe
   has a project and competes").
4. Friendly fire on? — proposed: on.
5. Keep Automata as a second world, or retire it and make this the flagship?
