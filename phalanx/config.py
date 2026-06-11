from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Arena (bounded axial hex grid, 6 directions). Kept small so the teams are
    # forced into contact — no running off to a corner to heal in peace.
    width: int = 16  # q extent
    height: int = 12  # r extent
    obstacle_density: float = 0.08  # fraction of cells that are cover/walls

    # Teams
    house_teams: int = 2
    team_size: int = 3  # tanks per team

    # Tank energy economy (energy = HP + fuel; 0 = destroyed). No regen: every hit
    # is permanent, so fights resolve by attrition instead of stalemating.
    start_energy: int = 80
    max_energy: int = 90
    cost_move: float = 0.4
    regen: float = 0.0  # damage sticks

    # Guns — fire AT a target; a shot lands if it's in range with line of sight.
    # Reliable hits mean the fight is decided by WHO you shoot, not whether you
    # can line up a ray — which is exactly the coordination (focus-fire) story.
    fire_range: int = 6  # max hex distance a shot carries
    shot_cost: float = 2.0  # energy spent per shot
    shot_damage: float = 12.0
    hit_reward: float = 2.0  # small sustain for landing hits (rewards focused fire)
    gun_cooldown: int = 2  # ticks between shots
    sensor_range: int = 8  # how far a tank sees (hex distance)
    friendly_fire: bool = False

    # Closing zone: a safe region around the center that shrinks over the match,
    # herding the teams together so fights actually happen (and matches resolve by
    # combat, not stalemate). Tanks caught outside it bleed energy.
    zone_start: int = 22  # tick the safe zone begins to close
    zone_close: int = 80  # tick it reaches its minimum radius
    zone_min: int = 2  # minimum safe radius, in hexes from center
    zone_damage: float = 4.0  # energy lost per tick spent outside the zone


DEFAULT = Config()
