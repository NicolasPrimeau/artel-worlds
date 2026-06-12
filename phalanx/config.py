from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Arena: a HEXAGON-shaped axial hex grid — every cell within map_radius of the center
    # cell (map_radius, map_radius). Kept small so the teams are forced into contact.
    map_radius: int = 7  # 169 cells
    obstacle_density: float = 0.11  # fraction of cells that are cover/walls

    @property
    def width(self) -> int:  # bounding box extent, for layouts and iteration
        return 2 * self.map_radius + 1

    @property
    def height(self) -> int:
        return 2 * self.map_radius + 1

    # Teams
    house_teams: int = 2
    team_size: int = 3  # tanks per team

    # Tank energy economy (energy = HP + fuel; 0 = destroyed). No regen: every hit
    # is permanent, so fights resolve by attrition instead of stalemating.
    start_energy: int = 80
    max_energy: int = 90
    cost_move: float = 0.0  # mobility is life: a wounded tank must always be able to maneuver
    repair: float = (
        2.0  # recovered per turn spent holding still, not firing, untouched, in the zone
    )

    # Guns — fire AT a target at a chosen POWER; a shot lands if the target is within
    # that power's range with line of sight. Power buys reach, not damage: long-range
    # poking is expensive, closing in is cheap — and the fight is still decided by WHO
    # you shoot, which is exactly the coordination (focus-fire) story.
    power_range: tuple = (3, 5, 7)  # max hex distance for power 1 / 2 / 3
    power_cost: tuple = (0.0, 2.0, 4.0)  # base shots are FREE; energy buys reach
    shot_damage: float = 12.0

    @property
    def fire_range(self) -> int:  # absolute max reach (highest power)
        return self.power_range[-1]

    @property
    def shot_cost(self) -> float:  # default-power cost, for back-compat math
        return self.power_cost[1]

    hit_reward: float = 2.0  # small sustain for landing hits (rewards focused fire)
    gun_cooldown: int = (
        1  # a ready gun fires every tick — anything slower reads as idle at 2.5s/tick
    )
    sensor_range: int = 8  # how far a tank sees (hex distance)
    friendly_fire: bool = False

    # Closing zone: a safe region around the center that shrinks over the match,
    # herding the teams together so fights actually happen (and matches resolve by
    # combat, not stalemate). Tanks caught outside it bleed energy.
    zone_start: int = 26  # tick the safe zone begins to close (1s ticks: fights develop first)
    zone_close: int = 60  # tick it reaches its minimum radius
    zone_min: int = 2  # minimum safe radius, in hexes from center
    zone_damage: float = 6.0  # energy lost per tick spent outside the zone
    match_max_ticks: int = 95  # hard cap — force a result so a match can never stall forever


DEFAULT = Config()
