from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Arena (bounded square grid, 8 compass directions)
    width: int = 32
    height: int = 24
    obstacle_density: float = 0.04  # fraction of cells that are cover/walls

    # Teams
    house_teams: int = 2
    team_size: int = 4  # tanks per team

    # Tank energy economy (energy = HP + fuel; 0 = destroyed)
    start_energy: int = 100
    max_energy: int = 120
    cost_move: float = 0.5
    regen: float = 0.3  # passive, per tick

    # Guns
    fire_min: float = 0.3
    fire_max: float = 3.0
    gun_cooldown: int = 2  # ticks between shots
    damage_per_power: float = 4.0
    reward_per_power: float = 3.0  # shooter regains this * power on a hit
    shell_speed: int = 2  # cells a shell travels per tick
    sensor_range: int = 7  # how far a tank sees (Chebyshev)
    friendly_fire: bool = True


DEFAULT = Config()
