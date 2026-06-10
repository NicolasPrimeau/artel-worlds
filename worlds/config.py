from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Grid
    width: int = 60
    height: int = 60
    toric: bool = True

    # Seed population (house organisms)
    initial_population: int = 80

    # Energy economy (tuned 'lean_consume': leaner energy keeps density off the
    # grid ceiling so spatial dynamics stay visible; evolution still drifts toward
    # saturation over long runs, so periodic reset is the intended homeostat).
    birth_energy: int = 50
    cost_base: int = 1
    cost_migration: int = 3
    cost_division: int = 10
    consumption_max: int = 5
    gain_per_nutrient: int = 1

    # Nutrient field
    nutrient_max: int = 100
    nutrient_initial: int = 50
    nutrient_regrowth: int = 1

    # Toxin field
    toxin_emission: int = 6
    toxin_degradation: int = 1
    toxin_lethal: int = 50
    toxin_max: int = 100

    # Genome / mutation
    max_genes: int = 8
    p_point: float = 0.1
    p_add: float = 0.1
    p_del: float = 0.1
    p_dup: float = 0.1
    p_swap: float = 0.1
    point_delta: int = 10


DEFAULT = Config()
