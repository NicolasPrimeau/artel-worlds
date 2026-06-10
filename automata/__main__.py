import argparse

from .config import DEFAULT
from .tick import step
from .world import World


def main():
    ap = argparse.ArgumentParser(description="Headless run — the 'is it alive?' gate.")
    ap.add_argument("--ticks", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--every", type=int, default=25)
    args = ap.parse_args()

    world = World(DEFAULT, seed=args.seed)
    world.seed(DEFAULT.initial_population)
    print(f"{'tick':>6} {'pop':>5} {'lineages':>9} {'avg_E':>7} {'avg_tox':>8} {'deaths':>7}")
    for _ in range(args.ticks):
        s = step(world)
        if s["tick"] % args.every == 0 or s["population"] == 0:
            print(
                f"{s['tick']:>6} {s['population']:>5} {s['lineages']:>9} "
                f"{s['avg_energy']:>7} {s['avg_toxin']:>8} {s['deaths']:>7}"
            )
        if s["population"] == 0:
            print("extinction")
            break


if __name__ == "__main__":
    main()
