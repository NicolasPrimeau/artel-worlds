import secrets

SEATS = {"PHALANX": 6, "PITCH": 6, "WATCHTOWER": 6, "VERGLAS": 8, "VIBEQUEST": 1, "AUTOMATA": 1}

pairs = []
for world, count in SEATS.items():
    ids = [f"{world.lower()}-{i}" for i in range(1, count + 1)]
    keys = [secrets.token_urlsafe(32) for _ in ids]
    pairs += [f"{i}:{k}" for i, k in zip(ids, keys)]
    print(f"{world}_IDS={','.join(ids)}")
    print(f"{world}_KEYS={','.join(keys)}")
print(f"ALL={','.join(pairs)}")
