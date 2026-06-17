from __future__ import annotations

from collections import Counter

# Deterministic meeting brains — the A/B baseline. A "decider" maps the board + meeting to each living
# agent's vote (target id, or -1 to skip). The whole experiment lives in one flag: can agents POOL what
# they saw (share=True, the Artel conduit) or does each vote on its own slice (share=False)? Pooling a
# single eyewitness account to the whole table convicts; siloed, that account is one vote drowned in
# "I didn't see anything" and the impostor walks. This is the load-bearing claim, made falsifiable.


def _victim_companions(game, mt):
    # who was seen with the victim near death — pooled "last seen with" testimony
    c: Counter = Counter()
    if mt.victim is None:
        return c
    v = game.by_id(mt.victim)
    last = max((s.tick for s in v.seen), default=0)
    for s in v.seen:
        if s.tick >= last - 1:
            for other in s.present:
                c[other] += 1
    for a in game.living(impostor=False):  # corroboration from survivors who saw the victim too
        for s in a.seen:
            if mt.victim in s.present and s.tick >= last - 1:
                for other in s.present:
                    if other != mt.victim:
                        c[other] += 1
    return c


def make_decider(share: bool):
    def decide(game, mt):
        living = game.living()
        if share:
            # POOLED: eyewitness accounts and "last seen with" testimony land on the whole table.
            score: Counter = Counter()
            for a in game.living(impostor=False):
                for imp in a.witnessed:
                    if game.by_id(imp).alive:
                        score[imp] += 3  # a direct account, told to everyone, is damning
            for cid, n in _victim_companions(game, mt).items():
                if game.by_id(cid).alive:
                    score[cid] += n
            if not score:
                return {a.id: -1 for a in living}  # nothing to go on → skip as a bloc
            suspect = max(score, key=score.get)
            votes = {}
            for a in living:
                if a.impostor:
                    votes[a.id] = (
                        a.id if a.id == suspect else suspect
                    )  # impostor can't dodge the bloc
                else:
                    votes[a.id] = suspect
            return votes
        # SILOED: each agent votes only on what IT alone saw; the impostor casts a deflection.
        votes = {}
        for a in living:
            if not a.impostor and a.witnessed:
                seen = [i for i in a.witnessed if game.by_id(i).alive]
                votes[a.id] = seen[0] if seen else -1
            elif a.impostor:
                targets = [c.id for c in living if not c.impostor]
                votes[a.id] = game.rng.choice(targets) if targets else -1
            else:
                votes[a.id] = -1  # saw nothing, abstain
        return votes

    return decide
