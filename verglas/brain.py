from __future__ import annotations

from collections import Counter, defaultdict

# Deterministic meeting brains — the A/B baseline and the floor the LLM has to beat. A "decider" maps
# the board + meeting to each living agent's vote (target id, or -1 to skip).
#
# Three regimes, increasingly honest:
#   share=False           — each agent votes only on its own slice (no Artel conduit). The floor.
#   share=True, doubt=False— testimony is pooled and BLINDLY trusted; the most-accused agent is convicted.
#                            Overstates the crew: real tables don't trust like this.
#   share=True, doubt=True — pooled AND adversarial: the impostor lies (denies, counter-accuses its own
#                            witness), and crew apply doubt — a lone uncorroborated claim won't convict,
#                            and an accuser who is themselves suspected gets discounted. Lies land,
#                            wrong ejections happen, careful impostors survive. This is what a fixed rule
#                            can do; reasoning past the lie is what the LLM is for.

MIN_CONVICT = 2.0  # an accusation must clear this credibility to overcome reasonable doubt
LEAD_MARGIN = 1.0  # ...and lead the next suspect by this much, else the table skips


def _victim_companions(game, mt):
    # circumstantial "last seen with the victim" testimony, pooled across everyone who saw them
    c: Counter = Counter()
    if mt.victim is None:
        return c
    v = game.by_id(mt.victim)
    last = max((s.tick for s in v.seen), default=0)
    witnesses = [v] + [a for a in game.living(impostor=False)]
    for a in witnesses:
        for s in a.seen:
            if (a is v or mt.victim in s.present) and s.tick >= last - 1:
                for other in s.present:
                    if other != mt.victim:
                        c[other] += 1
    return c


def _skip_all(living):
    return {a.id: -1 for a in living}


def make_decider(share: bool, doubt: bool = True):
    def decide(game, mt):
        living = game.living()

        if not share:
            # SILOED: each votes only on what IT alone saw; the impostor casts a blind deflection.
            votes = {}
            for a in living:
                if not a.impostor and a.witnessed:
                    seen = [i for i in a.witnessed if game.by_id(i).alive]
                    votes[a.id] = seen[0] if seen else -1
                elif a.impostor:
                    targets = [c.id for c in living if not c.impostor]
                    votes[a.id] = game.rng.choice(targets) if targets else -1
                else:
                    votes[a.id] = -1
            return votes

        # POOLED. Gather every accusation made to the table: who is accusing whom, and how hard.
        accusers: dict[int, list[int]] = defaultdict(list)  # target -> [accuser ids]
        for c in game.living(impostor=False):
            for imp in c.witnessed:  # truthful eyewitness: "I saw <imp> do it"
                if game.by_id(imp).alive:
                    accusers[imp].append(c.id)
        companions = _victim_companions(game, mt)

        # the impostor speaks too: it denies and counter-accuses — preferably whoever fingered it.
        for m in game.living(impostor=True):
            my_witnesses = [c.id for c in game.living(impostor=False) if m.id in c.witnessed]
            if my_witnesses:
                tgt = game.rng.choice(my_witnesses)  # discredit the witness ("they're covering")
            else:
                innocents = [c.id for c in living if not c.impostor]
                tgt = game.rng.choice(innocents) if innocents else None
            if tgt is not None:
                accusers[tgt].append(m.id)

        if not accusers and not companions:
            return _skip_all(living)

        if not doubt:
            # BLIND TRUST: convict whoever is named most (+ circumstantial). Overstates the crew.
            score: Counter = Counter()
            for t, acc in accusers.items():
                score[t] += 3 * len(set(acc))
            for cid, n in companions.items():
                if game.by_id(cid).alive:
                    score[cid] += n
            suspect = max(score, key=score.get)
            return _vote_bloc(game, living, suspect)

        # DOUBT: an accusation's weight depends on the accuser's own credibility. An accuser who is
        # themselves heavily accused (the impostor, once suspected) is discounted; corroboration and
        # physical consistency with where the body was add weight. Single hearsay doesn't convict.
        raw_susp = {t: len(set(a)) for t, a in accusers.items()}
        score: dict[int, float] = {}
        for t, acc in accusers.items():
            s = 0.0
            for a in set(acc):
                s += 0.5 if raw_susp.get(a, 0) >= 2 else 1.0  # discount a suspected accuser
            if companions.get(t):  # physical consistency: also placed near the victim
                s += min(companions[t], 2) * 0.5
            score[t] = s
        for cid, n in companions.items():  # circumstantial-only suspects carry little weight
            if cid not in score and game.by_id(cid).alive:
                score[cid] = min(n, 2) * 0.5

        if not score:
            return _skip_all(living)
        suspect = max(score, key=score.get)
        top = score[suspect]
        second = max((v for k, v in score.items() if k != suspect), default=0.0)
        if top < MIN_CONVICT or top - second < LEAD_MARGIN:
            return _skip_all(living)  # reasonable doubt: nobody is going out on this
        return _vote_bloc(game, living, suspect)

    return decide


def _vote_bloc(game, living, suspect):
    votes = {}
    for a in living:
        votes[a.id] = a.id if (a.impostor and a.id == suspect) else suspect
    return votes
