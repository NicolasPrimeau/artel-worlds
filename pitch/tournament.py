from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

ROLES = ["GK", "LB", "RB", "CM", "ST"]


# A single-elimination World Cup over 8 clubs: quarter-finals -> semi-finals -> (third place) ->
# final. The engine plays one tie at a time; the tournament records the result, advances winners
# along the bracket, and keeps the running Golden Boot. Rosters are fixed for the edition so a
# player's goals accumulate across the rounds (the parasocial layer). When the final is decided a
# champion is crowned and a fresh edition is drawn.
@dataclass
class Tie:
    rnd: str
    slot: int
    a: str | None = None  # club, or None until a feeder is decided
    b: str | None = None
    sa: int = 0
    sb: int = 0
    pa: int | None = None  # shoot-out scores when a knockout tie is level
    pb: int | None = None
    winner: str | None = None
    played: bool = False
    a_from: tuple[int, int] | None = None  # (round, slot) whose winner fills slot a
    b_from: tuple[int, int] | None = None
    a_loser: bool = False  # a_from feeds the LOSER (third-place playoff)
    b_loser: bool = False


@dataclass
class Tournament:
    clubs: list[str]
    edition: int = 1
    seed: int = 0
    rounds: list[list[Tie]] = field(default_factory=list)
    rosters: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    scorers: dict[str, dict] = field(default_factory=dict)
    order: list[tuple[int, int]] = field(default_factory=list)  # ties in play order
    cur: int = 0  # index into order
    champion: str | None = None
    _rng: Random = field(default_factory=lambda: Random(0))

    def __post_init__(self) -> None:
        self._rng = Random(f"cup:{self.seed}:{self.edition}")
        self._draw()

    def _draw(self) -> None:
        c = list(self.clubs)
        self._rng.shuffle(c)
        c = c[:8]
        names = list(NAME_POOL)
        self._rng.shuffle(names)
        for i, club in enumerate(c):
            self.rosters[club] = [(ROLES[k], names[i * 5 + k]) for k in range(5)]
        qf = [Tie("Quarter-final", i, c[2 * i], c[2 * i + 1]) for i in range(4)]
        sf = [
            Tie("Semi-final", 0, a_from=(0, 0), b_from=(0, 1)),
            Tie("Semi-final", 1, a_from=(0, 2), b_from=(0, 3)),
        ]
        third = Tie("Third place", 0, a_from=(1, 0), b_from=(1, 1), a_loser=True, b_loser=True)
        final = Tie("Final", 0, a_from=(1, 0), b_from=(1, 1))
        self.rounds = [qf, sf, [third], [final]]
        self.order = [(0, i) for i in range(4)] + [(1, 0), (1, 1)] + [(2, 0), (3, 0)]

    def tie_at(self, idx: tuple[int, int]) -> Tie:
        return self.rounds[idx[0]][idx[1]]

    def current(self) -> Tie | None:
        return self.tie_at(self.order[self.cur]) if self.cur < len(self.order) else None

    def roster_names(self, club: str) -> list[str]:
        return [f"{role} {name}" for role, name in self.rosters[club]]

    def record_goal(self, club: str, name: str) -> None:
        key = f"{club}|{name}"
        row = self.scorers.get(key)
        if row is None:
            self.scorers[key] = {"name": name, "club": club, "goals": 1}
        else:
            row["goals"] += 1

    def record_result(self, sa: int, sb: int) -> None:
        tie = self.current()
        if tie is None:
            return
        tie.sa, tie.sb, tie.played = sa, sb, True
        if sa == sb:  # knockout can't end level — settle on penalties
            pa, pb = self._rng.randint(3, 5), self._rng.randint(3, 5)
            while pa == pb:
                pb = self._rng.randint(2, 6)
            tie.pa, tie.pb = pa, pb
            tie.winner = tie.a if pa > pb else tie.b
        else:
            tie.winner = tie.a if sa > sb else tie.b
        loser = tie.b if tie.winner == tie.a else tie.a
        self._propagate(self.order[self.cur], tie.winner, loser)
        self.cur += 1
        nxt = self.current()
        if nxt is None:
            final = self.rounds[3][0]
            self.champion = final.winner

    def _propagate(self, src: tuple[int, int], winner: str | None, loser: str | None) -> None:
        for rnd in self.rounds:
            for tie in rnd:
                if tie.a_from == src:
                    tie.a = loser if tie.a_loser else winner
                if tie.b_from == src:
                    tie.b = loser if tie.b_loser else winner

    def standings(self) -> list[dict]:
        rows: dict[str, dict] = {}
        for rnd in self.rounds:
            for tie in rnd:
                if not tie.played:
                    continue
                for club, gf, ga in ((tie.a, tie.sa, tie.sb), (tie.b, tie.sb, tie.sa)):
                    if club is None:
                        continue
                    r = rows.setdefault(
                        club, {"club": club, "p": 0, "w": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0}
                    )
                    r["p"] += 1
                    r["gf"] += gf
                    r["ga"] += ga
                    won = tie.winner == club
                    r["w"] += 1 if won else 0
                    r["l"] += 0 if won else 1
                    r["pts"] += 3 if won else 0
        return sorted(rows.values(), key=lambda r: (-r["pts"], -(r["gf"] - r["ga"]), -r["gf"]))


# Player surnames drawn from the great soccer nations — Brazil, Argentina, Spain, Italy, England,
# France, Germany, Portugal, the Netherlands, Africa, East Asia, the Balkans, Scandinavia, Mexico
# — plus Québécois names for the Montréal angle, and a handful of footballer x machine-learning
# puns sprinkled in as star ringers. A big pool so each edition fields fresh, unique rosters.
NAME_POOL = [
    # footballer/ML pun ringers
    "Embappé",
    "Maradata",
    "Neuralmar",
    "Zidata",
    "Inferensta",
    "Overfittipaldi",
    # Brazil / Portugal
    "Silva",
    "Santos",
    "Souza",
    "Oliveira",
    "Costa",
    "Pereira",
    "Ribeiro",
    "Gomes",
    "Fernandes",
    "Carvalho",
    "Ferreira",
    "Mendes",
    "Leão",
    "Cancelo",
    # Argentina / Uruguay / Chile / Mexico
    "Fernández",
    "González",
    "Rodríguez",
    "Martínez",
    "López",
    "Álvarez",
    "Suárez",
    "Romero",
    "Cavani",
    "Forlán",
    "Jiménez",
    "Lozano",
    "Vela",
    "Reyes",
    "Vargas",
    # Spain
    "García",
    "Hernández",
    "Torres",
    "Busquets",
    "Morata",
    "Rodri",
    "Olmo",
    "Gavi",
    # Italy
    "Rossi",
    "Esposito",
    "Greco",
    "Ferrari",
    "Romano",
    "Verratti",
    "Barella",
    "Chiesa",
    # England
    "Smith",
    "Walker",
    "Kane",
    "Saka",
    "Foden",
    "Rice",
    "Stones",
    "Bellingham",
    # France
    "Dubois",
    "Moreau",
    "Giroud",
    "Kanté",
    "Dembélé",
    "Griezmann",
    "Tchouaméni",
    # Germany
    "Müller",
    "Schmidt",
    "Wagner",
    "Werner",
    "Kroos",
    "Havertz",
    "Wirtz",
    # Netherlands
    "de Jong",
    "van Dijk",
    "Depay",
    "Gakpo",
    "Frimpong",
    # Africa
    "Okafor",
    "Adeyemi",
    "Mensah",
    "Diallo",
    "Touré",
    "Hassan",
    "Mahrez",
    "Osimhen",
    "Koulibaly",
    "Mané",
    "Salah",
    "Hakimi",
    # East Asia
    "Tanaka",
    "Yamamoto",
    "Nakamura",
    "Sato",
    "Mitoma",
    "Son",
    "Kim",
    "Lee",
    "Park",
    "Kang",
    # Balkans / Scandinavia
    "Novak",
    "Petrović",
    "Modrić",
    "Vlahović",
    "Haaland",
    "Ødegaard",
    "Eriksen",
    "Larsson",
    "Isak",
    # Québec / Montréal
    "Tremblay",
    "Gagné",
    "Roy",
    "Bouchard",
    "Lévesque",
    "Côté",
    "Gauthier",
    "Pelletier",
    "Bélanger",
    "Lavoie",
    "Bergeron",
    "Fortin",
]
