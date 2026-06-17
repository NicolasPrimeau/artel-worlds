from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from .config import DEFAULT


# A single-elimination World Cup over 16 clubs: round of 16 -> quarter-finals -> semi-finals ->
# (third place) -> final. The engine plays one tie at a time; the tournament records, advances winners
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
    clubs: list[str] = field(default_factory=list)
    edition: int = 1
    seed: int = 0
    rounds: list[list[Tie]] = field(default_factory=list)
    rosters: dict[str, list[str]] = field(default_factory=dict)  # club -> surnames (team_size each)
    scorers: dict[str, dict] = field(default_factory=dict)
    order: list[tuple[int, int]] = field(default_factory=list)  # ties in play order
    cur: int = 0  # index into order
    champion: str | None = None
    artel_clubs: set[str] = field(default_factory=set)  # the coordinated (Artel-coached) entrants
    _rng: Random = field(default_factory=lambda: Random(0))

    def __post_init__(self) -> None:
        self._rng = Random(f"cup:{self.seed}:{self.edition}")
        self._draw()

    def _draw(self) -> None:
        # 16 club names, each a DISTINCT prefix (no two same-city/club sides in one cup) + a distinct
        # AI/ML suffix — so no team name ever repeats within an edition. Fresh field each tournament.
        pre = self._rng.sample(CLUB_PREFIXES, 16)
        suf = self._rng.sample(AI_SUFFIXES, 16)
        c = [f"{pre[i]} {suf[i]}" for i in range(16)]
        self.clubs = c
        # no Artel teams in pitch — every side runs the plain deterministic brain (pitch is a
        # spectacle, not an Artel demo; coordination code stays but never activates)
        self.artel_clubs: set[str] = set()
        # one shared pool of DISTINCT surnames dealt out across the squads — a surname is never on
        # two clubs in the same cup. Draws team_size * 16 unique names, then chunks them per club.
        squad = DEFAULT.team_size
        names = self._rng.sample(NAME_POOL, squad * 16)
        for i, club in enumerate(c):
            self.rosters[club] = names[i * squad : (i + 1) * squad]
        # 16-team single-elimination: Round of 16 -> quarter-finals -> semi-finals -> (third) -> final
        r16 = [Tie("Round of 16", i, c[2 * i], c[2 * i + 1]) for i in range(8)]
        qf = [Tie("Quarter-final", i, a_from=(0, 2 * i), b_from=(0, 2 * i + 1)) for i in range(4)]
        sf = [Tie("Semi-final", i, a_from=(1, 2 * i), b_from=(1, 2 * i + 1)) for i in range(2)]
        third = Tie("Third place", 0, a_from=(2, 0), b_from=(2, 1), a_loser=True, b_loser=True)
        final = Tie("Final", 0, a_from=(2, 0), b_from=(2, 1))
        self.rounds = [r16, qf, sf, [third], [final]]
        self.order = (
            [(0, i) for i in range(8)]
            + [(1, i) for i in range(4)]
            + [(2, 0), (2, 1)]
            + [(3, 0), (4, 0)]
        )

    def tie_at(self, idx: tuple[int, int]) -> Tie:
        return self.rounds[idx[0]][idx[1]]

    def current(self) -> Tie | None:
        return self.tie_at(self.order[self.cur]) if self.cur < len(self.order) else None

    def roster_names(self, club: str) -> list[str]:
        return list(self.rosters[club])

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
            self.champion = self.rounds[-1][0].winner  # the final is the last round

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


# Club name = one CLUB_PREFIX + one AI_SUFFIX, both unique within an edition. The joke only lands
# when the prefix already READS as football — a famous club word (Real, Inter, Borussia) or a city
# that instantly names a club (Manchester, Dortmund, Napoli) — so a generic world city ("Jakarta
# Quantizers") never gets paired. Suffixes are the punchy, recognisable AI/ML terms. The curated
# pairing gives a few hundred combinations, every one of them a real-sounding side.
CLUB_PREFIXES = [
    "Montréal",
    "Manchester",
    "Madrid",
    "Barcelona",
    "Liverpool",
    "Milan",
    "Turin",
    "Napoli",
    "Roma",
    "Munich",
    "Dortmund",
    "Leverkusen",
    "Paris",
    "Marseille",
    "Lisbon",
    "Porto",
    "Amsterdam",
    "Glasgow",
    "Real",
    "Inter",
    "Atlético",
    "Sporting",
    "Athletic",
    "Olympique",
    "Bayern",
    "Bayer",
    "Borussia",
    "Dynamo",
    "Ajax",
    "Benfica",
    "Celtic",
    "Boca",
    "River",
]
AI_SUFFIXES = [
    "Latency",
    "Tensor",
    "Gradient",
    "Softmax",
    "Neural",
    "Vector",
    "Overflow",
    "Backprop",
    "ReLU",
    "Dropout",
    "Epoch",
    "Pooling",
    "Bias",
    "Matrix",
    "Embedding",
    "Kernel",
    "Entropy",
    "Activation",
    "Attention",
    "Inference",
    "Variance",
    "Momentum",
]


# Player surnames — recognisable names from the great soccer nations so the Golden Boot reads like
# real stars, plus footballer x ML pun ringers and a few Québécois names for the Montréal angle. An
# edition fields 16 squads of nine = 144 players and NO surname is used twice in a cup, so the pool
# has to clear 144; sized just above (~170) it draws 144 each tournament with ~30 sitting out and
# rotating — familiar faces carry an edition-to-edition Golden Boot story without the field ever
# being identical.
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
    "Costa",
    "Pereira",
    "Fernandes",
    "Leão",
    "Cancelo",
    "Carvalho",
    "Casemiro",
    "Marquinhos",
    "Rodrygo",
    "Vinícius",
    "Richarlison",
    "Jesus",
    "Alisson",
    "Ederson",
    "Neves",
    "Dias",
    "Bernardo",
    "Jota",
    "Félix",
    # Argentina / Uruguay / Chile / Mexico
    "Fernández",
    "González",
    "Rodríguez",
    "Martínez",
    "Álvarez",
    "Suárez",
    "Cavani",
    "Forlán",
    "Jiménez",
    "Lozano",
    "Vela",
    "Di María",
    "Otamendi",
    "Paredes",
    "Mac Allister",
    "Dybala",
    "Molina",
    "Valverde",
    "Núñez",
    # Spain
    "García",
    "Torres",
    "Busquets",
    "Morata",
    "Rodri",
    "Olmo",
    "Gavi",
    "Pedri",
    "Carvajal",
    "Laporte",
    "Llorente",
    "Koke",
    "Merino",
    "Ferran",
    "Williams",
    # Italy
    "Rossi",
    "Esposito",
    "Verratti",
    "Barella",
    "Chiesa",
    "Donnarumma",
    "Bonucci",
    "Chiellini",
    "Jorginho",
    "Tonali",
    "Locatelli",
    "Immobile",
    "Insigne",
    "Bastoni",
    # England
    "Kane",
    "Saka",
    "Foden",
    "Rice",
    "Stones",
    "Bellingham",
    "Walker",
    "Sterling",
    "Grealish",
    "Mount",
    "Henderson",
    "Maguire",
    "Pickford",
    "Sancho",
    "Watkins",
    "Palmer",
    # France
    "Giroud",
    "Kanté",
    "Dembélé",
    "Griezmann",
    "Tchouaméni",
    "Benzema",
    "Pogba",
    "Varane",
    "Coman",
    "Thuram",
    "Camavinga",
    "Saliba",
    "Konaté",
    "Upamecano",
    "Maignan",
    # Germany
    "Müller",
    "Werner",
    "Kroos",
    "Havertz",
    "Wirtz",
    "Neuer",
    "Rüdiger",
    "Gnabry",
    "Sané",
    "Goretzka",
    "Gündoğan",
    "Musiala",
    "Kimmich",
    # Netherlands
    "de Jong",
    "van Dijk",
    "Depay",
    "Gakpo",
    "Frimpong",
    "Malen",
    "Koopmeiners",
    "Timber",
    "de Vrij",
    "de Ligt",
    "Dumfries",
    # Africa
    "Touré",
    "Adeyemi",
    "Mahrez",
    "Osimhen",
    "Koulibaly",
    "Mané",
    "Salah",
    "Hakimi",
    "Aubameyang",
    "Partey",
    "Kudus",
    "Ndidi",
    "Ziyech",
    "Onana",
    # East Asia
    "Tanaka",
    "Nakamura",
    "Mitoma",
    "Son",
    "Kim",
    "Lee",
    "Park",
    "Kubo",
    "Endo",
    "Doan",
    "Tomiyasu",
    # Balkans / Scandinavia
    "Modrić",
    "Vlahović",
    "Haaland",
    "Ødegaard",
    "Eriksen",
    "Larsson",
    "Isak",
    "Kovačić",
    "Brozović",
    "Perišić",
    "Gvardiol",
    "Kramarić",
    "Højlund",
    "Schmeichel",
    # Québec / Montréal
    "Tremblay",
    "Roy",
    "Gagné",
    "Bélanger",
]
