"""Rank decision graphs by Elo from round-robin deathmatch duels."""

import math
import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from itertools import combinations
from statistics import fmean

from decision_compose.doom import DEATHMATCH, DoomEnvironment, DoomTask, Episode, Episodes
from decision_compose.optimize import Outcome
from decision_compose.runtime import DecisionProgram

# Elo's conventions: 400 points is tenfold odds, and the field averages 1500.
ELO_SCALE = 400
ELO_MEAN = 1500
CONVERGED = 1e-10


@dataclass(frozen=True)
class Entrant:
    name: str
    program: DecisionProgram


def free_port() -> int:
    """A UDP port nothing is using, for one game's host to listen on."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def play(
    task: DoomTask, program: DecisionProgram, episode: int | Episode, game_args: str = ""
) -> Outcome:
    """One seat's episode. A seat whose graph fails leaves the game, and its opponent plays on."""
    with DoomEnvironment(task, game_args=game_args) as env:
        return Episodes(task, env)(program, episode)


def duel(
    task: DoomTask,
    a: DecisionProgram,
    b: DecisionProgram,
    episode: int | Episode,
    host_first: bool,
) -> list[Outcome]:
    """Both seats' episodes of one deathmatch, a's first, each graph in its own networked game."""
    port = free_port()
    order = [a, b] if host_first else [b, a]
    args = [f"-host 2 -port {port} {DEATHMATCH}", f"-join 127.0.0.1 -port {port}"]
    # The host waits at init until its opponent joins, so both seats start together.
    with ThreadPoolExecutor(2) as pool:
        futures = [
            pool.submit(play, task, program, episode, arg)
            for program, arg in zip(order, args, strict=True)
        ]
    seats = [future.result() for future in futures]
    return seats if host_first else seats[::-1]


class Duels:
    """Scores a graph by its reward in a deathmatch against a fixed opponent per episode."""

    def __init__(self, task: DoomTask, opponent: DecisionProgram):
        self.task = task
        self.opponent = opponent

    def __call__(self, program: DecisionProgram, episode: int | Episode) -> Outcome:
        # Alternating hosts by seed shares the joiner's latency handicap across episodes.
        hosted = self.task.episode(episode).seed % 2 == 0
        mine, theirs = duel(self.task, program, self.opponent, episode, hosted)
        own, rival = mine.record["stats"]["frags"], theirs.record["stats"]["frags"]
        result = "won" if own > rival else "lost" if own < rival else "drew"
        feedback = f"{mine.trace['Feedback']} The opponent scored {rival} frags, so you {result}."
        return replace(
            mine,
            record=mine.record | {"hosted": hosted, "opponent_frags": rival, "result": result},
            trace=mine.trace | {"Feedback": feedback},
        )


def match(task: DoomTask, a: Entrant, b: Entrant, episode: int | Episode, host_first: bool) -> dict:
    """One deathmatch between two entrants, scored for the tournament."""
    seats = duel(task, a.program, b.program, episode, host_first)
    frags = [seat.record["stats"]["frags"] for seat in seats]
    failed = [seat.error is not None for seat in seats]
    if failed[0] != failed[1]:
        score = [float(failed[1]), float(failed[0])]
    elif frags[0] == frags[1]:
        score = [0.5, 0.5]
    else:
        score = [float(frags[0] > frags[1]), float(frags[1] > frags[0])]
    episode = task.episode(episode)
    return {
        "players": [a.name, b.name],
        "seed": episode.seed,
        "map": episode.map,
        "host": (a if host_first else b).name,
        "frags": frags,
        "score": score,
        "seats": [seat.record for seat in seats],
    }


def ratings(matches: list[dict], names: list[str]) -> dict[str, float]:
    """Bradley-Terry strengths fit to every result at once, so match order never matters,
    reported on the Elo scale. Each pair also shares one virtual draw, which keeps an unbeaten
    player's rating finite."""
    points = {(x, y): 0.5 for x in names for y in names if x != y}
    for record in matches:
        (x, y), (earned_x, earned_y) = record["players"], record["score"]
        points[x, y] += earned_x
        points[y, x] += earned_y
    strength = dict.fromkeys(names, 1.0)
    while True:
        # Hunter's minorization-maximization update, which converges to the maximum likelihood.
        updated = {
            x: sum(points[x, y] for y in names if y != x)
            / sum(
                (points[x, y] + points[y, x]) / (strength[x] + strength[y]) for y in names if y != x
            )
            for x in names
        }
        center = math.exp(fmean(math.log(value) for value in updated.values()))
        updated = {x: value / center for x, value in updated.items()}
        done = max(abs(updated[x] - strength[x]) for x in names) < CONVERGED
        strength = updated
        if done:
            break
    return {x: ELO_MEAN + ELO_SCALE * math.log10(strength[x]) for x in names}


def tournament(task: DoomTask, entrants: list[Entrant], episodes: list[int | Episode]) -> dict:
    """Every pair of entrants plays every episode, swapping hosts from one episode to the next."""
    matches = []
    for a, b in combinations(entrants, 2):
        for index, episode in enumerate(episodes):
            matches.append(match(task, a, b, episode, host_first=index % 2 == 0))
    names = [entrant.name for entrant in entrants]
    rated = ratings(matches, names)
    records = {
        name: {"wins": 0, "draws": 0, "losses": 0, "frags": 0, "frags_against": 0} for name in names
    }
    for record in matches:
        for side, name in enumerate(record["players"]):
            earned, own, rival = (
                record["score"][side],
                record["frags"][side],
                record["frags"][1 - side],
            )
            records[name]["wins" if earned == 1 else "draws" if earned == 0.5 else "losses"] += 1
            records[name]["frags"] += own
            records[name]["frags_against"] += rival
    standings = sorted(names, key=rated.get, reverse=True)
    return {
        "ratings": {name: round(rated[name], 1) for name in standings},
        "records": {name: records[name] for name in standings},
        "matches": matches,
    }
