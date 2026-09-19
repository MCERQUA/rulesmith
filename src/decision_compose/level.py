"""Static Doom map geometry from a WAD, for routing the player to a level's exit."""

import heapq
import math
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Doom's movement limits and line conventions, from the original engine.
MAX_STEP = 24
PLAYER_HEIGHT = 56
# The route grid: cell size, a cell's minimum distance from walls to count as free (under
# the player's 16-unit radius, so narrow passages stay open), the distance under which a cell
# costs WALL_PENALTY more, how close counts as at the exit switch, how far a waypoint leads,
# and how many cells out the player's own cell is searched for.
CELL = 16
CLEARANCE = 12
PLAYER_RADIUS = 16
CROWDED = 24
WALL_PENALTY = 1.0
EXIT_REACH = 48
LOOKAHEAD = 4
SEARCH = 3
# Spacing of floor samples when checking a straight walk for steps.
SAMPLE = 4
IMPASSABLE = 0x0001
MULTIPLAYER = 0x0010
SUBSECTOR = 0x8000
EXITS = {11, 52, 197}
LOCKED = {26, 27, 28, 32, 33, 34, 99, 133, 134, 135, 136, 137}


@dataclass(frozen=True)
class Line:
    start: tuple[float, float]
    end: tuple[float, float]
    flags: int
    special: int
    front: int
    back: int | None

    @property
    def middle(self) -> tuple[float, float]:
        return ((self.start[0] + self.end[0]) / 2, (self.start[1] + self.end[1]) / 2)


# The lumps that follow a map's marker in a WAD, in Doom's fixed order.
MAP_LUMPS = {
    "THINGS",
    "LINEDEFS",
    "SIDEDEFS",
    "VERTEXES",
    "SEGS",
    "SSECTORS",
    "NODES",
    "SECTORS",
    "REJECT",
    "BLOCKMAP",
}


def lumps(path: Path, map_name: str) -> dict[str, bytes]:
    data = path.read_bytes()
    _, count, directory = struct.unpack_from("<4sii", data, 0)
    entries = [struct.unpack_from("<ii8s", data, directory + 16 * i) for i in range(count)]
    names = [name.rstrip(b"\0").decode() for _, _, name in entries]
    if map_name not in names:
        raise ValueError(f"{path.name} has no map {map_name}")
    result = {}
    start = names.index(map_name) + 1
    for (position, size, _), name in zip(entries[start:], names[start:], strict=True):
        if name not in MAP_LUMPS or name in result:
            break
        result[name] = data[position : position + size]
    return result


def rows(lump: bytes, layout: str) -> list[tuple]:
    size = struct.calcsize(layout)
    return [struct.unpack_from(layout, lump, offset) for offset in range(0, len(lump), size)]


ACTOR = re.compile(r"^\s*actor\s+(\w+)(?:\s*:\s*(\w+))?", re.IGNORECASE | re.MULTILINE)
EDITOR_NUMBER = re.compile(r"^\s*(\d+)\s*=\s*(\w+)", re.MULTILINE)
# ZDoom's table of Doom editor numbers, used by Doom and Freedoom maps.
DOOM_NUMBERS = "mapinfo/doomitems.txt"


def solid_types(pk3: Path) -> dict[int, tuple[str, float]]:
    """Doom editor numbers of solid, stationary actors, with their names and collision radii,
    from the engine's actor definitions and editor-number table."""
    actors = {}
    with zipfile.ZipFile(pk3) as archive:
        numbers = {
            int(number): name.lower()
            for number, name in EDITOR_NUMBER.findall(archive.read(DOOM_NUMBERS).decode("latin-1"))
        }
        for path in archive.namelist():
            if not (path.startswith("actors/") and path.endswith(".txt")):
                continue
            text = archive.read(path).decode("latin-1").replace("\r\n", "\n")
            for match in ACTOR.finditer(text):
                body = text[match.end() : text.find("\n}", match.end())]
                radius = re.search(r"^\s*radius\s+(\d+)", body, re.IGNORECASE | re.MULTILINE)
                flags = {flag.lower() for flag in re.findall(r"([+-]\w+)", body)}
                monster = re.search(r"^\s*monster\s*$", body, re.IGNORECASE | re.MULTILINE)
                actors[match.group(1).lower()] = {
                    "name": match.group(1),
                    "parent": (match.group(2) or "").lower() or None,
                    "radius": float(radius.group(1)) if radius else None,
                    "solid": True if "+solid" in flags else False if "-solid" in flags else None,
                    "monster": True if monster or "+ismonster" in flags else None,
                }

    def inherited(actor: dict, key: str, default):
        while actor is not None:
            if actor[key] is not None:
                return actor[key]
            actor = actors.get(actor["parent"])
        return default

    solid = {}
    for number, name in numbers.items():
        actor = actors.get(name)
        if actor is None or inherited(actor, "monster", False):
            continue
        if inherited(actor, "solid", False):
            # Doom's base actor radius is 20 when no definition in the chain sets one.
            solid[number] = (actor["name"], inherited(actor, "radius", 20.0))
    return solid


class Level:
    """A map's geometry, with walking distance to its exit from anywhere on the map."""

    def __init__(self, path: Path, map_name: str):
        parts = lumps(path, map_name)
        vertexes = rows(parts["VERTEXES"], "<2h")
        sides = rows(parts["SIDEDEFS"], "<2h8s8s8sh")
        self.heights = [
            (floor, ceiling) for floor, ceiling, *_ in rows(parts["SECTORS"], "<2h8s8s3h")
        ]
        self.lines = [
            Line(
                vertexes[v1],
                vertexes[v2],
                flags,
                special,
                sides[right][5],
                sides[left][5] if left != -1 else None,
            )
            for v1, v2, flags, special, _, right, left in rows(parts["LINEDEFS"], "<7h")
        ]
        self.nodes = rows(parts["NODES"], "<4h8h2H")
        subsectors = rows(parts["SSECTORS"], "<2H")
        segs = rows(parts["SEGS"], "<6h")
        # A subsector's sector is the sector on the facing side of its first seg's line.
        self.subsectors = []
        for _, first in subsectors:
            _, _, _, linedef, direction, _ = segs[first]
            line = self.lines[linedef]
            self.subsectors.append(line.back if direction else line.front)
        self.solid_types = solid_types(path.parent / "vizdoom.pk3")
        # Things placed for multiplayer only are absent in single-player games.
        self.obstacles = [
            ((float(x), float(y)), self.solid_types[kind][1])
            for x, y, _, kind, flags in rows(parts["THINGS"], "<5h")
            if kind in self.solid_types and not flags & MULTIPLAYER
        ]
        self.exits = [line for line in self.lines if line.special in EXITS]
        if not self.exits:
            raise ValueError(f"{map_name} has no exit line")
        self.grid()

    def locate(self, x: float, y: float) -> int:
        """The sector containing the point, found by walking the map's BSP tree as Doom does."""
        child = len(self.nodes) - 1
        while not child & SUBSECTOR:
            node_x, node_y, dx, dy, *_, right, left = self.nodes[child]
            # Doom's R_PointOnSide: the right child unless the point is behind the partition.
            behind = (y - node_y) * dx >= dy * (x - node_x)
            child = left if behind else right
        return self.subsectors[child & ~SUBSECTOR]

    def grid(self) -> None:
        """Walking distance to the exit for every cell of a grid over the map.

        Opening-to-opening routes cut corners through walls wherever a sector is not convex,
        so the route is a distance field over free cells instead: a cell is free when the
        player can stand in its sector and it is clear of every wall."""
        points = np.array([line.start for line in self.lines] + [line.end for line in self.lines])
        self.origin = points.min(axis=0) - CELL
        self.shape = tuple(((points.max(axis=0) + CELL - self.origin) // CELL).astype(int) + 1)
        xs = self.origin[0] + (np.arange(self.shape[0]) + 0.5) * CELL
        ys = self.origin[1] + (np.arange(self.shape[1]) + 0.5) * CELL
        centers = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
        sectors = np.array([self.locate(x, y) for x, y in centers])
        floors = np.array([floor for floor, _ in self.heights])[sectors]
        gaps = np.array([ceiling - floor for floor, ceiling in self.heights])[sectors]
        doors = gaps <= 0
        locked = {
            sector
            for line in self.lines
            if line.special in LOCKED
            for sector in (line.front, line.back)
            if sector is not None and self.heights[sector][1] <= self.heights[sector][0]
        }
        walls = [line for line in self.lines if line.back is None or line.flags & IMPASSABLE]
        self.wall_starts = np.array([line.start for line in walls], dtype=float)
        self.wall_ends = np.array([line.end for line in walls], dtype=float)
        clearance = np.full(len(centers), np.inf)
        for line in walls:
            clearance = np.minimum(clearance, segment_distance(centers, line.start, line.end))
        for (x, y), radius in self.obstacles:
            clearance = np.minimum(clearance, np.hypot(*(centers - (x, y)).T) - radius)
        free = (
            ((gaps >= PLAYER_HEIGHT) | doors)
            & ~np.isin(sectors, list(locked))
            & (clearance >= CLEARANCE)
        )
        self.centers, self.floors, self.doors, self.free = centers, floors, doors, free
        # A thin ledge can sit between two cell centers, so each move also checks the sector
        # at its midpoint: floor heights there, and whether the player can stand there.
        midway = {}
        for di, dj, _ in STEPS:
            points = centers + np.array([di, dj]) * CELL / 2
            mids = np.array([self.locate(x, y) for x, y in points])
            mid_floor = np.array([floor for floor, _ in self.heights])[mids]
            mid_gap = np.array([ceiling - floor for floor, ceiling in self.heights])[mids]
            midway[di, dj] = (mid_floor, (mid_gap >= PLAYER_HEIGHT) | (mid_gap <= 0))
        # Hugging a wall snags on door frames and corners, so cells near one cost more.
        crowded = 1 + WALL_PENALTY * (clearance < CROWDED)
        self.distance = np.full(len(centers), np.inf)
        # The next cell on each cell's legal path to the exit; -1 for exit cells.
        self.toward = np.full(len(centers), -1)
        frontier = []
        for line in self.exits:
            near = segment_distance(centers, line.start, line.end)
            for cell in np.flatnonzero(free & (near <= EXIT_REACH)):
                self.distance[cell] = near[cell]
                frontier.append((near[cell], cell))
        heapq.heapify(frontier)
        width = self.shape[1]
        while frontier:
            length, cell = heapq.heappop(frontier)
            if length > self.distance[cell]:
                continue
            i, j = divmod(cell, width)
            for di, dj, step in STEPS:
                ni, nj = i + di, j + dj
                if not (0 <= ni < self.shape[0] and 0 <= nj < width):
                    continue
                other = ni * width + nj
                if not free[other]:
                    continue
                # The player walks from other into cell, in the opposite direction, and must
                # not climb more than Doom's step height on either half of the move.
                mid_floor, standable = midway[-di, -dj]
                if (
                    not standable[other]
                    or mid_floor[other] - floors[other] > MAX_STEP
                    or floors[cell] - mid_floor[other] > MAX_STEP
                ):
                    continue
                total = length + step * CELL * crowded[other]
                if total < self.distance[other]:
                    self.distance[other] = total
                    self.toward[other] = cell
                    heapq.heappush(frontier, (total, other))

    def cell(self, x: float, y: float) -> int | None:
        """The reachable cell nearest the point, searching a few cells out, since the player
        can stand closer to a wall than a free cell's clearance."""
        i0, j0 = ((np.array([x, y]) - self.origin) // CELL).astype(int)
        best = None
        for radius in range(SEARCH + 1):
            for i in range(i0 - radius, i0 + radius + 1):
                for j in range(j0 - radius, j0 + radius + 1):
                    if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
                        continue
                    cell = i * self.shape[1] + j
                    if np.isfinite(self.distance[cell]) and self.walkable(
                        (x, y), tuple(self.centers[cell])
                    ):
                        cost = self.distance[cell] + math.dist((x, y), self.centers[cell])
                        if best is None or cost < best[0]:
                            best = (cost, cell)
            if best is not None:
                return best[1]
        return None

    def walkable(self, start: tuple[float, float], end: tuple[float, float]) -> bool:
        """Whether walking straight from start to end crosses no wall and no step too high."""
        (ax, ay), (bx, by) = start, end
        direction = np.array([bx - ax, by - ay])
        edges = self.wall_ends - self.wall_starts
        offset = self.wall_starts - np.array(start)
        denominator = cross(direction, edges)
        with np.errstate(divide="ignore", invalid="ignore"):
            along_path = cross(offset, edges) / denominator
            along_wall = cross(offset, direction) / denominator
        hits = (denominator != 0) & (along_path > 0) & (along_path < 1)
        if (hits & (along_wall >= 0) & (along_wall <= 1)).any():
            return False
        # The player is a body, not a point: the walk must clear wall corners by its radius.
        path = np.array([start, end], dtype=float)
        near_walls = np.minimum(
            np.minimum(
                points_to_segments(path[0], self.wall_starts, self.wall_ends),
                points_to_segments(path[1], self.wall_starts, self.wall_ends),
            ),
            np.minimum(
                segment_distance(self.wall_starts, start, end),
                segment_distance(self.wall_ends, start, end),
            ),
        )
        if (near_walls < PLAYER_RADIUS).any():
            return False
        for center, radius in self.obstacles:
            if segment_distance(np.array([center]), start, end)[0] < radius + CLEARANCE:
                return False
        samples = max(int(math.dist(start, end) / SAMPLE), 1)
        floors = [
            self.heights[self.locate(ax + (bx - ax) * k / samples, ay + (by - ay) * k / samples)][0]
            for k in range(samples + 1)
        ]
        return all(b - a <= MAX_STEP for a, b in zip(floors, floors[1:], strict=False))

    def remaining(self, x: float, y: float) -> float | None:
        """Walking distance from the point to an exit switch."""
        cell = self.cell(x, y)
        if cell is None:
            return None
        return float(self.distance[cell] + math.dist((x, y), self.centers[cell]))

    def waypoint(self, x: float, y: float) -> tuple[tuple[float, float], str] | None:
        """A point about LOOKAHEAD cells down the distance field, and what lies there:
        the exit, a door, or plain floor."""
        cell = self.cell(x, y)
        if cell is None:
            return None
        path = [cell]
        while len(path) <= LOOKAHEAD and self.toward[path[-1]] >= 0:
            path.append(int(self.toward[path[-1]]))
        # Head for the farthest cell ahead that a straight walk reaches, since a straight line
        # to a cell further along can clip a wall corner or a ledge the path steps around.
        reach = next(
            (
                k
                for k in range(len(path) - 1, 0, -1)
                if self.walkable((x, y), tuple(self.centers[path[k]]))
            ),
            min(1, len(path) - 1),
        )
        cell = path[reach]
        door = bool(self.doors[path[1 : reach + 1]].any()) if reach else False
        if self.distance[cell] <= EXIT_REACH:
            target = min(self.exits, key=lambda line: math.dist((x, y), line.middle))
            return target.middle, "exit"
        return tuple(self.centers[cell]), "door" if door else "path"


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The z component of 2-D cross products."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def points_to_segments(point: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Distance from one point to each segment from starts to ends."""
    spans = ends - starts
    lengths = np.maximum((spans * spans).sum(axis=1), 1e-9)
    along = np.clip(((point - starts) * spans).sum(axis=1) / lengths, 0, 1)
    return np.linalg.norm(point - (starts + along[:, None] * spans), axis=1)


def segment_distance(points: np.ndarray, start: tuple, end: tuple) -> np.ndarray:
    """Distance from each point to the segment from start to end."""
    a, b = np.array(start, dtype=float), np.array(end, dtype=float)
    span = b - a
    along = np.clip(((points - a) @ span) / max(span @ span, 1e-9), 0, 1)
    return np.linalg.norm(points - (a + along[:, None] * span), axis=1)


# Moves to the eight neighboring cells, with their length in cells.
STEPS = [(di, dj, math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
