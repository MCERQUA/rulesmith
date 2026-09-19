"""Static Doom map geometry from a WAD, for routing the player to a level's exit."""

import heapq
import math
import struct
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
CROWDED = 24
WALL_PENALTY = 1.0
EXIT_REACH = 48
LOOKAHEAD = 4
SEARCH = 3
IMPASSABLE = 0x0001
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
        clearance = np.full(len(centers), np.inf)
        for line in walls:
            clearance = np.minimum(clearance, segment_distance(centers, line.start, line.end))
        free = (
            ((gaps >= PLAYER_HEIGHT) | doors)
            & ~np.isin(sectors, list(locked))
            & (clearance >= CLEARANCE)
        )
        self.centers, self.floors, self.doors, self.free = centers, floors, doors, free
        # Hugging a wall snags on door frames and corners, so cells near one cost more.
        crowded = 1 + WALL_PENALTY * (clearance < CROWDED)
        self.distance = np.full(len(centers), np.inf)
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
                # Walking from other into cell must not climb more than Doom's step height.
                if not free[other] or floors[cell] - floors[other] > MAX_STEP:
                    continue
                total = length + step * CELL * crowded[other]
                if total < self.distance[other]:
                    self.distance[other] = total
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
                    if np.isfinite(self.distance[cell]):
                        cost = self.distance[cell] + math.dist((x, y), self.centers[cell])
                        if best is None or cost < best[0]:
                            best = (cost, cell)
            if best is not None:
                return best[1]
        return None

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
        width = self.shape[1]
        door = False
        for _ in range(LOOKAHEAD):
            i, j = divmod(cell, width)
            neighbors = [
                (i + di) * width + (j + dj)
                for di, dj, _ in STEPS
                if 0 <= i + di < self.shape[0] and 0 <= j + dj < width
            ]
            following = min(neighbors, key=lambda other: self.distance[other])
            if self.distance[following] >= self.distance[cell]:
                break
            cell = following
            door = door or bool(self.doors[cell])
        if self.distance[cell] <= EXIT_REACH:
            target = min(self.exits, key=lambda line: math.dist((x, y), line.middle))
            return target.middle, "exit"
        return tuple(self.centers[cell]), "door" if door else "path"


def segment_distance(points: np.ndarray, start: tuple, end: tuple) -> np.ndarray:
    """Distance from each point to the segment from start to end."""
    a, b = np.array(start, dtype=float), np.array(end, dtype=float)
    span = b - a
    along = np.clip(((points - a) @ span) / max(span @ span, 1e-9), 0, 1)
    return np.linalg.norm(points - (a + along[:, None] * span), axis=1)


# Moves to the eight neighboring cells, with their length in cells.
STEPS = [(di, dj, math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
