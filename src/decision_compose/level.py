"""Static Doom map geometry from a WAD, for routing the player to a level's exit."""

import heapq
import math
import struct
from dataclasses import dataclass
from pathlib import Path

# Doom's movement limits and line conventions, from the original engine.
MAX_STEP = 24
PLAYER_HEIGHT = 56
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
    """A map's sectors and the openings between them, with a shortest route to the exit."""

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
        # Openings the player can cross, in each direction, keyed by the sector they leave.
        self.openings: dict[int, list[tuple[Line, int]]] = {}
        for line in self.lines:
            if line.back is None or line.flags & IMPASSABLE or line.special in LOCKED:
                continue
            for here, there in ((line.front, line.back), (line.back, line.front)):
                if self.passable(here, there):
                    self.openings.setdefault(here, []).append((line, there))

    def passable(self, here: int, there: int) -> bool:
        floor, ceiling = self.heights[there]
        # A closed door has its ceiling on its floor; the player opens it with USE.
        closed_door = ceiling <= floor
        return closed_door or (
            floor - self.heights[here][0] <= MAX_STEP and ceiling - floor >= PLAYER_HEIGHT
        )

    def locate(self, x: float, y: float) -> int:
        """The sector containing the point, found by walking the map's BSP tree as Doom does."""
        child = len(self.nodes) - 1
        while not child & SUBSECTOR:
            node_x, node_y, dx, dy, *_, right, left = self.nodes[child]
            # Doom's R_PointOnSide: the right child unless the point is behind the partition.
            behind = (y - node_y) * dx >= dy * (x - node_x)
            child = left if behind else right
        return self.subsectors[child & ~SUBSECTOR]

    def route(self, x: float, y: float) -> tuple[list[Line], float] | None:
        """Openings to cross, in order, and the path length from the point to an exit switch."""
        start = self.locate(x, y)
        goals = {line.front: line for line in self.exits}
        frontier = [(0.0, 0, (x, y), start, [])]
        settled: set[tuple[int, tuple]] = set()
        counter = 0
        while frontier:
            length, _, point, sector, path = heapq.heappop(frontier)
            if sector in goals:
                exit_line = goals[sector]
                return [*path, exit_line], length + math.dist(point, exit_line.middle)
            if (sector, point) in settled:
                continue
            settled.add((sector, point))
            for line, there in self.openings.get(sector, []):
                counter += 1
                step = math.dist(point, line.middle)
                heapq.heappush(
                    frontier, (length + step, counter, line.middle, there, [*path, line])
                )
        return None
