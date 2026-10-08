"""A map: a graph whose units live in files of their own.

A `graph` node may name a unit by `path` instead of embedding its plan. Loading a map reads
each named unit, `.json` or `.rules`, relative to the file that names it, and embeds it, so
what comes back is an ordinary graph that the validator, the runtime and `draw` already know.
A unit may name units of its own; a map that reaches itself is refused.
"""

import json
from pathlib import Path

from rulesmith.graph import Example, Plan, Task
from rulesmith.rules import compile_rules


class MapError(ValueError):
    """A unit that cannot be found, or a map that contains itself."""


def units(path: Path) -> dict[str, str]:
    """The units a map names by path, by node: what a map is made of, before it is loaded. A
    rules file names none."""
    if path.suffix == ".rules":
        return {}
    graph = json.loads(path.read_text())
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    return {
        name: node["path"]
        for name, node in (nodes or {}).items()
        if isinstance(node, dict) and node.get("type") == "graph" and "path" in node
    }


def load(path: Path, labels: dict[str, str] | None = None) -> Plan:
    """The graph in `path`, with every unit it names embedded. `labels` are the task's, for a
    rules file that asks over them."""
    return Plan.model_validate(resolve(Path(path), labels, ()))


def resolve(path: Path, labels: dict[str, str] | None, chain: tuple[Path, ...]) -> dict:
    path = path.resolve()
    if path in chain:
        named = " -> ".join(step.name for step in (*chain, path))
        raise MapError(f"{path.name} contains itself: {named}")
    text = path.read_text()
    if path.suffix == ".rules":
        graph = compile_rules(text, labels).model_dump(mode="json", exclude_none=True)
    else:
        graph = json.loads(text)
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    for node in (nodes or {}).values():
        if isinstance(node, dict) and node.get("type") == "graph" and "path" in node:
            unit = path.parent / node.pop("path")
            if not unit.is_file():
                raise MapError(f"{path.name} names a unit at {unit} that does not exist")
            node["plan"] = resolve(unit, labels, (*chain, path))
    return graph


def unit_task(program, task: Task, unit: str) -> Task:
    """The task of one unit of a map: the examples the map hands it, each as the state it was
    handed, keeping the ones whose right answer is a label the unit can give. A case routed to
    the unit that needs an answer it cannot give is the router's mistake, not the unit's, and
    no change to the unit fixes it. `program` runs the whole map, on its judge."""
    labels = sorted(program.plan.domains()[unit], key=task.labels.index)
    # A unit handed part of the input can be handed the same thing for different cases; a task
    # holds each state once, so the first split to reach it keeps it.
    seen: set[str] = set()

    def cases(examples: list[Example]) -> list[Example]:
        found = []
        for example in examples:
            handed = program(state=example.state).handed
            if unit not in handed or example.label not in labels:
                continue
            key = json.dumps(handed[unit], sort_keys=True)
            if key not in seen:
                seen.add(key)
                found.append(example.model_copy(update={"state": handed[unit]}))
        return found

    splits = {split: cases(getattr(task, split)) for split in ("train", "validation", "test")}
    for split, examples in splits.items():
        if not examples:
            raise MapError(
                f"the map hands {unit!r} none of the task's {split} examples it can answer"
            )
    return Task.model_validate(
        task.model_dump(exclude={"train", "validation", "test", "calibration"})
        | {
            "description": f"{task.description} The cases the map hands to its unit {unit}.",
            "labels": labels,
            **{split: [e.model_dump() for e in examples] for split, examples in splits.items()},
            "calibration": [e.model_dump() for e in cases(task.calibration)],
        }
    )
