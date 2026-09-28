"""Rules mined from the training data before search writes any: a field that alone settles most
examples, like a star rating, is found in milliseconds rather than hoped for from a proposer.

A mined rule must be nearly pure on the training examples it covers and cover a real share of
them. The rules go ahead of the starting graph, which answers whatever they leave, so a judge the
starting graph asks is asked only about the cases no field settles.
"""

import re
from collections import Counter
from dataclasses import dataclass

from rulesmith.graph import Example, LiteralInput, Plan, Task
from rulesmith.rules import RuleError, render_rules

# A categorical field with more values than this is text, not a category a rule can name.
MOST_CATEGORIES = 12


@dataclass(frozen=True)
class Test:
    path: tuple[str, ...]
    op: str
    value: float | str

    def holds(self, state: dict) -> bool:
        value = scalars(state).get(self.path)
        if value is None or isinstance(value, str) != isinstance(self.value, str):
            return False
        if self.op == "eq":
            return value == self.value
        return value >= self.value if self.op == "ge" else value < self.value


def scalars(state, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], float | str]:
    """Every number and string in an input object, by path, two levels deep at most."""
    found = {}
    if not isinstance(state, dict) or len(prefix) > 1:
        return found
    for key, value in state.items():
        path = (*prefix, str(key))
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)) or (isinstance(value, str) and value):
            found[path] = value
        else:
            found |= scalars(value, path)
    return found


def columns(examples: list[Example]) -> dict[tuple[str, ...], list]:
    """The values each field takes, over examples whose input is an object. Paths the rules
    language cannot write, or strings too varied to be categories, are left out."""
    values: dict[tuple[str, ...], list] = {}
    for example in examples:
        for path, value in scalars(example.state).items():
            values.setdefault(path, []).append(value)
    usable = {}
    for path, seen in values.items():
        if not all(re.fullmatch(r"[A-Za-z_]\w*", part) for part in path):
            continue
        if all(isinstance(value, str) for value in seen):
            # A blank is no label a rule can name; the example simply lacks the field.
            named = [value for value in seen if value]
            written = all(not set(",[]\n") & set(value) for value in named)
            if named and len(set(named)) <= MOST_CATEGORIES and written:
                usable[path] = named
        elif all(isinstance(value, (int, float)) for value in seen):
            usable[path] = seen
    return usable


def fields(examples: list[Example]) -> dict[str, str]:
    """What each field in the inputs holds, in words a proposer can plan rules from."""
    summary = {}
    for path, seen in columns(examples).items():
        name = ".".join(path)
        if isinstance(seen[0], str):
            summary[name] = "one of " + ", ".join(sorted(set(seen)))
        else:
            summary[name] = f"numbers from {min(seen):g} to {max(seen):g}"
    return summary


def candidates(examples: list[Example]) -> list[Test]:
    tests = []
    for path, seen in columns(examples).items():
        distinct = sorted(set(seen))
        if isinstance(distinct[0], str):
            tests += [Test(path, "eq", value) for value in distinct]
        else:
            # Cutting at each value the field takes keeps thresholds to numbers the data holds.
            tests += [Test(path, op, value) for value in distinct[1:] for op in ("ge", "lt")]
    return tests


def rule_list(
    examples: list[Example], most: int, precision: float, share: float
) -> list[tuple[Test, str]]:
    """Greedy: the test covering the most remaining examples at the precision asked, again and
    again, each rule taking its examples out of what the next one sees."""
    left, rules = list(examples), []
    enough = max(3, round(share * len(examples)))
    tests = candidates(examples)
    while left and len(rules) < most:
        best = None
        for test in tests:
            covered = [e for e in left if test.holds(e.state)]
            if len(covered) < enough:
                continue
            label, hits = Counter(e.label for e in covered).most_common(1)[0]
            if hits / len(covered) >= precision and (best is None or len(covered) > best[2]):
                best = (test, label, len(covered))
        if best is None:
            break
        test, label, _ = best
        rules.append((test, label))
        left = [e for e in left if not test.holds(e.state)]
    return rules


def mine(
    task: Task,
    seed: Plan,
    most: int = 4,
    precision: float = 0.95,
    share: float = 0.05,
) -> Plan | None:
    """The seed with mined rules ahead of its label, or None when no field settles enough, the
    inputs are not objects, or the seed cannot be written as rules to build on."""
    try:
        render_rules(seed)
    except RuleError:
        return None
    rules = rule_list(task.train, most, precision, share)
    if not rules:
        return None
    graph = seed.model_dump(mode="json", exclude_none=True)
    nodes, sources, tests = graph["nodes"], [], []
    for test, _ in rules:
        read_node(nodes, test, [e.state for e in task.train])
    for index, (test, label) in enumerate(rules, start=1):
        name = fresh(nodes, f"rule_mined_{index}")
        condition = {
            "type": "compare",
            "node": read_name(test.path),
            "op": test.op,
            "value": test.value,
        }
        nodes[name] = {"type": "value", "value": label, "when": condition}
        sources.append(name)
        tests.append(condition)
    # A judgment runs whenever nothing gates it, so the seed's are asked only where no mined rule
    # already answered; otherwise the rules would save nothing.
    unsettled = {"type": "not", "condition": {"type": "any", "conditions": tests}}
    for node in nodes.values():
        if node["type"] == "call":
            gate = node.get("when")
            node["when"] = {"type": "all", "conditions": [unsettled, gate]} if gate else unsettled
    label = seed.outputs["label"]
    if isinstance(label, LiteralInput):
        fallback = fresh(nodes, "rule_seed")
        nodes[fallback] = {"type": "value", "value": label.value}
    else:
        fallback = label.node
    joined = fresh(nodes, "decision_mined")
    nodes[joined] = {"type": "coalesce", "sources": [*sources, fallback]}
    graph["outputs"]["label"] = {"type": "node", "node": joined}
    return Plan.model_validate(graph)


def read_name(path: tuple[str, ...]) -> str:
    return "read_" + "_".join(path)


def read_node(nodes: dict, test: Test, states: list) -> None:
    """A read for the tested field, sized to the values seen with room either side, so a value a
    little beyond the training range still reads rather than skipping the rule."""
    name = read_name(test.path)
    if name in nodes:
        return
    seen = [scalars(state)[test.path] for state in states if test.path in scalars(state)]
    if isinstance(test.value, str):
        domain = {"categories": sorted(set(seen))}
    else:
        low, high = min(seen), max(seen)
        room = max(high - low, 1)
        domain = {"bounds": [low - room, high + room]}
    nodes[name] = {"type": "read", "path": list(test.path), **domain}


def fresh(nodes: dict, base: str) -> str:
    name, count = base, 1
    while name in nodes:
        count += 1
        name = f"{base}_{count}"
    return name
