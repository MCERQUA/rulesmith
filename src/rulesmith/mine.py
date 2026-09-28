"""Rules mined from the training data before search writes any: a field that alone settles most
examples, like a star rating, is found in milliseconds rather than hoped for from a proposer.

A mined rule is one test on a field, or two together, and must be nearly pure on the training
examples it covers and cover a real share of them. A text field is tested by its length, so an
empty company profile can settle a case with nothing read. The rules go ahead of the starting
graph, which answers whatever they leave, so a judge the starting graph asks is asked only about
the cases no field settles.
"""

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

from rulesmith.graph import Example, LiteralInput, Plan, Task
from rulesmith.rules import RuleError, render_rules

# A categorical field with more values than this is text, not a category a rule can name.
MOST_CATEGORIES = 12
# A field's value, or how long it is: ("company_profile",), "length".
Feature = tuple[tuple[str, ...], str]


@dataclass(frozen=True)
class Test:
    feature: Feature
    op: str
    value: float | str

    def holds(self, row: dict[Feature, float | str]) -> bool:
        value = row.get(self.feature)
        if value is None or isinstance(value, str) != isinstance(self.value, str):
            return False
        if self.op == "eq":
            return value == self.value
        return value >= self.value if self.op == "ge" else value < self.value


Rule = tuple[Test, ...]


def scalars(state, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], float | str]:
    """Every number and string in an input object, by path, two levels deep at most."""
    found = {}
    if not isinstance(state, dict) or len(prefix) > 1:
        return found
    for key, value in state.items():
        path = (*prefix, str(key))
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float, str)):
            found[path] = value
        else:
            found |= scalars(value, path)
    return found


def columns(examples: list[Example]) -> dict[Feature, list]:
    """The values each feature takes, over examples whose input is an object. A field of few
    strings is a category; a field of many is text, known by its length; paths the rules
    language cannot write are left out."""
    values: dict[tuple[str, ...], list] = {}
    for example in examples:
        for path, value in scalars(example.state).items():
            values.setdefault(path, []).append(value)
    usable: dict[Feature, list] = {}
    for path, seen in values.items():
        if not all(re.fullmatch(r"[A-Za-z_]\w*", part) for part in path):
            continue
        if all(isinstance(value, str) for value in seen):
            # A blank is no label a rule can name, but it has a length: an empty field is often
            # the most telling thing about it.
            named = [value for value in seen if value]
            written = all(not set(",[]\n") & set(value) for value in named)
            if named and len(set(named)) <= MOST_CATEGORIES and written:
                usable[path, "value"] = named
            elif len(set(named)) > MOST_CATEGORIES:
                usable[path, "length"] = [len(value) for value in seen]
        elif all(isinstance(value, (int, float)) for value in seen):
            usable[path, "value"] = seen
    return usable


def row(state, features: Iterable[Feature]) -> dict[Feature, float | str]:
    """One input as the features the miner knows."""
    fields = scalars(state)
    found: dict[Feature, float | str] = {}
    for path, measure in features:
        value = fields.get(path)
        if value is None:
            continue
        if measure == "length":
            if isinstance(value, str):
                found[path, measure] = len(value)
        elif value != "":
            found[path, measure] = value
    return found


def fields(examples: list[Example]) -> dict[str, str]:
    """What each field in the inputs holds, in words a proposer can plan rules from."""
    summary = {}
    for (path, measure), seen in columns(examples).items():
        name = ".".join(path)
        if measure == "length":
            summary[name] = f"text of {min(seen)} to {max(seen)} characters"
        elif isinstance(seen[0], str):
            summary[name] = "one of " + ", ".join(sorted(set(seen)))
        else:
            summary[name] = f"numbers from {min(seen):g} to {max(seen):g}"
    return summary


def candidates(features: dict[Feature, list]) -> list[Test]:
    tests = []
    for feature, seen in features.items():
        distinct = sorted(set(seen))
        if isinstance(distinct[0], str):
            tests += [Test(feature, "eq", value) for value in distinct]
        else:
            # Cutting at each value the field takes keeps thresholds to numbers the data holds.
            tests += [Test(feature, op, value) for value in distinct[1:] for op in ("ge", "lt")]
    return tests


def rule_list(
    examples: list[Example], most: int, precision: float, share: float
) -> list[tuple[Rule, str]]:
    """Greedy: the rule covering the most remaining examples at the precision asked, again and
    again, each taking its examples out of what the next one sees. A single test is preferred;
    two together are tried only when no single one is pure enough, since each test a rule adds
    fits the training examples a little closer."""
    features = columns(examples)
    rows = [row(example.state, features) for example in examples]
    labels = [example.label for example in examples]
    tests = candidates(features)
    covers = {test: frozenset(i for i, r in enumerate(rows) if test.holds(r)) for test in tests}
    left, rules = frozenset(range(len(examples))), []
    enough = max(3, round(share * len(examples)))

    def best(options) -> tuple[Rule, str, frozenset] | None:
        found = None
        for rule, covered in options:
            covered &= left
            if len(covered) < enough or (found and len(covered) <= len(found[2])):
                continue
            label, hits = Counter(labels[i] for i in covered).most_common(1)[0]
            if hits / len(covered) >= precision:
                found = (rule, label, covered)
        return found

    while left and len(rules) < most:
        single = best(((test,), covers[test]) for test in tests)
        # Pairs are drawn from tests that alone cover enough of what is left; others cannot.
        wide = [test for test in tests if len(covers[test] & left) >= enough]
        pairs = (
            ((a, b), covers[a] & covers[b])
            for a, b in combinations(wide, 2)
            if a.feature != b.feature
        )
        found = single or best(pairs)
        if found is None:
            break
        rule, label, covered = found
        rules.append((rule, label))
        left -= covered
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
    nodes, sources, conditions = graph["nodes"], [], []
    for rule, _ in rules:
        for test in rule:
            read_node(nodes, test, [e.state for e in task.train])
    for index, (rule, label) in enumerate(rules, start=1):
        name = fresh(nodes, f"rule_mined_{index}")
        tests = [
            {"type": "compare", "node": read_name(t.feature), "op": t.op, "value": t.value}
            for t in rule
        ]
        condition = tests[0] if len(tests) == 1 else {"type": "all", "conditions": tests}
        nodes[name] = {"type": "value", "value": label, "when": condition}
        sources.append(name)
        conditions.append(condition)
    # A judgment runs whenever nothing gates it, so the seed's are asked only where no mined rule
    # already answered; otherwise the rules would save nothing.
    unsettled = {"type": "not", "condition": {"type": "any", "conditions": conditions}}
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


def read_name(feature: Feature) -> str:
    path, measure = feature
    return "read_" + ("length_" if measure == "length" else "") + "_".join(path)


def read_node(nodes: dict, test: Test, states: list) -> None:
    """A read for the tested feature, sized to the values seen with room either side, so a value
    a little beyond the training range still reads rather than skipping the rule."""
    name = read_name(test.feature)
    if name in nodes:
        return
    path, measure = test.feature
    rows = [row(state, [test.feature]) for state in states]
    seen = [found[test.feature] for found in rows if test.feature in found]
    if isinstance(test.value, str):
        domain = {"categories": sorted(set(seen))}
    else:
        low, high = min(seen), max(seen)
        room = max(high - low, 1)
        domain = {"bounds": [low - room, high + room]}
    extra = {"measure": "length"} if measure == "length" else {}
    nodes[name] = {"type": "read", "path": list(path), **domain, **extra}


def fresh(nodes: dict, base: str) -> str:
    name, count = base, 1
    while name in nodes:
        count += 1
        name = f"{base}_{count}"
    return name
