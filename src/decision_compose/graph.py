"""A typed, finite DAG language for model calls and deterministic data flow."""

from __future__ import annotations

import json
from collections import Counter
from graphlib import CycleError, TopologicalSorter
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
State = str | dict[str, JsonValue] | list[JsonValue]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Choice(StrictModel):
    type: Literal["choice"] = "choice"
    instructions: Text
    criteria: dict[Text, Text] = Field(min_length=2, max_length=255)


class Score(StrictModel):
    type: Literal["score"] = "score"
    instructions: Text
    criteria: list[Text] = Field(min_length=2, max_length=10)

    @model_validator(mode="after")
    def distinct_levels(self) -> Self:
        if len(set(self.criteria)) != len(self.criteria):
            raise ValueError("score criteria must be distinct")
        return self


class Noul(StrictModel):
    type: Literal["noul"] = "noul"
    instructions: Text


Question = Annotated[Choice | Score | Noul, Field(discriminator="type")]


class StateInput(StrictModel):
    type: Literal["state"] = "state"


class LiteralInput(StrictModel):
    type: Literal["literal"] = "literal"
    value: JsonValue


class NodeInput(StrictModel):
    type: Literal["node"] = "node"
    node: Text
    view: Literal["value", "answer"] = "value"
    optional: bool = False


Source = Annotated[StateInput | LiteralInput | NodeInput, Field(discriminator="type")]


class Compare(StrictModel):
    type: Literal["compare"] = "compare"
    node: Text
    op: Literal["eq", "lt", "ge"]
    value: str | float


class Present(StrictModel):
    type: Literal["present"] = "present"
    node: Text


class All(StrictModel):
    type: Literal["all"] = "all"
    conditions: list[Condition] = Field(min_length=1)


class Any(StrictModel):
    type: Literal["any"] = "any"
    conditions: list[Condition] = Field(min_length=1)


class Not(StrictModel):
    type: Literal["not"] = "not"
    condition: Condition


Condition = Annotated[Compare | Present | All | Any | Not, Field(discriminator="type")]
All.model_rebuild()
Any.model_rebuild()
Not.model_rebuild()


def reachable(dependencies: dict[str, set[str]], outputs) -> set[str]:
    """The nodes an output depends on, following references as far as they go."""
    used: set[str] = set()
    pending = [source.node for source in outputs if isinstance(source, NodeInput)]
    while pending:
        key = pending.pop()
        if key not in used and key in dependencies:
            used.add(key)
            pending.extend(dependencies[key])
    return used


def references(value: JsonValue) -> set[str]:
    """Node names a raw node or output mentions, wherever they sit in its JSON."""
    if isinstance(value, dict):
        named = {value["node"]} if isinstance(value.get("node"), str) else set()
        if isinstance(value.get("source"), str):
            named.add(value["source"])
        named.update(s for s in value.get("sources") or [] if isinstance(s, str))
        return named.union(*(references(v) for v in value.values())) if value else named
    if isinstance(value, list):
        return set().union(*(references(v) for v in value)) if value else set()
    return set()


def predicates(condition: Condition | None) -> list[Compare | Present]:
    if condition is None:
        return []
    if isinstance(condition, (Compare, Present)):
        return [condition]
    if isinstance(condition, Not):
        return predicates(condition.condition)
    return [leaf for child in condition.conditions for leaf in predicates(child)]


class Guarded(StrictModel):
    when: Condition | None = None


class Call(Guarded):
    type: Literal["call"] = "call"
    question: Question
    inputs: dict[Text, Source] = Field(min_length=1)


class Switch(Guarded):
    type: Literal["switch"] = "switch"
    source: Text
    cases: dict[Text, Text] = Field(min_length=1)


class Threshold(Guarded):
    type: Literal["threshold"] = "threshold"
    source: Text
    cutoff: float
    below: Text
    above: Text


class Merge(Guarded):
    type: Literal["merge"] = "merge"
    inputs: dict[Text, Source] = Field(min_length=1)


class Coalesce(Guarded):
    type: Literal["coalesce"] = "coalesce"
    sources: list[Text] = Field(min_length=1)


class Value(Guarded):
    type: Literal["value"] = "value"
    value: JsonValue


class Read(Guarded):
    """A value taken from the original input by path, with its declared domain."""

    type: Literal["read"] = "read"
    path: list[str | int] = Field(min_length=1)
    bounds: tuple[float, float] | None = None
    categories: list[Text] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def one_domain(self) -> Self:
        if (self.bounds is None) == (self.categories is None):
            raise ValueError("a read declares exactly one of bounds or categories")
        if self.bounds is not None and self.bounds[0] > self.bounds[1]:
            raise ValueError("read bounds must be increasing")
        return self


Node = Annotated[
    Call | Switch | Threshold | Merge | Coalesce | Value | Read, Field(discriminator="type")
]
Domain = frozenset[str] | tuple[float, float] | None


def literal_domain(value: JsonValue) -> Domain:
    if isinstance(value, str):
        return frozenset([value])
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (value, value)
    return None


class Plan(StrictModel):
    nodes: dict[Text, Node]
    outputs: dict[Text, Source] = Field(min_length=1)

    def dependencies(self) -> dict[str, set[str]]:
        result = {}
        for key, node in self.nodes.items():
            refs = {condition.node for condition in predicates(node.when)}
            if isinstance(node, (Call, Merge)):
                refs.update(s.node for s in node.inputs.values() if isinstance(s, NodeInput))
            elif isinstance(node, (Switch, Threshold)):
                refs.add(node.source)
            elif isinstance(node, Coalesce):
                refs.update(node.sources)
            result[key] = refs
        return result

    def contributing(self) -> set[str]:
        """Every node an output depends on, directly or through another node."""
        return reachable(self.dependencies(), self.outputs.values())

    def order(self) -> tuple[str, ...]:
        dependencies = self.dependencies()
        return tuple(
            TopologicalSorter(
                {k: sorted(dependencies[k]) for k in sorted(self.nodes)}
            ).static_order()
        )

    def domains(self) -> dict[str, Domain]:
        """Infer finite label sets and numeric bounds without running model calls."""
        result: dict[str, Domain] = {}
        for key in self.order():
            node = self.nodes[key]
            domain = None
            if isinstance(node, Call):
                q = node.question
                domain = (
                    frozenset(q.criteria)
                    if isinstance(q, Choice)
                    else (0, len(q.criteria) - 1 if isinstance(q, Score) else 1)
                )
            elif isinstance(node, Switch):
                domain = frozenset(node.cases.values())
            elif isinstance(node, Threshold):
                domain = frozenset([node.below, node.above])
            elif isinstance(node, Value):
                domain = literal_domain(node.value)
            elif isinstance(node, Read):
                domain = frozenset(node.categories) if node.categories else node.bounds
            elif isinstance(node, Coalesce):
                inputs = [result[source] for source in node.sources]
                if all(isinstance(d, frozenset) for d in inputs):
                    domain = frozenset().union(*inputs)
                elif all(isinstance(d, tuple) for d in inputs):
                    domain = (min(d[0] for d in inputs), max(d[1] for d in inputs))
            result[key] = domain
        return result

    @model_validator(mode="after")
    def typed_references(self) -> Self:
        dependencies = self.dependencies()
        sources = list(self.outputs.values())
        for node in self.nodes.values():
            if isinstance(node, (Call, Merge)):
                sources.extend(node.inputs.values())
        references = {r for parents in dependencies.values() for r in parents}
        references.update(s.node for s in sources if isinstance(s, NodeInput))
        unknown = references - self.nodes.keys()
        if unknown:
            raise ValueError(f"unknown nodes: {sorted(unknown)}")
        for source in sources:
            if (
                isinstance(source, NodeInput)
                and source.view == "answer"
                and not isinstance(self.nodes[source.node], Call)
            ):
                raise ValueError("answer view requires a call node")
        try:
            self.order()
        except CycleError as error:
            raise ValueError(f"graph contains a cycle: {error.args[1]}") from error

        if self.contributing() != self.nodes.keys():
            raise ValueError("every node must contribute to a graph output")

        domains = self.domains()
        for node in self.nodes.values():
            if isinstance(node, Switch):
                domain = domains[node.source]
                if not isinstance(domain, frozenset) or set(node.cases) != domain:
                    raise ValueError("switch cases must exactly cover the source categories")
            elif isinstance(node, Threshold):
                domain = domains[node.source]
                if not isinstance(domain, tuple):
                    raise ValueError("threshold requires a numeric source")
                if not domain[0] <= node.cutoff <= domain[1]:
                    raise ValueError(f"cutoff must be within source bounds {domain}")
            for condition in predicates(node.when):
                if isinstance(condition, Present):
                    continue
                domain = domains[condition.node]
                if isinstance(condition.value, str) and condition.op == "eq":
                    if not isinstance(domain, frozenset) or condition.value not in domain:
                        raise ValueError("equality requires a declared source category")
                elif not isinstance(domain, tuple) or isinstance(condition.value, str):
                    raise ValueError("numeric comparison requires a numeric source and value")
        return self

    @classmethod
    def constant(cls, label: str) -> Plan:
        """A call-free graph that always outputs one label."""
        return cls(nodes={}, outputs={"label": LiteralInput(value=label)})

    def check(self, labels: list[str], max_nodes: int, max_depth: int) -> None:
        """Keep classification targets and search bounds outside mutable candidates."""
        if len(self.nodes) > max_nodes:
            raise ValueError(f"node count exceeds {max_nodes}")
        dependencies, depths = self.dependencies(), {}
        for key in self.order():
            depths[key] = 1 + max((depths[p] for p in dependencies[key]), default=0)
        if max(depths.values(), default=0) > max_depth:
            raise ValueError(f"graph depth exceeds {max_depth}")
        label = self.outputs.get("label")
        domain = None
        if isinstance(label, LiteralInput):
            domain = literal_domain(label.value)
        elif isinstance(label, NodeInput) and label.view == "value" and not label.optional:
            domain = self.domains()[label.node]
        if not isinstance(domain, frozenset) or not domain <= set(labels):
            raise ValueError(
                "label output must have a required categorical value within task labels"
            )


class Example(StrictModel):
    id: Text
    state: State
    label: Text


class Task(StrictModel):
    description: Text
    labels: list[Text] = Field(min_length=2)
    train: list[Example] = Field(min_length=1)
    validation: list[Example] = Field(min_length=1)
    test: list[Example] = Field(min_length=1)

    @model_validator(mode="after")
    def disjoint_splits(self) -> Self:
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("task labels must be distinct")
        ids, states = set(), set()
        for example in self.train + self.validation + self.test:
            if example.label not in self.labels:
                raise ValueError(f"unknown label in example {example.id}: {example.label}")
            state = json.dumps(example.state, sort_keys=True, ensure_ascii=False)
            if example.id in ids:
                raise ValueError(f"duplicate example id: {example.id}")
            if state in states:
                raise ValueError(f"duplicate state in example {example.id}")
            ids.add(example.id)
            states.add(state)
        return self

    def baseline_plan(self) -> Plan:
        """A call-free majority-label seed, so search can start from the task alone."""
        return Plan.constant(Counter(e.label for e in self.train).most_common(1)[0][0])
