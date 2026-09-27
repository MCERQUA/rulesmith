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


class Relation(StrictModel):
    """One value compared with another, where Compare holds a node against a constant."""

    type: Literal["relation"] = "relation"
    node: Text
    op: Literal["eq", "lt", "ge"]
    other: Text


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


Condition = Annotated[Compare | Relation | Present | All | Any | Not, Field(discriminator="type")]
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
    """References in graph syntax; literal payloads and question text are opaque."""
    if isinstance(value, list):
        return set().union(*(references(v) for v in value))
    if not isinstance(value, dict):
        return set()
    named: set[str] = set()
    kind = value.get("type")
    if kind in ("node", "compare", "relation", "present") and isinstance(value.get("node"), str):
        named.add(value["node"])
    if kind == "relation" and isinstance(value.get("other"), str):
        named.add(value["other"])
    if kind in ("switch", "threshold", "confidence") and isinstance(value.get("source"), str):
        named.add(value["source"])
    if kind in ("coalesce", "vote") and isinstance(value.get("sources"), list):
        named.update(source for source in value["sources"] if isinstance(source, str))
    if kind == "sum" and isinstance(value.get("terms"), list):
        named.update(
            term["node"]
            for term in value["terms"]
            if isinstance(term, dict) and isinstance(term.get("node"), str)
        )
    for key in ("when", "condition", "conditions"):
        named.update(references(value.get(key)))
    if kind in ("call", "merge") and isinstance(value.get("inputs"), dict):
        for source in value["inputs"].values():
            named.update(references(source))
    return named


def predicates(condition: Condition | None) -> list[Compare | Relation | Present]:
    if condition is None:
        return []
    if isinstance(condition, (Compare, Relation, Present)):
        return [condition]
    if isinstance(condition, Not):
        return predicates(condition.condition)
    return [leaf for child in condition.conditions for leaf in predicates(child)]


def tested(condition: Condition | None) -> set[str]:
    """The nodes a condition reads."""
    named = set()
    for leaf in predicates(condition):
        named.add(leaf.node)
        if isinstance(leaf, Relation):
            named.add(leaf.other)
    return named


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


class Vote(Guarded):
    """Several judgments settled into one value. `majority` takes the value most of them gave,
    breaking a tie towards the earlier source; `agree` reports whether they all said the same,
    which is how a second opinion verifies a first."""

    type: Literal["vote"] = "vote"
    sources: list[Text] = Field(min_length=2)
    rule: Literal["majority", "agree"] = "majority"
    same: Text = "yes"
    different: Text = "no"


class Confidence(Guarded):
    """How sure a judgment was, as a number a rule can test. A choice or a score answer carries
    its own confidence; a probability answers with the probability itself, so it has none to
    report and this is absent for one."""

    type: Literal["confidence"] = "confidence"
    source: Text


class Term(StrictModel):
    node: Text
    weight: float = 1


class Sum(Guarded):
    """A weighted sum of numeric values plus a constant: a margin, a total, a scaled
    difference. It is skipped when any term is, since a missing term is not a zero."""

    type: Literal["sum"] = "sum"
    terms: list[Term] = Field(min_length=1)
    constant: float = 0


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
    Call | Switch | Threshold | Merge | Coalesce | Value | Read | Confidence | Vote | Sum,
    Field(discriminator="type"),
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
            refs = tested(node.when)
            if isinstance(node, (Call, Merge)):
                refs.update(s.node for s in node.inputs.values() if isinstance(s, NodeInput))
            elif isinstance(node, (Switch, Threshold, Confidence)):
                refs.add(node.source)
            elif isinstance(node, (Coalesce, Vote)):
                refs.update(node.sources)
            elif isinstance(node, Sum):
                refs.update(term.node for term in node.terms)
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
            elif isinstance(node, Confidence):
                domain = (0.0, 1.0)
            elif isinstance(node, Sum):
                ranges = [result[term.node] for term in node.terms]
                if all(isinstance(r, tuple) for r in ranges):
                    # Each term spans weight * low to weight * high, whichever way the sign runs.
                    ends = [
                        sorted((term.weight * low, term.weight * high))
                        for term, (low, high) in zip(node.terms, ranges, strict=True)
                    ]
                    domain = (
                        node.constant + sum(low for low, _ in ends),
                        node.constant + sum(high for _, high in ends),
                    )
            elif isinstance(node, Vote):
                if node.rule == "agree":
                    domain = frozenset([node.same, node.different])
                else:
                    voted = [result[source] for source in node.sources]
                    if all(isinstance(d, frozenset) for d in voted):
                        domain = frozenset().union(*voted)
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
        # A vote counts labels. A probability or a rubric position is a float, and two floats
        # agreeing exactly is a coincidence; a merged object cannot be counted at all.
        domains = self.domains()
        for key, node in self.nodes.items():
            if isinstance(node, Vote) and not all(
                isinstance(domains[source], frozenset) for source in node.sources
            ):
                raise ValueError(
                    f"vote {key!r} counts labels, so every source needs a categorical value"
                )
            if isinstance(node, Sum) and not all(
                isinstance(domains[term.node], tuple) for term in node.terms
            ):
                raise ValueError(f"sum {key!r} adds numbers, so every term needs a numeric value")

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
                if isinstance(condition, Relation):
                    pair = domains[condition.node], domains[condition.other]
                    kinds = {type(domain) for domain in pair}
                    if len(kinds) != 1 or None in pair:
                        raise ValueError("a relation compares two values of the same kind")
                    if condition.op != "eq" and kinds != {tuple}:
                        raise ValueError("an ordering relation requires numeric sources")
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
    # Absent only when a grader, not a label, says whether an answer is right.
    label: Text | None = None
    accepted: list[Text] = Field(
        default_factory=list, description="Other labels that are also right for this example"
    )

    @model_validator(mode="after")
    def accepted_beside_a_label(self) -> Self:
        if self.accepted and self.label is None:
            raise ValueError(f"example {self.id} accepts other labels but has none of its own")
        if self.label in self.accepted:
            raise ValueError(f"example {self.id} lists its own label as another accepted one")
        return self


class Task(StrictModel):
    description: Text
    labels: list[Text] = Field(min_length=2)
    train: list[Example] = Field(min_length=1)
    validation: list[Example] = Field(min_length=1)
    test: list[Example] = Field(min_length=1)
    costs: dict[Text, dict[Text, Annotated[float, Field(ge=0)]]] = Field(
        default_factory=dict,
        description="costs[expected][answered]: what a wrong answer costs; 1 where unlisted",
    )

    @model_validator(mode="after")
    def disjoint_splits(self) -> Self:
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("task labels must be distinct")
        ids, states = set(), set()
        for example in self.train + self.validation + self.test:
            for label in [example.label, *example.accepted]:
                if label is not None and label not in self.labels:
                    raise ValueError(f"unknown label in example {example.id}: {label}")
            state = json.dumps(example.state, sort_keys=True, ensure_ascii=False)
            if example.id in ids:
                raise ValueError(f"duplicate example id: {example.id}")
            if state in states:
                raise ValueError(f"duplicate state in example {example.id}")
            ids.add(example.id)
            states.add(state)
        for expected, row in self.costs.items():
            for answered in [expected, *row]:
                if answered not in self.labels:
                    raise ValueError(f"costs name a label the task does not have: {answered}")
            if expected in row:
                raise ValueError(f"costs give {expected!r} a cost for answering itself")
        if self.worst == 0:
            raise ValueError("costs make every mistake free, so no answer is worse than another")
        return self

    def cost(self, expected: str, answered: str) -> float:
        if expected == answered:
            return 0.0
        return self.costs.get(expected, {}).get(answered, 1.0)

    @property
    def worst(self) -> float:
        """The dearest mistake the task can make, which scores zero."""
        return max(self.cost(e, a) for e in self.labels for a in self.labels if e != a)

    def baseline_plan(self) -> Plan:
        """A call-free majority-label seed, so search can start from the task alone. Graded
        tasks may label nothing, and then the seed is the first label allowed."""
        counts = Counter(e.label for e in self.train if e.label is not None)
        return Plan.constant(counts.most_common(1)[0][0] if counts else self.labels[0])
