"""A finite tree language over a shared bank of typed questions."""

from __future__ import annotations

import json
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


class Return(StrictModel):
    type: Literal["return"] = "return"
    label: Text


class Switch(StrictModel):
    type: Literal["switch"] = "switch"
    question: Text
    cases: dict[Text, Node]


class Threshold(StrictModel):
    type: Literal["threshold"] = "threshold"
    question: Text
    cutoff: float
    below: Node
    above: Node


Node = Annotated[Return | Switch | Threshold, Field(discriminator="type")]
Switch.model_rebuild()
Threshold.model_rebuild()


def children(node: Node) -> list[Node]:
    if isinstance(node, Switch):
        return list(node.cases.values())
    if isinstance(node, Threshold):
        return [node.below, node.above]
    return []


class Plan(StrictModel):
    questions: dict[Text, Question]
    root: Node

    @model_validator(mode="after")
    def typed_references(self) -> Self:
        used = set()
        pending = [self.root]
        while pending:
            node = pending.pop()
            if isinstance(node, Return):
                continue
            if node.question not in self.questions:
                raise ValueError(f"unknown question: {node.question}")
            question = self.questions[node.question]
            used.add(node.question)
            if isinstance(node, Switch):
                if not isinstance(question, Choice):
                    raise ValueError("switch requires a Choice question")
                if set(node.cases) != set(question.criteria):
                    raise ValueError("switch cases must match the Choice options exactly")
            else:
                if isinstance(question, Choice):
                    raise ValueError("threshold requires a Noul or Score question")
                upper = len(question.criteria) - 1 if isinstance(question, Score) else 1
                if not 0 <= node.cutoff <= upper:
                    raise ValueError(f"cutoff for {node.question} must be between 0 and {upper}")
            pending.extend(children(node))
        if used != set(self.questions):
            raise ValueError("every declared question must be referenced by the tree")
        return self

    def check(self, labels: list[str], max_questions: int, max_depth: int) -> None:
        """Keep the task contract and search bounds outside the mutable candidate."""
        if len(self.questions) > max_questions:
            raise ValueError(f"question count exceeds {max_questions}")
        pending = [(self.root, 0)]
        while pending:
            node, depth = pending.pop()
            if depth > max_depth:
                raise ValueError(f"tree depth exceeds {max_depth}")
            if isinstance(node, Return) and node.label not in labels:
                raise ValueError(f"output label is outside the task: {node.label}")
            pending.extend((child, depth + 1) for child in children(node))

    def decide(self, values: dict[str, str | float]) -> tuple[str, list[str]]:
        node = self.root
        path = []
        while not isinstance(node, Return):
            path.append(node.question)
            value = values[node.question]
            if isinstance(node, Switch):
                node = node.cases[value]
            else:
                node = node.above if value >= node.cutoff else node.below
        return node.label, path


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
