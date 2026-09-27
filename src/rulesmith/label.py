"""Draft labels for unlabeled inputs with a generative model, for human review."""

import json
import random
from collections.abc import Callable
from typing import Literal

import dspy
from pydantic import Field

from rulesmith.graph import State, StrictModel, Task, Text
from rulesmith.optimize import text_adapter


class Unlabeled(StrictModel):
    description: Text
    labels: list[Text] = Field(min_length=2)
    inputs: list[State] = Field(min_length=3)


class Labeler(dspy.Module):
    def __init__(self, lm: dspy.LM, description: str, labels: list[str]):
        super().__init__()
        self.lm = lm
        signature = dspy.make_signature(
            {
                "state": (State, dspy.InputField()),
                # A Literal output makes the adapter reject labels outside the task.
                "label": (Literal[tuple(labels)], dspy.OutputField()),
            },
            description,
        )
        self.predict = dspy.Predict(signature)

    def forward(self, state: State) -> str:
        with dspy.context(lm=self.lm, adapter=text_adapter()):
            return self.predict(state=state).label


def draft_task(
    description: str,
    labels: list[str],
    states: list[State],
    label: Callable[[State], str],
    validation: int,
    test: int,
    seed: int,
) -> Task:
    # Every label is paid for, so a task that cannot be built is refused before the first one.
    if validation < 1 or test < 1:
        raise ValueError("validation and test splits each need at least one input")
    if validation + test >= len(states):
        raise ValueError("validation and test splits must leave at least one training input")
    written = [json.dumps(state, sort_keys=True) for state in states]
    if len(set(written)) != len(written):
        raise ValueError("inputs repeat; a task may hold each input once")
    examples = [{"id": str(index), "state": state} for index, state in enumerate(states)]
    random.Random(seed).shuffle(examples)
    for example in examples:
        example["label"] = label(example["state"])
    return Task(
        description=description,
        labels=labels,
        validation=examples[:validation],
        test=examples[validation : validation + test],
        train=examples[validation + test :],
    )
