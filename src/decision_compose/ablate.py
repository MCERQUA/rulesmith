"""What each input is worth: switch one off and count the decisions that change.

A reading that changes nothing is either a path the state never had, a rule an earlier one always
beats to the decision, or a threshold real data never trips. All three look identical from a score
-- the graph runs, decides something, and scores like a graph that simply decides badly.
"""

import json
from dataclasses import dataclass

from decision_compose.graph import Call, Plan, Read, State
from decision_compose.runtime import (
    MISSING,
    DecisionProgram,
    ExecutionConfig,
    GraphExecutionError,
    read_path,
)


@dataclass(frozen=True)
class Effect:
    """One input's share of the decisions it changes when it is taken away."""

    node: str
    kind: str
    changed: int
    decided: int
    reason: str = ""

    @property
    def share(self) -> float:
        return self.changed / self.decided if self.decided else 0.0


class Remembered:
    """A judge that answers each question about each state once. A replay with a reading taken
    away asks the same questions of the same states, so it gets the answers the baseline paid
    for; a call whose inputs did change is a different question and goes to the judge."""

    def __init__(self, client):
        self.client = client
        self.answers: dict[str, object] = {}

    def system_one(self, *, state, questions, model):
        key = json.dumps([state, questions, model], sort_keys=True)
        if key not in self.answers:
            self.answers[key] = self.client.system_one(
                state=state, questions=questions, model=model
            )
        return self.answers[key]


def decisions(program: DecisionProgram, states: list[State]) -> list[object | None]:
    """The label the graph gives each state, or None where it cannot decide at all."""
    labels = []
    for state in states:
        try:
            labels.append(program(state=state).label)
        except GraphExecutionError:
            labels.append(None)
    return labels


def effects(plan: Plan, states: list[State], client=None, model: str = "none") -> list[Effect]:
    """Every read and every judgment, ranked by how much the graph would change without it.

    The judge is asked once per state for the baseline and never again: taking away a reading
    cannot change what a judge said about the same state, so the replays reuse those answers and
    a graph full of questions ablates for free."""
    inputs = {
        name: "read" if isinstance(node, Read) else "ask"
        for name, node in plan.nodes.items()
        if isinstance(node, (Read, Call))
    }
    judge = Remembered(client) if client is not None else None
    baseline = decisions(DecisionProgram(plan, judge, model), states)
    found = []
    for name, kind in inputs.items():
        suppressed = DecisionProgram(plan, judge, model, ExecutionConfig(suppress=(name,)))
        changed = sum(
            mine != theirs for mine, theirs in zip(baseline, decisions(suppressed, states), strict=True)
        )
        found.append(Effect(node=name, kind=kind, changed=changed, decided=len(states),
                            reason=blank(plan, name, states) if not changed else ""))
    return sorted(found, key=lambda effect: (-effect.changed, effect.node))


def blank(plan: Plan, name: str, states: list[State]) -> str:
    """Why an input changed nothing, when the graph can tell. A read whose path is not in the
    state is a different problem from a rule that never wins, and only one of them is a typo."""
    node = plan.nodes[name]
    if not isinstance(node, Read):
        return "the graph decides the same without it"
    present = sum(read_path(node, state) is not MISSING for state in states)
    if not present:
        return "this path is never in the state"
    return f"present in {present} of {len(states)} states and changes no decision"
