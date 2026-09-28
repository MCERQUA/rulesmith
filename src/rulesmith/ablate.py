"""What each input is worth: switch one off and count the decisions that change.

A reading that changes nothing is either a path the state never had, a rule an earlier one always
beats to the decision, or a threshold real data never trips. All three look identical from a score
-- the graph runs, decides something, and scores like a graph that simply decides badly.
"""

from dataclasses import dataclass

from rulesmith.graph import Call, Plan, Read, State, Sum
from rulesmith.rules import named_condition
from rulesmith.runtime import (
    MISSING,
    DecisionProgram,
    ExecutionConfig,
    GraphExecutionError,
    Remembered,
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


def decisions(program: DecisionProgram, states: list[State]) -> list[dict | None]:
    """Every output the graph gives each state, or None where it cannot decide at all. A graph
    may decide more than a label, and a change to any output is a changed decision."""
    decided = []
    for state in states:
        try:
            decided.append(program(state=state).outputs)
        except GraphExecutionError:
            decided.append(None)
    return decided


def effects(
    plan: Plan,
    states: list[State],
    client=None,
    model: str = "none",
    execution: ExecutionConfig | None = None,
) -> list[Effect]:
    """Every read, judgment and name, ranked by how much the graph would change without it.

    The judge is asked once per state for the baseline and never again: taking away a reading
    cannot change what a judge said about the same state, so the replays reuse those answers and
    a graph full of questions ablates for free."""
    inputs = {}
    for name, node in plan.nodes.items():
        if isinstance(node, (Read, Call)):
            inputs[name] = "read" if isinstance(node, Read) else "ask"
        elif isinstance(node, Sum) or named_condition(node):
            inputs[name] = "let"
    judge = Remembered(client) if client is not None else None
    execution = execution or ExecutionConfig()
    baseline = decisions(DecisionProgram(plan, judge, model, execution), states)
    found = []
    for name, kind in inputs.items():
        without = execution.model_copy(update={"suppress": (name,)})
        suppressed = DecisionProgram(plan, judge, model, without)
        changed = sum(
            mine != theirs
            for mine, theirs in zip(baseline, decisions(suppressed, states), strict=True)
        )
        found.append(
            Effect(
                node=name,
                kind=kind,
                changed=changed,
                decided=len(states),
                reason=blank(plan, name, states) if not changed else "",
            )
        )
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
