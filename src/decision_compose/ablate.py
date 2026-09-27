"""What each input is worth: switch one off and count the decisions that change.

A reading that changes nothing is either a path the state never had, a rule an earlier one always
beats to the decision, or a threshold real data never trips. All three look identical from a score
-- the graph runs, decides something, and scores like a graph that simply decides badly.
"""

from dataclasses import dataclass

from decision_compose.graph import Call, Plan, Read, State
from decision_compose.runtime import DecisionProgram, ExecutionConfig, GraphExecutionError


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
    """Every read and every judgement, ranked by how much the graph would change without it.

    The judge is asked once per state for the baseline and never again: taking away a reading
    cannot change what a judge said about the same state, so the replays reuse those answers and
    a graph full of questions ablates for free."""
    inputs = {
        name: "read" if isinstance(node, Read) else "ask"
        for name, node in plan.nodes.items()
        if isinstance(node, (Read, Call))
    }
    baseline = decisions(DecisionProgram(plan, client, model), states)
    found = []
    for name, kind in inputs.items():
        suppressed = DecisionProgram(plan, client, model, ExecutionConfig(suppress={name}))
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
    from decision_compose.runtime import MISSING, read_path

    present = sum(read_path(node, state) is not MISSING for state in states)
    if not present:
        return "this path is never in the state"
    return f"present in {present} of {len(states)} states and changes no decision"
