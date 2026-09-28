"""Fitting how sure a judge should read. A model that says 0.97 and is right three times in five
is overconfident, and a rule testing `confidence >= 0.9` on it means nothing. One temperature per
question, fitted on labeled examples, makes its probabilities say what they claim."""

import math

from rulesmith.graph import Call, Choice, Plan, Task
from rulesmith.runtime import DecisionProgram, GraphExecutionError, calibrated

# Temperatures tried, evenly spaced in log from a fifth to five: calibration corrects a judge's
# confidence, it does not invert it.
TEMPERATURES = [math.exp(math.log(0.2) + index * (math.log(25) / 80)) for index in range(81)]


def fit_temperature(pairs: list[tuple[dict[str, float], str]]) -> float:
    """The temperature under which these answers give the right labels the most probability,
    judged by log loss. `pairs` holds each answer's probabilities and the label that was right."""

    def loss(temperature: float) -> float:
        total = 0.0
        for probabilities, right in pairs:
            tempered = calibrated(
                {"type": "choice", "value": right, "probabilities": probabilities}, temperature
            )
            total -= math.log(max(tempered["probabilities"][right], 1e-12))
        return total

    return round(min(TEMPERATURES, key=loss), 3)


def calibrate(plan: Plan, task: Task, client, model: str) -> tuple[Plan, dict[str, float]]:
    """The plan with a temperature fitted for every choice question over the task's labels, from
    how its answers on the training examples compare with their labels. Other questions have no
    labels to be right about and keep theirs."""
    fitted_nodes = {
        name: node
        for name, node in plan.nodes.items()
        if isinstance(node, Call)
        and isinstance(node.question, Choice)
        and set(node.question.criteria) <= set(task.labels)
    }
    raw = plan.model_copy(
        update={
            "nodes": plan.nodes
            | {
                name: node.model_copy(update={"temperature": 1})
                for name, node in fitted_nodes.items()
            }
        }
    )
    program = DecisionProgram(raw, client, model)
    pairs: dict[str, list] = {name: [] for name in fitted_nodes}
    for example in task.train:
        try:
            answers = program(state=example.state).answers
        except GraphExecutionError as failure:
            answers = failure.prediction.answers
        for name, node in fitted_nodes.items():
            if name in answers and example.label in node.question.criteria:
                pairs[name].append((answers[name]["probabilities"], example.label))
    temperatures = {name: fit_temperature(found) for name, found in pairs.items() if found}
    nodes = plan.nodes | {
        name: plan.nodes[name].model_copy(update={"temperature": temperature})
        for name, temperature in temperatures.items()
    }
    return Plan.model_validate(plan.model_copy(update={"nodes": nodes}).model_dump()), temperatures
