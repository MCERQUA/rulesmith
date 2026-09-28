"""Fitting how sure a judge should read, and measuring how often each rule is right. A model
that says 0.97 and is right three times in five is overconfident, and a rule testing
`confidence >= 0.9` on it means nothing. One temperature per question, fitted on labeled examples,
makes its probabilities say what they claim. A rule claims nothing at all until it is measured:
counted on labeled examples, its decisions are as sure as it was right, and inputs that reach it
far more or less often than those did are no longer the inputs it was measured on."""

import math
from collections import Counter

from rulesmith.graph import (
    Call,
    Choice,
    Coalesce,
    Measured,
    NodeInput,
    Plan,
    Task,
    Threshold,
    Value,
)
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


def outcomes(plan: Plan, examples: list, client, model: str) -> list[dict]:
    """Each labeled example as the plan decided it: the node that decided, the judgments asked,
    and whether the answer was one the example accepts."""
    program = DecisionProgram(plan, client, model)
    found = []
    for example in examples:
        try:
            prediction = dict(program(state=example.state))
        except GraphExecutionError as failure:
            prediction = dict(failure.prediction)
        found.append(
            {
                "label": prediction.get("label"),
                "decided_by": prediction.get("decided_by"),
                "calls": prediction["calls"],
                "correct": prediction.get("label") in (example.label, *example.accepted),
            }
        )
    return found


def tally(records: list[dict]) -> dict[str | None, tuple[int, int]]:
    """For each node that decided an answer, how many of its answers were right and how many it
    gave. A constant graph decides with no node, under None."""
    counts: dict[str | None, tuple[int, int]] = {}
    for record in records:
        if record.get("label") is None:
            continue
        right, decided = counts.get(record["decided_by"], (0, 0))
        counts[record["decided_by"]] = (right + bool(record["correct"]), decided + 1)
    return counts


def rules(plan: Plan) -> list[str]:
    """The rules the label is decided by: every value or threshold in its list of fallbacks."""
    source = plan.outputs["label"]
    pending, found = [source.node] if isinstance(source, NodeInput) else [], []
    while pending:
        name = pending.pop(0)
        node = plan.nodes[name]
        if isinstance(node, Coalesce):
            pending = list(node.sources) + pending
        elif isinstance(node, (Value, Threshold)):
            found.append(name)
    return found


def measure(plan: Plan, task: Task, client, model: str) -> Plan:
    """The plan with every rule that decides its label counted on the task's labeled training
    and validation examples. A rule decides only some of them, so it is counted on all it can be;
    they are examples search has seen, so the counts flatter a searched graph, and a task of
    fresh examples measures it honestly."""
    examples = [e for e in [*task.train, *task.validation] if e.label is not None]
    if not examples or not rules(plan):
        return plan
    counts = tally(outcomes(plan, examples, client, model))
    nodes = plan.nodes | {
        name: plan.nodes[name].model_copy(
            update={
                "measured": Measured(
                    right=counts.get(name, (0, 0))[0],
                    decided=counts.get(name, (0, 0))[1],
                    examples=len(examples),
                )
            }
        )
        for name in rules(plan)
    }
    return Plan.model_validate(plan.model_copy(update={"nodes": nodes}).model_dump())


def unmeasured(plan: Plan) -> Plan:
    """The plan without its rules' counts, which describe this graph and no rewrite of it."""
    nodes = {
        name: node.model_copy(update={"measured": None})
        if isinstance(node, (Value, Threshold))
        else node
        for name, node in plan.nodes.items()
    }
    return plan.model_copy(update={"nodes": nodes})


def binomial_p(count: int, trials: int, share: float) -> float:
    """Exact two-sided binomial test: the chance, over `trials` examples each reaching a rule
    with probability `share`, of a count no likelier than `count`."""
    if share in (0, 1):
        return float(count == trials * share)

    def log_chance(k: int) -> float:
        ways = math.lgamma(trials + 1) - math.lgamma(k + 1) - math.lgamma(trials - k + 1)
        return ways + k * math.log(share) + (trials - k) * math.log1p(-share)

    # A relative tolerance, as exact tests use, so a tie with the observed count is not lost
    # to rounding.
    observed = log_chance(count) + 1e-7
    return min(1.0, sum(math.exp(c) for k in range(trials + 1) if (c := log_chance(k)) <= observed))


def drift(plan: Plan, records: list[dict], alpha: float) -> list[dict]:
    """Every measured rule these examples reach at a rate its measurement makes unlikely, below
    `alpha`: the inputs have moved from the ones it was counted on, and so may how often it is
    right."""
    reached = Counter(record.get("decided_by") for record in records)
    flagged = []
    for name, node in plan.nodes.items():
        measured = getattr(node, "measured", None)
        if measured is None:
            continue
        p = binomial_p(reached[name], len(records), measured.share)
        if p < alpha:
            flagged.append(
                {
                    "rule": name,
                    "measured_share": measured.share,
                    "share": reached[name] / len(records),
                    "p": p,
                }
            )
    return flagged
