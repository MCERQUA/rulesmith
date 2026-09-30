"""Fitting how sure a judge should read, and measuring how often each rule is right. A model
that says 0.97 and is right three times in five is overconfident, and a rule testing
`confidence >= 0.9` on it means nothing. One temperature per question, fitted on labeled examples,
makes its probabilities say what they claim. A rule claims nothing at all until it is measured:
counted on labeled examples, its decisions are as sure as it was right, and inputs that reach it
far more or less often than those did are no longer the inputs it was measured on. Where a rule
draws its line on a judgment, `scam >= 0.5`, is a number the proposer guessed; fitted here, it is
the line between the judge's answers that gets the most labeled examples right and holds up on
examples held out of the fit."""

import math
from collections import Counter
from dataclasses import dataclass

from rulesmith.graph import (
    All,
    Any,
    Call,
    Choice,
    Coalesce,
    Compare,
    Condition,
    Confidence,
    Measured,
    NodeInput,
    Not,
    Noul,
    Plan,
    Score,
    Task,
    Threshold,
    Value,
)
from rulesmith.runtime import DecisionProgram, GraphExecutionError, calibrated, sureness

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


@dataclass(frozen=True)
class Cut:
    """One numeric line a rule draws on a judgment: a `>=` or `<` in a rule's condition, reached
    by `path` through its and/or/not groups, or a threshold node's cutoff (`path` None). `tests`
    is the judgment or confidence the line is drawn on."""

    node: str
    path: tuple[int, ...] | None
    tests: str

    def value(self, plan: Plan) -> float:
        if self.path is None:
            return plan.nodes[self.node].cutoff
        return descend(plan.nodes[self.node].when, self.path).value

    def moved(self, plan: Plan, value: float) -> Plan:
        """The plan with this line drawn at `value` and nothing else changed."""
        data = plan.model_dump()
        node = data["nodes"][self.node]
        if self.path is None:
            node["cutoff"] = value
        else:
            condition = node["when"]
            for index in self.path:
                condition = (
                    condition["conditions"][index]
                    if "conditions" in condition
                    else condition["condition"]
                )
            condition["value"] = value
        return Plan.model_validate(data)


def descend(condition: Condition, path: tuple[int, ...]) -> Compare:
    for index in path:
        condition = (
            condition.conditions[index]
            if isinstance(condition, (All, Any))
            else condition.condition
        )
    return condition


def cuts(plan: Plan) -> list[Cut]:
    """Every line a rule draws on a number the judge produced: a probability, a score's expected
    level, or how sure a judgment was. Lines drawn on readings are what the rule says about the
    input, not about the judge, and stay where they were written."""
    judged = {
        name
        for name, node in plan.nodes.items()
        if (isinstance(node, Call) and isinstance(node.question, (Noul, Score)))
        or isinstance(node, Confidence)
    }

    def compares(condition: Condition, path: tuple[int, ...]):
        if isinstance(condition, Compare):
            if condition.op != "eq" and condition.node in judged:
                yield path, condition.node
        elif isinstance(condition, Not):
            yield from compares(condition.condition, (*path, 0))
        elif isinstance(condition, (All, Any)):
            for index, child in enumerate(condition.conditions):
                yield from compares(child, (*path, index))

    found = []
    for name, node in plan.nodes.items():
        if isinstance(node, Threshold) and node.source in judged:
            found.append(Cut(name, None, node.source))
        if getattr(node, "when", None) is not None:
            found += [Cut(name, path, tests) for path, tests in compares(node.when, ())]
    return found


FOLDS = 5


def fit_cutoffs(plan: Plan, task: Task, client, model: str) -> tuple[Plan, list[dict]]:
    """The plan with every cut moved to where it gets the most training examples right, one cut
    at a time, holding the questions and the rules' words still. A move is kept only when it
    also wins held out: the training examples are dealt into folds, the best line is chosen
    without each fold in turn and scored on it, and those scores together must beat the line as
    it stands. One example on the far side of a line is not evidence to move it; a few are.
    Each kept move is reported with the line's old and new place and the training accuracy
    before and after; a refused one, with the same and `kept` false."""
    examples = [e for e in task.train if e.label is not None]
    lines = cuts(plan)
    if not examples or not lines:
        return plan, []
    seen = observed(plan, examples, client, model)
    report = []
    for _ in range(3):
        moved = False
        for cut in lines:
            current = cut.value(plan)
            candidates = midpoints(seen.get(cut.tests, []))
            if not candidates:
                continue
            scores = {
                value: correct(cut.moved(plan, value), examples, client, model)
                for value in candidates
            }
            scores[current] = correct(plan, examples, client, model)
            best = choose(scores, current)
            kept = best != current and held_out_wins(scores, current, len(examples))
            report.append(
                {
                    "rule": cut.node,
                    "tests": cut.tests,
                    "from": current,
                    "to": best if kept else current,
                    "tried": len(candidates),
                    "train_before": sum(scores[current]) / len(examples),
                    "train_after": sum(scores[best if kept else current]) / len(examples),
                    "kept": kept,
                }
            )
            if kept:
                plan = cut.moved(plan, best)
                moved = True
        if not moved:
            break
    return plan, report


def observed(plan: Plan, examples: list, client, model: str) -> dict[str, list[float]]:
    """The numbers each judgment and confidence took on the examples, the values a line can be
    drawn between."""
    program = DecisionProgram(plan, client, model)
    seen: dict[str, list[float]] = {}
    for example in examples:
        try:
            answers = program(state=example.state).answers
        except GraphExecutionError as failure:
            answers = failure.prediction.answers
        for name, node in plan.nodes.items():
            if (
                isinstance(node, Call)
                and name in answers
                and isinstance(node.question, (Noul, Score))
            ):
                seen.setdefault(name, []).append(answers[name]["value"])
            if isinstance(node, Confidence) and node.source in answers:
                seen.setdefault(name, []).append(sureness(answers[node.source]))
    return seen


def midpoints(values: list[float]) -> list[float]:
    """A line between each pair of neighboring values, rounded where rounding still separates
    them: any line between the same two answers decides the same examples, and a short number
    reads as a choice rather than an accident."""
    distinct = sorted(set(values))
    found = []
    for low, high in zip(distinct, distinct[1:], strict=False):
        exact = (low + high) / 2
        short = round(exact, 4)
        found.append(short if low < short < high else exact)
    return found


def correct(plan: Plan, examples: list, client, model: str) -> list[bool]:
    return [record["correct"] for record in outcomes(plan, examples, client, model)]


def choose(scores: dict[float, list[bool]], current: float, fold: int | None = None) -> float:
    """The line with the most right answers, outside `fold` if one is named; among equals, the
    one nearest to where the line stands, so a tie never moves it."""

    def total(value: float) -> int:
        return sum(right for index, right in enumerate(scores[value]) if index % FOLDS != fold)

    return max(scores, key=lambda value: (total(value), -abs(value - current)))


def held_out_wins(scores: dict[float, list[bool]], current: float, count: int) -> bool:
    """Whether choosing the line without each fold and scoring it on that fold beats, over all the
    folds, the line as it stands."""
    folds = min(FOLDS, count)
    held_out = 0
    for fold in range(folds):
        chosen = choose(scores, current, fold)
        held_out += sum(
            right for index, right in enumerate(scores[chosen]) if index % FOLDS == fold
        )
    return held_out > sum(scores[current])


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
