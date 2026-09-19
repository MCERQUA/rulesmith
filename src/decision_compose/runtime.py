"""Execute typed DAGs as DSPy modules through the official TypeSafe client."""

import json
import math
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from graphlib import TopologicalSorter

import dspy
from pydantic import ConfigDict, Field, TypeAdapter
from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer, TypeSafeClient

from decision_compose.graph import (
    All,
    Call,
    Choice,
    Coalesce,
    Compare,
    Condition,
    LiteralInput,
    Merge,
    Node,
    NodeInput,
    Not,
    Noul,
    Plan,
    Present,
    Question,
    Read,
    Score,
    Source,
    State,
    StateInput,
    StrictModel,
    Switch,
    Value,
)


class ResponseError(ValueError):
    """The provider returned an answer that violates the requested contract."""


def in_range(value: float, upper: float) -> bool:
    return math.isfinite(value) and 0 <= value <= upper


def read_answer(question: Question, answer: NoulAnswer | ChoiceAnswer | ScoreAnswer) -> dict:
    expected = {Noul: NoulAnswer, Choice: ChoiceAnswer, Score: ScoreAnswer}
    if not isinstance(answer, expected[type(question)]):
        raise ResponseError("answer type does not match question type")
    if isinstance(answer, NoulAnswer):
        if not in_range(answer.noul, 1):
            raise ResponseError("Noul probability must be finite and between 0 and 1")
        return {"type": "noul", "value": answer.noul}

    keys = (
        set(question.criteria)
        if isinstance(question, Choice)
        else set(range(len(question.criteria)))
    )
    probabilities = answer.probabilities
    if set(probabilities) != keys or not all(in_range(p, 1) for p in probabilities.values()):
        raise ResponseError(
            "probabilities must cover exactly the declared options and lie in [0, 1]"
        )
    # The hosted service may round individual probabilities in its response.
    if not math.isclose(sum(probabilities.values()), 1, abs_tol=0.02):
        raise ResponseError("probabilities do not sum to 1 within rounding tolerance")
    if not in_range(answer.confidence, 1):
        raise ResponseError("confidence must be finite and between 0 and 1")

    if isinstance(answer, ChoiceAnswer):
        if answer.choice not in keys:
            raise ResponseError("selected option is outside the declared options")
        value = answer.choice
    else:
        if not in_range(answer.score, len(question.criteria) - 1):
            raise ResponseError("score is outside the declared rubric")
        if answer.legend != dict(enumerate(question.criteria)):
            raise ResponseError("returned rubric differs from the requested rubric")
        value = answer.score
    return {
        "type": question.type,
        "value": value,
        "probabilities": probabilities,
        "confidence": answer.confidence,
    }


class GraphExecutionError(ValueError):
    """A valid graph failed to produce a required output for this input."""

    def __init__(self, message: str, prediction: dspy.Prediction):
        super().__init__(message)
        self.prediction = prediction


class ExecutionConfig(StrictModel):
    max_workers: int = Field(default=4, ge=1)


# JSON null is a valid value, so absence needs a distinct sentinel.
MISSING = object()


def matches(condition: Condition | None, values: dict) -> bool:
    if condition is None:
        return True
    if isinstance(condition, Present):
        return condition.node in values
    if isinstance(condition, Compare):
        if condition.node not in values:
            return False
        value = values[condition.node]
        if condition.op == "eq":
            return value == condition.value
        return value < condition.value if condition.op == "lt" else value >= condition.value
    if isinstance(condition, Not):
        return not matches(condition.condition, values)
    if isinstance(condition, All):
        return all(matches(child, values) for child in condition.conditions)
    return any(matches(child, values) for child in condition.conditions)


def resolve(source: Source, state: State, values: dict, answers: dict):
    if isinstance(source, StateInput):
        return state
    if isinstance(source, LiteralInput):
        return deepcopy(source.value)
    return (answers if source.view == "answer" else values).get(source.node, MISSING)


def resolve_inputs(inputs: dict[str, Source], state: State, values: dict, answers: dict):
    result = {}
    for name, source in inputs.items():
        value = resolve(source, state, values, answers)
        if value is MISSING:
            if isinstance(source, NodeInput) and source.optional:
                continue
            return MISSING
        result[name] = value
    return result


def read_path(node: Read, state: State):
    value = state
    for step in node.path:
        if isinstance(step, int) and isinstance(value, list) and -len(value) <= step < len(value):
            value = value[step]
        elif isinstance(step, str) and isinstance(value, dict) and step in value:
            value = value[step]
        else:
            return MISSING
    if node.categories is not None:
        allowed = isinstance(value, str) and value in node.categories
    else:
        low, high = node.bounds
        allowed = isinstance(value, int | float) and not isinstance(value, bool)
        allowed = allowed and low <= value <= high
    # A value outside the declared domain is skipped like a missing path, so every value
    # downstream nodes see stays within the domain validation relied on.
    return value if allowed else MISSING


def resolve_node(node: Node, state: State, values: dict, answers: dict):
    """Resolve deterministic values, or a call's input object; skipped nodes stay absent."""
    if not matches(node.when, values):
        return MISSING
    if isinstance(node, Value):
        return deepcopy(node.value)
    if isinstance(node, (Call, Merge)):
        return resolve_inputs(node.inputs, state, values, answers)
    if isinstance(node, Coalesce):
        return next((values[source] for source in node.sources if source in values), MISSING)
    if isinstance(node, Read):
        return read_path(node, state)
    if node.source not in values:
        return MISSING
    value = values[node.source]
    if isinstance(node, Switch):
        return node.cases[value]
    return node.above if value >= node.cutoff else node.below


class DecisionProgram(dspy.Module):
    def __init__(
        self,
        plan: Plan,
        client: TypeSafeClient | None,
        model: str,
        execution: ExecutionConfig | None = None,
    ):
        super().__init__()
        self.plan = plan
        self.client = client
        self.model = model
        self.execution = execution or ExecutionConfig()
        if client is None and any(isinstance(node, Call) for node in plan.nodes.values()):
            raise ValueError("graphs containing calls require a TypeSafe client")

    def _ask(self, calls: list[tuple[str, Call]], state: dict) -> dict[str, tuple[dict, dict]]:
        """Ask every call sharing one input object in a single request."""
        questions = {key: node.question.model_dump() for key, node in calls}
        response = self.client.system_one(state=state, questions=questions, model=self.model)
        if set(response.answers) != set(questions):
            raise ResponseError("answer IDs do not match the requested question IDs")
        results = {}
        for index, (key, node) in enumerate(calls):
            # A request is billed once, so its usage is attributed to its first call only.
            first = index == 0
            usage = {
                "model": response.model,
                "input_tokens": response.usage.input_tokens if first else 0,
                "output_tokens": response.usage.output_tokens if first else 0,
                "request": sorted(questions),
            }
            results[key] = read_answer(node.question, response.answers[key]), usage
        return results

    def forward(self, state: State) -> dspy.Prediction:
        state = TypeAdapter(State, config=ConfigDict(allow_inf_nan=False)).validate_python(state)
        values, answers, calls = {}, {}, {}
        sorter = TopologicalSorter(self.plan.dependencies())
        sorter.prepare()
        ready, running = [], {}
        with ThreadPoolExecutor(max_workers=self.execution.max_workers) as pool:
            while sorter.is_active():
                # Drain locally executable nodes before waiting on network work.
                progressed = True
                while progressed:
                    ready.extend(sorter.get_ready())
                    deferred, progressed, requests = [], False, {}
                    for key in sorted(ready):
                        node = self.plan.nodes[key]
                        value = resolve_node(node, state, values, answers)
                        if value is MISSING:
                            sorter.done(key)
                            progressed = True
                        elif isinstance(node, Call):
                            group = json.dumps(value, sort_keys=True)
                            requests.setdefault(group, (value, []))[1].append((key, node))
                        else:
                            values[key] = value
                            sorter.done(key)
                            progressed = True
                    for value, group in requests.values():
                        if len(running) < self.execution.max_workers:
                            running[pool.submit(self._ask, group, value)] = group[0][0]
                        else:
                            deferred.extend(key for key, _ in group)
                    ready = deferred
                if running:
                    completed, _ = wait(running, return_when=FIRST_COMPLETED)
                    for future in sorted(completed, key=running.get):
                        running.pop(future)
                        for key, (answer, usage) in sorted(future.result().items()):
                            answers[key], calls[key] = answer, usage
                            values[key] = answer["value"]
                            sorter.done(key)

        outputs, missing = {}, []
        for name, source in self.plan.outputs.items():
            value = resolve(source, state, values, answers)
            if value is not MISSING:
                outputs[name] = value
            elif not source.optional:
                missing.append(name)
        order = self.plan.order()
        result = dict(
            outputs=outputs,
            node_values={key: values[key] for key in order if key in values},
            answers={key: answers[key] for key in order if key in answers},
            skipped=[key for key in order if key not in values],
            calls={key: calls[key] for key in order if key in calls},
            models=sorted({call["model"] for call in calls.values()}),
            input_tokens=sum(call["input_tokens"] for call in calls.values()),
            output_tokens=sum(call["output_tokens"] for call in calls.values()),
        )
        if "label" in outputs:
            result["label"] = outputs["label"]
        prediction = dspy.Prediction(**result)
        if missing:
            raise GraphExecutionError(f"required graph outputs were skipped: {missing}", prediction)
        return prediction
