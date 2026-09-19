"""GEPA searches the graph language; the evaluator owns the task contract."""

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import Protocol

import dspy
import gepa
from dspy.utils.exceptions import AdapterParseError
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from pydantic import Field, JsonValue, ValidationError
from typesafe_sdk import TypeSafeClient

from decision_compose.graph import Example, Plan, StrictModel
from decision_compose.runtime import DecisionProgram, ExecutionConfig, GraphExecutionError


def text_adapter() -> dspy.ChatAdapter:
    """Plain-text fields work on any chat endpoint; JSON mode is not universal (LM Studio)."""
    return dspy.ChatAdapter(use_json_adapter_fallback=False)


class GraphProposal(dspy.Signature):
    """Design a complete decision DAG that satisfies the supplied schema and feedback."""

    request: str = dspy.InputField()
    # One whole-graph field: models asked for nodes and outputs separately tend to
    # nest the complete graph inside nodes.
    graph: dict[str, JsonValue] = dspy.OutputField(
        desc='The complete graph: {"nodes": {...}, "outputs": {...}}'
    )


class GraphProposer(dspy.Module):
    def __init__(self, lm: dspy.LM):
        super().__init__()
        self.lm = lm
        self.predict = dspy.Predict(GraphProposal)

    def forward(self, prompt: str | list[dict]) -> str:
        request = prompt if isinstance(prompt, str) else json.dumps(prompt)
        with dspy.context(lm=self.lm, adapter=text_adapter()):
            try:
                result = self.predict(request=request)
            except (AdapterParseError, ValidationError) as error:
                # Syntax failures remain candidates so GEPA can score and reflect on them.
                return json.dumps({"generation_error": str(error)})
        try:
            return json.dumps(result.graph, allow_nan=False)
        except ValueError as error:
            return json.dumps({"generation_error": str(error)})


@dataclass(frozen=True)
class Outcome:
    """One scored example: its report row and the record reflection sees."""

    score: float
    record: dict
    trace: dict
    models: list[str]
    error: str | None = None


def classify(program: DecisionProgram, example: Example) -> Outcome:
    try:
        prediction, error = program(state=example.state), None
    except GraphExecutionError as failure:
        prediction, error = failure.prediction, f"Graph execution failed: {failure}"
    score = float(error is None and prediction.label == example.label)
    return Outcome(
        score=score,
        record={
            "id": example.id,
            "expected": example.label,
            "correct": bool(score),
            **dict(prediction),
        },
        trace={
            "Inputs": example.state,
            "Generated Outputs": dict(prediction),
            "Feedback": error or f"Expected label: {example.label}. Exact-match score: {score}.",
        },
        models=prediction.models,
        error=error,
    )


Scorer = Callable[[DecisionProgram, object], Outcome]


class SearchTask(Protocol):
    """What search needs from a task; scoring its examples is the scorer's job."""

    description: str
    labels: list[str]
    train: Sequence
    validation: Sequence

    def model_dump_json(self) -> str: ...


class SearchConfig(StrictModel):
    max_metric_calls: int = Field(default=100, ge=1)
    max_nodes: int = Field(default=16, ge=1)
    max_depth: int = Field(default=8, ge=1)
    reflection_batch: int = Field(default=3, ge=1)
    repair_attempts: int = Field(
        default=2, ge=0, description="Times an invalid proposal is sent back with its error"
    )
    seed: int = 0


class GraphAdapter(GEPAAdapter[object, dict, dict]):
    def __init__(
        self,
        task: SearchTask,
        client: TypeSafeClient,
        model: str,
        config: SearchConfig,
        execution: ExecutionConfig | None = None,
        score: Scorer = classify,
    ):
        self.task = task
        self.score = score
        self.client = client
        self.model = model
        self.config = config
        self.execution = execution or ExecutionConfig()
        self.failure: Exception | None = None
        self.resolved_models: set[str] = set()
        # GEPA discards proposals that do not beat their parent, along with their
        # results; keep the latest one so the next reflection can learn from it.
        self.proposed: str | None = None
        self.proposal_feedback: dict | None = None
        self.invalid_candidate_evaluations = 0

    def parse(self, candidate: dict[str, str]) -> Plan:
        plan = Plan.model_validate_json(candidate["graph"])
        plan.check(self.task.labels, self.config.max_nodes, self.config.max_depth)
        return plan

    def evaluate(self, batch, candidate, capture_traces=False):
        if self.failure is not None:
            raise self.failure
        outputs, scores, records = [], [], []
        try:
            plan = self.parse(candidate)
        except ValueError as error:
            self.invalid_candidate_evaluations += 1
            feedback = {"Feedback": f"Candidate validation failed: {error}"}
            return self.batch(
                candidate,
                [None] * len(batch),
                [0.0] * len(batch),
                [feedback] * len(batch) if capture_traces else None,
            )
        program = DecisionProgram(plan, self.client, self.model, self.execution)
        for example in batch:
            try:
                outcome = self.score(program, example)
            except Exception as error:
                # GEPA catches some adapter failures internally, even with raise_on_exception.
                self.failure = error
                raise
            self.resolved_models.update(outcome.models)
            outputs.append(outcome.record)
            scores.append(outcome.score)
            records.append(outcome.trace)
        return self.batch(candidate, outputs, scores, records if capture_traces else None)

    def batch(self, candidate, outputs, scores, trajectories) -> EvaluationBatch:
        if trajectories is not None and candidate["graph"] == self.proposed:
            self.proposal_feedback = {
                "graph": candidate["graph"],
                "scores": scores,
                "results": trajectories,
            }
            self.proposed = None
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {component: eval_batch.trajectories for component in components_to_update}


def optimize_plan(
    task: SearchTask,
    initial: Plan,
    client: TypeSafeClient,
    model: str,
    reflection_lm: Callable,
    config: SearchConfig,
    execution: ExecutionConfig | None = None,
    score: Scorer = classify,
) -> tuple[Plan, dict]:
    initial.check(task.labels, config.max_nodes, config.max_depth)
    batch_size = min(config.reflection_batch, len(task.train))
    minimum_budget = 2 * (len(task.validation) + batch_size)
    if config.max_metric_calls < minimum_budget:
        raise ValueError(f"max_metric_calls must be at least {minimum_budget} for one search round")
    adapter = GraphAdapter(task, client, model, config, execution, score)
    reflection_calls = 0
    repairs = 0
    syntax_example = Plan.model_validate(
        {
            "nodes": {
                "decision": {
                    "type": "call",
                    "question": {
                        "type": "choice",
                        "instructions": task.description,
                        "criteria": {
                            label: f"When the input calls for {label}." for label in task.labels[:2]
                        },
                    },
                    "inputs": {"request": {"type": "state"}},
                }
            },
            "outputs": {"label": {"type": "node", "node": "decision"}},
        }
    )

    def reflect(prompt):
        nonlocal reflection_calls
        if adapter.failure is not None:
            raise adapter.failure
        previous = adapter.proposal_feedback
        if previous is not None:
            feedback = (
                "Your previous proposal was scored on training examples (1 is best per "
                "example) and discarded unless it beat the current graph. Do not repeat its "
                f"mistakes.\nPrevious proposal:\n{previous['graph']}\n"
                f"Scores: {previous['scores']}\nResults: {json.dumps(previous['results'])}"
            )
            prompt = (
                [*prompt, {"role": "user", "content": feedback}]
                if isinstance(prompt, list)
                else f"{prompt}\n\n{feedback}"
            )
            adapter.proposal_feedback = None
        nonlocal repairs
        for attempt in range(config.repair_attempts + 1):
            try:
                reflection_calls += 1
                proposal = reflection_lm(prompt).strip()
            except Exception as error:
                # Preserve the provider failure if GEPA treats it as an unsuccessful mutation.
                adapter.failure = error
                raise
            try:
                adapter.parse({"graph": proposal})
                break
            except ValueError as error:
                if attempt == config.repair_attempts:
                    break
                repairs += 1
                # Validation is instant, so fix schema slips before spending episodes on them.
                repair = (
                    f"This graph failed validation and was not run:\n{proposal}\n"
                    f"Error: {error}\nReturn the complete corrected graph."
                )
                prompt = (
                    [*prompt, {"role": "user", "content": repair}]
                    if isinstance(prompt, list)
                    else f"{prompt}\n\n{repair}"
                )
        adapter.proposed = proposal
        # GEPA keeps only the text between the first and last ``` of a reply, so fence the
        # proposal ourselves; backticks inside the graph then survive intact.
        return f"```\n{adapter.proposed}\n```"

    prompt = (
        "Improve this directed acyclic graph using observed execution failures. Return only "
        "the complete replacement JSON object, with nodes and outputs. Add, remove, or rewrite "
        "nodes, questions, input wiring, and conditions. Start from the current graph: keep "
        "the nodes, IDs, and conditions that work, and change only what the feedback shows is "
        "failing, since rewriting a working graph from scratch usually loses what worked. "
        "Each call receives its mapped inputs, "
        "which can reference original state, literal data, or earlier node values/answers. "
        "Ready calls with identical inputs share one request; others run concurrently. "
        "A call's question MUST be an object nested inside the call, containing lowercase "
        "type, instructions, and criteria when applicable. Choice criteria map each option, "
        "which is the value the node outputs, to a description of when it applies. "
        "Never put instructions or criteria "
        'beside question. Original input uses {"type":"state"}; it is not a node named input. '
        'Every input and output source needs its type, e.g. {"type":"node","node":"decision"}; '
        "a source without type is rejected. A choice call's value is the option it picks; a "
        "score call's value is the expected position on its ordered levels; a noul call's value "
        "is the probability that its yes/no statement is true. Calls judge; they cannot extract "
        "or compute numbers from their inputs. "
        "A read node takes one value from the original input by path, e.g. "
        '{"type":"read","path":["rangefinder",3],"bounds":[0,2048]}, declaring bounds or '
        "categories; use reads plus threshold, switch, and conditions for exact checks, and "
        "calls only for judgment. A read's path must end at a number or string, never an "
        "object or list. A missing path or a value outside the declared domain skips the "
        "read, so test whether a field exists with "
        '{"type":"present","node":"<read of a scalar inside it>"}. The label must always get '
        "a value: end the final coalesce with an unconditional value node as the default. "
        "Only declared node IDs may be referenced; there is no input_wiring field. "
        "Use view=value for a label output, not view=answer (which returns an object). "
        "Node and control references must form a DAG; every node must feed an output. "
        "Conditions gate execution after dependencies settle. Missing required inputs skip "
        "a node; optional missing inputs are omitted. merge builds an object. coalesce waits "
        "for all sources and picks the first available in listed order. All skipped sources "
        "skip the coalesce. A required missing output invalidates that execution. "
        "Switch cases must cover the source's categorical values; threshold uses >= for above. "
        "The label output must have a statically known domain within the allowed labels. "
        "Do not encode example-specific answers. Prefer small generalizable graphs.\n"
        f"Task: {task.description}\nAllowed label values: {json.dumps(task.labels)}\n"
        f"At most {config.max_nodes} nodes and dependency depth {config.max_depth}.\n"
        f"JSON schema: {json.dumps(Plan.model_json_schema())}\n"
        "Minimal syntax example using two labels (not a solution):\n"
        f"{syntax_example.model_dump_json(exclude_none=True)}\n"
        "Use the task's full label set when appropriate. You may replace this topology "
        "with any valid DAG, including multiple calls, branches, and joins.\n"
        "Current graph:\n<curr_param>\nObserved examples and feedback:\n<side_info>"
    )
    result = gepa.optimize(
        seed_candidate={"graph": initial.model_dump_json()},
        trainset=task.train,
        valset=task.validation,
        adapter=adapter,
        reflection_lm=reflect,
        reflection_prompt_template=prompt,
        reflection_minibatch_size=batch_size,
        max_metric_calls=config.max_metric_calls,
        seed=config.seed,
        display_progress_bar=False,
        raise_on_exception=True,
        stop_callbacks=lambda state: adapter.failure is not None,
    )
    if adapter.failure is not None:
        raise adapter.failure
    best = adapter.parse(result.best_candidate)
    report = {
        "model": model,
        "resolved_models": sorted(adapter.resolved_models),
        "task_sha256": hashlib.sha256(task.model_dump_json().encode()).hexdigest(),
        "seed_sha256": hashlib.sha256(initial.model_dump_json().encode()).hexdigest(),
        "versions": {name: version(name) for name in ("dspy", "gepa", "typesafe-sdk")},
        "search": config.model_dump(),
        "execution": adapter.execution.model_dump(),
        "seed_validation_score": result.val_aggregate_scores[0],
        "validation_score": result.val_aggregate_scores[result.best_idx],
        "validation_scores": result.val_aggregate_scores,
        "candidates": len(result.candidates),
        "reflection_calls": reflection_calls,
        "repairs": repairs,
        "invalid_candidate_evaluations": adapter.invalid_candidate_evaluations,
        "metric_calls": result.total_metric_calls,
    }
    return best, report


def evaluate_plan(
    plan: Plan,
    examples: Sequence,
    client: TypeSafeClient | None,
    model: str,
    execution: ExecutionConfig | None = None,
    score: Scorer = classify,
) -> dict:
    program = DecisionProgram(plan, client, model, execution)
    outcomes = [score(program, example) for example in examples]
    return {
        "score": sum(outcome.score for outcome in outcomes) / len(outcomes),
        "examples": [outcome.record for outcome in outcomes],
    }
