"""GEPA searches the graph language; the evaluator owns the task contract."""

import hashlib
import json
from collections.abc import Callable
from importlib.metadata import version

import dspy
import gepa
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from pydantic import Field
from typesafe_sdk import TypeSafeClient

from decision_compose.graph import Example, Plan, StrictModel, Task
from decision_compose.runtime import DecisionProgram, ExecutionConfig, GraphExecutionError


class SearchConfig(StrictModel):
    max_metric_calls: int = Field(default=100, ge=1)
    max_nodes: int = Field(default=16, ge=1)
    max_depth: int = Field(default=8, ge=1)
    reflection_batch: int = Field(default=3, ge=1)
    seed: int = 0


class GraphAdapter(GEPAAdapter[Example, dict, dspy.Prediction]):
    def __init__(
        self,
        task: Task,
        client: TypeSafeClient,
        model: str,
        config: SearchConfig,
        execution: ExecutionConfig | None = None,
    ):
        self.task = task
        self.client = client
        self.model = model
        self.config = config
        self.execution = execution or ExecutionConfig()
        self.failure: Exception | None = None
        self.resolved_models: set[str] = set()

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
            return EvaluationBatch(
                outputs=[None] * len(batch),
                scores=[0.0] * len(batch),
                trajectories=[{"Feedback": f"Candidate validation failed: {error}"} for _ in batch]
                if capture_traces
                else None,
            )
        program = DecisionProgram(plan, self.client, self.model, self.execution)
        for example in batch:
            try:
                prediction = program(state=example.state)
            except GraphExecutionError as error:
                self.resolved_models.update(error.prediction.models)
                outputs.append(error.prediction)
                scores.append(0.0)
                if capture_traces:
                    records.append(
                        {
                            "Inputs": example.state,
                            "Generated Outputs": dict(error.prediction),
                            "Feedback": f"Graph execution failed: {error}",
                        }
                    )
                continue
            except Exception as error:
                # GEPA catches some adapter failures internally, even with raise_on_exception.
                self.failure = error
                raise
            score = float(prediction.label == example.label)
            self.resolved_models.update(prediction.models)
            outputs.append(prediction)
            scores.append(score)
            if capture_traces:
                records.append(
                    {
                        "Inputs": example.state,
                        "Generated Outputs": dict(prediction),
                        "Feedback": f"Expected label: {example.label}. Exact-match score: {score}.",
                    }
                )
        return EvaluationBatch(
            outputs=outputs, scores=scores, trajectories=records if capture_traces else None
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {component: eval_batch.trajectories for component in components_to_update}


def optimize_plan(
    task: Task,
    initial: Plan,
    client: TypeSafeClient,
    model: str,
    reflection_lm: Callable,
    config: SearchConfig,
    execution: ExecutionConfig | None = None,
) -> tuple[Plan, dict]:
    initial.check(task.labels, config.max_nodes, config.max_depth)
    batch_size = min(config.reflection_batch, len(task.train))
    minimum_budget = 2 * (len(task.validation) + batch_size)
    if config.max_metric_calls < minimum_budget:
        raise ValueError(f"max_metric_calls must be at least {minimum_budget} for one search round")
    adapter = GraphAdapter(task, client, model, config, execution)

    def reflect(prompt):
        if adapter.failure is not None:
            raise adapter.failure
        try:
            return reflection_lm(prompt)
        except Exception as error:
            # Preserve the provider failure if GEPA treats it as an unsuccessful mutation.
            adapter.failure = error
            raise

    prompt = (
        "Improve this directed acyclic graph using observed execution failures. Return only "
        "the complete replacement JSON object, with nodes and outputs. Add, remove, or rewrite "
        "nodes, questions, input wiring, and conditions. Each call receives its mapped inputs, "
        "which can reference original state, literal data, or earlier node values/answers. "
        "Each call is a separate request; independent ready calls run concurrently. "
        "Node and control references must form a DAG; every node must feed an output. "
        "Conditions gate execution after dependencies settle. Missing required inputs skip "
        "a node; optional missing inputs are omitted. merge builds an object. coalesce waits "
        "for all sources and picks the first available in listed order. All skipped sources "
        "skip the coalesce. A required missing output invalidates that execution. "
        "Switch cases must cover the source's categorical values; threshold uses >= for above. "
        "The label output must have a statically known domain within the allowed labels. "
        "Do not encode example-specific answers. Prefer small generalizable graphs.\n"
        f"Task: {task.description}\nAllowed output labels: {json.dumps(task.labels)}\n"
        f"At most {config.max_nodes} nodes and dependency depth {config.max_depth}.\n"
        f"JSON schema: {json.dumps(Plan.model_json_schema())}\n"
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
        "seed_validation_accuracy": result.val_aggregate_scores[0],
        "validation_accuracy": result.val_aggregate_scores[result.best_idx],
        "validation_scores": result.val_aggregate_scores,
        "candidates": len(result.candidates),
        "metric_calls": result.total_metric_calls,
    }
    return best, report


def evaluate_plan(
    plan: Plan,
    examples: list[Example],
    client: TypeSafeClient | None,
    model: str,
    execution: ExecutionConfig | None = None,
) -> dict:
    program = DecisionProgram(plan, client, model, execution)
    rows = []
    for example in examples:
        prediction = program(state=example.state)
        rows.append(
            {
                "id": example.id,
                "expected": example.label,
                "correct": prediction.label == example.label,
                **dict(prediction),
            }
        )
    return {"accuracy": sum(row["correct"] for row in rows) / len(rows), "examples": rows}
