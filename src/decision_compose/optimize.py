"""GEPA searches the tree language; the evaluator owns the task contract."""

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
from decision_compose.runtime import DecisionProgram


class SearchConfig(StrictModel):
    max_metric_calls: int = Field(default=100, ge=1)
    max_questions: int = Field(default=8, ge=1)
    max_depth: int = Field(default=6, ge=1)
    reflection_batch: int = Field(default=3, ge=1)
    seed: int = 0


class TreeAdapter(GEPAAdapter[Example, dict, dspy.Prediction]):
    def __init__(self, task: Task, client: TypeSafeClient, model: str, config: SearchConfig):
        self.task = task
        self.client = client
        self.model = model
        self.config = config
        self.failure: Exception | None = None
        self.resolved_models: set[str] = set()

    def parse(self, candidate: dict[str, str]) -> Plan:
        plan = Plan.model_validate_json(candidate["graph"])
        plan.check(self.task.labels, self.config.max_questions, self.config.max_depth)
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
        program = DecisionProgram(plan, self.client, self.model)
        for example in batch:
            try:
                prediction = program(state=example.state)
            except Exception as error:
                # GEPA catches some adapter failures internally, even with raise_on_exception.
                self.failure = error
                raise
            score = float(prediction.label == example.label)
            if prediction.model is not None:
                self.resolved_models.add(prediction.model)
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
) -> tuple[Plan, dict]:
    initial.check(task.labels, config.max_questions, config.max_depth)
    batch_size = min(config.reflection_batch, len(task.train))
    minimum_budget = 2 * (len(task.validation) + batch_size)
    if config.max_metric_calls < minimum_budget:
        raise ValueError(f"max_metric_calls must be at least {minimum_budget} for one search round")
    adapter = TreeAdapter(task, client, model, config)

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
        "Improve this decision tree using observed execution failures. Return only the complete "
        "replacement JSON object, with questions and root. Add, remove, or rewrite questions "
        "and branches. Every question sees only the original input state, never sibling answers. "
        "Switch cases must cover every Choice option; thresholds use >= for above. "
        "Do not encode example-specific answers. Prefer small generalizable trees.\n"
        f"Task: {task.description}\nAllowed output labels: {json.dumps(task.labels)}\n"
        f"At most {config.max_questions} questions and {config.max_depth} decision levels.\n"
        f"JSON schema: {json.dumps(Plan.model_json_schema())}\n"
        "Current tree:\n<curr_param>\nObserved examples and feedback:\n<side_info>"
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
        "seed_validation_accuracy": result.val_aggregate_scores[0],
        "validation_accuracy": result.val_aggregate_scores[result.best_idx],
        "validation_scores": result.val_aggregate_scores,
        "candidates": len(result.candidates),
        "metric_calls": result.total_metric_calls,
    }
    return best, report


def evaluate_plan(plan: Plan, examples: list[Example], client: TypeSafeClient, model: str) -> dict:
    program = DecisionProgram(plan, client, model)
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
