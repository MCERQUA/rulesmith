"""GEPA searches the graph language; the evaluator owns the task contract."""

import hashlib
import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib.metadata import version
from typing import Protocol

import dspy
import gepa
from dspy.utils.exceptions import AdapterParseError
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from pydantic import Field, ValidationError
from typesafe_sdk import TypeSafeClient

from decision_compose.graph import Example, Plan, StrictModel
from decision_compose.rules import RuleError, parts, whole
from decision_compose.runtime import DecisionProgram, ExecutionConfig, GraphExecutionError


def text_adapter() -> dspy.ChatAdapter:
    """Plain-text fields work on any chat endpoint; JSON mode is not universal (LM Studio)."""
    return dspy.ChatAdapter(use_json_adapter_fallback=False)


class GraphProposal(dspy.Signature):
    """Revise the requested part of a decision program, using the feedback supplied."""

    request: str = dspy.InputField()
    # Rules are text, and so is the whole-graph JSON a program beyond rules falls back to.
    graph: str = dspy.OutputField(desc="The requested part, complete and alone")


class GraphProposer(dspy.Module):
    def __init__(self, lm: dspy.LM):
        super().__init__()
        self.lm = lm
        self.predict = dspy.Predict(GraphProposal)

    def forward(self, prompt: str | list[dict]) -> str:
        request = prompt if isinstance(prompt, str) else json.dumps(prompt)
        # Syntax failures remain candidates so GEPA can score and reflect on them.
        with dspy.context(lm=self.lm, adapter=text_adapter()):
            try:
                answer = self.predict(request=request).graph
            except AdapterParseError as error:
                answer = headless_part(error)
            except ValidationError as error:
                answer = json.dumps({"generation_error": str(error)})
        return unfenced(answer if isinstance(answer, str) else json.dumps(answer))


def headless_part(error: AdapterParseError) -> str:
    """The answer from a reply that carries no field header. Models with thinking off answer
    with the part alone, sometimes with a sentence either side of it."""
    return error.lm_response


def unfenced(answer: str) -> str:
    """The answer without the code fence models wrap it in, and without the sentence some
    write before a whole-graph JSON reply."""
    text = answer.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.rsplit("```", 1)[0].strip()
    if not text.startswith("{") and '"nodes"' in text:
        start = text.find("{")
        while start != -1:
            try:
                graph, end = json.JSONDecoder().raw_decode(text[start:])
            except json.JSONDecodeError:
                start = text.find("{", start + 1)
                continue
            if isinstance(graph, dict) and "nodes" in graph:
                return text[start : start + end]
            start = text.find("{", start + 1)
    return text


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
    parallel_examples: int = Field(
        default=1, ge=1, description="Examples scored at once; scorers sharing one game need 1"
    )
    merges: int = Field(
        default=5, ge=0, description="Times search may combine two candidates' parts"
    )


def score_all(score: Scorer, program: DecisionProgram, examples: Sequence, parallel: int) -> list:
    """Every example's outcome, in order, scoring up to `parallel` examples at once."""
    with ThreadPoolExecutor(parallel) as pool:
        return list(pool.map(lambda example: score(program, example), examples))


def components(plan: Plan) -> dict[str, str]:
    """The candidate a proposer revises: rules, not the graph they compile to."""
    return parts(plan)


def assemble(candidate: dict[str, str]) -> Plan:
    """The plan a candidate describes, however its parts were rewritten."""
    try:
        return whole(candidate)
    except RuleError as error:
        raise ValueError(str(error)) from error


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
        # A graph scores the same however often GEPA asks, and the champion is re-scored
        # every round; without this the seed pays for its own validation over and over.
        self.scored: dict[tuple[str, str], Outcome] = {}

    def parse(self, candidate: dict[str, str]) -> Plan:
        plan = assemble(candidate)
        plan.check(self.task.labels, self.config.max_nodes, self.config.max_depth)
        return plan

    def remember(self, plan: Plan, program: DecisionProgram, batch: Sequence) -> list:
        """Every example's outcome, scoring only the ones this graph has not seen."""
        graph = plan.model_dump_json()
        wanted = [e for e in batch if (graph, str(e)) not in self.scored]
        fresh = score_all(self.score, program, wanted, self.config.parallel_examples)
        self.scored.update({(graph, str(e)): o for e, o in zip(wanted, fresh, strict=True)})
        return [self.scored[graph, str(example)] for example in batch]

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
        try:
            outcomes = self.remember(plan, program, batch)
        except Exception as error:
            # GEPA catches some adapter failures internally, even with raise_on_exception.
            self.failure = error
            raise
        for outcome in outcomes:
            self.resolved_models.update(outcome.models)
            outputs.append(outcome.record)
            scores.append(outcome.score)
            records.append(outcome.trace)
        return self.batch(candidate, outputs, scores, records if capture_traces else None)

    def batch(self, candidate, outputs, scores, trajectories) -> EvaluationBatch:
        if trajectories is not None and self.proposed in candidate.values():
            self.proposal_feedback = {
                "graph": self.proposed,
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

    def reflect(prompt, candidate, component):
        nonlocal reflection_calls
        if adapter.failure is not None:
            raise adapter.failure
        # Name the part, and show the rest, so a rule can use a reading that already exists.
        context = "\n".join(f"{part}:\n{text}" for part, text in candidate.items() if text)
        prompt = (
            f"{prompt}\n\nReturn the part named {component}.\n"
            f"The whole program as it stands:\n{context}"
        )
        previous = adapter.proposal_feedback
        if previous is not None:
            feedback = (
                "Your previous proposal was scored on training examples (1 is best per "
                "example) and discarded unless it beat the current graph. Do not repeat its "
                f"mistakes.\nPrevious proposal:\n{previous['graph']}\n"
                f"Scores: {previous['scores']}\nResults: {json.dumps(previous['results'])}"
            )
            prompt = f"{prompt}\n\n{feedback}"
            adapter.proposal_feedback = None
        nonlocal repairs
        request = prompt
        for attempt in range(config.repair_attempts + 1):
            try:
                reflection_calls += 1
                proposal = reflection_lm(request).strip()
            except Exception as error:
                # Preserve the provider failure if GEPA treats it as an unsuccessful mutation.
                adapter.failure = error
                raise
            try:
                adapter.parse(candidate | {component: proposal})
                break
            except ValueError as error:
                if attempt == config.repair_attempts:
                    break
                repairs += 1
                # Validation is instant, so fix schema slips before spending episodes on them.
                repair = (
                    f"This graph failed validation and was not run:\n{proposal}\n"
                    f"Error: {error}\nReturn only the corrected {component} component."
                )
                # Only the latest attempt matters; earlier failures would bloat the prompt.
                request = f"{prompt}\n\n{repair}"
        adapter.proposed = proposal
        return proposal

    prompt = (
        "Improve this directed acyclic graph using observed execution failures. Return only "
        "the replacement component JSON object, with nodes and, for policy, outputs. "
        "Add, remove, or rewrite "
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
        "categories. Reads give exact values for thresholds, switches, and conditions; calls "
        "let the judge weigh the whole input, which suits decisions no single value settles, "
        "such as whether to fight or fall back. Use whichever scores better. "
        "A call runs on every input that does not skip it, so in a game it runs every step; "
        "gate calls with when-conditions on reads, such as only while a target is present, "
        "so the judge is asked only where no read settles the decision. "
        "A read's path must end at a number or string, never an "
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
        "Use the task's full label set when appropriate.\n"
        "Write the program as rules, tried in order, the first whose condition holds "
        "deciding. Declare what it may look at: 'read target.x in [-1, 1]', "
        "'read route.next one of [path, door, exit]', or a judgement "
        '\'ask threat = "question" [fight: "when it applies"] [clear: "when it applies"]\', '
        "which may end 'when <condition>' to be asked only then. Then the rules: "
        "'when <condition>: <action>' or 'always: <action>', where an action is a label, an "
        "ask written the same way, or 'route.bearing >= 0 ? TURN_LEFT : TURN_RIGHT'. "
        "Conditions join with 'and', name a reading to test that it has a value, or compare "
        "one with >=, < or ==. The readings part holds declarations and the rules part holds "
        "rules; return the part you are asked for, complete and alone, except that a rules "
        "part may declare any reading its rules need. A program that rules cannot express "
        "may be returned as whole graph JSON instead, matching the schema above.\n"
        "Part to revise:\n<curr_param>\nObserved examples and feedback:\n<side_info>"
    )

    def propose(candidate, reflective_dataset, components_to_update):
        return {
            component: reflect(
                prompt.replace("<curr_param>", candidate[component]).replace(
                    "<side_info>", json.dumps(reflective_dataset[component])
                ),
                candidate,
                component,
            )
            for component in components_to_update
        }

    adapter.propose_new_texts = propose
    result = gepa.optimize(
        seed_candidate=components(initial),
        trainset=task.train,
        valset=task.validation,
        adapter=adapter,
        reflection_minibatch_size=batch_size,
        max_metric_calls=config.max_metric_calls,
        seed=config.seed,
        use_merge=config.merges > 0,
        max_merge_invocations=config.merges,
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
    parallel_examples: int = 1,
) -> dict:
    program = DecisionProgram(plan, client, model, execution)
    outcomes = score_all(score, program, examples, parallel_examples)
    return {
        "score": sum(outcome.score for outcome in outcomes) / len(outcomes),
        "examples": [outcome.record for outcome in outcomes],
    }
