"""Public benchmarks: rulesmith against the baselines a reader would reach for first.

Each run draws a task from a public dataset with a seed, scores every method on the same held-out
test split, and records accuracy with the judge calls and tokens each example cost. A method a
run has already recorded is skipped, so an interrupted benchmark resumes where it stopped.
"""

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Literal

# pyarrow must load before dspy: DSPy 3.3 swaps numpy for a lazy proxy that breaks a compiled
# extension importing numpy afterwards ("numpy._core.multiarray failed to import").
import pyarrow.parquet as parquet  # isort: skip
import dspy
from dspy.utils.exceptions import AdapterParseError

from rulesmith.graph import Plan, State, Task
from rulesmith.optimize import (
    GraphProposer,
    Labels,
    SearchConfig,
    evaluate_plan,
    optimize_plan,
    text_adapter,
)
from rulesmith.rules import compile_rules
from rulesmith.runtime import DecisionProgram

Rows = dict[str, list[tuple[State, str]]]
COVERAGES = (0.8, 0.5)


@dataclass(frozen=True)
class Dataset:
    """A public dataset as rows of (input, label), pooled by the dataset's own splits. Training
    and validation draw from its 'train' pool and the test split from its 'test' pool, so nothing
    a search sees overlaps what it is scored on."""

    name: str
    description: str
    rows: Callable[[], Rows]
    describe: Callable[[str], str]
    # Draw every label equally, for a dataset whose rare label is the one that matters.
    balanced: bool = False
    # The rules a person would write in a few minutes, where a field obviously decides most cases.
    handwritten: str | None = None

    def labels(self) -> dict[str, str]:
        """Every label the dataset trains on, with what the direct question says about it."""
        found = sorted({label for _, label in self.rows()["train"]})
        return {label: self.describe(label) for label in found}


def sample(dataset: Dataset, train: int, validation: int, test: int, seed: int) -> Task:
    """A task of the requested sizes, the same for the same seed. Repeated inputs are dropped,
    since a task holds each input once."""
    pools = dataset.rows()
    rng = random.Random(seed)
    seen: set[str] = set()

    def order(pool: str) -> list[tuple[State, str]]:
        rows = list(pools[pool])
        rng.shuffle(rows)
        if not dataset.balanced:
            return rows
        by_label: dict[str, list] = {}
        for row in rows:
            by_label.setdefault(row[1], []).append(row)
        # Take one of each label in turn, so any prefix holds every label about equally.
        groups = [by_label[label] for label in sorted(by_label)]
        return [
            group[i] for i in range(max(map(len, groups))) for group in groups if i < len(group)
        ]

    def draw(pool: str, count: int) -> list[dict]:
        drawn = []
        for state, label in order(pool):
            key = json.dumps(state, sort_keys=True)
            if key not in seen:
                seen.add(key)
                drawn.append({"state": state, "label": label})
            if len(drawn) == count:
                return drawn
        raise ValueError(f"the {pool} pool of {dataset.name} holds fewer than {count} inputs")

    held = draw("test", test)
    searched = draw("train", train + validation)
    # A balanced draw alternates labels, so split it without handing one split every even row.
    rng.shuffle(searched)
    splits = {"train": searched[:train], "validation": searched[train:], "test": held}
    described = dataset.labels()
    return Task(
        description=dataset.description,
        labels=sorted(described),
        descriptions=described,
        **{
            split: [{"id": f"{split}-{index}", **row} for index, row in enumerate(rows)]
            for split, rows in splits.items()
        },
    )


def direct_plan(task: Task) -> Plan:
    """The simplest thing a judge can do: one choice question over every label, no graph."""
    return Plan.model_validate(
        {
            "nodes": {
                "label": {
                    "type": "call",
                    "question": {
                        "type": "choice",
                        "instructions": task.description,
                        "criteria": task.criteria(),
                    },
                    "inputs": {"observation": {"type": "state"}},
                }
            },
            "outputs": {"label": {"type": "node", "node": "label"}},
        }
    )


@dataclass(frozen=True)
class Answer:
    """One test example as a method answered it, with what the answer cost."""

    correct: bool
    # How sure the method was, to rank answers by; None for a method that cannot say.
    confidence: float | None = None
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    invalid: bool = False


def confidence(record: dict) -> float:
    """How sure a graph was of an answer: the least sure of the judgments it rests on. An answer
    the rules settled without asking rests on none and counts as sure, which is the point of
    writing the rule; a graph that failed to answer is not sure at all."""
    if record.get("error"):
        return 0.0
    sure = []
    for answer in record.get("answers", {}).values():
        if answer["type"] == "choice":
            sure.append(answer["probabilities"][answer["value"]])
        elif answer["type"] == "score":
            sure.append(answer["confidence"])
        else:
            sure.append(max(answer["value"], 1 - answer["value"]))
    return min(sure, default=1.0)


def graph_answers(records: list[dict]) -> list[Answer]:
    return [
        Answer(
            correct=bool(record["correct"]),
            confidence=confidence(record),
            calls=len(record.get("calls", {})),
            input_tokens=record.get("input_tokens", 0),
            output_tokens=record.get("output_tokens", 0),
            invalid=bool(record.get("error")),
        )
        for record in records
    ]


def selective(answers: list[Answer], coverage: float) -> float | None:
    """Accuracy on the share of examples a method was surest of: what it scores when it may pass
    the rest to a person. None for a method with no confidence to rank by."""
    if any(answer.confidence is None for answer in answers):
        return None
    ranked = sorted(answers, key=lambda answer: -answer.confidence)
    kept = ranked[: max(1, math.ceil(coverage * len(ranked)))]
    return mean(answer.correct for answer in kept)


def result(method: str, seed: int, answers: list[Answer], seconds: float, **extra) -> dict:
    count = len(answers)
    return {
        "method": method,
        "seed": seed,
        "accuracy": mean(answer.correct for answer in answers),
        **{f"accuracy_at_{round(k * 100)}": selective(answers, k) for k in COVERAGES},
        "calls_per_example": sum(answer.calls for answer in answers) / count,
        "tokens_per_example": sum(answer.input_tokens for answer in answers) / count,
        "output_tokens_per_example": sum(answer.output_tokens for answer in answers) / count,
        "seconds_per_example": seconds / count,
        "invalid_rate": mean(answer.invalid for answer in answers),
        # Kept per example, so two methods can be compared on the very same examples.
        "correct": [answer.correct for answer in answers],
        **extra,
    }


def record(method: str, seed: int, task: Task, plan: Plan, client, model: str, **extra) -> dict:
    """A graph scored on the task's test split."""
    started = time.perf_counter()
    scored = evaluate_plan(plan, task.test, client, model, Labels(task))
    seconds = time.perf_counter() - started
    return result(method, seed, graph_answers(scored["examples"]), seconds, **extra)


def majority(seed: int, task: Task) -> dict:
    run = record("majority", seed, task, task.baseline_plan(), None, "none")
    # One label for everything ranks nothing, so it has no coverage to trade.
    return run | {f"accuracy_at_{round(k * 100)}": None for k in COVERAGES}


def text(state: State) -> str:
    return state if isinstance(state, str) else json.dumps(state, sort_keys=True)


def tfidf(seed: int, task: Task) -> dict:
    """Logistic regression on word and word-pair counts, trained on everything search sees."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    seen = task.train + task.validation
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=2000),
    )
    model.fit([text(e.state) for e in seen], [e.label for e in seen])
    started = time.perf_counter()
    odds = model.predict_proba([text(e.state) for e in task.test])
    seconds = time.perf_counter() - started
    classes = list(model.classes_)
    answers = [
        Answer(
            correct=classes[max(range(len(row)), key=row.__getitem__)] == e.label,
            confidence=float(max(row)),
        )
        for row, e in zip(odds, task.test, strict=True)
    ]
    return result("tfidf", seed, answers, seconds)


def search(
    seed: int, task: Task, client, model: str, proposer, budget: int, call_cost: float
) -> dict:
    """rulesmith's search, seeded with the direct call, so the comparison that matters is
    whether a graph improves on asking the judge once."""
    # 50 validation examples reward a lucky candidate; keep the direct call unless the gain holds.
    config = SearchConfig(
        max_metric_calls=budget,
        seed=seed,
        call_cost=call_cost,
        significance=SIGNIFICANCE,
        mine=True,
    )
    best, report = optimize_plan(task, direct_plan(task), client, model, proposer, config)
    return record(
        "search",
        seed,
        task,
        best,
        client,
        model,
        validation=report["validation_score"],
        seed_validation=report["seed_validation_score"],
        kept_seed=report["kept_seed"],
        mined=report["mined"],
        nodes=len(best.nodes),
        graph=best.model_dump(mode="json"),
    )


def dspy_examples(split) -> list[dspy.Example]:
    return [dspy.Example(text=text(e.state), label=e.label).with_inputs("text") for e in split]


def dspy_answers(program: dspy.Module, task: Task, student: dspy.LM) -> tuple[list[Answer], float]:
    """A DSPy program's answers on the test split. A reply that names no allowed label is an
    invalid answer, counted wrong, since a typed judgment cannot give one."""
    answers, started = [], time.perf_counter()
    with dspy.context(lm=student, adapter=text_adapter()):
        for example in dspy_examples(task.test):
            before = len(student.history)
            try:
                label = program(text=example.text).label
            except AdapterParseError:
                label = None
            # A reply naming no allowed label, or a failure the program reported as no answer.
            invalid = label is None
            usage = [entry.get("usage") or {} for entry in student.history[before:]]
            answers.append(
                Answer(
                    correct=label == example.label,
                    calls=len(usage),
                    input_tokens=sum(u.get("prompt_tokens", 0) for u in usage),
                    output_tokens=sum(u.get("completion_tokens", 0) for u in usage),
                    invalid=invalid,
                )
            )
    return answers, time.perf_counter() - started


class Failsafe(dspy.Module):
    """A DSPy program whose failure on one example is an answer of None, scored zero, as DSPy's
    GEPA means a failed example to be. Its integration instead crashes the whole optimization
    when an example comes back missing, which ended runs over one dropped connection or one
    error inside DSPy. Each failure is logged; GEPA scores copies of the program, so a count kept
    here would miss theirs."""

    def __init__(self, program: dspy.Module):
        super().__init__()
        self.program = program
        self.failures = 0

    def forward(self, text: str) -> dspy.Prediction:
        try:
            return self.program(text=text)
        except Exception as error:  # any failure: DSPy counts it at failure_score, not a crash
            self.failures += 1
            logging.getLogger(__name__).warning("DSPy example failed: %r", error)
            return dspy.Prediction(label=None)


def dspy_baseline(
    seed: int,
    task: Task,
    student: dspy.LM,
    reflection: dspy.LM,
    budget: int,
    reasoning: bool = False,
) -> dict:
    """A standard DSPy classifier on a text-generating model, optimized with DSPy's own GEPA.
    With `reasoning` it thinks step by step first, the stronger and costlier standard form."""
    signature = dspy.make_signature(
        {
            "text": (str, dspy.InputField()),
            "label": (Literal[tuple(task.labels)], dspy.OutputField()),
        },
        task.description,
    )
    program = Failsafe(dspy.ChainOfThought(signature) if reasoning else dspy.Predict(signature))

    def metric(gold, predicted, trace=None, pred_name=None, pred_trace=None):
        right = predicted.label == gold.label
        return dspy.Prediction(
            score=float(right),
            feedback="Correct." if right else f"Expected {gold.label}, answered {predicted.label}.",
        )

    with dspy.context(lm=student, adapter=text_adapter()):
        optimizer = dspy.GEPA(
            metric=metric,
            max_metric_calls=budget,
            reflection_lm=reflection,
            seed=seed,
            num_threads=DSPY_THREADS,
        )
        tuned = optimizer.compile(
            program, trainset=dspy_examples(task.train), valset=dspy_examples(task.validation)
        )
    answers, seconds = dspy_answers(tuned, task, student)
    return result("dspy-cot" if reasoning else "dspy", seed, answers, seconds)


def mcnemar(first: list[bool], second: list[bool]) -> float:
    """Exact two-sided McNemar p-value: could two methods scored on the same examples differ
    this much by chance? Only the examples exactly one of them got right count."""
    only_first = sum(a and not b for a, b in zip(first, second, strict=True))
    only_second = sum(b and not a for a, b in zip(first, second, strict=True))
    split = only_first + only_second
    tail = sum(math.comb(split, k) for k in range(min(only_first, only_second) + 1)) / 2**split
    return min(1.0, 2 * tail)


def summarize(runs: list[dict]) -> dict[str, dict]:
    """Each method averaged over seeds, with the spread of its accuracy between seeds."""
    methods: dict[str, list[dict]] = {}
    for run in runs:
        methods.setdefault(run["method"], []).append(run)

    def averaged(done: list[dict], key: str) -> float | None:
        values = [run.get(key) for run in done]
        return None if any(value is None for value in values) else mean(values)

    fields = (
        "calls_per_example",
        "tokens_per_example",
        "output_tokens_per_example",
        "seconds_per_example",
        "invalid_rate",
        *(f"accuracy_at_{round(k * 100)}" for k in COVERAGES),
    )
    return {
        method: {
            "seeds": len(done),
            "accuracy": mean(run["accuracy"] for run in done),
            "spread": stdev(run["accuracy"] for run in done) if len(done) > 1 else 0.0,
            **{field: averaged(done, field) for field in fields},
        }
        for method, done in methods.items()
    }


def against(runs: list[dict], method: str) -> dict[str, float]:
    """McNemar p-values of `method` against every other, over the seeds both finished, their
    test examples pooled."""
    by_seed = {(run["method"], run["seed"]): run["correct"] for run in runs}
    found = {}
    for other in sorted({run["method"] for run in runs} - {method}):
        seeds = sorted(s for m, s in by_seed if m == method and (other, s) in by_seed)
        if seeds:
            mine = [c for s in seeds for c in by_seed[method, s]]
            theirs = [c for s in seeds for c in by_seed[other, s]]
            found[other] = mcnemar(mine, theirs)
    return found


def report(results: dict) -> str:
    """The results as a Markdown table, best accuracy first, and whether search's differences
    from each other method could be chance."""

    def shown(value: float | None, digits: str = ".3f") -> str:
        return "n/a" if value is None else format(value, digits)

    rows = sorted(summarize(results["runs"]).items(), key=lambda item: -item[1]["accuracy"])
    lines = [
        f"{results['dataset']}, {results['test']} test examples per seed",
        "",
        "| Method | Accuracy | At 80% | At 50% | Seeds | Judge or LM calls | Input tokens "
        "| Output tokens | Seconds | Invalid |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for method, row in rows:
        lines.append(
            f"| {method} | {row['accuracy']:.3f} ± {row['spread']:.3f} | "
            f"{shown(row['accuracy_at_80'])} | {shown(row['accuracy_at_50'])} | {row['seeds']} | "
            f"{shown(row['calls_per_example'], '.2f')} | {shown(row['tokens_per_example'], '.0f')} "
            f"| {shown(row['output_tokens_per_example'], '.0f')} | "
            f"{shown(row['seconds_per_example'], '.2f')} | {shown(row['invalid_rate'], '.1%')} |"
        )
    compared = against(results["runs"], "search")
    if compared:
        lines += ["", "Search against each method, exact McNemar p over pooled test examples:", ""]
        lines += [f"- {other}: p = {p:.3g}" for other, p in compared.items()]
    return "\n".join(lines)


@dataclass(frozen=True)
class Models:
    """What the model-backed methods run on. Unused by majority and tfidf."""

    client: object = None
    judge: str = "none"
    proposer: object = None
    student: dspy.LM | None = None
    reflection: dspy.LM | None = None
    budget: int = 0
    call_cost: float = 0


def run(
    dataset: Dataset,
    seeds: list[int],
    methods: list[str],
    sizes: dict[str, int],
    output: Path,
    models: dict[str, str],
    resources: Models | None = None,
    deadline: float = math.inf,
) -> list[dict]:
    """Every method on every seed, written to `output` after each one finishes. Seeds go in
    order, each finishing every method before the next begins, so a run cut short by `deadline`
    (seconds since the epoch) loses whole seeds rather than whole methods."""
    resources = resources or Models()
    results = {"dataset": dataset.name, **sizes, "runs": []}
    if output.exists():
        results = json.loads(output.read_text())
        if {key: results[key] for key in sizes} != sizes:
            raise ValueError(f"{output} holds runs of other sizes; write to a new file")
    done = {(run["method"], run["seed"]) for run in results["runs"]}
    for seed in seeds:
        task = sample(dataset, sizes["train"], sizes["validation"], sizes["test"], seed)
        for method in methods:
            if (method, seed) in done:
                continue
            if time.time() > deadline:
                return results["runs"]
            # Per run, since a resumed file may be finished with other models than started it.
            result = measure(method, seed, task, dataset, resources) | {"models": models}
            results["runs"].append(result)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(results, indent=2) + "\n")
    return results["runs"]


def measure(method: str, seed: int, task: Task, dataset: Dataset, resources: Models) -> dict:
    if method == "majority":
        return majority(seed, task)
    if method == "tfidf":
        return tfidf(seed, task)
    if method == "direct":
        plan = direct_plan(task)
        return record("direct", seed, task, plan, resources.client, resources.judge)
    if method == "handwritten":
        plan = compile_rules(dataset.handwritten, task.criteria())
        return record("handwritten", seed, task, plan, resources.client, resources.judge)
    if method == "search":
        return search(
            seed,
            task,
            resources.client,
            resources.judge,
            resources.proposer,
            resources.budget,
            resources.call_cost,
        )
    return dspy_baseline(
        seed,
        task,
        resources.student,
        resources.reflection,
        resources.budget,
        reasoning=method == "dspy-cot",
    )


def elo_against(reference: float, score: float) -> float | None:
    """The rating that expects `score` against an opponent rated `reference`, by the logistic
    Elo curve. A clean sweep either way has no finite answer."""
    if not 0 < score < 1:
        return None
    return reference + 400 * math.log10(score / (1 - score))


def chess_reference(
    task, graphs: list[Path], engine, games: int, rating: float, output: Path
) -> list[dict]:
    """Every graph against one engine over the same seeds, colors alternating by seed, written
    to `output` after each graph. A graph already there is skipped."""
    from rulesmith.chess import Games
    from rulesmith.cli import read_plan

    results = json.loads(output.read_text()) if output.exists() else {"runs": []}
    done = {run["graph"] for run in results["runs"]}
    cli = argparse.ArgumentParser()
    for path in graphs:
        if path.stem in done:
            continue
        program = DecisionProgram(read_plan(cli, path), None, "none")
        scorer = Games(task, engine)
        records = [scorer(program, seed).record for seed in range(games)]
        points = [record["points"] for record in records]
        score = mean(points)
        results["runs"].append(
            {
                "graph": path.stem,
                "games": games,
                "won": points.count(1.0),
                "drew": points.count(0.5),
                "lost": points.count(0.0),
                "score": score,
                "material": mean(record["material"] for record in records),
                # A graph that loses every game still differs in how long it holds out.
                "plies": mean(record["plies"] for record in records),
                "elo": elo_against(rating, score),
            }
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2) + "\n")
    return results["runs"]


def language_model(
    model: str | None,
    base_url: str | None,
    options: str | None,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> dspy.LM | None:
    """A generative model, given its DSPy id. `options` is extra request JSON, such as switching
    off a model's thinking so a one-word label does not cost a page of reasoning. A failed request
    is retried: a local server under several runs' load drops connections, and DSPy's GEPA ends
    the whole run on a single missing answer."""
    if model is None:
        return None
    extra = {"extra_body": json.loads(options)} if options else {}
    endpoint = {"api_base": base_url} if base_url else {}
    return dspy.LM(
        model,
        cache=False,
        num_retries=retries,
        max_tokens=max_tokens,
        timeout=timeout,
        **endpoint,
        **extra,
    )


def judge_client(base_url: str, timeout: float):
    """The local judge, waiting as long as a generative request may: every run of a benchmark
    queues on the same judge, and the SDK's ten seconds ended runs that were only waiting."""
    from typesafe_sdk import TypeSafeClient

    return TypeSafeClient(base_url=base_url, api_key="local-no-auth", timeout=timeout)


def main() -> None:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    bench = commands.add_parser("run", help="Run methods over seeds, resuming a results file")
    bench.add_argument("dataset", choices=sorted(DATASETS))
    bench.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    bench.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    for split, default in (("train", 50), ("validation", 50), ("test", 300)):
        bench.add_argument(f"--{split}", type=int, default=default)
    bench.add_argument("--budget", type=int, default=150, help="Metric calls per search")
    bench.add_argument(
        "--call-cost",
        type=float,
        default=0.02,
        help="Score search takes off per judge question: a whole call is worth two points",
    )
    bench.add_argument("--judge-base-url", default="http://127.0.0.1:8010")
    bench.add_argument("--judge-model", default="default")
    for role in ("proposer", "student", "reflection"):
        bench.add_argument(f"--{role}-model")
        bench.add_argument(f"--{role}-base-url")
        bench.add_argument(f"--{role}-options", help="Extra request JSON for this model")
    bench.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Per reply; a proposal restating 77 labels with its reasoning needs room",
    )
    bench.add_argument(
        "--student-max-tokens",
        type=int,
        default=2048,
        help="Per DSPy reply: a label with its reasoning fits, and a local server reserves "
        "memory for this many tokens in every reply it is generating at once",
    )
    bench.add_argument(
        "--note", help="How the judge was served, e.g. RULING_ROTATIONS=1, kept with the results"
    )
    bench.add_argument(
        "--timeout",
        type=float,
        default=3600,
        help="Seconds a generative request may take; runs sharing one local model queue behind "
        "each other",
    )
    bench.add_argument(
        "--hours",
        type=float,
        help="Start nothing new after this many hours; finished results are kept either way",
    )
    bench.add_argument(
        "--retries", type=int, default=10, help="Retries for a failed generative request"
    )
    bench.add_argument("--output", type=Path, required=True)
    games = commands.add_parser("chess", help="Play chess graphs against a fixed-strength engine")
    games.add_argument("task", type=Path, help="A chess task; its move limit and openings apply")
    games.add_argument("graphs", type=Path, nargs="+")
    games.add_argument("--engine", required=True, help="Path to a UCI engine such as stockfish")
    games.add_argument("--skill", type=int, default=0, help="Stockfish Skill Level, 0 to 20")
    games.add_argument(
        "--rating", type=float, default=1320, help="The engine's rating at that skill"
    )
    games.add_argument("--seconds", type=float, default=0.05, help="Engine time per move")
    games.add_argument("--games", type=int, default=40)
    games.add_argument("--output", type=Path, required=True)
    shown = commands.add_parser("report", help="Print a results file as a Markdown table")
    shown.add_argument("results", type=Path)
    args = root.parse_args()
    if args.command == "report":
        print(report(json.loads(args.results.read_text())))
        return
    if args.command == "chess":
        from rulesmith.chess import ChessTask, Engine

        task = ChessTask.model_validate_json(args.task.read_text())
        with Engine(args.engine, args.skill, args.seconds) as engine:
            runs = chess_reference(task, args.graphs, engine, args.games, args.rating, args.output)
        print(json.dumps(runs, indent=2))
        return
    lms = {
        role: language_model(
            getattr(args, f"{role}_model"),
            getattr(args, f"{role}_base_url"),
            getattr(args, f"{role}_options"),
            args.student_max_tokens if role == "student" else args.max_tokens,
            args.timeout,
            args.retries,
        )
        for role in ("proposer", "student", "reflection")
    }
    needed = {"search": "proposer", "dspy": "student", "dspy-cot": "student"}
    for method, role in needed.items():
        if method in args.methods and lms[role] is None:
            root.error(f"{method} needs --{role}-model")
    dataset = DATASETS[args.dataset]
    if "handwritten" in args.methods and dataset.handwritten is None:
        root.error(f"{dataset.name} has no hand-written rules to run")
    sizes = {split: getattr(args, split) for split in ("train", "validation", "test")}
    models = {"judge": args.judge_model, "note": args.note or "", "call_cost": args.call_cost} | {
        role: getattr(args, f"{role}_model") for role in lms if getattr(args, f"{role}_model")
    }
    with judge_client(args.judge_base_url, args.timeout) as client:
        resources = Models(
            client=client,
            judge=args.judge_model,
            proposer=GraphProposer(lms["proposer"]) if lms["proposer"] else None,
            student=lms["student"],
            reflection=lms["reflection"] or lms["proposer"],
            budget=args.budget,
            call_cost=args.call_cost,
        )
        deadline = time.time() + args.hours * 3600 if args.hours else math.inf
        run(dataset, args.seeds, args.methods, sizes, args.output, models, resources, deadline)
    print(report(json.loads(args.output.read_text())))


def cached(url: str, name: str, sha1: str | None = None) -> Path:
    """A downloaded file, fetched once. Datasets stay out of the repo; their licenses allow
    use, and fetching from the source keeps attribution where the publisher put it. With a
    published checksum, a file that differs is refused rather than measured."""
    path = Path(os.environ.get("RULESMITH_CACHE", Path.home() / ".cache" / "rulesmith")) / name
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if sha1 is not None and hashlib.sha1(data).hexdigest() != sha1:
            raise ValueError(f"{url} does not match its published checksum {sha1}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path


def banking77() -> Rows:
    """PolyAI's Banking77 (CC BY 4.0): 13,083 banking queries across 77 intents."""
    base = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
    rows = {}
    for split in ("train", "test"):
        with cached(f"{base}/{split}.csv", f"banking77/{split}.csv").open(newline="") as file:
            rows[split] = [(row["text"], row["category"]) for row in csv.DictReader(file)]
    return rows


def hub(repo: str, revision: str, path: str) -> Path:
    """A file from a Hugging Face dataset, pinned to the revision these results came from."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, path, repo_type="dataset", revision=revision))


def clothing() -> Rows:
    """Women's E-Commerce Clothing Reviews (CC0): whether a reviewer recommends the item, from
    their rating, age, the product's department and the review itself. The rating settles most
    cases; a three-star review needs its text read."""
    repo, revision = "saattrupdan/womens-clothing-ecommerce-reviews", CLOTHING_REVISION
    files = {
        "train": "data/train-00000-of-00001-44ddebcb331919e6.parquet",
        "test": "data/test-00000-of-00001-90b9c3d0edf6fc3c.parquet",
    }
    rows = {}
    for split, path in files.items():
        table = parquet.read_table(hub(repo, revision, path)).to_pylist()
        rows[split] = [
            (
                {
                    "rating": row["rating"],
                    "age": row["age"],
                    "helpful_votes": row["positive_feedback_count"],
                    "division": row["division_name"],
                    "department": row["department_name"],
                    "category": row["class_name"],
                    "review": row["review_text"],
                },
                "recommended" if row["recommended_ind"] else "not_recommended",
            )
            for row in table
        ]
    return rows


def trec() -> Rows:
    """TREC question classification (Li and Roth), coarse: the kind of answer a question asks
    for. A small, fast sanity check; its license is not stated, so it is for research use."""
    rows = {}
    for split in ("train", "test"):
        with hub("SetFit/TREC-QC", TREC_REVISION, f"{split}.jsonl").open() as file:
            lines = [json.loads(line) for line in file]
        rows[split] = [(line["text"], line["label_coarse_text"]) for line in lines]
    return rows


def jobs() -> Rows:
    """Real or Fake Job Postings (CC0): whether a posting is a scam, from its structured fields
    (company logo, screening questions, salary, employment type) and its text. The source has one
    file, so every fifth posting is held out as the test pool."""
    path = hub(JOBS_REPO, JOBS_REVISION, "fake_job_postings.csv")
    with path.open(newline="") as file:
        postings = list(csv.DictReader(file))
    rows: Rows = {"train": [], "test": []}
    for index, row in enumerate(postings):
        state = {
            "title": row["title"],
            "location": row["location"],
            "salary_range": row["salary_range"],
            "employment_type": row["employment_type"],
            "required_experience": row["required_experience"],
            "required_education": row["required_education"],
            "industry": row["industry"],
            "function": row["function"],
            "telecommuting": int(row["telecommuting"]),
            "has_company_logo": int(row["has_company_logo"]),
            "has_questions": int(row["has_questions"]),
            "company_profile": row["company_profile"],
            "description": row["description"],
        }
        label = "fraudulent" if row["fraudulent"] == "1" else "genuine"
        rows["test" if index % 5 == 0 else "train"].append((state, label))
    return rows


def news() -> Rows:
    """Which channel a Mashable article ran in, from its title and sixteen numbers about its text
    (length, links, images, sentiment): UCI's Online News Popularity (CC BY 4.0) as the
    multimodal AutoML benchmark of Shi et al. (2021) prepared it. Chosen before any result, and
    used for nothing while rulesmith was built, as a check on the other datasets."""
    base = "https://automl-mm-bench.s3.amazonaws.com/news_channel"
    rows = {}
    for split, digest in NEWS_SHA1.items():
        path = cached(f"{base}/{split}.csv", f"news_channel/{split}.csv", digest)
        with path.open(newline="") as file:
            table = [
                {key.strip(): value.strip() for key, value in row.items()}
                for row in csv.DictReader(file)
            ]
        rows[split] = [
            (
                {"title": row.pop("article_title")}
                | {key: float(value) for key, value in row.items() if key != "channel"},
                CHANNELS[row["channel"].removeprefix("data_channel_is_")],
            )
            for row in table
        ]
    return rows


def spelled(label: str) -> str:
    """A label described by its own words, the most a direct question gets for free."""
    return label.replace("_", " ")


CLOTHING_REVISION = "4e788183125a6143d8bdba24e58f1efda2e4b48f"
TREC_REVISION = "0a34640b3cac3affef6f0abdfb81e3d0ecffcf92"
JOBS_REPO = "victor/real-or-fake-fake-jobposting-prediction"
JOBS_REVISION = "a751f55136a26da3c36e26aa207a9e187ca24b45"
SIGNIFICANCE = 0.1
# DSPy evaluates eight examples at once by default; five datasets doing that at once swamp one
# local model server, and a connection it drops ends the run inside DSPy's GEPA.
DSPY_THREADS = 4
NEWS_SHA1 = {
    "train": "ab226210b6a878b449d01f33a195014c65c22311",
    "test": "a71516784ce6e168bd9933e9ec50080f65cb05fd",
}
CHANNELS = {
    "bus": "business",
    "entertainment": "entertainment",
    "lifestyle": "lifestyle",
    "socmed": "social_media",
    "tech": "tech",
    "world": "world",
}
METHODS = ("majority", "tfidf", "direct", "handwritten", "search", "dspy", "dspy-cot")
DATASETS: dict[str, Dataset] = {
    "banking77": Dataset(
        "banking77",
        "Which intent does this online banking customer's message express?",
        banking77,
        spelled,
    ),
    "clothing": Dataset(
        "clothing",
        "Would this shopper recommend the clothing item they reviewed?",
        clothing,
        spelled,
        handwritten="\n".join(
            [
                "read rating in [1, 5]",
                "when rating >= 4: recommended",
                "when rating < 3: not_recommended",
                'always: ask "Would this shopper recommend the clothing item they reviewed?" '
                '[recommended: "They would recommend it."] '
                '[not_recommended: "They would not recommend it."]',
            ]
        ),
    ),
    "jobs": Dataset(
        "jobs",
        "Is this job posting a scam?",
        jobs,
        spelled,
        balanced=True,
        handwritten="\n".join(
            [
                "read has_company_logo in [0, 1]",
                "when has_company_logo >= 1: genuine",
                'always: ask "Is this job posting a scam?" '
                '[fraudulent: "It is a scam."] [genuine: "It is a real job."]',
            ]
        ),
    ),
    "news": Dataset("news", "Which channel did this article run in?", news, spelled),
    "trec": Dataset("trec", "What kind of answer does this question ask for?", trec, spelled),
}

if __name__ == "__main__":
    main()
