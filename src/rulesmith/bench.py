"""Public benchmarks: rulesmith against the baselines a reader would reach for first.

Each run draws a task from a public dataset with a seed, scores every method on the same held-out
test split, and records accuracy with the judge calls and tokens each example cost. A method a
run has already recorded is skipped, so an interrupted benchmark resumes where it stopped.
"""

import argparse
import csv
import json
import math
import os
import random
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

from rulesmith.graph import Plan, State, Task
from rulesmith.optimize import (
    GraphProposer,
    Labels,
    SearchConfig,
    evaluate_plan,
    optimize_plan,
    text_adapter,
)
from rulesmith.runtime import DecisionProgram

Rows = dict[str, list[tuple[State, str]]]


@dataclass(frozen=True)
class Dataset:
    """A public dataset as rows of (input, label), pooled by the dataset's own splits. Training
    and validation draw from its 'train' pool and the test split from its 'test' pool, so nothing
    a search sees overlaps what it is scored on."""

    name: str
    description: str
    rows: Callable[[], Rows]
    describe: Callable[[str], str]

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

    def draw(pool: str, count: int) -> list[dict]:
        rows = list(pools[pool])
        rng.shuffle(rows)
        drawn = []
        for state, label in rows:
            key = json.dumps(state, sort_keys=True)
            if key not in seen:
                seen.add(key)
                drawn.append({"state": state, "label": label})
            if len(drawn) == count:
                return drawn
        raise ValueError(f"the {pool} pool of {dataset.name} holds fewer than {count} inputs")

    held = draw("test", test)
    searched = draw("train", train + validation)
    splits = {"train": searched[:train], "validation": searched[train:], "test": held}
    return Task(
        description=dataset.description,
        labels=sorted(dataset.labels()),
        **{
            split: [{"id": f"{split}-{index}", **row} for index, row in enumerate(rows)]
            for split, rows in splits.items()
        },
    )


def direct_plan(task: Task, labels: dict[str, str]) -> Plan:
    """The simplest thing a judge can do: one choice question over every label, no graph."""
    return Plan.model_validate(
        {
            "nodes": {
                "label": {
                    "type": "call",
                    "question": {
                        "type": "choice",
                        "instructions": task.description,
                        "criteria": {label: labels[label] for label in task.labels},
                    },
                    "inputs": {"observation": {"type": "state"}},
                }
            },
            "outputs": {"label": {"type": "node", "node": "label"}},
        }
    )


def tokens_and_calls(records: list[dict]) -> tuple[float, float]:
    """Judge questions answered and input tokens billed, each per example."""
    calls = sum(len(record.get("calls", {})) for record in records)
    tokens = sum(record.get("input_tokens", 0) for record in records)
    return calls / len(records), tokens / len(records)


def result(method: str, seed: int, accuracy: float, records: list[dict], **extra) -> dict:
    calls, tokens = tokens_and_calls(records)
    return {
        "method": method,
        "seed": seed,
        "accuracy": accuracy,
        "calls_per_example": calls,
        "tokens_per_example": tokens,
        **extra,
    }


def record(method: str, seed: int, task: Task, plan: Plan, client, model: str, **extra) -> dict:
    """A graph scored on the task's test split."""
    scored = evaluate_plan(plan, task.test, client, model, Labels(task))
    return result(method, seed, scored["score"], scored["examples"], **extra)


def majority(seed: int, task: Task) -> dict:
    return record("majority", seed, task, task.baseline_plan(), None, "none")


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
    guessed = model.predict([text(e.state) for e in task.test])
    right = sum(guess == e.label for guess, e in zip(guessed, task.test, strict=True))
    return result("tfidf", seed, right / len(task.test), [{}] * len(task.test))


def search(
    seed: int, task: Task, labels: dict[str, str], client, model: str, proposer, budget: int
) -> dict:
    """rulesmith's search, seeded with the direct call, so the comparison that matters is
    whether a graph improves on asking the judge once."""
    config = SearchConfig(max_metric_calls=budget, seed=seed)
    best, report = optimize_plan(task, direct_plan(task, labels), client, model, proposer, config)
    return record(
        "search",
        seed,
        task,
        best,
        client,
        model,
        validation=report["validation_score"],
        seed_validation=report["seed_validation_score"],
        nodes=len(best.nodes),
    )


def dspy_baseline(seed: int, task: Task, student: dspy.LM, reflection: dspy.LM, budget: int):
    """A standard DSPy classifier on a text-generating model, optimized with DSPy's own GEPA."""
    signature = dspy.make_signature(
        {
            "text": (str, dspy.InputField()),
            "label": (Literal[tuple(task.labels)], dspy.OutputField()),
        },
        task.description,
    )
    program = dspy.Predict(signature)

    def metric(gold, predicted, trace=None, pred_name=None, pred_trace=None):
        right = predicted.label == gold.label
        return dspy.Prediction(
            score=float(right),
            feedback="Correct." if right else f"Expected {gold.label}, answered {predicted.label}.",
        )

    def examples(split):
        return [dspy.Example(text=text(e.state), label=e.label).with_inputs("text") for e in split]

    with dspy.context(lm=student, adapter=text_adapter()):
        optimizer = dspy.GEPA(
            metric=metric, max_metric_calls=budget, reflection_lm=reflection, seed=seed
        )
        tuned = optimizer.compile(
            program, trainset=examples(task.train), valset=examples(task.validation)
        )
        calls_before = len(student.history)
        right = sum(
            tuned(text=example.text).label == example.label for example in examples(task.test)
        )
        used = student.history[calls_before:]
    tokens = sum((entry.get("usage") or {}).get("prompt_tokens", 0) for entry in used)
    return {
        "method": "dspy",
        "seed": seed,
        "accuracy": right / len(task.test),
        "calls_per_example": len(used) / len(task.test),
        "tokens_per_example": tokens / len(task.test),
    }


def summarize(runs: list[dict]) -> dict[str, dict]:
    """Each method's test accuracy averaged over seeds, with the spread between seeds."""
    methods: dict[str, list[dict]] = {}
    for run in runs:
        methods.setdefault(run["method"], []).append(run)
    return {
        method: {
            "seeds": len(done),
            "accuracy": mean(run["accuracy"] for run in done),
            "spread": stdev(run["accuracy"] for run in done) if len(done) > 1 else 0.0,
            "calls_per_example": mean(run["calls_per_example"] for run in done),
            "tokens_per_example": mean(run["tokens_per_example"] for run in done),
        }
        for method, done in methods.items()
    }


def report(results: dict) -> str:
    """The results as a Markdown table, best accuracy first."""
    rows = sorted(summarize(results["runs"]).items(), key=lambda item: -item[1]["accuracy"])
    lines = [
        f"{results['dataset']}, {results['test']} test examples per seed",
        "",
        "| Method | Test accuracy | Seeds | Judge calls / example | Input tokens / example |",
        "| --- | --- | --- | --- | --- |",
    ]
    for method, row in rows:
        lines.append(
            f"| {method} | {row['accuracy']:.3f} ± {row['spread']:.3f} | {row['seeds']} | "
            f"{row['calls_per_example']:.2f} | {math.ceil(row['tokens_per_example'])} |"
        )
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


def run(
    dataset: Dataset,
    seeds: list[int],
    methods: list[str],
    sizes: dict[str, int],
    output: Path,
    models: dict[str, str],
    resources: Models | None = None,
) -> list[dict]:
    """Every method on every seed, written to `output` after each one finishes."""
    resources = resources or Models()
    results = {"dataset": dataset.name, **sizes, "models": models, "runs": []}
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
            results["runs"].append(measure(method, seed, task, dataset, resources))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(results, indent=2) + "\n")
    return results["runs"]


def measure(method: str, seed: int, task: Task, dataset: Dataset, resources: Models) -> dict:
    if method == "majority":
        return majority(seed, task)
    if method == "tfidf":
        return tfidf(seed, task)
    if method == "direct":
        plan = direct_plan(task, dataset.labels())
        return record("direct", seed, task, plan, resources.client, resources.judge)
    if method == "search":
        return search(
            seed,
            task,
            dataset.labels(),
            resources.client,
            resources.judge,
            resources.proposer,
            resources.budget,
        )
    return dspy_baseline(seed, task, resources.student, resources.reflection, resources.budget)


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
    model: str | None, base_url: str | None, options: str | None, max_tokens: int
) -> dspy.LM | None:
    """A generative model, given its DSPy id. `options` is extra request JSON, such as switching
    off a model's thinking so a one-word label does not cost a page of reasoning."""
    if model is None:
        return None
    extra = {"extra_body": json.loads(options)} if options else {}
    endpoint = {"api_base": base_url} if base_url else {}
    return dspy.LM(model, cache=False, num_retries=0, max_tokens=max_tokens, **endpoint, **extra)


def main() -> None:
    from typesafe_sdk import TypeSafeClient

    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    bench = commands.add_parser("run", help="Run methods over seeds, resuming a results file")
    bench.add_argument("dataset", choices=sorted(DATASETS))
    bench.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    bench.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    for split, default in (("train", 50), ("validation", 50), ("test", 300)):
        bench.add_argument(f"--{split}", type=int, default=default)
    bench.add_argument("--budget", type=int, default=150, help="Metric calls per search")
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
        "--note", help="How the judge was served, e.g. RULING_ROTATIONS=1, kept with the results"
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
            args.max_tokens,
        )
        for role in ("proposer", "student", "reflection")
    }
    needed = {"search": "proposer", "dspy": "student"}
    for method, role in needed.items():
        if method in args.methods and lms[role] is None:
            root.error(f"{method} needs --{role}-model")
    sizes = {split: getattr(args, split) for split in ("train", "validation", "test")}
    models = {"judge": args.judge_model, "note": args.note or ""} | {
        role: getattr(args, f"{role}_model") for role in lms if getattr(args, f"{role}_model")
    }
    with TypeSafeClient(base_url=args.judge_base_url, api_key="local-no-auth") as client:
        resources = Models(
            client=client,
            judge=args.judge_model,
            proposer=GraphProposer(lms["proposer"]) if lms["proposer"] else None,
            student=lms["student"],
            reflection=lms["reflection"] or lms["proposer"],
            budget=args.budget,
        )
        run(DATASETS[args.dataset], args.seeds, args.methods, sizes, args.output, models, resources)
    print(report(json.loads(args.output.read_text())))


def cached(url: str, name: str) -> Path:
    """A downloaded file, fetched once. Datasets stay out of the repo; their licenses allow
    use, and fetching from the source keeps attribution where the publisher put it."""
    path = Path(os.environ.get("RULESMITH_CACHE", Path.home() / ".cache" / "rulesmith")) / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
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


def spelled(label: str) -> str:
    """A label described by its own words, the most a direct question gets for free."""
    return label.replace("_", " ")


CLOTHING_REVISION = "4e788183125a6143d8bdba24e58f1efda2e4b48f"
TREC_REVISION = "0a34640b3cac3affef6f0abdfb81e3d0ecffcf92"
METHODS = ("majority", "tfidf", "direct", "search", "dspy")
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
    ),
    "trec": Dataset("trec", "What kind of answer does this question ask for?", trec, spelled),
}

if __name__ == "__main__":
    main()
