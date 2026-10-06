"""Escalation measured: for each benchmark seed, the tuned 4B judge alone, Clef-flash alone, and
the 4B escalating to Clef-flash where it is unsure, the cutoff fitted by calibrate against a
price on the Clef call. No search: the graphs are the direct question, written three ways.

    python3 benchmarks/escalation.py build DATASET SEED DIR   the task and three graphs, under DIR
    python3 benchmarks/escalation.py collect ROOT      results/escalation.json from every ROOT/*/
    python3 benchmarks/escalation.py summarize         the table, from that file

The driver is benchmarks/escalation.sh.
"""

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import pyarrow.parquet  # noqa: F401  # before dspy's lazy numpy proxy, which breaks pyarrow

from rulesmith.bench import DATASETS, mcnemar, sample

SIZES = (50, 50, 300)
CUTOFF = 0.8  # where the escalation line starts; calibrate moves it
RESULTS = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "escalation.json"
GRAPHS = (
    ("small", "tuned 4B alone"),
    ("large", "Clef-flash alone"),
    ("escalate", "4B, escalating to Clef"),
)


def build(dataset: str, seed: int, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    task = sample(DATASETS[dataset], *SIZES, seed)
    (out / "task.json").write_text(task.model_dump_json(indent=2, exclude_none=True))
    asked = f'ask "{task.description}" [task labels]'
    (out / "small.rules").write_text(f"always: {asked}\n")
    (out / "large.rules").write_text(f"always: {asked} using large\n")
    question = f"ask q = {asked.removeprefix('ask ')}"
    sure = [f'when q.confidence >= {CUTOFF} and q == "{label}": {label}' for label in task.labels]
    fallback = f"always: {asked} using large"
    (out / "escalate.rules").write_text("\n".join([question, *sure, fallback]) + "\n")


def collect(root: Path) -> None:
    """One record per dataset, seed and graph: the score, each row's rightness, the calls each
    judge answered per row, and the escalation cutoffs as calibrate left them."""
    runs = []
    for path in sorted(root.glob("*/*.evaluation.json")):
        dataset, seed = path.parent.name.rsplit("-", 1)
        graph = path.name.split(".")[0]
        evaluation = json.loads(path.read_text())
        paid = Counter(
            ("large" if call.get("judge") else "small")
            for row in evaluation["examples"]
            for call in row["calls"].values()
        )
        record = {
            "dataset": dataset,
            "seed": int(seed),
            "graph": graph,
            "accuracy": evaluation["score"],
            "correct": [row["correct"] for row in evaluation["examples"]],
            "calls_per_row": {
                who: paid[who] / len(evaluation["examples"]) for who in ("small", "large")
            },
            "models": evaluation["resolved_models"],
        }
        fitted = path.parent / "escalate.calibrated.rules"
        if graph == "escalate" and fitted.exists():
            record["rules"] = fitted.read_text()
        runs.append(record)
    RESULTS.write_text(
        json.dumps(
            {"sizes": SIZES, "cutoff": CUTOFF, "price": {"large": 0.1}, "runs": runs}, indent=1
        )
        + "\n"
    )
    print(f"{len(runs)} runs -> {RESULTS}")


def summarize() -> None:
    runs = json.loads(RESULTS.read_text())["runs"]
    by = {(r["dataset"], r["seed"], r["graph"]): r for r in runs}
    for dataset in sorted({r["dataset"] for r in runs}):
        print(f"\n### {dataset}\n")
        print(
            "| judge | accuracy by seed | mean | 4B calls/row | Clef calls/row | vs 4B alone (p) |"
        )
        print("|---|---|---:|---:|---:|---:|")
        seeds = sorted({r["seed"] for r in runs if r["dataset"] == dataset})
        for graph, name in GRAPHS:
            found = [by[dataset, s, graph] for s in seeds if (dataset, s, graph) in by]
            if not found:
                continue
            base = [
                c
                for s in seeds
                if (dataset, s, "small") in by
                for c in by[dataset, s, "small"]["correct"]
            ]
            now = [c for r in found for c in r["correct"]]
            p = "" if graph == "small" or len(now) != len(base) else f"{mcnemar(now, base):.2g}"
            each = ", ".join(f"{r['accuracy']:.3f}" for r in found)
            small = mean(r["calls_per_row"]["small"] for r in found)
            large = mean(r["calls_per_row"]["large"] for r in found)
            accuracy = mean(r["accuracy"] for r in found)
            print(f"| {name} | {each} | {accuracy:.3f} | {small:.2f} | {large:.2f} | {p} |")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]))
    elif sys.argv[1] == "collect":
        collect(Path(sys.argv[2]))
    else:
        summarize()
