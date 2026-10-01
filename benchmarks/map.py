"""The demonstration map: three decisions of a marketplace's trust-and-safety work, each a unit
search found on a public dataset, behind a router that costs nothing.

    python3 benchmarks/map.py build       # write examples/map from the benchmark results
    python3 benchmarks/map.py summarize benchmarks/results/map.json

`build` takes the graph search found on the jobs, clothing and Banking77 benchmarks (the seed
with the best validation score of each, from benchmarks/results) and writes them as unit
files, a map that hands a job posting to the scam unit, a review to the recommendation unit and
anything else to the intent unit, and a task of held-out rows from all three datasets with
every dataset's labels. Scoring the map on
that task is `rulesmith evaluate`; `summarize` reads its output and says how the map did as a
whole and unit by unit.
"""

import json
import sys
from pathlib import Path

import pyarrow  # noqa: F401  # before dspy, whose lazy numpy proxy breaks pyarrow loaded after it

from rulesmith.bench import DATASETS, sample
from rulesmith.graph import Plan
from rulesmith.rules import render_rules

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "examples" / "map"
# Unit, the benchmark it was searched on, and the field that marks an input as its kind.
UNITS = {
    "postings": ("jobs", "has_company_logo", [0, 1]),
    "reviews": ("clothing", "rating", [1, 5]),
}
FALLBACK = ("intents", "banking77")
ROWS_PER_DATASET = {"train": 4, "validation": 4, "test": 40}


def searched(dataset: str) -> Plan:
    """The graph search found on a benchmark: of its three seeds, the one that scored best on
    validation, which is the one a person would take forward."""
    results = json.loads((ROOT / "benchmarks" / "results" / f"{dataset}.json").read_text())
    runs = [r for r in results["runs"] if r["method"] == "search"]
    return Plan.model_validate(max(runs, key=lambda r: r["validation"])["graph"])


def build() -> None:
    (MAP / "units").mkdir(parents=True, exist_ok=True)
    nodes: dict = {}
    sources = []
    for unit, (dataset, field, bounds) in UNITS.items():
        (MAP / "units" / f"{unit}.rules").write_text(render_rules(searched(dataset)) + "\n")
        nodes[field] = {"type": "read", "path": [field], "bounds": bounds}
        nodes[unit] = {
            "type": "graph",
            "path": f"units/{unit}.rules",
            "when": {"type": "present", "node": field},
        }
        sources.append(unit)
    unit, dataset = FALLBACK
    (MAP / "units" / f"{unit}.rules").write_text(render_rules(searched(dataset)) + "\n")
    nodes[unit] = {"type": "graph", "path": f"units/{unit}.rules"}
    sources.append(unit)
    nodes["decision"] = {"type": "coalesce", "sources": sources}
    graph = {"nodes": nodes, "outputs": {"label": {"type": "node", "node": "decision"}}}
    (MAP / "trust-safety.json").write_text(json.dumps(graph, indent=2) + "\n")
    (MAP / "task.json").write_text(json.dumps(task(), indent=2) + "\n")


def task() -> dict:
    """Held-out rows from every dataset, labeled in each dataset's own words, so the map is
    scored on the whole of what it routes. The rows are the benchmark's own test samples, so
    no unit has seen them."""
    datasets = [dataset for dataset, _, _ in UNITS.values()] + [FALLBACK[1]]
    labels: list[str] = []
    splits: dict[str, list] = {split: [] for split in ROWS_PER_DATASET}
    for dataset in datasets:
        sampled = sample(DATASETS[dataset], 50, 50, 300, 0)
        labels += [label for label in sampled.labels if label not in labels]
        for split, count in ROWS_PER_DATASET.items():
            for example in getattr(sampled, split)[:count]:
                splits[split].append(
                    {
                        "id": f"{dataset}-{example.id}",
                        "state": example.state,
                        "label": example.label,
                    }
                )
    return {
        "description": "Trust and safety for a marketplace: is a job posting a scam, would a "
        "reviewer recommend the item, and which intent does a customer's message express? "
        "Each input is one of those three, told apart by the fields it has.",
        "labels": labels,
        **splits,
    }


def summarize(evaluation: dict) -> dict:
    """How the map did on the whole, and per unit: which unit decided each example is in
    `decided_by`, and each row's id says which dataset it came from."""
    per_unit: dict[str, dict] = {}
    for example in evaluation["examples"]:
        unit = (example.get("decided_by") or "none").split(".")[0]
        tally = per_unit.setdefault(unit, {"examples": 0, "correct": 0, "calls": 0})
        tally["examples"] += 1
        tally["correct"] += bool(example.get("correct"))
        tally["calls"] += len(example.get("calls", {}))
    for tally in per_unit.values():
        tally["accuracy"] = tally["correct"] / tally["examples"]
        tally["calls_per_example"] = tally["calls"] / tally["examples"]
    examples = evaluation["examples"]
    return {
        "examples": len(examples),
        "accuracy": evaluation["score"],
        "calls_per_example": sum(len(e.get("calls", {})) for e in examples) / len(examples),
        "units": per_unit,
    }


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    elif sys.argv[1] == "summarize":
        print(json.dumps(summarize(json.loads(Path(sys.argv[2]).read_text())), indent=2))
