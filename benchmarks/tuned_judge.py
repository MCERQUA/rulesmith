"""The tuned-judge setting: a judge fine-tuned, with Ruling, on exactly the 100 examples search
sees, one adapter per dataset and seed, scored beside the stock judges.

    python3 benchmarks/tuned_judge.py build      # training and holdout records under runs/lora/data
    python3 benchmarks/tuned_judge.py compare    # the comparison tables from benchmarks/results

`build` writes, for each dataset and seed, what `rulesmith tune` writes for one task:
a training file of the 100 training and validation examples and a holdout file of the 300
test examples, each asked the direct question with the right label, in the record shape
`ruling train` reads. The holdout file is what the trainer checks for overlap, so no test
example can leak into an adapter. `compare`
reads the stock results, the stock Instruct results and the tuned results and prints, per
dataset and method, accuracy over seeds, selective accuracy, calls per row, and an exact
McNemar test of tuned against the stock Instruct judge on the same test rows. The driver that
trains the adapters and runs the methods is benchmarks/tuned-judge.sh."""

import json
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy  # noqa: F401  imported before dspy's lazy import of numpy re-enters it
import pyarrow.parquet  # noqa: F401

from rulesmith.bench import DATASETS, mcnemar, sample
from rulesmith.tuning import records

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs" / "lora" / "data"
SIZES = {"train": 50, "validation": 50, "test": 300}
SEEDS = (0, 1, 2)
TUNABLE = ("jobs", "clothing", "trec", "banking77", "news")
SETTINGS = {
    "Qwen3.5 stock": "benchmarks/results",
    "Instruct stock": "benchmarks/results/instruct",
    "Instruct tuned": "benchmarks/results/tuned",
}
METHODS = ("direct", "search", "dspy-typed")


def build() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name in TUNABLE:
        for seed in SEEDS:
            task = sample(DATASETS[name], SIZES["train"], SIZES["validation"], SIZES["test"], seed)
            for split, rows in records(task).items():
                lines = "\n".join(json.dumps(row) for row in rows) + "\n"
                (DATA / f"{name}-{seed}-{split}.jsonl").write_text(lines)
    print(f"{len(list(DATA.glob('*.jsonl')))} files under {DATA}")


def runs(setting: str, dataset: str) -> dict[tuple[str, int], dict]:
    path = ROOT / SETTINGS[setting] / f"{dataset}.json"
    if not path.exists():
        return {}
    found = json.loads(path.read_text())["runs"]
    return {(r["method"], r["seed"]): r for r in found if r["method"] in METHODS}


def cell(found: dict, method: str) -> str:
    rows = [r for (m, _), r in sorted(found.items()) if m == method]
    if not rows:
        return ""
    accuracy = [r["accuracy"] for r in rows]
    spread = stdev(accuracy) if len(accuracy) > 1 else 0.0
    text = f"{mean(accuracy):.3f} ± {spread:.3f}"
    surest = [r["accuracy_at_80"] for r in rows if r.get("accuracy_at_80") is not None]
    if surest:
        text += f", at 80% {mean(surest):.3f}"
    if method == "search":
        text += f", calls {mean(r['calls_per_example'] for r in rows):.2f}"
    return text


def compare() -> str:
    lines = []
    for dataset in TUNABLE:
        by = {setting: runs(setting, dataset) for setting in SETTINGS}
        lines += [
            f"### {dataset}",
            "",
            "| Method | " + " | ".join(SETTINGS) + " | Tuned vs stock Instruct, p |",
            "| --- | " + " | ".join("---" for _ in SETTINGS) + " | --- |",
        ]
        for method in METHODS:
            stock, tuned = by["Instruct stock"], by["Instruct tuned"]
            seeds = sorted(s for (m, s) in tuned if m == method and (method, s) in stock)
            p = ""
            if seeds:
                mine = [c for s in seeds for c in tuned[method, s]["correct"]]
                theirs = [c for s in seeds for c in stock[method, s]["correct"]]
                p = f"{mcnemar(mine, theirs):.2g}"
            cells = " | ".join(cell(by[setting], method) for setting in SETTINGS)
            lines.append(f"| {method} | {cells} | {p} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if command == "build":
        build()
    elif command == "compare":
        print(compare())
    else:
        sys.exit(f"unknown command {command!r}; use build or compare")
