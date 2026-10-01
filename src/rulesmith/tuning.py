"""Records for tuning the judge on a task's examples, in the shape Ruling's trainer reads.
rulesmith never trains a model; it writes what a trainer needs: every training and validation
example asked the direct question over the task's labels, with the label it carries, and the
test examples in a holdout file the trainer refuses to overlap, so a tuned judge is scored on
rows it has never seen."""

import json
import subprocess
from pathlib import Path

from rulesmith.graph import Task

FILES = ("train", "holdout")
# What the benchmark validated for a hundred examples: a light tune, the best of ten validation
# checkpoints kept, on a pure-attention base that Ruling trains at full speed.
RECIPE = ("--steps", "200", "--eval-every", "20", "--valid-fraction", "0.2")
BASE = "mlx-community/Qwen3-4B-Instruct-2507-4bit"


def direct_question(task: Task) -> dict:
    """The simplest question a judge can be asked about a task: a choice over every label, each
    as the task describes it."""
    return {"type": "choice", "instructions": task.description, "criteria": task.criteria()}


def records(task: Task) -> dict[str, list[dict]]:
    question = direct_question(task)

    def record(example) -> dict:
        return {
            "state": example.state,
            "questions": {"label": question},
            "labels": {"label": example.label},
        }

    return {
        "train": [record(e) for e in [*task.train, *task.validation] if e.label is not None],
        "holdout": [record(e) for e in task.test if e.label is not None],
    }


def export(task: Task, directory: Path) -> dict:
    """Write the task's records under `directory` and say what was written and how to train
    on it. Nothing is overwritten: a tuning set is an input to a run worth keeping."""
    paths = {name: directory / f"{name}.jsonl" for name in FILES}
    taken = [str(path) for path in paths.values() if path.exists()]
    if taken:
        raise FileExistsError(f"already exists: {', '.join(taken)}")
    found = records(task)
    directory.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        path.write_text("".join(json.dumps(row) + "\n" for row in found[name]))
    return {name: {"path": str(paths[name]), "examples": len(found[name])} for name in FILES}


def trainer(ruling_dir: Path) -> list[str] | None:
    """How to invoke Ruling from its checkout: its own environment when it has one, else through
    uv; None where there is no checkout."""
    executable = ruling_dir / ".venv" / "bin" / "ruling"
    if executable.exists():
        return [str(executable)]
    if (ruling_dir / "pyproject.toml").exists():
        return ["uv", "run", "--directory", str(ruling_dir), "ruling"]
    return None


def tune(task: Task, directory: Path, ruling_dir: Path, base: str, extra: list[str]) -> dict:
    """Export the task's records and train an adapter on them with Ruling, under `directory`.
    The recipe comes first and `extra` after it, so anything passed through overrides it. The
    report says where the adapter is and how to serve it."""
    report = export(task, directory)
    command = trainer(ruling_dir)
    if command is None:
        raise FileNotFoundError(
            f"no Ruling checkout at {ruling_dir}; the records are written under {directory}, "
            f"and `ruling train --data {directory / 'train.jsonl'} --holdout "
            f"{directory / 'holdout.jsonl'} --output {directory / 'adapter'}` trains on them"
        )
    adapter = directory / "adapter"
    subprocess.run(
        [
            *command,
            "train",
            "--data",
            report["train"]["path"],
            "--holdout",
            report["holdout"]["path"],
            "--model",
            base,
            *RECIPE,
            *extra,
            "--output",
            str(adapter),
        ],
        check=True,
    )
    return report | {
        "adapter": str(adapter),
        "serve_command": f"RULING_MODEL={base} RULING_ADAPTER={adapter} {' '.join(command)} serve",
    }
