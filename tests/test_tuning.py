"""Exporting a task's examples for tuning the judge, in the shape Ruling's trainer reads."""

import json
import pathlib

import pytest
from test_cli import invoke, write
from test_graph import task_data

from rulesmith.graph import Task
from rulesmith.tuning import export, records


def labeled_task() -> Task:
    data = task_data()
    data["train"] = [
        {"id": f"t{i}", "state": f"request {i}", "label": "accept" if i % 2 else "review"}
        for i in range(6)
    ]
    data["validation"] = [{"id": "v0", "state": "request v", "label": "review"}]
    data["test"] = [{"id": f"h{i}", "state": f"held {i}", "label": "accept"} for i in range(3)]
    return Task.model_validate(data)


def test_records_ask_the_direct_question_with_the_label_and_keep_the_test_split_apart():
    # The adapter is trained on exactly the examples search sees, each asked the direct
    # question with the right label, and the holdout file holds every test example so the
    # trainer can refuse any overlap: a judge tuned on a test row would void the whole setting.
    task = labeled_task()
    found = records(task)
    assert len(found["train"]) == 7 and len(found["holdout"]) == 3
    question = found["train"][0]["questions"]["label"]
    assert question == {
        "type": "choice",
        "instructions": task.description,
        "criteria": task.criteria(),
    }
    assert [r["labels"]["label"] for r in found["train"]] == [
        e.label for e in [*task.train, *task.validation]
    ]
    states = {json.dumps(r["state"], sort_keys=True) for r in found["train"]}
    assert not states & {json.dumps(r["state"], sort_keys=True) for r in found["holdout"]}


def test_export_writes_both_files_once(tmp_path):
    report = export(labeled_task(), tmp_path / "tuning")
    lines = (tmp_path / "tuning" / "train.jsonl").read_text().splitlines()
    assert len([json.loads(line) for line in lines]) == report["train"]["examples"] == 7
    assert report["holdout"]["examples"] == 3
    with pytest.raises(FileExistsError, match="already exists"):
        export(labeled_task(), tmp_path / "tuning")


def fake_ruling(tmp_path) -> pathlib.Path:
    """A Ruling checkout whose trainer records what it was asked and writes an adapter, so the
    whole hand-off is exercised without a model."""
    checkout = tmp_path / "ruling"
    (checkout / ".venv" / "bin").mkdir(parents=True)
    script = checkout / ".venv" / "bin" / "ruling"
    script.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$@" > "$(dirname "$0")/../../argv.txt"\n'
        'for i in "$@"; do [ "$prev" = "--output" ] && out="$i"; prev="$i"; done\n'
        'mkdir -p "$out" && echo fake > "$out/adapters.safetensors"\n'
    )
    script.chmod(0o755)
    return checkout


def test_tune_runs_ruling_on_the_records_and_says_how_to_serve_the_adapter(tmp_path):
    checkout = fake_ruling(tmp_path)
    task = write(tmp_path / "task.json", labeled_task().model_dump(mode="json"))
    out = tmp_path / "tuning"
    result = invoke("tune", task, "--output", out, "--ruling-dir", checkout, "--", "--steps", "50")
    assert result.returncode == 0, result.stderr
    printed = json.loads(result.stdout)
    argv = (checkout / "argv.txt").read_text().splitlines()
    assert argv[0] == "train"
    assert argv[argv.index("--data") + 1] == str(out / "train.jsonl")
    assert argv[argv.index("--holdout") + 1] == str(out / "holdout.jsonl")
    # The recipe is there, and the passed-through argument comes after it, so it wins.
    assert "--eval-every" in argv and argv.index("--steps", argv.index("--steps") + 1) > 0
    assert argv[-2:] == ["--output", str(out / "adapter")]
    assert (out / "adapter" / "adapters.safetensors").exists()
    assert printed["adapter"] == str(out / "adapter")
    assert "RULING_ADAPTER=" in printed["serve_command"] and " serve" in printed["serve_command"]


def test_tune_refuses_a_game_an_existing_output_and_a_missing_checkout(tmp_path):
    checkout = fake_ruling(tmp_path)
    task = write(tmp_path / "task.json", labeled_task().model_dump(mode="json"))
    doom = write(tmp_path / "doom.json", {"scenario": "basic", "labels": ["ATTACK"]})
    assert (
        invoke("tune", doom, "--output", tmp_path / "a", "--ruling-dir", checkout).returncode != 0
    )
    missing = invoke("tune", task, "--output", tmp_path / "b", "--ruling-dir", tmp_path / "nowhere")
    assert missing.returncode != 0 and "no Ruling checkout" in missing.stderr
    # The records were still written, and running again on the same output is refused.
    assert (tmp_path / "b" / "train.jsonl").exists()
    again = invoke("tune", task, "--output", tmp_path / "b", "--ruling-dir", checkout)
    assert again.returncode != 0 and "already exists" in again.stderr
