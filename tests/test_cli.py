import json
import subprocess
import sys
from pathlib import Path

from test_graph import binary_plan, task_data


def invoke(*args):
    return subprocess.run(
        [sys.executable, "-m", "jev_compose.cli", *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )


def write(path, value):
    path.write_text(json.dumps(value))
    return path


def test_validate_works_without_credentials(tmp_path):
    task = write(tmp_path / "task.json", task_data())
    plan = write(tmp_path / "plan.json", binary_plan())
    result = invoke("validate", task, plan)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["questions"] == 1


def test_invalid_plan_exits_before_attempting_network(tmp_path):
    task = write(tmp_path / "task.json", task_data())
    plan = write(
        tmp_path / "plan.json", {"questions": {}, "root": {"type": "return", "label": "bad"}}
    )
    result = invoke("validate", task, plan)
    assert result.returncode != 0


def test_cli_evaluation_persists_actual_sdk_results(tmp_path, provider, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-placeholder")
    task = write(tmp_path / "task.json", task_data())
    plan = write(tmp_path / "plan.json", binary_plan())
    output = tmp_path / "result.json"
    provider.body["answers"] = {"eligible": {"type": "noul", "noul": 0.9}}
    result = invoke(
        "evaluate",
        task,
        plan,
        "--base-url",
        provider.url,
        "--model",
        "fixture-model",
        "--output",
        output,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["split"] == "test" and report["accuracy"] == 1.0
    assert report["examples"][0]["id"] == "c"
    assert provider.requests[0][1]["state"] == "request C"
    assert len(provider.requests) == 1
    repeated = invoke("evaluate", task, plan, "--base-url", provider.url, "--output", output)
    assert repeated.returncode != 0
    assert len(provider.requests) == 1


def test_repository_example_matches_the_schema():
    root = Path(__file__).resolve().parents[1]
    result = invoke("validate", root / "examples/task.json", root / "examples/seed.json")
    assert result.returncode == 0, result.stderr


def test_cli_optimization_uses_dspy_lm_and_saves_a_reloadable_tree(tmp_path, provider, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    data = task_data()
    data["validation"][0]["label"] = "accept"
    task = write(tmp_path / "task.json", data)
    initial = write(
        tmp_path / "initial.json",
        {
            "questions": {},
            "root": {"type": "return", "label": "review"},
        },
    )
    provider.completion = json.dumps(binary_plan())
    provider.body["answers"] = {"eligible": {"type": "noul", "noul": 0.9}}
    output = tmp_path / "run"
    result = invoke(
        "optimize",
        task,
        initial,
        "--output",
        output,
        "--base-url",
        provider.url,
        "--model",
        "fixture-model",
        "--reflection-model",
        "openai/fixture-model",
        "--reflection-base-url",
        provider.url + "/v1",
        "--max-metric-calls",
        4,
        "--reflection-batch",
        1,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((output / "plan.json").read_text()) == binary_plan()
    report = json.loads((output / "report.json").read_text())
    assert report["validation_accuracy"] == 1.0
    assert report["seed_validation_accuracy"] == 0.0
    assert report["candidates"] == 2
    assert sum(path == "/v1/chat/completions" for path, _ in provider.requests) == 1
    assert invoke("validate", task, output / "plan.json").returncode == 0


def test_invalid_search_budget_does_not_reserve_output_or_make_calls(
    tmp_path, provider, monkeypatch
):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-placeholder")
    task = write(tmp_path / "task.json", task_data())
    plan = write(tmp_path / "plan.json", binary_plan())
    output = tmp_path / "run"
    result = invoke(
        "optimize",
        task,
        plan,
        "--output",
        output,
        "--base-url",
        provider.url,
        "--reflection-model",
        "openai/fixture-model",
        "--max-metric-calls",
        1,
    )
    assert result.returncode != 0
    assert provider.requests == []
    assert not output.exists()
