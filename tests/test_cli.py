import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_graph import binary_plan, task_data

from decision_compose.graph import Plan


def invoke(*args):
    return subprocess.run(
        [sys.executable, "-m", "decision_compose.cli", *map(str, args)],
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
    assert json.loads(result.stdout)["nodes"] == 2


def test_invalid_plan_exits_before_attempting_network(tmp_path):
    task = write(tmp_path / "task.json", task_data())
    plan = write(
        tmp_path / "plan.json",
        {"nodes": {}, "outputs": {"label": {"type": "literal", "value": "bad"}}},
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
    assert provider.requests[0][1]["state"] == {"state": "request C"}
    assert len(provider.requests) == 1
    repeated = invoke("evaluate", task, plan, "--base-url", provider.url, "--output", output)
    assert repeated.returncode != 0
    assert len(provider.requests) == 1


@pytest.mark.parametrize("example", ["seed.json", "dag.json"])
def test_repository_example_matches_the_schema(example):
    root = Path(__file__).resolve().parents[1]
    result = invoke("validate", root / "examples/task.json", root / "examples" / example)
    assert result.returncode == 0, result.stderr


def test_cli_optimization_uses_dspy_lm_and_saves_a_reloadable_graph(
    tmp_path, provider, monkeypatch
):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    data = task_data()
    data["validation"][0]["label"] = "accept"
    task = write(tmp_path / "task.json", data)
    initial = write(
        tmp_path / "initial.json",
        {
            "nodes": {},
            "outputs": {"label": {"type": "literal", "value": "review"}},
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
        "--max-workers",
        2,
        "--max-metric-calls",
        4,
        "--reflection-batch",
        1,
    )
    assert result.returncode == 0, result.stderr
    assert Plan.model_validate_json((output / "plan.json").read_text()) == Plan.model_validate(
        binary_plan()
    )
    report = json.loads((output / "report.json").read_text())
    assert report["validation_accuracy"] == 1.0
    assert report["seed_validation_accuracy"] == 0.0
    assert report["candidates"] == 2
    assert report["execution"] == {"max_workers": 2}
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


def test_cli_run_supports_generic_named_outputs_without_a_task(tmp_path):
    plan = write(
        tmp_path / "graph.json",
        {
            "nodes": {
                "joined": {
                    "type": "merge",
                    "inputs": {
                        "request": {"type": "state"},
                        "policy": {"type": "literal", "value": {"version": 3}},
                    },
                }
            },
            "outputs": {"payload": {"type": "node", "node": "joined"}},
        },
    )
    state = write(tmp_path / "state.json", {"account": "example"})
    result = invoke("run", plan, state, "--max-workers", 2)
    assert result.returncode == 0, result.stderr
    prediction = json.loads(result.stdout)
    assert prediction["outputs"] == {
        "payload": {"request": {"account": "example"}, "policy": {"version": 3}}
    }
    assert prediction["calls"] == {}
    assert "label" not in prediction


def test_ruling_backend_uses_local_endpoint_without_typesafe_credentials(
    tmp_path, provider, monkeypatch
):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("RULING_API_KEY", raising=False)
    monkeypatch.setenv("TYPESAFE_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("RULING_BASE_URL", provider.url)
    plan = write(tmp_path / "plan.json", binary_plan())
    state = write(tmp_path / "state.json", "Please refund the duplicate charge.")
    provider.body["answers"] = {"eligible": {"type": "noul", "noul": 0.9}}
    result = invoke("run", plan, state, "--backend", "ruling")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["label"] == "accept"
    assert provider.requests[0][1]["model"] == "default"


@pytest.mark.parametrize("backend", ["ruling", "typesafe"])
def test_backend_configuration_keeps_endpoints_and_credentials_separate(backend, monkeypatch):
    from decision_compose.cli import resolve_backend

    monkeypatch.setenv("RULING_BASE_URL", "http://127.0.0.1:8010")
    monkeypatch.setenv("RULING_API_KEY", "local-test-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://provider.example.com")
    monkeypatch.setenv("TYPESAFE_API_KEY", "hosted-test-key")
    model, connection = resolve_backend(backend)
    assert connection == {
        "base_url": "http://127.0.0.1:8010"
        if backend == "ruling"
        else "https://provider.example.com",
        "api_key": "local-test-key" if backend == "ruling" else "hosted-test-key",
    }
    assert model == ("default" if backend == "ruling" else "jev-latest")
    model, connection = resolve_backend(
        backend, model="chosen-model", base_url="http://127.0.0.1:9"
    )
    assert model == "chosen-model" and connection["base_url"] == "http://127.0.0.1:9"


def test_empty_ruling_configuration_never_falls_back_to_hosted_endpoint(monkeypatch):
    from decision_compose.cli import BACKENDS, resolve_backend

    monkeypatch.setenv("RULING_BASE_URL", "  ")
    monkeypatch.setenv("RULING_API_KEY", "  ")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://provider.example.com")
    monkeypatch.setenv("TYPESAFE_API_KEY", "hosted-test-key")
    _, connection = resolve_backend("ruling")
    assert connection == {
        "base_url": BACKENDS["ruling"].base_url,
        "api_key": BACKENDS["ruling"].api_key,
    }
