"""A search that checkpoints as it goes, stops gracefully, and resumes from where it stopped."""

import json

import pytest
from conftest import chat_reply
from test_cli import invoke, write
from test_graph import binary_plan, task_data

from rulesmith.graph import Plan, Task
from rulesmith.optimize import SearchConfig, optimize_plan


def routed(provider):
    def answer_by_request(request):
        noul = 0.9 if "request A" in json.dumps(request["state"]) else 0.1
        provider.body["answers"] = {"eligible": {"type": "noul", "noul": noul}}

    provider.on_request = answer_by_request


def small_task() -> Task:
    data = task_data()
    data["train"].append({"id": "d", "state": "request D", "label": "review"})
    return Task.model_validate(data)


def test_a_stopped_search_resumes_with_its_candidates_and_spends_the_rest_of_its_budget(
    tmp_path, provider, client
):
    task = small_task()
    routed(provider)
    proposals = []
    run_dir = tmp_path / "search"

    def propose(prompt):
        proposals.append(prompt)
        # Ask GEPA to stop after this proposal is scored, the way a deadline would.
        (run_dir / "gepa.stop").write_text("")
        return json.dumps(binary_plan())

    config = SearchConfig(max_metric_calls=20, reflection_batch=2)
    plan, report = optimize_plan(
        task, task.baseline_plan(), client, "fixture", propose, config, run_dir=run_dir
    )
    assert report["stopped_early"] and report["metric_calls"] < 20
    found = report["candidates"]
    assert (run_dir / "gepa_state.bin").exists()
    (run_dir / "gepa.stop").unlink()

    def propose_more(prompt):
        return json.dumps(binary_plan(cutoff=0.4))

    plan, report = optimize_plan(
        task, task.baseline_plan(), client, "fixture", propose_more, config, run_dir=run_dir
    )
    # Resumed, not restarted: the candidates found before are still there, in order, and the
    # budget continues from where it stood rather than from zero.
    assert report["resumed"] and not report["stopped_early"]
    assert report["candidates"] >= found and report["discovered_at"][:found] == sorted(
        report["discovered_at"][:found]
    )
    assert report["metric_calls"] >= 20


def cli_search(task, output, provider, *extra):
    return invoke(
        "optimize",
        task,
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
        "--reflection-batch",
        2,
        *extra,
    )


def test_the_cli_resumes_an_output_directory_and_refuses_to_restart_one(
    tmp_path, provider, monkeypatch
):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    task = write(tmp_path / "task.json", small_task().model_dump(mode="json"))
    provider.completion = chat_reply(graph=binary_plan())
    routed(provider)
    output = tmp_path / "run"
    first = cli_search(task, output, provider, "--max-metric-calls", 6)
    assert first.returncode == 0, first.stderr
    calls = json.loads((output / "report.json").read_text())["metric_calls"]
    assert (output / "search" / "gepa_state.bin").exists()
    # The same directory again is a mistake unless it is a resume.
    again = cli_search(task, output, provider, "--max-metric-calls", 6)
    assert again.returncode != 0 and "resume" in again.stderr
    # Resumed with a larger budget, the search runs on and the report says so.
    longer = cli_search(task, output, provider, "--max-metric-calls", 12, "--resume")
    assert longer.returncode == 0, longer.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["resumed"] and report["metric_calls"] > calls
    assert Plan.model_validate_json((output / "plan.json").read_text())


def test_a_resume_must_continue_the_same_search(tmp_path, provider, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    task = write(tmp_path / "task.json", small_task().model_dump(mode="json"))
    provider.completion = chat_reply(graph=binary_plan())
    routed(provider)
    output = tmp_path / "run"
    assert cli_search(task, output, provider, "--max-metric-calls", 6).returncode == 0
    # Another task, or another starting graph, is another search; it may not borrow this one.
    other = task_data()
    other["description"] = "Something else entirely."
    other_task = write(tmp_path / "other.json", other)
    result = cli_search(other_task, output, provider, "--max-metric-calls", 12, "--resume")
    assert result.returncode != 0 and "another" in result.stderr
    # And resuming needs a checkpoint to resume from.
    empty = tmp_path / "empty"
    empty.mkdir()
    result = cli_search(task, empty, provider, "--max-metric-calls", 6, "--resume")
    assert result.returncode != 0 and "checkpoint" in result.stderr


def test_search_config_still_refuses_a_budget_too_small_for_one_round():
    with pytest.raises(ValueError, match="max_metric_calls"):
        optimize_plan(
            small_task(),
            Plan.constant("review"),
            None,
            "none",
            lambda _: "always: review",
            SearchConfig(max_metric_calls=1),
        )
