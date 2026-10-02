import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import chat_reply
from test_graph import binary_plan, task_data

from rulesmith.graph import Plan, Task


def invoke(*args):
    return subprocess.run(
        [sys.executable, "-m", "rulesmith.cli", *map(str, args)],
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


def test_invalid_opponent_is_rejected_before_judge_credentials(tmp_path, monkeypatch, capsys):
    from test_arena import duel_data

    from rulesmith.cli import main

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    task = write(tmp_path / "task.json", duel_data())
    candidate = binary_plan()
    candidate["nodes"]["decision"].update(below="TURN_LEFT", above="ATTACK")
    plan = write(tmp_path / "plan.json", candidate)
    opponent = write(tmp_path / "opponent.json", {"nodes": {}, "outputs": {}})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rulesmith",
            "evaluate",
            str(task),
            str(plan),
            "--opponent",
            str(opponent),
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    with pytest.raises(SystemExit):
        main()
    assert "outputs" in capsys.readouterr().err


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
    assert report["split"] == "test" and report["score"] == 1.0
    # Two evaluations with different judges must be told apart by their reports.
    assert report["model"] == "fixture-model" and report["resolved_models"] == ["fixture-model"]
    assert report["examples"][0]["id"] == "c"
    assert provider.requests[0][1]["state"] == {"state": "request C"}
    assert len(provider.requests) == 1
    repeated = invoke("evaluate", task, plan, "--base-url", provider.url, "--output", output)
    assert repeated.returncode != 0
    assert len(provider.requests) == 1


@pytest.mark.parametrize("example", ["seed.json", "dag.json"])
def test_repository_example_matches_the_schema(example):
    root = Path(__file__).resolve().parents[1]
    result = invoke(
        "validate",
        root / "examples/support-routing/task.json",
        root / "examples" / "support-routing" / example,
    )
    assert result.returncode == 0, result.stderr


def test_cli_optimization_generates_a_graph_from_the_task_alone(tmp_path, provider, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    data = task_data()
    data["train"].append({"id": "d", "state": "request D", "label": "review"})
    task = write(tmp_path / "task.json", data)
    provider.completion = chat_reply(graph=binary_plan())

    def answer_by_request(request):
        noul = 0.9 if "request A" in json.dumps(request["state"]) else 0.1
        provider.body["answers"] = {"eligible": {"type": "noul", "noul": noul}}

    provider.on_request = answer_by_request
    output = tmp_path / "run"
    result = invoke(
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
        "--max-metric-calls",
        6,
        "--reflection-batch",
        2,
    )
    assert result.returncode == 0, result.stderr
    assert Plan.model_validate_json((output / "plan.json").read_text()) == Plan.model_validate(
        binary_plan()
    )
    report = json.loads((output / "report.json").read_text())
    assert report["validation_score"] == 1.0
    assert report["seed_validation_score"] == 0.0
    assert report["candidates"] == 2
    # When each candidate was found, in metric calls: the seed first, for nothing.
    assert report["discovered_at"][0] == 0 and len(report["discovered_at"]) == 2
    # suppress is empty for a real run; it only carries nodes when measuring what one is worth.
    assert report["execution"] == {"max_workers": 2, "suppress": []}
    # The proposer is asked for a part, and the graph it answers with is the whole program.
    assert sum(path == "/v1/chat/completions" for path, _ in provider.requests) >= 1
    assert invoke("validate", task, output / "plan.json").returncode == 0
    # This graph keys its judgment's input 'state', which rules cannot write, so no rules file.
    assert not (output / "plan.rules").exists() and report["rules"] is None


def test_cli_optimization_saves_the_graph_as_rules_too(tmp_path, provider, monkeypatch):
    # The rules are what search wrote and what a person edits; saving only the JSON loses that.
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    data = task_data()
    data["train"].append({"id": "d", "state": "request D", "label": "review"})
    task = write(tmp_path / "task.json", data)
    rules = (
        'ask eligible = "Should this request be accepted?"\n'
        "when eligible >= 0.5: accept\nalways: review\n"
    )
    seed = tmp_path / "seed.rules"
    seed.write_text(rules)
    provider.completion = chat_reply(graph=rules)

    def answer_by_request(request):
        noul = 0.9 if "request A" in json.dumps(request["state"]) else 0.1
        provider.body["answers"] = {"eligible": {"type": "noul", "noul": noul}}

    provider.on_request = answer_by_request
    output = tmp_path / "run"
    result = invoke(
        "optimize",
        task,
        seed,
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
        6,
        "--reflection-batch",
        2,
    )
    assert result.returncode == 0, result.stderr
    saved = Plan.model_validate_json((output / "plan.json").read_text())
    from rulesmith.rules import compile_rules

    assert compile_rules((output / "plan.rules").read_text()) == saved
    assert json.loads((output / "report.json").read_text())["rules"] == "plan.rules"


def test_cli_drafts_a_reviewable_task_from_unlabeled_inputs(tmp_path, provider, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    states = ["request A", {"request": "B"}, "request C", "request D"]
    inputs = write(
        tmp_path / "inputs.json",
        {"description": "Classify requests.", "labels": ["accept", "review"], "inputs": states},
    )
    provider.completion = chat_reply(label="review")
    output = tmp_path / "task.json"
    result = invoke(
        "draft-labels",
        inputs,
        "--output",
        output,
        "--labeler-model",
        "openai/fixture-model",
        "--labeler-base-url",
        provider.url + "/v1",
        "--validation",
        1,
        "--test",
        1,
    )
    assert result.returncode == 0, result.stderr
    task = Task.model_validate_json(output.read_text())
    examples = task.train + task.validation + task.test
    assert sorted(json.dumps(e.state) for e in examples) == sorted(map(json.dumps, states))
    assert {e.label for e in examples} == {"review"}
    assert len(provider.requests) == len(states)
    plan = write(tmp_path / "plan.json", binary_plan())
    assert invoke("validate", output, plan).returncode == 0


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
    from rulesmith.cli import resolve_backend

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
    from rulesmith.cli import BACKENDS, resolve_backend

    monkeypatch.setenv("RULING_BASE_URL", "  ")
    monkeypatch.setenv("RULING_API_KEY", "  ")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://provider.example.com")
    monkeypatch.setenv("TYPESAFE_API_KEY", "hosted-test-key")
    _, connection = resolve_backend("ruling")
    assert connection == {
        "base_url": BACKENDS["ruling"].base_url,
        "api_key": BACKENDS["ruling"].api_key,
    }


@pytest.mark.parametrize("retries,succeeds", [(0, False), (1, True)])
def test_judge_retries_ride_out_a_transient_failure(
    tmp_path, provider, monkeypatch, retries, succeeds
):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-placeholder")
    task = write(tmp_path / "task.json", task_data())
    plan = write(tmp_path / "plan.json", binary_plan())
    provider.body["answers"] = {"eligible": {"type": "noul", "noul": 0.9}}

    def fail_first(request):
        provider.status = 503 if len(provider.requests) == 1 else 200

    provider.on_request = fail_first
    result = invoke(
        "evaluate",
        task,
        plan,
        "--base-url",
        provider.url,
        "--retries",
        retries,
        "--output",
        tmp_path / "out.json",
    )
    assert (result.returncode == 0) == succeeds, result.stderr


def test_a_hung_proposer_fails_the_search_instead_of_stalling(tmp_path, provider, monkeypatch):
    import threading

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    data = task_data()
    data["train"].append({"id": "d", "state": "request D", "label": "review"})
    task = write(tmp_path / "task.json", data)
    released = threading.Event()
    provider.completion = chat_reply(graph=binary_plan())

    def hang_on_proposals(request):
        released.wait(timeout=30)

    provider.on_chat = hang_on_proposals
    try:
        result = invoke(
            "optimize",
            task,
            "--output",
            tmp_path / "run",
            "--base-url",
            provider.url,
            "--reflection-model",
            "openai/fixture",
            "--reflection-base-url",
            provider.url + "/v1",
            "--reflection-timeout",
            1,
            "--max-metric-calls",
            6,
            "--reflection-batch",
            2,
            "--repair-attempts",
            0,
        )
    finally:
        released.set()
    assert result.returncode != 0
    assert "Timeout" in result.stderr


def test_a_graph_can_be_written_as_rules_and_run_from_the_file(tmp_path):
    task = write(tmp_path / "task.json", task_data())
    rules = tmp_path / "policy.rules"
    rules.write_text("always: review\n")
    state = write(tmp_path / "state.json", "Please refund the duplicate charge.")
    assert invoke("validate", task, rules).returncode == 0
    result = invoke("run", rules, state)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["label"] == "review"


def test_rules_that_will_not_compile_name_the_line(tmp_path):
    task = write(tmp_path / "task.json", task_data())
    rules = tmp_path / "policy.rules"
    rules.write_text("always: review\nwhen nonsense >= 1: accept\n")
    result = invoke("validate", task, rules)
    assert result.returncode != 0 and "line 2" in result.stderr


def test_ablate_refuses_to_overwrite_a_report(tmp_path):
    task = write(tmp_path / "task.json", task_data())
    plan = write(
        tmp_path / "plan.json",
        {"nodes": {}, "outputs": {"label": {"type": "literal", "value": "accept"}}},
    )
    kept = tmp_path / "ablation.json"
    kept.write_text("keep")
    result = invoke("ablate", task, plan, "--output", kept)
    assert result.returncode != 0 and kept.read_text() == "keep"


@pytest.mark.parametrize(
    "extra",
    [
        ("--grader-base-url", "http://127.0.0.1:9"),
        ("--grader-model", "openai/fixture-model"),
        ("--max-workers", "0"),
        ("--parallel-examples", "0"),
    ],
)
def test_flags_that_would_be_ignored_or_break_later_are_refused_up_front(tmp_path, provider, extra):
    # The grader must not be the judge the graph asks, and a base URL with no grader grades
    # nothing; both, and zero workers, are refused before any model is called.
    task = write(tmp_path / "task.json", task_data())
    plan = write(tmp_path / "plan.json", binary_plan())
    result = invoke(
        "evaluate",
        task,
        plan,
        "--base-url",
        provider.url,
        "--model",
        "fixture-model",
        "--output",
        tmp_path / "out.json",
        *extra,
    )
    assert result.returncode != 0 and "Traceback" not in result.stderr
    assert not provider.requests


def test_a_broken_rules_file_is_reported_without_a_traceback(tmp_path):
    plan = tmp_path / "plan.rules"
    plan.write_text("read x in [0, 1]\nwhen nothing >= 1: a\nalways: b")
    result = invoke("draw", plan)
    assert result.returncode != 0 and "line 2" in result.stderr and "Traceback" not in result.stderr


STOCKFISH = shutil.which("stockfish") or os.environ.get("RULESMITH_STOCKFISH")
CHESS = Path(__file__).resolve().parents[1] / "examples/chess"


@pytest.mark.skipif(not STOCKFISH, reason="needs a stockfish binary on PATH or RULESMITH_STOCKFISH")
def test_a_chess_graph_is_evaluated_against_an_engine(tmp_path):
    output = tmp_path / "vs-engine.json"
    result = invoke(
        "evaluate",
        CHESS / "duel.json",
        CHESS / "graphs/greedy.rules",
        "--engine",
        STOCKFISH,
        "--engine-skill",
        "0",
        "--engine-seconds",
        "0.01",
        "--output",
        output,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["opponent"] == f"engine {STOCKFISH} at skill 0"
    assert all("engine" in game["policies"] for game in report["examples"])


@pytest.mark.parametrize(
    "extra",
    [
        ("--engine", "stockfish", "--opponent", str(CHESS / "graphs/greedy.rules")),
        ("--engine-skill", "3"),
    ],
)
def test_an_engine_is_the_opponent_or_nothing(tmp_path, extra):
    result = invoke(
        "evaluate",
        CHESS / "duel.json",
        CHESS / "graphs/greedy.rules",
        "--output",
        tmp_path / "out.json",
        *extra,
    )
    assert result.returncode != 0 and "Traceback" not in result.stderr


def test_an_engine_only_plays_chess(tmp_path):
    task = write(tmp_path / "task.json", task_data())
    plan = write(tmp_path / "plan.json", binary_plan())
    result = invoke("evaluate", task, plan, "--engine", "stockfish", "--output", tmp_path / "o")
    assert result.returncode != 0 and "chess" in result.stderr


ROOT = Path(__file__).resolve().parents[1]
# The packages only the doom, chess and bench extras install.
EXTRAS_ONLY = (
    "vizdoom",
    "gymnasium",
    "numpy",
    "PIL",
    "imageio_ffmpeg",
    "chess",
    "pyarrow",
    "sklearn",
)


def invoke_core(*args):
    """The command line as a core install runs it: a None entry in sys.modules makes importing
    that package fail just as it does when the package is not installed."""
    script = (
        "import sys\n"
        f"sys.modules.update(dict.fromkeys({EXTRAS_ONLY!r}))\n"
        f"sys.argv = ['rulesmith', *{list(map(str, args))!r}]\n"
        "from rulesmith.cli import main\n"
        "main()\n"
    )
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)


def test_a_core_install_validates_and_runs_a_classification_graph(tmp_path):
    validated = invoke_core(
        "validate",
        ROOT / "examples/support-routing/task.json",
        ROOT / "examples/support-routing/seed.json",
    )
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["labels"] == ["billing", "technical", "sales"]
    plan = tmp_path / "plan.rules"
    plan.write_text("read order.total in [0, 1000]\nwhen order.total < 50: ship\nalways: hold")
    state = write(tmp_path / "state.json", {"order": {"total": 20}})
    ran = invoke_core("run", plan, state)
    assert ran.returncode == 0, ran.stderr
    assert json.loads(ran.stdout)["label"] == "ship"


@pytest.mark.parametrize(
    ("extra", "task", "plan"),
    [
        ("doom", ROOT / "examples/doom/duel.json", ROOT / "examples/doom/graphs/duel-attack.json"),
        ("chess", CHESS / "duel.json", CHESS / "graphs/greedy.rules"),
    ],
)
def test_a_game_task_without_its_extra_names_the_install(extra, task, plan):
    result = invoke_core("validate", task, plan)
    assert result.returncode != 0 and "Traceback" not in result.stderr
    assert f"pip install 'rulesmith[{extra}]'" in result.stderr


def test_reflection_options_reach_the_proposers_requests(tmp_path, provider, monkeypatch):
    # A hosted thinking model left alone reasons past any token limit; a cap has to reach it.
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    data = task_data()
    data["train"].append({"id": "d", "state": "request D", "label": "review"})
    task = write(tmp_path / "task.json", data)
    provider.completion = chat_reply(graph=binary_plan())
    provider.on_request = lambda request: provider.body.update(
        answers={"eligible": {"type": "noul", "noul": 0.9}}
    )
    result = invoke(
        "optimize",
        task,
        "--output",
        tmp_path / "run",
        "--base-url",
        provider.url,
        "--model",
        "fixture-model",
        "--reflection-model",
        "openai/fixture-model",
        "--reflection-base-url",
        provider.url + "/v1",
        "--max-metric-calls",
        6,
        "--reflection-batch",
        2,
        "--reflection-options",
        '{"reasoning": {"max_tokens": 64}}',
    )
    assert result.returncode == 0, result.stderr
    chats = [body for path, body in provider.requests if path == "/v1/chat/completions"]
    assert chats and all(body.get("reasoning") == {"max_tokens": 64} for body in chats)
