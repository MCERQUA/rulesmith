"""A stronger model grades a graph's answers where no label says what is right."""

import json

import dspy
import pytest
from conftest import chat_reply
from test_cli import invoke, write
from test_graph import binary_plan, constant_plan, task_data

from rulesmith.grade import Graded, Grader
from rulesmith.graph import Plan, Task
from rulesmith.runtime import DecisionProgram

UNLABELED = {
    "description": "Route each ticket.",
    "labels": ["accept", "review"],
    "train": [{"id": "a", "state": "request A"}, {"id": "d", "state": "request D"}],
    "validation": [{"id": "b", "state": "request B"}],
    "test": [{"id": "c", "state": "request C"}],
}


def grader(provider, model="openai/grader-model") -> Grader:
    lm = dspy.LM(
        model,
        api_base=provider.url + "/v1",
        api_key="test-placeholder",
        cache=False,
        num_retries=0,
    )
    return Grader(lm, "Route each ticket.")


def chats(provider) -> list[dict]:
    return [request for path, request in provider.requests if path == "/v1/chat/completions"]


def always(label: str) -> DecisionProgram:
    return DecisionProgram(Plan.model_validate(constant_plan(label)), None, "none")


def test_the_grade_is_the_score_and_its_reason_reaches_search(provider):
    provider.completion = chat_reply(score=0.75, reason="Refunds go to review.")
    example = Task.model_validate(UNLABELED).train[0]
    outcome = Graded(grader(provider))(always("accept"), example)
    assert outcome.score == 0.75
    assert "Refunds go to review." in outcome.trace["Feedback"]
    assert outcome.record["grade"] == 0.75


def test_each_answer_is_graded_once(provider):
    # Search re-scores the same answers every round; a grade is paid for the first time only.
    provider.completion = chat_reply(score=1, reason="Right.")
    example = Task.model_validate(UNLABELED).train[0]
    graded = Graded(grader(provider))
    for _ in range(3):
        graded(always("accept"), example)
    graded(always("review"), example)
    assert len(chats(provider)) == 2


def test_the_grader_is_shown_the_label_when_there_is_one(provider):
    provider.completion = chat_reply(score=0, reason="A person said review.")
    example = Task.model_validate(task_data()).validation[0]
    Graded(grader(provider))(always("accept"), example)
    shown = chats(provider)[-1]["messages"][-1]["content"]
    assert f"[[ ## reference ## ]]\n{example.label}" in shown


def test_a_grade_outside_zero_to_one_is_refused(provider):
    provider.completion = chat_reply(score=7, reason="Very good.")
    example = Task.model_validate(UNLABELED).train[0]
    with pytest.raises(ValueError, match="7"):
        Graded(grader(provider))(always("accept"), example)


def test_the_grader_cannot_be_the_model_the_graph_asks(provider, client):
    # Otherwise search finds the graph that asks the grader's own question and scores perfectly.
    provider.completion = chat_reply(score=1, reason="Right.")
    provider.body["answers"] = {"eligible": {"type": "noul", "noul": 0.9}}
    program = DecisionProgram(Plan.model_validate(binary_plan()), client, "fixture-model")
    example = Task.model_validate(UNLABELED).train[0]
    with pytest.raises(ValueError, match="fixture-model"):
        Graded(grader(provider, "openai/fixture-model"))(program, example)
    assert not chats(provider)


def test_a_graph_that_fails_scores_zero_without_a_grade(provider):
    plan = Plan.model_validate(
        {
            "nodes": {"never": {"type": "read", "path": ["missing"], "categories": ["accept"]}},
            "outputs": {"label": {"type": "node", "node": "never"}},
        }
    )
    example = Task.model_validate(UNLABELED).train[0]
    outcome = Graded(grader(provider))(DecisionProgram(plan, None, "none"), example)
    assert outcome.score == 0 and outcome.error
    assert not chats(provider)


def test_cli_evaluates_an_unlabeled_task_with_a_grader(tmp_path, provider, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    provider.completion = chat_reply(score=0.5, reason="Half right.")
    task = write(tmp_path / "task.json", UNLABELED)
    plan = write(tmp_path / "plan.json", constant_plan("accept"))
    common = ("--base-url", provider.url, "--model", "fixture-model")
    refused = invoke("evaluate", task, plan, *common, "--output", tmp_path / "none.json")
    assert refused.returncode != 0 and "grader" in refused.stderr
    assert not provider.requests
    output = tmp_path / "graded.json"
    result = invoke(
        "evaluate",
        task,
        plan,
        *common,
        "--grader-model",
        "openai/grader-model",
        "--grader-base-url",
        provider.url + "/v1",
        "--output",
        output,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    # A graded score is not accuracy, and the report has to say which it is.
    assert report["score"] == 0.5 and report["scored_by"] == "grader openai/grader-model"


def test_cli_searches_an_unlabeled_task_against_a_grader(tmp_path, provider, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-placeholder")
    proposal = chat_reply(graph=constant_plan("review"))

    def respond(request):
        # One endpoint plays both models: the grader favors review, and the proposer offers it.
        prompt = json.dumps(request["messages"])
        if "Grade" in prompt:
            provider.completion = chat_reply(
                score=1.0 if '\\"review\\"' in prompt else 0.0, reason="Tickets need review."
            )
        else:
            provider.completion = proposal

    provider.on_chat = respond
    task = write(tmp_path / "task.json", UNLABELED)
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
        "openai/proposer-model",
        "--reflection-base-url",
        provider.url + "/v1",
        "--grader-model",
        "openai/grader-model",
        "--grader-base-url",
        provider.url + "/v1",
        "--max-metric-calls",
        6,
        "--reflection-batch",
        2,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["seed_validation_score"] == 0.0 and report["validation_score"] == 1.0
    assert report["scored_by"] == "grader openai/grader-model"
    assert Plan.model_validate_json((output / "plan.json").read_text()) == Plan.constant("review")
