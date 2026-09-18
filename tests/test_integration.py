import json

import pytest
from test_graph import binary_plan, task_data

from decision_compose.graph import Plan, Task
from decision_compose.optimize import SearchConfig, TreeAdapter, optimize_plan
from decision_compose.runtime import DecisionProgram, ResponseError


def test_official_sdk_http_dspy_and_tree_execute_together(provider, client):
    plan = Plan.model_validate(binary_plan())
    provider.body["answers"] = {"eligible": {"type": "noul", "noul": 0.9}}
    result = DecisionProgram(plan, client, "fixture-model")(state={"request": "Example"})
    assert result.label == "accept"
    assert result.path == ["eligible"]
    assert result.model == "fixture-model" and result.input_tokens == 12
    assert provider.requests == [
        (
            "/v1/systemone",
            {
                "state": {"request": "Example"},
                "model": "fixture-model",
                "questions": plan.model_dump()["questions"],
            },
        )
    ]


def test_provider_contract_failure_does_not_become_a_bad_candidate(provider, client):
    task = Task.model_validate(task_data())
    adapter = TreeAdapter(task, client, "fixture-model", SearchConfig())
    with pytest.raises(ResponseError, match="IDs"):
        adapter.evaluate(
            task.train, {"graph": Plan.model_validate(binary_plan()).model_dump_json()}
        )


def test_invalid_candidate_produces_actionable_feedback_without_http(provider, client):
    task = Task.model_validate(task_data())
    adapter = TreeAdapter(task, client, "fixture-model", SearchConfig())
    batch = adapter.evaluate(task.train, {"graph": "{}"}, capture_traces=True)
    assert batch.scores == [0.0]
    assert batch.outputs == [None]
    feedback = adapter.make_reflective_dataset({"graph": "{}"}, batch, ["graph"])
    assert "validation" in feedback["graph"][0]["Feedback"].lower()
    assert provider.requests == []


def test_real_gepa_evolves_structure_through_the_official_sdk(provider, client):
    data = task_data()
    data["validation"][0]["label"] = "accept"
    task = Task.model_validate(data)
    initial = Plan.model_validate(
        {
            "questions": {},
            "root": {"type": "return", "label": "review"},
        }
    )
    proposed = Plan.model_validate(binary_plan())
    provider.body["answers"] = {"eligible": {"type": "noul", "noul": 0.9}}
    reflections = []

    def scripted_reflection(prompt):
        # A fixed proposal isolates optimizer wiring from paid model quality.
        reflections.append(prompt)
        return proposed.model_dump_json()

    plan, report = optimize_plan(
        task,
        initial,
        client,
        "fixture-model",
        scripted_reflection,
        SearchConfig(max_metric_calls=4, reflection_batch=1),
    )
    assert len(reflections) == 1
    assert plan == proposed
    assert report["validation_accuracy"] == 1.0
    assert report["seed_validation_accuracy"] == 0.0
    assert report["candidates"] == 2
    assert report["metric_calls"] == 4
    assert report["resolved_models"] == ["fixture-model"]
    assert len(report["task_sha256"]) == 64
    assert len(report["seed_sha256"]) == 64
    assert set(report["versions"]) == {"dspy", "gepa", "typesafe-sdk"}
    assert {body["state"] for _, body in provider.requests} == {"request A", "request B"}
    assert "request C" not in json.dumps(reflections)


def test_constant_program_never_calls_provider(provider, client):
    plan = Plan.model_validate({"questions": {}, "root": {"type": "return", "label": "review"}})
    assert DecisionProgram(plan, client, "fixture-model")(state="Example").label == "review"
    assert provider.requests == []


def test_reflection_outage_is_propagated_and_stops_search(client):
    task = Task.model_validate(task_data())
    initial = Plan.model_validate(
        {
            "questions": {},
            "root": {"type": "return", "label": "review"},
        }
    )
    calls = []

    def unavailable(prompt):
        calls.append(prompt)
        raise RuntimeError("reflection provider unavailable")

    with pytest.raises(RuntimeError, match="reflection provider unavailable"):
        optimize_plan(
            task,
            initial,
            client,
            "fixture-model",
            unavailable,
            SearchConfig(max_metric_calls=4, reflection_batch=1),
        )
    assert len(calls) == 1


def test_choice_score_and_noul_share_one_sdk_request(provider, client, mixed_plan):
    provider.body["answers"] = {
        "team": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.8,
            "probabilities": {"billing": 0.9, "technical": 0.1},
        },
        "urgency": {
            "type": "score",
            "score": 0.9,
            "confidence": 0.8,
            "probabilities": {"0": 0.1, "1": 0.9},
            "legend": {"0": "Routine request", "1": "Immediate assistance needed"},
        },
        "refund": {"type": "noul", "noul": 0.9},
    }
    prediction = DecisionProgram(mixed_plan, client, "fixture-model")(
        state="Refund this charge now."
    )
    assert prediction.label == "accept"
    assert prediction.path == ["team", "urgency", "refund"]
    assert len(provider.requests) == 1
    assert provider.requests[0][1]["questions"] == mixed_plan.model_dump()["questions"]
