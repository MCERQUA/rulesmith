import json

import pytest
from test_graph import binary_plan, call, constant_plan, ref, task_data

from decision_compose.graph import Plan, Task
from decision_compose.optimize import GraphAdapter, SearchConfig, optimize_plan
from decision_compose.runtime import DecisionProgram, ResponseError


def test_official_sdk_http_dspy_and_dag_execute_together(provider, client):
    plan = Plan.model_validate(binary_plan())
    provider.body["answers"] = {"eligible": {"type": "noul", "noul": 0.9}}
    result = DecisionProgram(plan, client, "fixture-model")(state={"request": "Example"})
    assert result.label == "accept"
    assert result.outputs == {"label": "accept"}
    assert result.models == ["fixture-model"] and result.input_tokens == 12
    assert provider.requests == [
        (
            "/v1/systemone",
            {
                "state": {"state": {"request": "Example"}},
                "model": "fixture-model",
                "questions": {"eligible": plan.nodes["eligible"].question.model_dump()},
            },
        )
    ]


def test_provider_contract_failure_does_not_become_a_bad_candidate(provider, client):
    task = Task.model_validate(task_data())
    adapter = GraphAdapter(task, client, "fixture-model", SearchConfig())
    with pytest.raises(ResponseError, match="IDs"):
        adapter.evaluate(
            task.train, {"graph": Plan.model_validate(binary_plan()).model_dump_json()}
        )


def test_invalid_candidate_produces_actionable_feedback_without_http(provider, client):
    task = Task.model_validate(task_data())
    adapter = GraphAdapter(task, client, "fixture-model", SearchConfig())
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
    initial = Plan.model_validate(constant_plan())
    proposal = binary_plan()
    proposal["nodes"].update(left=call(), right=call())
    proposal["nodes"]["eligible"]["inputs"].update(left=ref("left"), right=ref("right"))
    proposed = Plan.model_validate(proposal)
    provider.answers = {key: {"type": "noul", "noul": 0.9} for key in ("left", "right", "eligible")}
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
    assert len(provider.requests) == 6
    assert report["resolved_models"] == ["fixture-model"]
    assert len(report["task_sha256"]) == 64
    assert len(report["seed_sha256"]) == 64
    assert set(report["versions"]) == {"dspy", "gepa", "typesafe-sdk"}
    assert {body["state"]["state"] for _, body in provider.requests} == {"request A", "request B"}
    assert "request C" not in json.dumps(reflections)


def test_constant_program_never_calls_provider(provider, client):
    plan = Plan.model_validate(constant_plan())
    assert DecisionProgram(plan, client, "fixture-model")(state="Example").label == "review"
    assert provider.requests == []


def test_reflection_outage_is_propagated_and_stops_search(client):
    task = Task.model_validate(task_data())
    initial = Plan.model_validate(constant_plan())
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


def test_choice_score_and_noul_execute_as_independent_sdk_calls(provider, client, mixed_plan):
    provider.answers = {
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
    assert set(prediction.answers) == {"team", "urgency", "refund"}
    assert len(provider.requests) == 3
    assert prediction.input_tokens == 36
    for _, request in provider.requests:
        (key,) = request["questions"]
        assert request["questions"][key] == mixed_plan.nodes[key].question.model_dump()


def test_diamond_routes_values_and_answer_records_and_runs_shared_nodes_once(provider, client):
    from threading import Barrier

    from decision_compose.runtime import ExecutionConfig

    plan = Plan.model_validate(
        {
            "nodes": {
                "root": call(),
                "left": call(parent=ref("root")),
                "right": call(parent=ref("root")),
                "join": call(
                    left=ref("left"),
                    right=ref("right"),
                    evidence=ref("root", view="answer"),
                    original={"type": "state"},
                ),
                "independent": call(),
            },
            "outputs": {"joined": ref("join"), "other": ref("independent")},
        }
    )
    expected = {"root": 0.4, "left": 0.6, "right": 0.8, "join": 0.9, "independent": 0.1}
    provider.answers = {key: {"type": "noul", "noul": value} for key, value in expected.items()}
    siblings = Barrier(2)

    def synchronize(request):
        (key,) = request["questions"]
        if key in {"left", "right"}:
            # The timeout only bounds a deadlock if parallel execution regresses.
            siblings.wait(timeout=5)

    provider.on_request = synchronize
    result = DecisionProgram(plan, client, "fixture-model", ExecutionConfig(max_workers=2))(
        state={"ticket": "input"}
    )
    requests = {next(iter(body["questions"])): body for _, body in provider.requests}
    assert len(provider.requests) == len(plan.nodes)
    assert requests["left"]["state"] == requests["right"]["state"] == {"parent": 0.4}
    assert requests["join"]["state"] == {
        "left": 0.6,
        "right": 0.8,
        "evidence": {"type": "noul", "value": 0.4},
        "original": {"ticket": "input"},
    }
    assert result.outputs == {"joined": 0.9, "other": 0.1}
    assert result.node_values == expected
    assert result.input_tokens == 12 * len(plan.nodes)


@pytest.mark.parametrize("probability,active", [(0.2, "low"), (0.8, "high")])
def test_conditional_calls_skip_unused_branches_and_join_available_outputs(
    probability,
    active,
    provider,
    client,
):
    condition = {"type": "compare", "node": "gate", "op": "ge", "value": 0.5}
    plan = Plan.model_validate(
        {
            "nodes": {
                "gate": call(),
                "high": {
                    **call(),
                    "when": {
                        "type": "all",
                        "conditions": [condition, {"type": "present", "node": "gate"}],
                    },
                },
                "low": {
                    **call(),
                    "when": {
                        "type": "any",
                        "conditions": [
                            {"type": "not", "condition": condition},
                            {"type": "compare", "node": "gate", "op": "lt", "value": 0.1},
                        ],
                    },
                },
                "joined": {"type": "coalesce", "sources": ["high", "low"]},
                "final": call(
                    result=ref("joined"),
                    high=ref("high", optional=True),
                    low=ref("low", optional=True),
                ),
            },
            "outputs": {"result": ref("final")},
        }
    )
    provider.answers = {key: {"type": "noul", "noul": probability} for key in plan.nodes}
    result = DecisionProgram(plan, client, "fixture-model")(state="input")
    requests = {next(iter(body["questions"])): body for _, body in provider.requests}
    assert set(requests) == {"gate", active, "final"}
    assert requests["final"]["state"] == {"result": probability, active: probability}
    assert result.skipped == ["low" if active == "high" else "high"]


def test_scheduler_releases_dependents_without_waiting_for_unrelated_roots(provider, client):
    from threading import Event

    from decision_compose.runtime import ExecutionConfig

    plan = Plan.model_validate(
        {
            "nodes": {
                "slow": call(),
                "fast": call(),
                "child": call(parent=ref("fast")),
            },
            "outputs": {"slow": ref("slow"), "child": ref("child")},
        }
    )
    provider.answers = {key: {"type": "noul", "noul": 0.6} for key in plan.nodes}
    child_started = Event()

    def coordinate(request):
        (key,) = request["questions"]
        if key == "child":
            child_started.set()
        elif key == "slow":
            assert child_started.wait(timeout=5), "independent root blocked a ready descendant"

    provider.on_request = coordinate
    result = DecisionProgram(plan, client, "fixture-model", ExecutionConfig(max_workers=2))(
        state="input"
    )
    assert result.outputs == {"slow": 0.6, "child": 0.6}
    assert len(provider.requests) == 3


def test_coalesce_uses_declared_priority_and_program_does_not_cache_previous_input(
    provider, client
):
    plan = Plan.model_validate(
        {
            "nodes": {
                "a": call(),
                "z": call(),
                "join": {"type": "coalesce", "sources": ["z", "a"]},
            },
            "outputs": {"value": ref("join")},
        }
    )
    program = DecisionProgram(plan, client, "fixture-model")
    for probability in (0.3, 0.7):
        provider.answers = {
            "a": {"type": "noul", "noul": 0.1},
            "z": {"type": "noul", "noul": probability},
        }
        assert program(state=str(probability)).outputs == {"value": probability}
    assert len(provider.requests) == 4


def test_candidate_with_unavailable_label_gets_feedback_without_poisoning_search(provider, client):
    data = binary_plan()
    data["nodes"]["decision"]["when"] = {
        "type": "compare",
        "node": "eligible",
        "op": "ge",
        "value": 0.5,
    }
    plan = Plan.model_validate(data)
    task = Task.model_validate(task_data())
    provider.body["answers"] = {"eligible": {"type": "noul", "noul": 0.1}}
    adapter = GraphAdapter(task, client, "fixture-model", SearchConfig())
    batch = adapter.evaluate(task.train, {"graph": plan.model_dump_json()}, capture_traces=True)
    assert batch.scores == [0.0]
    assert "label" in batch.trajectories[0]["Feedback"]
    assert adapter.failure is None
    trace = batch.trajectories[0]["Generated Outputs"]
    assert trace["node_values"]["eligible"] == 0.1
    assert trace["input_tokens"] == 12
    valid = adapter.evaluate(
        task.train, {"graph": Plan.model_validate(constant_plan("accept")).model_dump_json()}
    )
    assert valid.scores == [1.0]


def test_provider_error_prevents_dependent_calls(provider, client):
    plan = Plan.model_validate(
        {
            "nodes": {
                "root": call(),
                "dependent": call(parent=ref("root")),
            },
            "outputs": {"result": ref("dependent")},
        }
    )
    with pytest.raises(ResponseError, match="IDs"):
        DecisionProgram(plan, client, "fixture-model")(state="input")
    assert len(provider.requests) == 1


@pytest.mark.parametrize(
    "team,severity,label",
    [
        ("support", 2, "review"),
        ("billing", 2, "urgent"),
        ("billing", 1, "review"),
    ],
)
def test_switch_routes_to_a_conditional_score_and_rejoins(team, severity, label, provider, client):
    plan = Plan.model_validate(
        {
            "nodes": {
                "team": {
                    "type": "call",
                    "inputs": {"state": {"type": "state"}},
                    "question": {
                        "type": "choice",
                        "instructions": "Choose a team.",
                        "criteria": {"billing": "Payments", "support": "Other requests"},
                    },
                },
                "route": {
                    "type": "switch",
                    "source": "team",
                    "cases": {"billing": "assess", "support": "review"},
                },
                "severity": {
                    "type": "call",
                    "inputs": {"state": {"type": "state"}},
                    "question": {
                        "type": "score",
                        "instructions": "Assess severity.",
                        "criteria": ["Routine", "Disrupted", "Blocked"],
                    },
                    "when": {"type": "compare", "node": "route", "op": "eq", "value": "assess"},
                },
                "decision": {
                    "type": "threshold",
                    "source": "severity",
                    "cutoff": 1.5,
                    "below": "review",
                    "above": "urgent",
                },
                "fallback": {
                    "type": "value",
                    "value": "review",
                    "when": {"type": "compare", "node": "route", "op": "eq", "value": "review"},
                },
                "joined": {"type": "coalesce", "sources": ["decision", "fallback"]},
            },
            "outputs": {"label": ref("joined")},
        }
    )
    provider.answers = {
        "team": {
            "type": "choice",
            "choice": team,
            "confidence": 1.0,
            "probabilities": {key: float(key == team) for key in ("billing", "support")},
        },
        "severity": {
            "type": "score",
            "score": severity,
            "confidence": 1.0,
            "probabilities": {str(i): float(i == severity) for i in range(3)},
            "legend": {
                str(i): level for i, level in enumerate(["Routine", "Disrupted", "Blocked"])
            },
        },
    }
    result = DecisionProgram(plan, client, "fixture-model")(state="input")
    assert result.label == label
    assert len(provider.requests) == (1 if team == "support" else 2)
    if team == "support":
        assert set(result.skipped) == {"severity", "decision"}
