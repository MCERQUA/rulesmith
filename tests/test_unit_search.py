"""Searching one unit of a map against its own task, with the rest of the map frozen."""

import hashlib
import json

from conftest import chat_reply
from test_cli import invoke, write
from test_graph import task_data
from test_map import BILLING, router

from rulesmith.graph import Plan
from rulesmith.rules import compile_rules

# A billing task: the unit's own labels, over tickets shaped like what the router hands it.
BILLING_TASK = {
    "description": "Settle a billing ticket.",
    "labels": ["auto_refund", "review"],
    "train": [
        {
            "id": "a",
            "state": {"charge": {"amount": 40}, "message": "twice"},
            "label": "auto_refund",
        },
        {"id": "d", "state": {"charge": {"amount": 900}, "message": "all"}, "label": "review"},
    ],
    "validation": [
        {"id": "b", "state": {"charge": {"amount": 30}, "message": "dup"}, "label": "auto_refund"}
    ],
    "test": [
        {"id": "c", "state": {"charge": {"amount": 12}, "message": "seat"}, "label": "review"}
    ],
}
# The proposal: the unit with a cheaper rule, asking no judge.
FOUND = "read charge.amount in [0, 100000]\nwhen charge.amount < 50: auto_refund\nalways: review"


def search(tmp_path, provider, monkeypatch, map_graph: dict, proposal: str, task_data=None):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    (tmp_path / "billing.rules").write_text(BILLING)
    map_file = write(tmp_path / "support.json", map_graph)
    task = write(tmp_path / "billing-task.json", task_data or BILLING_TASK)
    provider.completion = chat_reply(graph=proposal)
    # The judge doubts every ticket, so the seed unit reviews the refund it should grant and
    # the proposal, which reads the amount instead, beats it.
    provider.answers = {"genuine": {"type": "noul", "noul": 0.1}}
    output = tmp_path / "run"
    result = invoke(
        "optimize",
        task,
        map_file,
        "--unit",
        "billing",
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
    return result, output


def test_a_unit_is_searched_from_its_own_graph_and_written_as_a_unit(
    tmp_path, provider, monkeypatch
):
    result, output = search(tmp_path, provider, monkeypatch, router("billing.rules"), FOUND)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "report.json").read_text())
    unit = compile_rules(BILLING)
    # Search started from the unit's graph, not the map's, and says which unit it searched.
    assert report["seed_sha256"] == hashlib.sha256(unit.model_dump_json().encode()).hexdigest()
    assert report["unit"] == {"name": "billing", "path": "billing.rules", "placed": True}
    found = Plan.model_validate_json((output / "plan.json").read_text())
    assert found == compile_rules(FOUND)
    # The winner drops in where the unit was, and the rest of the map is untouched.
    assert (output / "plan.rules").read_text().strip() == FOUND


def test_a_winner_the_map_cannot_take_is_refused(tmp_path, provider, monkeypatch):
    graph = router("billing.rules")
    graph["nodes"]["refunded"] = {
        "type": "value",
        "value": "engineer",
        "when": {"type": "compare", "node": "billing", "op": "eq", "value": "review"},
    }
    graph["nodes"]["decision"]["sources"] = ["refunded", "billing", "escalate"]
    # The unit's task now wants big charges held, and the winner never answers 'review' any
    # more, which the router's rule still tests for.
    task = json.loads(json.dumps(BILLING_TASK))
    task["labels"].append("hold")
    task["train"][1]["label"] = "hold"
    proposal = (
        "read charge.amount in [0, 100000]\nwhen charge.amount < 50: auto_refund\nalways: hold"
    )
    result, output = search(tmp_path, provider, monkeypatch, graph, proposal, task)
    assert result.returncode != 0
    assert "billing == 'review'" in result.stderr
    assert not (output / "plan.json").exists()


def test_the_unit_must_be_a_unit_of_the_map(tmp_path, provider, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-placeholder")
    (tmp_path / "billing.rules").write_text(BILLING)
    map_file = write(tmp_path / "support.json", router("billing.rules"))
    task = write(tmp_path / "billing-task.json", BILLING_TASK)
    result = invoke(
        "optimize",
        task,
        map_file,
        "--unit",
        "team",
        "--output",
        tmp_path / "run",
        "--reflection-model",
        "openai/fixture-model",
    )
    assert result.returncode != 0 and "team" in result.stderr and "unit" in result.stderr
    plain = write(tmp_path / "plain.json", task_data())
    result = invoke(
        "optimize",
        plain,
        tmp_path / "billing.rules",
        "--unit",
        "billing",
        "--output",
        tmp_path / "run2",
        "--reflection-model",
        "openai/fixture-model",
    )
    assert result.returncode != 0 and "unit" in result.stderr


MAP_TASK = {
    "description": "Answer a support ticket.",
    "labels": ["auto_refund", "review", "engineer"],
    "train": [
        {"id": "m1", "state": {"charge": {"amount": 900}, "message": "all"}, "label": "review"}
    ],
    "validation": [
        {"id": "m2", "state": {"charge": {"amount": 30}, "message": "dup"}, "label": "auto_refund"},
        {"id": "m3", "state": {"charge": {"amount": 700}, "message": "big"}, "label": "review"},
    ],
    "test": [
        {"id": "m4", "state": {"charge": {"amount": 12}, "message": "seat"}, "label": "review"}
    ],
}


def routed_to_billing(provider):
    provider.answers = {
        "team": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.9,
            "probabilities": {"billing": 0.9, "technical": 0.1},
        },
        "genuine": {"type": "noul", "noul": 0.1},
    }


def search_with_map_task(tmp_path, provider, monkeypatch, map_task: dict):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    (tmp_path / "billing.rules").write_text(BILLING)
    map_file = write(tmp_path / "support.json", router("billing.rules"))
    task = write(tmp_path / "billing-task.json", BILLING_TASK)
    whole = write(tmp_path / "support-task.json", map_task)
    provider.completion = chat_reply(graph=FOUND)
    routed_to_billing(provider)
    output = tmp_path / "run"
    result = invoke(
        "optimize",
        task,
        map_file,
        "--unit",
        "billing",
        "--map-task",
        whole,
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
    return result, output


def test_the_whole_map_is_scored_before_and_after_the_unit_changes(tmp_path, provider, monkeypatch):
    result, output = search_with_map_task(tmp_path, provider, monkeypatch, MAP_TASK)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "report.json").read_text())
    # The old unit reviewed the small refund the map wanted granted; the new one grants it.
    assert report["unit"]["map_validation"] == {"before": 0.5, "after": 1.0}


def test_a_unit_that_is_better_alone_but_worse_for_the_map_is_refused(
    tmp_path, provider, monkeypatch
):
    # The map's owners want every ticket reviewed, whatever the billing unit's own task says.
    map_task = json.loads(json.dumps(MAP_TASK))
    map_task["validation"][0]["label"] = "review"
    result, output = search_with_map_task(tmp_path, provider, monkeypatch, map_task)
    assert result.returncode != 0 and "map" in result.stderr and "0.5" in result.stderr
    assert not (output / "plan.json").exists()
