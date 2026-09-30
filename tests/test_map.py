"""A map: a graph whose units live in files of their own, loaded by path."""

import json
from pathlib import Path

import pytest
from test_cli import invoke, write
from test_graph import task_data

from rulesmith.graph import Plan
from rulesmith.maps import MapError, load
from rulesmith.rules import compile_rules

BILLING = (
    "read charge.amount in [0, 100000]\n"
    'ask genuine = "The message describes a mistake the customer did not cause."\n'
    "when charge.amount < 50 and genuine >= 0.8: auto_refund\n"
    "always: review\n"
)


def router(unit: str) -> dict:
    return {
        "nodes": {
            "team": {
                "type": "call",
                "question": {
                    "type": "choice",
                    "instructions": "Which team should answer this ticket?",
                    "criteria": {"billing": "Charges and refunds.", "technical": "Bugs."},
                },
                "inputs": {"observation": {"type": "state"}},
            },
            "billing": {
                "type": "graph",
                "path": unit,
                "when": {"type": "compare", "node": "team", "op": "eq", "value": "billing"},
            },
            "escalate": {"type": "value", "value": "engineer"},
            "decision": {"type": "coalesce", "sources": ["billing", "escalate"]},
        },
        "outputs": {"label": {"type": "node", "node": "decision"}},
    }


def test_a_unit_is_loaded_from_its_own_file_relative_to_the_map(tmp_path):
    (tmp_path / "units").mkdir()
    (tmp_path / "units" / "billing.rules").write_text(BILLING)
    (tmp_path / "maps").mkdir()
    map_file = write(tmp_path / "maps" / "support.json", router("../units/billing.rules"))
    plan = load(map_file)
    # The loaded map is an ordinary graph with the unit embedded, so everything else is unchanged.
    embedded = router("")
    embedded["nodes"]["billing"] = {
        "type": "graph",
        "plan": compile_rules(BILLING).model_dump(mode="json", exclude_none=True),
        "when": embedded["nodes"]["billing"]["when"],
    }
    assert plan == Plan.model_validate(embedded)


def test_a_unit_file_may_be_json_and_hold_units_of_its_own(tmp_path):
    inner = compile_rules("read x in [0, 9]\nwhen x >= 5: high\nalways: low")
    write(tmp_path / "inner.json", inner.model_dump(mode="json", exclude_none=True))
    write(
        tmp_path / "middle.json",
        {
            "nodes": {"depth": {"type": "graph", "path": "inner.json"}},
            "outputs": {"label": {"type": "node", "node": "depth"}},
        },
    )
    map_file = write(tmp_path / "outer.json", router("middle.json"))
    plan = load(map_file)
    assert plan.nodes["billing"].plan.nodes["depth"].plan == inner


def test_a_map_that_contains_itself_is_refused(tmp_path):
    write(tmp_path / "loop.json", router("loop.json"))
    with pytest.raises(MapError, match="loop.json"):
        load(tmp_path / "loop.json")


def test_a_unit_path_that_reaches_nothing_says_which_map_named_it(tmp_path):
    map_file = write(tmp_path / "support.json", router("missing.rules"))
    with pytest.raises(MapError, match="missing.rules") as failure:
        load(map_file)
    assert "support.json" in str(failure.value)


def test_a_graph_read_without_loading_its_units_is_refused():
    # Validation alone cannot see into a file, so it says what to do instead of guessing.
    with pytest.raises(ValueError, match="load"):
        Plan.model_validate(router("billing.rules"))


def test_the_cli_takes_a_map_wherever_it_takes_a_graph(tmp_path):
    (tmp_path / "billing.rules").write_text(BILLING)
    map_file = write(tmp_path / "support.json", router("billing.rules"))
    data = task_data() | {"labels": ["auto_refund", "review", "engineer"]}
    for split in ("train", "validation", "test"):
        for example in data[split]:
            example["label"] = "review"
    task = write(tmp_path / "task.json", data)
    result = invoke("validate", task, map_file)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["nodes"] == 4
    drawn = invoke("draw", map_file)
    assert drawn.returncode == 0 and "subgraph" in drawn.stdout


def test_the_example_map_loads_from_its_files():
    root = Path(__file__).resolve().parents[1]
    plan = load(root / "examples" / "support-map.json")
    assert plan.nodes["billing"].plan.nodes


def test_validate_reports_the_units_and_refuses_a_hand_off_the_unit_cannot_meet(tmp_path):
    (tmp_path / "billing.rules").write_text(BILLING)
    graph = router("billing.rules")
    # A rule after the unit tests its answer against a label the unit never gives.
    graph["nodes"]["refunded"] = {
        "type": "value",
        "value": "engineer",
        "when": {"type": "compare", "node": "billing", "op": "eq", "value": "approved"},
    }
    graph["nodes"]["decision"]["sources"] = ["refunded", "billing", "escalate"]
    map_file = write(tmp_path / "support.json", graph)
    data = task_data() | {"labels": ["auto_refund", "review", "engineer"]}
    for split in ("train", "validation", "test"):
        for example in data[split]:
            example["label"] = "review"
    task = write(tmp_path / "task.json", data)
    result = invoke("validate", task, map_file)
    assert result.returncode != 0 and "approved" in result.stderr
    graph["nodes"]["refunded"]["when"]["value"] = "auto_refund"
    write(map_file, graph)
    result = invoke("validate", task, map_file)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["units"] == {"billing": "billing.rules"}


def test_a_map_can_be_drawn_with_its_units_closed(tmp_path):
    (tmp_path / "billing.rules").write_text(BILLING)
    map_file = write(tmp_path / "support.json", router("billing.rules"))
    opened = invoke("draw", map_file)
    closed = invoke("draw", map_file, "--closed")
    assert opened.returncode == 0 and closed.returncode == 0, closed.stderr
    assert "subgraph" in opened.stdout and "genuine" in opened.stdout
    # Closed, a unit is one double-edged box, the sign for a program of its own, and its
    # insides are not drawn.
    assert "subgraph" not in closed.stdout and "genuine" not in closed.stdout
    assert '[["billing<br/>a graph of its own"]]' in closed.stdout


def test_a_map_whose_only_judge_calls_are_inside_a_unit_still_gets_a_judge(tmp_path, provider):
    # The map's top level asks nothing; the unit does. Evaluate and run must still bring a judge.
    (tmp_path / "billing.rules").write_text(BILLING)
    graph = {
        "nodes": {
            "billing": {"type": "graph", "path": "billing.rules"},
        },
        "outputs": {"label": {"type": "node", "node": "billing"}},
    }
    map_file = write(tmp_path / "support.json", graph)
    ticket = write(tmp_path / "ticket.json", {"charge": {"amount": 40}, "message": "twice"})
    provider.answers = {"genuine": {"type": "noul", "noul": 0.9}}
    result = invoke(
        "run",
        map_file,
        ticket,
        "--backend",
        "ruling",
        "--base-url",
        provider.url,
        "--max-workers",
        1,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["label"] == "auto_refund"
