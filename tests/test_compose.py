"""Graphs made of graphs: a unit's decision handed to another unit, typed across the boundary."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rulesmith.diagram import mermaid
from rulesmith.graph import Plan
from rulesmith.rules import compile_rules
from rulesmith.runtime import DecisionProgram

BILLING = compile_rules(
    "read charge.amount in [0, 100000]\n"
    'ask genuine = "The message describes a mistake the customer did not cause."\n'
    "when charge.amount < 50 and genuine >= 0.8: auto_refund\n"
    "always: review"
).model_dump(mode="json", exclude_none=True)


def support_map() -> dict:
    """A router that asks which team a ticket is for, and hands billing tickets to the billing
    graph, which answers from the ticket's charge and its own judgment."""
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
            # Handed the whole ticket, so its reads are the ticket's own fields.
            "billing": {
                "type": "graph",
                "plan": BILLING,
                "input": {"type": "state"},
                "when": {"type": "compare", "node": "team", "op": "eq", "value": "billing"},
            },
            "escalate": {"type": "value", "value": "engineer"},
            "decision": {"type": "coalesce", "sources": ["billing", "escalate"]},
        },
        "outputs": {"label": {"type": "node", "node": "decision"}},
    }


def test_a_unit_answers_with_what_its_graph_decides_and_its_judgments_roll_up(provider, client):
    provider.answers = {
        "team": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.9,
            "probabilities": {"billing": 0.9, "technical": 0.1},
        },
        "genuine": {"type": "noul", "noul": 0.95},
    }
    program = DecisionProgram(Plan.model_validate(support_map()), client, "fixture")
    ticket = {"charge": {"amount": 40}, "message": "Charged twice for one seat."}
    result = program(state=ticket)
    assert result.label == "auto_refund"
    # The billing graph's judgment is reported under the unit it belongs to, and paid for.
    assert set(result.answers) == {"team", "billing.genuine"}
    assert len(result.calls) == 2 and result.input_tokens > 0
    assert result.decided_by == "billing.rule_1"
    # No surer than the least sure judgment on the way: 0.9 for the team, 0.95 for genuine.
    assert result.confidence == pytest.approx(0.9)


def test_a_unit_not_reached_is_not_run(provider, client):
    provider.answers = {
        "team": {
            "type": "choice",
            "choice": "technical",
            "confidence": 0.9,
            "probabilities": {"billing": 0.1, "technical": 0.9},
        }
    }
    program = DecisionProgram(Plan.model_validate(support_map()), client, "fixture")
    result = program(state={"charge": {"amount": 40}, "message": "The export crashes."})
    assert result.label == "engineer" and "billing" in result.skipped
    # The billing graph asked nothing: only the router's question went to the judge.
    assert [key for key, _ in provider.requests] == ["/v1/systemone"]
    assert list(result.calls) == ["team"]


def test_the_boundary_is_typed():
    # A rule may test the unit's answer only against labels its graph can give.
    graph = support_map()
    graph["nodes"]["refunded"] = {
        "type": "value",
        "value": "yes",
        "when": {"type": "compare", "node": "billing", "op": "eq", "value": "auto_refund"},
    }
    graph["outputs"]["refunded"] = {"type": "node", "node": "refunded"}
    Plan.model_validate(graph)
    graph["nodes"]["refunded"]["when"]["value"] = "approved"
    # The refusal names the hand-off and what the unit can give, since a map has many.
    with pytest.raises(ValidationError, match="billing == 'approved'.*auto_refund.*review"):
        Plan.model_validate(graph)
    # A unit must answer with an output its graph has.
    graph = support_map()
    graph["nodes"]["billing"]["output"] = "priority"
    with pytest.raises(ValidationError, match="priority"):
        Plan.model_validate(graph)


def test_a_map_survives_json_and_counts_its_units_toward_the_task_labels():
    plan = Plan.model_validate(support_map())
    assert Plan.model_validate_json(plan.model_dump_json()) == plan
    plan.check(["auto_refund", "review", "engineer"], 16, 8)
    with pytest.raises(ValueError, match="label"):
        plan.check(["auto_refund", "review"], 16, 8)


def test_a_unit_is_drawn_as_a_box_you_can_see_into():
    drawn = mermaid(Plan.model_validate(support_map()))
    assert "subgraph" in drawn and drawn.count("judge:") == 2
    # The unit's own judgment sits inside its box; the router's outside.
    inside = drawn[drawn.index("subgraph") : drawn.index("\n    end")]
    assert "genuine" in inside and "Which team" not in inside


def test_the_example_map_loads_with_its_unit_in_a_file_of_its_own():
    from rulesmith.maps import load

    path = Path(__file__).resolve().parents[1] / "examples" / "support-map.json"
    plan = load(path)
    assert any(node.type == "graph" for node in plan.nodes.values())
    assert json.loads(path.read_text())["nodes"]["billing"]["path"] == "units/billing.rules"


def test_a_gate_on_a_unit_points_at_the_whole_unit_and_its_answer_leaves_once():
    drawn = mermaid(Plan.model_validate(support_map()))
    # The condition gates the unit as a whole, so its dashed arrow lands on the unit's box,
    # not on a node inside it.
    gate = next(line for line in drawn.splitlines() if line.strip().startswith("c0 -.->"))
    assert gate.strip().endswith("_unit")
    # The unit's chosen output is its exit node; drawing it as an output circle as well said the
    # same thing twice, so the map's own output is the only circle left.
    assert drawn.count('(("label"))') == 1
    assert "==> n1" in drawn


def test_a_presence_test_says_so():
    plan = Plan.model_validate(
        {
            "nodes": {
                "rating": {"type": "read", "path": ["rating"], "bounds": [1, 5]},
                "reviewed": {
                    "type": "value",
                    "value": "reviewed",
                    "when": {"type": "present", "node": "rating"},
                },
            },
            "outputs": {"label": {"type": "node", "node": "reviewed"}},
        }
    )
    assert '["rating is present"]' in mermaid(plan)
