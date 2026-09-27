"""A plan drawn as a Mermaid flowchart."""

from pathlib import Path

import pytest
from test_graph import binary_plan

from rulesmith.diagram import mermaid
from rulesmith.graph import Plan
from rulesmith.rules import compile_rules

GRAPHS = sorted((Path(__file__).resolve().parents[1] / "examples/doom/graphs").glob("*.json"))


def test_every_node_and_edge_appears():
    plan = compile_rules(
        "read variables.HEALTH in [0, 200]\n"
        "when variables.HEALTH < 20: MOVE_BACKWARD\n"
        "always: ATTACK"
    )
    drawn = mermaid(plan)
    assert drawn.startswith("flowchart TD")
    for name in plan.nodes:
        assert name in drawn
    # The guard on a rule is written along the arrow that reaches it.
    assert "variables.HEALTH < 20" in drawn
    assert "label" in drawn


@pytest.mark.parametrize("path", GRAPHS, ids=lambda path: path.stem)
def test_every_repository_graph_draws(path):
    plan = Plan.model_validate_json(path.read_text())
    drawn = mermaid(plan)
    # Mermaid takes a label between shape markers, so a bracket inside one breaks the diagram.
    for line in drawn.splitlines()[1:]:
        if '"' in line:
            assert line.count('"') == 2, line


def test_a_graph_rules_cannot_express_still_draws():
    # The case the diagram is for: merges and several outputs have no rule form.
    plan = Plan.model_validate(binary_plan())
    drawn = mermaid(plan)
    assert drawn.count("-->") >= 1


def test_a_guard_keeps_its_grouping():
    # Dropping the parentheses turns 'not (a or b)' into '(not a) or b', a different rule.
    plan = compile_rules(
        "read a in [0, 9]\nread b in [0, 9]\nwhen not (a >= 1 or b >= 1): QUIET\nalways: LOUD"
    )
    assert "not (a >= 1 or b >= 1)" in mermaid(plan)
