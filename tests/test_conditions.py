"""Named conditions, and conditions that compare one reading with another."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from test_graph import call, ref

from rulesmith.diagram import mermaid
from rulesmith.graph import Plan
from rulesmith.rules import RuleError, compile_rules, parts, render_rules, whole
from rulesmith.runtime import DecisionProgram

READINGS = "read ammo in [0, 50]\nread health in [0, 200]\n"
NAMED = (
    READINGS + "let armed = ammo >= 1 and health >= 25\n"
    "let desperate = not armed and health < 10\n"
    "when armed: ATTACK\n"
    "when desperate: MOVE_BACKWARD\n"
    "always: MOVE_FORWARD"
)
INLINE = (
    READINGS + "when ammo >= 1 and health >= 25: ATTACK\n"
    "when not (ammo >= 1 and health >= 25) and health < 10: MOVE_BACKWARD\n"
    "always: MOVE_FORWARD"
)


def label(plan: Plan, state: dict) -> str:
    return DecisionProgram(plan, None, "none")(state).label


@given(st.integers(0, 50), st.integers(0, 200))
def test_a_named_condition_decides_exactly_what_it_stands_for(ammo, health):
    state = {"ammo": ammo, "health": health}
    assert label(compile_rules(NAMED), state) == label(compile_rules(INLINE), state)


def test_a_name_survives_the_round_trip():
    # Expanding the name inline would hand search back the long form it was written to avoid.
    once = render_rules(compile_rules(NAMED))
    assert "let armed = " in once
    assert "when armed: ATTACK" in once
    assert render_rules(compile_rules(once)) == once


def test_a_judgment_may_be_asked_only_under_a_named_condition():
    text = (
        READINGS + "let armed = ammo >= 1 and health >= 25\n"
        'ask threat = "Fight?" [fight: "Lined up."] [flee: "Outgunned."] when armed\n'
        'when threat == "fight": ATTACK\n'
        "always: MOVE_FORWARD"
    )
    plan = compile_rules(text)
    once = render_rules(plan)
    assert render_rules(compile_rules(once)) == once
    # Search revises readings and rules apart, so splitting and rejoining must not reorder a
    # declaration past the one it depends on.
    assert whole(parts(plan)) == plan


def test_a_name_that_is_already_taken_is_refused():
    with pytest.raises(RuleError, match="health"):
        compile_rules(READINGS + "let health = ammo >= 1\nwhen health: ATTACK\nalways: STOP")


def test_a_rules_part_may_restate_a_judgment_and_the_later_one_stands():
    # Search joins the readings part to a rules part that may declare what it needs again.
    asked = 'ask threat = "{}" [fight: "Lined up."] [flee: "Outgunned."]'
    plan = compile_rules(
        f"{asked.format('Fight?')}\n{asked.format('Fight now?')}\n"
        'when threat == "fight": ATTACK\nalways: STOP'
    )
    assert plan.nodes["threat"].question.instructions == "Fight now?"


def test_a_rule_cannot_name_a_condition_before_declaring_it():
    with pytest.raises(RuleError, match="armed"):
        compile_rules(READINGS + "when armed: ATTACK\nlet armed = ammo >= 1\nalways: STOP")


@given(st.integers(-9, 9), st.integers(0, 9))
def test_one_reading_is_compared_with_another(gain, worst):
    plan = compile_rules(
        "read capture.best_gain in [-9, 9]\nread hanging.worst in [0, 9]\n"
        "when capture.best_gain > hanging.worst: CAPTURE\n"
        "when capture.best_gain == hanging.worst: CHECK\n"
        "always: SAVE"
    )
    state = {"capture": {"best_gain": gain}, "hanging": {"worst": worst}}
    expected = "CAPTURE" if gain > worst else "CHECK" if gain == worst else "SAVE"
    assert label(plan, state) == expected


def test_a_comparison_fails_when_either_side_is_missing():
    plan = compile_rules(
        "read a in [0, 9]\nread b in [0, 9]\nwhen a >= b: HIGH\nwhen not a >= b: LOW\nalways: NONE"
    )
    assert label(plan, {"a": 1}) == "LOW"
    assert label(plan, {"b": 1}) == "LOW"


def test_a_comparison_between_readings_round_trips():
    text = "read a in [0, 9]\nread b in [0, 9]\nwhen a < b: LOW\nalways: HIGH"
    once = render_rules(compile_rules(text))
    assert "when a < b: LOW" in once
    assert render_rules(compile_rules(once)) == once


def test_labels_compare_with_labels_and_numbers_with_numbers():
    labels = "read x one of [a, b]\nread y one of [b, c]\n"
    assert label(compile_rules(labels + "when x == y: SAME\nalways: DIFF"), {"x": "b", "y": "b"})
    with pytest.raises(RuleError, match="numeric"):
        compile_rules(labels + "when x >= y: SAME\nalways: DIFF")
    with pytest.raises(RuleError, match="same kind"):
        compile_rules("read x one of [a, b]\nread n in [0, 1]\nwhen x == n: SAME\nalways: DIFF")


def test_a_comparison_draws_a_gate_from_both_sides():
    plan = compile_rules("read a in [0, 9]\nread b in [0, 9]\nwhen a < b: LOW\nalways: HIGH")
    drawn = mermaid(plan)
    assert "read_a -.->" in drawn and "read_b -.->" in drawn


def test_a_graph_node_rules_cannot_write_is_refused_rather_than_dropped():
    # A threshold nothing acts on, only tested by a condition, has no line of its own. Leaving
    # it out would render rules that no longer compile to the graph they came from.
    plan = Plan.model_validate(
        {
            "nodes": {
                "odds": call("Is it fraud?"),
                "gate": {
                    "type": "threshold",
                    "source": "odds",
                    "cutoff": 0.5,
                    "below": "no",
                    "above": "yes",
                },
                "flag": {
                    "type": "value",
                    "value": "review",
                    "when": {"type": "compare", "node": "gate", "op": "eq", "value": "yes"},
                },
            },
            "outputs": {"label": ref("flag")},
        }
    )
    with pytest.raises(RuleError, match="gate"):
        render_rules(plan)


@given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False))
def test_every_number_a_condition_compares_with_comes_back_exactly(value):
    text = f"read x in [-1000000, 1000000]\nwhen x >= {value!r}: HIGH\nalways: LOW"
    once = render_rules(compile_rules(text))
    assert compile_rules(once).nodes["rule_1"].when.value == value
    # Whole numbers read as whole numbers: a proposer copies '>= 1.0' into every rule it writes.
    if value.is_integer():
        assert f">= {int(value)}:" in once


@given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False))
def test_every_cutoff_and_bound_comes_back_exactly(value):
    text = f"read x in [{value!r}, 1000001]\nalways: x >= {value!r} ? HIGH : LOW"
    plan = compile_rules(render_rules(compile_rules(text)))
    assert plan.nodes["read_x"].bounds[0] == value
    assert plan.nodes["rule_1"].cutoff == value
