"""Values the rules compute from readings: a margin, a total, a scaled difference."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rulesmith.graph import Plan
from rulesmith.rules import RuleError, compile_rules, render_rules
from rulesmith.runtime import DecisionProgram

MARGIN = (
    "read gain = capture.best_gain in [-9, 9]\n"
    "read worst = hanging.worst in [0, 9]\n"
    "let margin = 2 * gain - worst + 1\n"
    "when margin >= 3: CAPTURE\n"
    "always: SAVE"
)


def label(plan: Plan, state: dict) -> str:
    return DecisionProgram(plan, None, "none")(state).label


@given(st.integers(-9, 9), st.integers(0, 9))
def test_a_computed_value_is_the_arithmetic_it_names(gain, worst):
    state = {"capture": {"best_gain": gain}, "hanging": {"worst": worst}}
    run = DecisionProgram(compile_rules(MARGIN), None, "none")(state)
    assert run.node_values["margin"] == 2 * gain - worst + 1
    assert run.label == ("CAPTURE" if 2 * gain - worst + 1 >= 3 else "SAVE")


def test_a_computed_value_round_trips():
    once = render_rules(compile_rules(MARGIN))
    assert "let margin = 2 * gain - worst + 1" in once
    assert render_rules(compile_rules(once)) == once
    negated = render_rules(compile_rules(MARGIN.replace("2 * gain - worst + 1", "-worst - 2")))
    assert render_rules(compile_rules(negated)) == negated


def test_a_value_computed_from_a_missing_reading_is_missing():
    # Treating the absent term as zero would invent a margin the state never showed.
    assert label(compile_rules(MARGIN), {"capture": {"best_gain": 9}}) == "SAVE"


def test_only_numbers_can_be_added():
    with pytest.raises(RuleError, match="numeric"):
        compile_rules(
            "read x one of [a, b]\nread n in [0, 1]\nlet t = x + n\nwhen t >= 1: A\nalways: B"
        )


def test_the_range_of_a_computed_value_follows_from_its_terms():
    # The margin runs from 2 * -9 - 9 + 1 = -26 to 2 * 9 - 0 + 1 = 19, so a threshold at 20 can
    # never trip, and validation says so rather than letting search keep a dead rule.
    with pytest.raises(RuleError, match="bounds"):
        compile_rules(
            MARGIN.replace("when margin >= 3: CAPTURE\n", "always: margin >= 20 ? A : B\n")
        )
    compile_rules(MARGIN.replace("when margin >= 3: CAPTURE\n", "always: margin >= 19 ? A : B\n"))
