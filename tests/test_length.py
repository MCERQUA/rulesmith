"""Reading how long a text or a list is, as a number rules can test."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rulesmith.graph import Plan
from rulesmith.rules import RuleError, compile_rules, render_rules
from rulesmith.runtime import DecisionProgram

RULES = (
    "read length of company_profile in [0, 10000]\n"
    "when length of company_profile < 20: fraudulent\n"
    "always: genuine"
)


def label(plan: Plan, state) -> str:
    return DecisionProgram(plan, None, "none")(state).label


@given(st.text(max_size=60))
def test_a_length_is_the_number_of_characters(profile):
    expected = "fraudulent" if len(profile) < 20 else "genuine"
    assert label(compile_rules(RULES), {"company_profile": profile}) == expected


def test_a_list_has_a_length_and_a_number_does_not():
    plan = compile_rules("read length of tags in [0, 99]\nwhen length of tags >= 2: many\nalways: few")
    assert label(plan, {"tags": ["a", "b"]}) == "many"
    # A number has no length, so the reading is missing and the rule does not fire.
    assert label(plan, {"tags": 7}) == "few"


def test_a_length_reads_back_as_written_and_can_be_named():
    once = render_rules(compile_rules(RULES))
    assert "read length of company_profile in [0, 10000]" in once
    assert render_rules(compile_rules(once)) == once
    named = compile_rules(
        "read profile = length of company_profile in [0, 10000]\nwhen profile < 20: a\nalways: b"
    )
    assert label(named, {"company_profile": ""}) == "a"
    assert render_rules(compile_rules(render_rules(named))) == render_rules(named)


def test_a_length_is_a_number_with_bounds():
    with pytest.raises(RuleError):
        compile_rules("read length of name one of [a, b]\nwhen length of name == \"a\": x\nalways: y")
