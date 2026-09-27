"""Rules that decide more than one thing at once."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from test_graph import task_data

from rulesmith.graph import Task
from rulesmith.rules import RuleError, compile_rules, parts, render_rules, whole
from rulesmith.runtime import DecisionProgram

TWO = (
    "read severity in [0, 3]\n"
    "read vip one of [yes, no]\n"
    "when severity >= 2: review\n"
    "always: accept\n"
    "output priority:\n"
    'when severity >= 3 or vip == "yes": high\n'
    "always: low"
)


@given(st.integers(0, 3), st.sampled_from(["yes", "no"]))
def test_each_output_is_decided_by_its_own_rules(severity, vip):
    run = DecisionProgram(compile_rules(TWO), None, "none")({"severity": severity, "vip": vip})
    assert run.outputs["label"] == ("review" if severity >= 2 else "accept")
    assert run.outputs["priority"] == ("high" if severity >= 3 or vip == "yes" else "low")


def test_several_outputs_round_trip_and_split_for_search():
    plan = compile_rules(TWO)
    once = render_rules(plan)
    assert "output priority:" in once
    assert render_rules(compile_rules(once)) == once
    assert whole(parts(plan)) == plan


def test_an_output_may_be_a_constant():
    plan = compile_rules("always: accept\noutput team:\nalways: billing")
    assert DecisionProgram(plan, None, "none")("anything").outputs == {
        "label": "accept",
        "team": "billing",
    }
    assert render_rules(compile_rules(render_rules(plan))) == render_rules(plan)


def test_an_output_is_declared_once():
    with pytest.raises(RuleError, match="priority"):
        compile_rules("always: a\noutput priority:\nalways: b\noutput priority:\nalways: c")


def test_an_output_without_rules_is_refused():
    with pytest.raises(RuleError, match="priority"):
        compile_rules("always: a\noutput priority:")


def test_the_label_is_still_what_a_task_checks():
    # Search scores the label; an extra output rides along but must not stand in for it.
    task = Task.model_validate(task_data())
    compile_rules(TWO).check(task.labels, 16, 8)
    with pytest.raises(ValueError, match="label"):
        compile_rules("output priority:\nalways: high").check(task.labels, 16, 8)
