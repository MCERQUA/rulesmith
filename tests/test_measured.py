"""Rules measured on labeled examples: how sure a decision they make is, whether new inputs still
look like the ones they were measured on, and whether search may let them answer alone."""

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st
from test_cli import invoke, write
from test_mine import reviews, task

from rulesmith.calibrate import binomial_p, drift, measure
from rulesmith.graph import Measured, Plan
from rulesmith.rules import RuleError, compile_rules, render_rules
from rulesmith.runtime import DecisionProgram

MEASURED = "read rating in [1, 5]\nwhen rating >= 4: yes calibrated 37/40 of 100\n"


def test_a_measured_rule_reads_back_as_written():
    text = MEASURED + "always: rating >= 2 ? maybe : no calibrated 3/5 of 100"
    plan = compile_rules(text)
    assert plan.nodes["rule_1"].measured == Measured(right=37, decided=40, examples=100)
    assert plan.nodes["rule_2"].measured == Measured(right=3, decided=5, examples=100)
    once = render_rules(plan)
    assert compile_rules(once) == plan and render_rules(compile_rules(once)) == once


@pytest.mark.parametrize("counts", ["41/40 of 100", "37/140 of 100"])
def test_a_rule_is_never_right_more_often_than_it_decided(counts):
    # A confidence above one, or a share above all, would rank every other answer below it.
    with pytest.raises(ValueError):
        compile_rules(
            f"read rating in [1, 5]\nwhen rating >= 4: yes calibrated {counts}\nalways: no"
        )


def test_a_label_that_would_read_back_as_a_measurement_is_refused():
    plan = Plan.constant("yes calibrated 1/2 of 3")
    with pytest.raises(RuleError):
        render_rules(plan)


def test_a_decision_is_as_sure_as_the_rule_or_judgment_that_made_it(provider, client):
    provider.answers = {
        "rule_3": {
            "type": "choice",
            "choice": "no",
            "confidence": 0.7,
            "probabilities": {"yes": 0.3, "no": 0.7},
        }
    }
    plan = compile_rules(
        MEASURED.replace("37/40 of 100", "9/10 of 20")
        + 'when rating < 2: no\nalways: ask "Recommended?" [yes: "Yes."] [no: "No."]'
    )
    program = DecisionProgram(plan, client, "fixture")
    settled = program(state={"rating": 5})
    # Right nine times in ten, smoothed toward even odds: a small count claims less than it saw.
    assert (settled.decided_by, settled.confidence) == ("rule_1", pytest.approx(10 / 12))
    # A rule nobody measured claims nothing beyond the judgments it rests on, here none.
    unmeasured = program(state={"rating": 1})
    assert (unmeasured.decided_by, unmeasured.confidence) == ("rule_2", 1.0)
    asked = program(state={"rating": 3})
    assert (asked.decided_by, asked.confidence) == ("rule_3", pytest.approx(0.7))


def test_a_decision_is_no_surer_than_the_least_sure_judgment_it_rests_on(provider, client):
    provider.answers = {
        "urgent": {"type": "noul", "noul": 0.2},
        "team": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.9,
            "probabilities": {"billing": 0.9, "sales": 0.1},
        },
    }
    plan = compile_rules(
        'ask urgent = "It is urgent."\nask team = "Which?" [billing: "B"] [sales: "S"]\n'
        'when urgent < 0.5 and team == "billing": billing\nalways: sales'
    )
    # 0.2 that it is urgent is 0.8 that it is not, which is surer than nothing but not than 0.9.
    assert DecisionProgram(plan, client, "fixture")(state="x").confidence == pytest.approx(0.8)


def test_calibrating_counts_each_rule_on_every_labeled_example():
    plan = compile_rules(
        "read rating in [1, 5]\nwhen rating >= 4: yes\nwhen rating >= 3: yes\nalways: no"
    )
    measured = measure(plan, task(), None, "none")
    counts = {name: node.measured for name, node in measured.nodes.items() if node.type == "value"}
    # Fifty training and validation reviews: twenty of four stars or more, ten of three, of
    # which four were recommended, and twenty of one or two, none recommended.
    assert counts == {
        "rule_1": Measured(right=20, decided=20, examples=50),
        "rule_2": Measured(right=4, decided=10, examples=50),
        "rule_3": Measured(right=20, decided=20, examples=50),
    }


def test_a_rule_deciding_at_another_rate_than_it_was_measured_on_is_flagged():
    plan = compile_rules(MEASURED.replace("37/40", "40/40") + "always: no calibrated 55/60 of 100")
    same = [{"decided_by": "rule_1"}] * 40 + [{"decided_by": "rule_2"}] * 60
    assert drift(plan, same, 0.01) == []
    # The jobs benchmark's logo rule: forty in a hundred on balanced samples, fourteen at the
    # real rate of scams.
    shifted = [{"decided_by": "rule_1"}] * 14 + [{"decided_by": "rule_2"}] * 86
    flagged = drift(plan, shifted, 0.01)
    assert {found["rule"] for found in flagged} == {"rule_1", "rule_2"}
    assert all(found["p"] < 0.01 for found in flagged)


@given(st.integers(1, 3000), st.floats(0.001, 0.999))
def test_the_likeliest_count_is_never_surprising(examples, share):
    # Every count is at most as likely as the most likely one, so seeing it has p = 1.
    likeliest = min(int((examples + 1) * share), examples)
    assert binomial_p(likeliest, examples, share) == pytest.approx(1)
    assert 0 <= binomial_p(0, examples, share) <= 1


def test_calibrate_then_evaluate_reports_a_rule_whose_rate_moved(tmp_path, provider):
    rows = reviews()
    data = {
        "description": "Would the reviewer recommend it?",
        "labels": ["no", "yes"],
        "train": rows[:40],
        "validation": rows[40:50],
        # Every test review has five stars, where training had one in five.
        "test": [
            {"id": f"t{i}", "state": {"rating": 5, "review": f"great {i}"}, "label": "yes"}
            for i in range(30)
        ],
    }
    task_file = write(tmp_path / "task.json", data)
    plan = tmp_path / "plan.rules"
    plan.write_text("read rating in [1, 5]\nwhen rating >= 4: yes\nalways: no\n")
    calibrated = tmp_path / "calibrated.rules"
    backend = ("--backend", "ruling", "--base-url", provider.url)
    fitted = invoke("calibrate", task_file, plan, *backend, "--output", calibrated)
    assert fitted.returncode == 0, fitted.stderr
    assert compile_rules(calibrated.read_text()).nodes["rule_1"].measured == Measured(
        right=20, decided=20, examples=50
    )
    report = tmp_path / "test.json"
    scored = invoke("evaluate", task_file, calibrated, *backend, "--output", report)
    assert scored.returncode == 0, scored.stderr
    written = json.loads(report.read_text())
    assert {found["rule"] for found in written["drift"]} == {"rule_1", "rule_2"}
    assert written["examples"][0]["confidence"] == pytest.approx(21 / 22)


def test_search_lets_rules_answer_alone_only_as_surely_as_asked(provider, client):
    from rulesmith.optimize import SearchConfig, optimize_plan

    provider.answers = {
        "q": {
            "type": "choice",
            "choice": "no",
            "confidence": 0.6,
            "probabilities": {"no": 0.6, "yes": 0.4},
        }
    }
    seed = compile_rules(
        'ask q = "Recommended?" [no: "No."] [yes: "Yes."]\nwhen q == "yes": yes\nalways: no'
    )
    rules = "read rating in [1, 5]\nwhen rating >= 4: yes\nalways: no"

    def propose(prompt):
        # Rules that need no judge, which score better on validation than the seed does.
        return rules if "Return the part named rules" in prompt else seed_readings

    seed_readings = 'ask q = "Recommended?" [no: "No."] [yes: "Yes."]'
    config = {"max_metric_calls": 30, "reflection_batch": 3}
    # 'always: no' is right on 26 of the 30 reviews it settles alone: 27/32 once smoothed.
    strict = SearchConfig(**config, keep_judge=0.9)
    plan, report = optimize_plan(task(), seed, client, "fixture", propose, strict)
    assert plan == seed and report["keep_judge_passed_over"] >= 1
    loose = SearchConfig(**config, keep_judge=0.8)
    plan, report = optimize_plan(task(), seed, client, "fixture", propose, loose)
    assert plan == compile_rules(rules) and report["keep_judge_passed_over"] == 0
