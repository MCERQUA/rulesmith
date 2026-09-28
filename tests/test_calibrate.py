"""Calibrating a judge's probabilities, and passing an example to a person."""

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from test_cli import invoke, write
from test_graph import task_data

from rulesmith.calibrate import fit_temperature
from rulesmith.graph import Plan, Task
from rulesmith.optimize import Labels
from rulesmith.rules import compile_rules, render_rules
from rulesmith.runtime import DecisionProgram, calibrated

CHOICE = {
    "type": "choice",
    "value": "a",
    "probabilities": {"a": 0.8, "b": 0.15, "c": 0.05},
    "confidence": 0.6,
}


@given(st.floats(0.2, 5))
def test_a_temperature_reshapes_probabilities_without_changing_the_answer(temperature):
    answer = calibrated(CHOICE, temperature)
    assert answer["value"] == "a"
    assert sum(answer["probabilities"].values()) == pytest.approx(1)
    top = answer["probabilities"]["a"]
    # Above one flattens an overconfident judge; below one sharpens a timid one.
    assert (top < 0.8) if temperature > 1.001 else (top > 0.8) if temperature < 0.999 else True
    noul = calibrated({"type": "noul", "value": 0.9}, temperature)
    assert (noul["value"] < 0.9) == (temperature > 1.001) or abs(temperature - 1) < 0.001


def test_a_score_moves_with_its_calibrated_levels():
    answer = calibrated(
        {
            "type": "score",
            "value": 1.737,
            "probabilities": {"0": 0.066, "1": 0.131, "2": 0.803},
            "confidence": 0.6,
        },
        3.0,
    )
    levels = answer["probabilities"]
    assert answer["value"] == pytest.approx(sum(int(k) * p for k, p in levels.items()))
    assert answer["value"] < 1.737


def test_an_overconfident_judge_is_fitted_a_temperature_above_one():
    # Sure of every answer, right on only three in five: its certainty needs flattening.
    sure = {"a": 0.97, "b": 0.03}
    pairs = [(sure, "a")] * 3 + [(sure, "b")] * 2
    assert fit_temperature(pairs) > 1
    timid = {"a": 0.6, "b": 0.4}
    assert fit_temperature([(timid, "a")] * 20) < 1


def test_a_calibrated_question_reads_back_as_written():
    text = 'ask q = "Which?" [a: "A"] [b: "B"] calibrated 1.7\nwhen q == "a": A\nalways: B'
    plan = compile_rules(text)
    assert plan.nodes["q"].temperature == 1.7
    once = render_rules(plan)
    assert "calibrated 1.7" in once
    assert render_rules(compile_rules(once)) == once


def test_a_temperature_must_be_positive():
    with pytest.raises(ValueError, match="temperature"):
        compile_rules(
            'ask q = "Which?" [a: "A"] [b: "B"] calibrated 0\nwhen q == "a": A\nalways: B'
        )


def test_the_cli_fits_a_temperature_on_training_examples(tmp_path, provider):
    data = task_data()
    data["train"] = [
        {"id": str(i), "state": f"request {i}", "label": "accept" if i % 5 < 3 else "review"}
        for i in range(10)
    ]
    task = write(tmp_path / "task.json", data)
    plan = tmp_path / "plan.rules"
    plan.write_text(
        'ask route = "Accept or review?" [accept: "Fine."] [review: "Check it."]\n'
        'when route == "accept": accept\nalways: review'
    )
    provider.answers = {
        "route": {
            "type": "choice",
            "choice": "accept",
            "confidence": 0.95,
            "probabilities": {"accept": 0.97, "review": 0.03},
        }
    }
    output = tmp_path / "calibrated.rules"
    result = invoke(
        "calibrate",
        task,
        plan,
        "--backend",
        "ruling",
        "--base-url",
        provider.url,
        "--output",
        output,
    )
    assert result.returncode == 0, result.stderr
    fitted = compile_rules(output.read_text()).nodes["route"].temperature
    assert fitted > 1
    assert json.loads(result.stdout)["temperatures"]["route"] == fitted


def test_passing_to_a_person_costs_what_the_task_says():
    data = task_data() | {"labels": ["accept", "review", "human"]}
    data["abstain"] = {"label": "human", "cost": 0.3}
    task = Task.model_validate(data)
    program = DecisionProgram(Plan.constant("human"), None, "none")
    outcome = Labels(task)(program, task.train[0])
    assert outcome.score == pytest.approx(0.7)
    assert "person" in outcome.trace["Feedback"]


@pytest.mark.parametrize(
    "abstain,labels",
    [
        ({"label": "human", "cost": 0.3}, ["accept", "review"]),
        ({"label": "human", "cost": 1.5}, ["accept", "review", "human"]),
    ],
)
def test_an_abstain_label_must_be_a_task_label_that_costs_less_than_a_mistake(abstain, labels):
    with pytest.raises(ValidationError):
        Task.model_validate(task_data() | {"labels": labels, "abstain": abstain})


def test_no_example_may_expect_a_person():
    data = task_data() | {"labels": ["accept", "review", "human"]}
    data["abstain"] = {"label": "human", "cost": 0.3}
    data["train"][0]["label"] = "human"
    with pytest.raises(ValidationError, match="human"):
        Task.model_validate(data)


def test_the_proposer_is_told_how_to_pass_an_example_on():
    from rulesmith.optimize import SearchConfig, proposer_prompt

    data = task_data() | {"labels": ["accept", "review", "human"]}
    data["abstain"] = {"label": "human", "cost": 0.3}
    prompt = proposer_prompt(Task.model_validate(data), SearchConfig())
    assert "human" in prompt and "0.3" in prompt
