import json

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from jev_compose.graph import Plan, Task


def binary_plan(cutoff=0.5):
    return {
        "questions": {"eligible": {"type": "noul", "instructions": "Is the request eligible?"}},
        "root": {
            "type": "threshold",
            "question": "eligible",
            "cutoff": cutoff,
            "below": {"type": "return", "label": "review"},
            "above": {"type": "return", "label": "accept"},
        },
    }


@given(
    st.floats(min_value=0, max_value=1, allow_nan=False),
    st.floats(min_value=0, max_value=1, allow_nan=False),
)
def test_threshold_partition_and_serialization(value, cutoff):
    plan = Plan.model_validate(binary_plan(cutoff))
    restored = Plan.model_validate_json(plan.model_dump_json())
    label, path = restored.decide({"eligible": value})
    assert label == ("accept" if value >= cutoff else "review")
    assert path == ["eligible"]


def test_nested_choice_and_score_select_the_actual_path():
    plan = Plan.model_validate(
        {
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "Choose a team.",
                    "criteria": {"billing": "Payments", "support": "Other requests"},
                },
                "severity": {
                    "type": "score",
                    "instructions": "Assess severity.",
                    "criteria": ["Routine", "Disrupted", "Blocked"],
                },
            },
            "root": {
                "type": "switch",
                "question": "team",
                "cases": {
                    "support": {"type": "return", "label": "review"},
                    "billing": {
                        "type": "threshold",
                        "question": "severity",
                        "cutoff": 1.5,
                        "below": {"type": "return", "label": "review"},
                        "above": {"type": "return", "label": "urgent"},
                    },
                },
            },
        }
    )
    assert plan.decide({"team": "support", "severity": 2}) == ("review", ["team"])
    assert plan.decide({"team": "billing", "severity": 2}) == ("urgent", ["team", "severity"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["root"].update(question="missing"),
        lambda p: p["root"].update(cutoff=1.1),
        lambda p: p["root"].update(cutoff=float("nan")),
        lambda p: p["root"].update(typo=True),
        lambda p: p["questions"]["eligible"].update(instructions="   "),
        lambda p: p["questions"].update(unused={"type": "noul", "instructions": "Unused?"}),
        lambda p: p["questions"]["eligible"].update(
            type="choice", criteria={"yes": "Yes", "no": "No"}
        ),
    ],
)
def test_invalid_graphs_are_rejected_before_execution(mutation):
    data = binary_plan()
    mutation(data)
    with pytest.raises(ValidationError):
        Plan.model_validate(data)


def test_switch_requires_exact_coverage_of_choice_options():
    data = {
        "questions": {
            "q": {"type": "choice", "instructions": "Choose.", "criteria": {"a": "A", "b": "B"}}
        },
        "root": {
            "type": "switch",
            "question": "q",
            "cases": {"a": {"type": "return", "label": "ok"}},
        },
    }
    with pytest.raises(ValidationError, match="cases"):
        Plan.model_validate(data)


def test_constant_tree_is_a_valid_baseline():
    plan = Plan.model_validate({"questions": {}, "root": {"type": "return", "label": "x"}})
    assert plan.decide({}) == ("x", [])


@pytest.mark.parametrize(
    "labels,questions,depth",
    [(["other"], 4, 4), (["accept", "review"], 0, 4), (["accept", "review"], 4, 0)],
)
def test_search_constraints_are_enforced(labels, questions, depth):
    with pytest.raises(ValueError):
        Plan.model_validate(binary_plan()).check(labels, questions, depth)


def task_data():
    return {
        "description": "Classify requests.",
        "labels": ["accept", "review"],
        "train": [{"id": "a", "state": "request A", "label": "accept"}],
        "validation": [{"id": "b", "state": "request B", "label": "review"}],
        "test": [{"id": "c", "state": "request C", "label": "accept"}],
    }


@pytest.mark.parametrize(
    "change",
    [
        lambda t: t["test"][0].update(id="a"),
        lambda t: t["test"][0].update(state="request A"),
        lambda t: t["test"][0].update(label="invented"),
        lambda t: t.update(validation=[]),
        lambda t: t.update(labels=["accept", "accept"]),
    ],
)
def test_task_rejects_leakage_and_invalid_targets(change):
    data = task_data()
    change(data)
    with pytest.raises(ValidationError):
        Task.model_validate(data)


def test_task_detects_duplicate_json_states_regardless_of_key_order():
    data = task_data()
    data["train"][0]["state"] = {"a": 1, "b": 2}
    data["test"][0]["state"] = {"b": 2, "a": 1}
    with pytest.raises(ValidationError, match="state"):
        Task.model_validate_json(json.dumps(data))
