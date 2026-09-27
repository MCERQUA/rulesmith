"""Scoring against labels: some mistakes cost more, and some examples have more than one answer."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from test_graph import constant_plan, task_data

from rulesmith.graph import Plan, Task
from rulesmith.optimize import Labels, evaluate_plan
from rulesmith.runtime import DecisionProgram

LABELS = ["accept", "review", "decline"]


def fraud_task(**changes) -> Task:
    data = task_data() | {"labels": LABELS}
    data["train"][0]["label"] = "review"
    return Task.model_validate(data | changes)


def always(label: str) -> DecisionProgram:
    return DecisionProgram(Plan.model_validate(constant_plan(label)), None, "none")


costs = st.dictionaries(
    st.sampled_from(LABELS),
    # Costs are written by people in a task file, so three decimal places is plenty; a cost of
    # 1e-104 is indistinguishable from free once divided by the worst.
    st.dictionaries(st.sampled_from(LABELS), st.floats(0, 100).map(lambda cost: round(cost, 3))),
).map(lambda table: {e: {a: c for a, c in row.items() if a != e} for e, row in table.items()})


@given(costs, st.sampled_from(LABELS), st.sampled_from(LABELS), st.sampled_from(LABELS))
def test_a_score_falls_as_the_mistake_costs_more(table, expected, first, second):
    task = fraud_task(costs=table)
    example = task.train[0].model_copy(update={"label": expected})
    score = Labels(task)
    one, other = score(always(first), example), score(always(second), example)
    assert 0 <= one.score <= 1
    assert (one.score == 1) == (first == expected or task.cost(expected, first) == 0)
    if task.cost(expected, first) < task.cost(expected, second):
        assert one.score > other.score


@given(st.sampled_from(LABELS), st.sampled_from(LABELS))
def test_without_costs_the_score_is_exact_match(expected, answered):
    task = fraud_task()
    example = task.train[0].model_copy(update={"label": expected})
    assert Labels(task)(always(answered), example).score == float(expected == answered)


def test_an_accepted_answer_is_fully_right():
    task = fraud_task(
        train=[{"id": "a", "state": "request A", "label": "review", "accepted": ["decline"]}]
    )
    assert Labels(task)(always("decline"), task.train[0]).score == 1
    assert Labels(task)(always("accept"), task.train[0]).score == 0


def test_search_is_told_what_a_mistake_cost():
    task = fraud_task(costs={"review": {"accept": 5}})
    outcome = Labels(task)(always("accept"), task.train[0])
    assert outcome.score == 0 and outcome.record["cost"] == 5
    assert "5" in outcome.trace["Feedback"]


def test_the_cheaper_mistake_scores_higher_where_exact_match_cannot_tell():
    # Missing fraud (answering accept when it was review) is five times worse than a false alarm.
    data = task_data() | {"labels": LABELS, "costs": {"review": {"accept": 5, "decline": 1}}}
    data["test"] = [{"id": "c", "state": "request C", "label": "review"}]
    task = Task.model_validate(data)
    score = Labels(task)
    lenient, strict = (
        evaluate_plan(
            Plan.model_validate(constant_plan(label)), task.test, None, "none", score=score
        )
        for label in ("accept", "decline")
    )
    assert strict["score"] > lenient["score"]


@pytest.mark.parametrize(
    "changes",
    [
        {"costs": {"invented": {"accept": 1}}},
        {"costs": {"review": {"invented": 1}}},
        {"costs": {"review": {"review": 1}}},
        {"costs": {"review": {"accept": -1}}},
        {"train": [{"id": "a", "state": "A", "label": "review", "accepted": ["invented"]}]},
        {"train": [{"id": "a", "state": "A", "label": "review", "accepted": ["review"]}]},
        {"train": [{"id": "a", "state": "A", "accepted": ["review"]}]},
    ],
)
def test_costs_and_accepted_answers_only_name_task_labels(changes):
    with pytest.raises(ValidationError):
        fraud_task(**changes)


def test_an_example_without_a_label_cannot_be_scored_against_labels():
    task = fraud_task(train=[{"id": "a", "state": "request A"}])
    with pytest.raises(ValueError, match="a"):
        Labels(task)(always("accept"), task.train[0])
    # With no training labels to count, the starting graph is the first label the task allows.
    assert task.baseline_plan() == Plan.constant("accept")
