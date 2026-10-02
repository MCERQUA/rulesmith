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


# A scam task sampled half and half, where scams are one posting in twenty in real traffic.
def scam_task(**changes) -> Task:
    labels = ["fraudulent", "genuine"]
    rows = lambda split, n: [  # noqa: E731
        {"id": f"{split}{i}", "state": f"{split} posting {i}", "label": labels[i % 2]}
        for i in range(n)
    ]
    data = {
        "description": "Is it a scam?",
        "labels": labels,
        "train": rows("t", 10),
        "validation": rows("v", 10),
        "test": rows("h", 10),
    }
    return Task.model_validate(data | changes)


REAL = {"fraudulent": 0.05, "genuine": 0.95}


def test_prevalence_must_name_every_label_and_sum_to_one():
    with pytest.raises(ValidationError, match="prevalence"):
        scam_task(prevalence={"fraudulent": 0.05})
    with pytest.raises(ValidationError, match="prevalence"):
        scam_task(prevalence={"fraudulent": 0.5, "genuine": 0.6})
    assert scam_task(prevalence=REAL).weights() == pytest.approx(
        {"fraudulent": 0.1, "genuine": 1.9}
    )


def test_at_the_real_rate_a_false_flag_costs_more_than_a_miss():
    # Sampled half and half, one wrong answer of each kind costs the same; at one scam in twenty,
    # flagging a genuine posting is the mistake that happens nineteen times as often.
    balanced, real = Labels(scam_task()), Labels(scam_task(prevalence=REAL))
    genuine = scam_task().train[1]
    scam = scam_task().train[0]
    assert balanced(always("fraudulent"), genuine).score == balanced(always("genuine"), scam).score
    assert real(always("fraudulent"), genuine).score < real(always("genuine"), scam).score


def test_search_and_evaluation_rank_programs_as_real_traffic_would():
    # Flagging everything is right on half the balanced sample and on one posting in twenty for
    # real; never flagging is right on half, and on nineteen in twenty for real.
    task = scam_task(prevalence=REAL)
    flag, never = constant_plan("fraudulent"), constant_plan("genuine")
    for split in (task.validation, task.test):
        flagged = evaluate_plan(Plan.model_validate(flag), split, None, "none", Labels(task))
        cleared = evaluate_plan(Plan.model_validate(never), split, None, "none", Labels(task))
        assert flagged["score"] < cleared["score"]
    unweighted = Labels(scam_task())
    flagged = evaluate_plan(Plan.model_validate(flag), task.test, None, "none", unweighted)
    cleared = evaluate_plan(Plan.model_validate(never), task.test, None, "none", unweighted)
    assert flagged["score"] == cleared["score"]


def test_evaluation_reports_precision_and_recall_at_the_real_rate():
    # Half and half: a rule flags four of five scams and one of five genuine postings, so its
    # flags look 80% precise. At one scam in twenty, that one false flag stands for 19 genuine
    # postings for every 1 the four scams stand for, and real precision is 0.4 / 2.3.
    from rulesmith.calibrate import at_prevalence

    task = scam_task(prevalence=REAL)
    records = [
        {"expected": "fraudulent", "label": "fraudulent" if i < 4 else "genuine"} for i in range(5)
    ] + [{"expected": "genuine", "label": "fraudulent" if i < 1 else "genuine"} for i in range(5)]
    found = at_prevalence(task, records)
    weights = task.weights()
    flagged = 4 * weights["fraudulent"] + 1 * weights["genuine"]
    assert found["labels"]["fraudulent"]["precision"] == pytest.approx(
        4 * weights["fraudulent"] / flagged
    )
    assert found["labels"]["fraudulent"]["recall"] == pytest.approx(0.8)
    right = 4 * weights["fraudulent"] + 4 * weights["genuine"]
    assert found["accuracy"] == pytest.approx(
        right / (5 * weights["fraudulent"] + 5 * weights["genuine"])
    )
    # Without a prevalence there is no real rate to report.
    assert at_prevalence(scam_task(), records) is None
