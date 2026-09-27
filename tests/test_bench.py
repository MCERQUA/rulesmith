"""The benchmark harness: tasks drawn from public data, baselines, and the report."""

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rulesmith.bench import (
    Dataset,
    direct_plan,
    elo_against,
    majority,
    record,
    report,
    run,
    sample,
    summarize,
    tfidf,
    tokens_and_calls,
)
from rulesmith.graph import Task

LABELS = {"billing": "Charges and refunds", "technical": "Bugs and errors", "sales": "Buying"}


def toy(train_rows: int = 60, test_rows: int = 30) -> Dataset:
    words = {"billing": "refund charge", "technical": "crash error", "sales": "price quote"}
    rows = {
        split: [
            (f"{words[label]} {split} {index}", label)
            for index in range(count)
            for label in [list(LABELS)[index % 3]]
        ]
        for split, count in (("train", train_rows), ("test", test_rows))
    }
    return Dataset("toy", "Route each ticket.", lambda: rows, LABELS.__getitem__)


@given(st.integers(0, 2**16), st.integers(3, 20), st.integers(3, 10), st.integers(3, 10))
def test_a_sample_is_disjoint_sized_and_reproducible(seed, train, validation, test):
    dataset = toy()
    task = sample(dataset, train, validation, test, seed)
    assert (len(task.train), len(task.validation), len(task.test)) == (train, validation, test)
    # Task validation refuses a state in two splits, so building it at all proves disjointness.
    assert Task.model_validate(task.model_dump()) == task
    assert sample(dataset, train, validation, test, seed) == task
    # The test split comes from the dataset's own test pool, never from what search may see.
    assert all(" test " in example.state for example in task.test)


def test_a_sample_larger_than_the_pool_is_refused():
    with pytest.raises(ValueError, match="test"):
        sample(toy(test_rows=5), 3, 3, 10, 0)


def test_the_direct_call_asks_one_question_over_every_label(provider, client):
    task = sample(toy(), 6, 3, 3, 0)
    plan = direct_plan(task, LABELS)
    plan.check(task.labels, 16, 8)
    provider.answers = {
        "label": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.9,
            "probabilities": {label: 1 / 3 for label in LABELS},
        }
    }
    result = record("direct", 0, task, plan, client, "fixture-model")
    assert result["calls_per_example"] == 1
    assert result["accuracy"] == sum(e.label == "billing" for e in task.test) / len(task.test)


def test_majority_needs_no_model():
    task = sample(toy(), 9, 3, 6, 1)
    result = majority(0, task)
    assert result["calls_per_example"] == 0 and 0 <= result["accuracy"] <= 1


def test_a_trained_classifier_separates_what_words_separate():
    result = tfidf(0, sample(toy(), 30, 6, 12, 2))
    assert result["accuracy"] == 1.0 and result["calls_per_example"] == 0


def test_calls_and_tokens_are_counted_per_example():
    records = [
        {"calls": {"a": {"input_tokens": 10}, "b": {"input_tokens": 0}}, "input_tokens": 10},
        {"calls": {}, "input_tokens": 0},
    ]
    assert tokens_and_calls(records) == (1.0, 5.0)


def test_the_summary_averages_each_method_over_seeds():
    runs = [
        {
            "method": "direct",
            "seed": 0,
            "accuracy": 0.5,
            "calls_per_example": 1,
            "tokens_per_example": 100,
        },
        {
            "method": "direct",
            "seed": 1,
            "accuracy": 0.7,
            "calls_per_example": 1,
            "tokens_per_example": 120,
        },
        {
            "method": "majority",
            "seed": 0,
            "accuracy": 0.1,
            "calls_per_example": 0,
            "tokens_per_example": 0,
        },
    ]
    summary = summarize(runs)
    assert summary["direct"]["seeds"] == 2
    assert summary["direct"]["accuracy"] == pytest.approx(0.6)
    assert summary["direct"]["spread"] == pytest.approx(0.1414, abs=1e-4)
    assert summary["direct"]["tokens_per_example"] == 110
    assert summary["majority"]["spread"] == 0
    assert "direct" in report({"dataset": "toy", "test": 300, "runs": runs})


def test_a_benchmark_resumes_without_rerunning_what_it_recorded(tmp_path):
    # Runs take hours; a result already on disk is kept, not recomputed or lost.
    output = tmp_path / "toy.json"
    sizes = {"train": 30, "validation": 6, "test": 12}
    kept = {
        "method": "majority",
        "seed": 0,
        "accuracy": 0.123,
        "calls_per_example": 0,
        "tokens_per_example": 0,
    }
    output.write_text(json.dumps({"dataset": "toy", **sizes, "runs": [kept]}))
    runs = run(toy(), [0], ["majority", "tfidf"], sizes, output, models={})
    assert kept in runs and any(r["method"] == "tfidf" for r in runs)
    assert json.loads(output.read_text())["runs"] == runs


def test_a_results_file_for_other_sizes_is_refused(tmp_path):
    output = tmp_path / "toy.json"
    output.write_text(
        json.dumps({"dataset": "toy", "train": 1, "validation": 1, "test": 1, "runs": []})
    )
    with pytest.raises(ValueError, match="sizes"):
        run(toy(), [0], ["majority"], {"train": 30, "validation": 6, "test": 12}, output, {})


def test_an_elo_estimate_follows_the_score_against_a_rated_reference():
    # Level with the reference is its own rating; a score of 0 or 1 says only that it is far away.
    assert elo_against(1320, 0.5) == 1320
    assert elo_against(1320, 0.75) > 1320 > elo_against(1320, 0.25)
    assert elo_against(1320, 0.0) is None and elo_against(1320, 1.0) is None
