"""The benchmark harness: tasks drawn from public data, baselines, and the report."""

import json
import time
from statistics import mean
from typing import Literal

import pytest
from conftest import chat_reply
from hypothesis import given
from hypothesis import strategies as st

from rulesmith.bench import (
    Answer,
    Dataset,
    Models,
    direct_plan,
    dspy_answers,
    elo_against,
    graph_answers,
    majority,
    mcnemar,
    measure,
    record,
    report,
    run,
    sample,
    selective,
    summarize,
    tfidf,
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
    plan = direct_plan(task)
    assert plan.nodes["label"].question.criteria == LABELS
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


def test_a_graph_answer_carries_its_cost_and_how_sure_it_was():
    asked = {
        "correct": True,
        "calls": {"a": {}, "b": {}},
        "input_tokens": 10,
        "answers": {
            "a": {"type": "choice", "value": "x", "probabilities": {"x": 0.7, "y": 0.3}},
            "b": {"type": "noul", "value": 0.2},
        },
    }
    settled = {"correct": False, "calls": {}, "input_tokens": 0, "answers": {}}
    failed = {"correct": False, "error": "skipped", "answers": {}}
    first, second, third = graph_answers([asked, settled, failed])
    # The least sure judgment decides; rules alone count as sure; a failure as not sure at all.
    assert (first.calls, first.input_tokens, first.confidence) == (2, 10, 0.7)
    assert second.confidence == 1.0 and third.confidence == 0.0 and third.invalid


@given(st.lists(st.tuples(st.booleans(), st.floats(0, 1)), min_size=1, max_size=50))
def test_answering_only_the_surest_never_scores_below_ranking_by_chance(pairs):
    answers = [Answer(correct=c, confidence=f) for c, f in pairs]
    assert selective(answers, 1.0) == mean(c for c, _ in pairs)
    # When confidence tracks correctness exactly, keeping the surest can only help.
    ordered = [Answer(correct=c, confidence=float(c)) for c, _ in pairs]
    assert selective(ordered, 0.5) >= selective(ordered, 1.0)
    assert selective([Answer(correct=True)], 0.5) is None


def test_mcnemar_tells_a_real_difference_from_chance():
    same = [True, False] * 50
    assert mcnemar(same, same) == 1.0
    # Ten examples only the first got right, none the other way: p = 2 / 2**10.
    first = [True] * 10 + [False] * 90
    second = [False] * 100
    assert mcnemar(first, second) == pytest.approx(2 / 2**10)
    assert mcnemar(first, second) == mcnemar(second, first)


def test_a_balanced_dataset_draws_every_label_about_equally():
    rows = {
        split: [(f"{split} rare {i}", "rare") for i in range(40)]
        + [(f"{split} common {i}", "common") for i in range(400)]
        for split in ("train", "test")
    }
    dataset = Dataset("skewed", "Which?", lambda: rows, str, balanced=True)
    task = sample(dataset, 20, 20, 40, 0)
    assert sum(e.label == "rare" for e in task.test) == 20
    assert abs(sum(e.label == "rare" for e in task.train + task.validation) - 20) <= 1


def test_hand_written_rules_are_scored_like_any_graph():
    rows = toy().rows
    dataset = Dataset("toy", "Route.", rows, LABELS.__getitem__, handwritten="always: billing")
    task = sample(dataset, 6, 3, 12, 0)
    run = measure("handwritten", 0, task, dataset, Models())
    assert run["accuracy"] == sum(e.label == "billing" for e in task.test) / len(task.test)
    assert run["calls_per_example"] == 0 and len(run["correct"]) == len(task.test)


def test_a_dspy_reply_naming_no_allowed_label_is_invalid(provider):
    import dspy

    provider.completion = chat_reply(label="bogus")
    task = sample(toy(), 6, 3, 3, 0)
    fields = {
        "text": (str, dspy.InputField()),
        "label": (Literal[tuple(task.labels)], dspy.OutputField()),
    }
    signature = dspy.make_signature(fields, task.description)
    student = dspy.LM(
        "openai/fixture-model",
        api_base=provider.url + "/v1",
        api_key="test-placeholder",
        cache=False,
        num_retries=0,
    )
    answers, _ = dspy_answers(dspy.Predict(signature), task, student)
    assert all(answer.invalid and not answer.correct for answer in answers)


def test_the_summary_averages_each_method_over_seeds():
    def run(method, seed, accuracy, calls):
        return {
            "method": method,
            "seed": seed,
            "accuracy": accuracy,
            "calls_per_example": calls,
            "tokens_per_example": 100 * calls,
            "accuracy_at_80": None,
            "correct": [True],
        }

    runs = [run("direct", 0, 0.5, 1), run("direct", 1, 0.7, 1), run("majority", 0, 0.1, 0)]
    summary = summarize(runs)
    assert summary["direct"]["seeds"] == 2
    assert summary["direct"]["accuracy"] == pytest.approx(0.6)
    assert summary["direct"]["spread"] == pytest.approx(0.1414, abs=1e-4)
    assert summary["direct"]["tokens_per_example"] == 100
    assert summary["majority"]["spread"] == 0
    # A figure a method cannot give stays missing rather than averaged as zero.
    assert summary["direct"]["accuracy_at_80"] is None
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


def test_nothing_new_starts_after_the_deadline(tmp_path):
    # An overnight run has a hard stop; work already written stays, and nothing half-starts.
    output = tmp_path / "toy.json"
    sizes = {"train": 30, "validation": 6, "test": 12}
    runs = run(toy(), [0, 1], ["majority", "tfidf"], sizes, output, {}, deadline=time.time() - 1)
    assert runs == [] and not output.exists()
    runs = run(toy(), [0], ["majority"], sizes, output, {}, deadline=time.time() + 3600)
    assert [r["method"] for r in runs] == ["majority"]


def test_a_download_that_is_not_the_published_file_is_refused(tmp_path, monkeypatch):
    # A mirror that changed its data would change every number measured on it, silently.
    from rulesmith.bench import cached

    source = tmp_path / "data.csv"
    source.write_text("a,b\n1,2\n")
    monkeypatch.setenv("RULESMITH_CACHE", str(tmp_path / "cache"))
    with pytest.raises(ValueError, match="checksum"):
        cached(source.as_uri(), "data.csv", sha1="0" * 40)
    assert not (tmp_path / "cache" / "data.csv").exists()
    import hashlib

    digest = hashlib.sha1(source.read_bytes()).hexdigest()
    assert cached(source.as_uri(), "data.csv", sha1=digest).read_text() == "a,b\n1,2\n"


def test_the_judge_waits_as_long_as_the_run_allows():
    # Four datasets, chess and DSPy queue on one local judge; the SDK's ten seconds ended runs.
    from rulesmith.bench import judge_client

    with judge_client("http://127.0.0.1:9", 3600) as client:
        assert client._config.timeout == 3600


def test_a_dropped_connection_to_a_generative_model_is_retried():
    # One dropped connection to a local server under load ended a whole overnight dataset.
    from rulesmith.bench import language_model

    lm = language_model("openai/fixture", "http://127.0.0.1:9", None, 64, 5, retries=3)
    assert lm.num_retries == 3


def test_a_failed_example_scores_zero_instead_of_ending_the_run():
    # DSPy means a failed example to score its failure_score; its GEPA integration instead
    # crashes when one is missing, so the program under search reports failures as answers.
    import dspy

    from rulesmith.bench import Failsafe

    class Broken(dspy.Module):
        def forward(self, text):
            raise TypeError("unsupported operand type(s) for /: 'list' and 'list'")

    guarded = Failsafe(Broken())
    predicted = guarded(text="anything")
    assert predicted.label is None and guarded.failures == 1


def test_each_result_names_the_models_that_produced_it(tmp_path):
    # A resumed run may use other models than the one that started the file; a single header
    # written at the start would credit every result to the first run's models.
    output = tmp_path / "toy.json"
    sizes = {"train": 30, "validation": 6, "test": 12}
    run(toy(), [0], ["majority"], sizes, output, {"proposer": "first"})
    runs = run(toy(), [0, 1], ["majority"], sizes, output, {"proposer": "second"})
    assert [r["models"]["proposer"] for r in runs] == ["first", "second"]
