import dspy
import pytest
from conftest import chat_reply
from dspy.utils.exceptions import AdapterParseError
from hypothesis import assume, given
from hypothesis import strategies as st

from rulesmith.label import Labeler, draft_task


@given(
    st.integers(min_value=3, max_value=30).flatmap(
        lambda n: st.tuples(
            st.just(n), st.integers(1, n - 2), st.integers(1, n - 2), st.integers(0, 2**32)
        )
    )
)
def test_every_input_lands_in_exactly_one_split(case):
    count, validation, test, seed = case
    assume(validation + test < count)
    states = [f"input {i}" for i in range(count)]
    task = draft_task("Classify.", ["a", "b"], states, lambda state: "a", validation, test, seed)
    splits = (task.train, task.validation, task.test)
    assert sorted(e.state for split in splits for e in split) == sorted(states)
    assert (len(task.validation), len(task.test)) == (validation, test)
    again = draft_task("Classify.", ["a", "b"], states, lambda state: "a", validation, test, seed)
    assert again == task


def test_split_sizes_are_checked_before_any_label_is_requested():
    requested = []
    with pytest.raises(ValueError, match="training"):
        draft_task("Classify.", ["a", "b"], ["x", "y"], requested.append, 1, 1, 0)
    assert requested == []


def test_labeler_rejects_a_label_outside_the_task(provider):
    provider.completion = chat_reply(label="invented")
    labeler = Labeler(
        dspy.LM(
            "openai/fixture-model",
            api_base=provider.url + "/v1",
            api_key="test-placeholder",
            cache=False,
            num_retries=0,
        ),
        "Classify requests.",
        ["accept", "review"],
    )
    with pytest.raises(AdapterParseError, match="invented"):
        labeler("request A")


@pytest.mark.parametrize(
    "states,validation,test",
    [
        (["x", "y", "z"], 0, 1),
        (["x", "y", "z"], 1, 0),
        (["x", "y", "z", "w"], -1, 1),
        (["x", "x", "y", "z"], 1, 1),
    ],
)
def test_a_task_that_cannot_be_built_is_refused_before_anything_is_labeled(
    states, validation, test
):
    # Labels are paid for; a split that fails afterwards throws every one of them away.
    requested = []
    with pytest.raises(ValueError):
        draft_task("Classify.", ["a", "b"], states, requested.append, validation, test, 0)
    assert requested == []
