import pytest
from hypothesis import given
from hypothesis import strategies as st
from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer

from jev_compose.graph import Choice, Noul, Score
from jev_compose.runtime import ResponseError, read_answer


@given(st.floats(min_value=0, max_value=1, allow_nan=False))
def test_noul_probability_is_preserved(value):
    assert read_answer(Noul(instructions="Eligible?"), NoulAnswer(noul=value))["value"] == value


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_probability_is_a_provider_error(value):
    with pytest.raises(ResponseError):
        read_answer(Noul(instructions="Eligible?"), NoulAnswer(noul=value))


def test_answer_type_must_match_question():
    with pytest.raises(ResponseError, match="type"):
        read_answer(Score(instructions="Severity?", criteria=["Low", "High"]), NoulAnswer(noul=0.5))


@pytest.mark.parametrize(
    "choice,probabilities,confidence",
    [
        ("outside", {"a": 0.6, "b": 0.4}, 0.2),
        ("a", {"a": 0.6, "c": 0.4}, 0.2),
        ("a", {"a": -0.1, "b": 1.1}, 0.2),
        ("a", {"a": 0.1, "b": 0.1}, 0.2),
        ("a", {"a": 0.6, "b": 0.4}, 1.2),
    ],
)
def test_choice_contract(choice, probabilities, confidence):
    with pytest.raises(ResponseError):
        read_answer(
            Choice(instructions="Pick.", criteria={"a": "A", "b": "B"}),
            ChoiceAnswer(choice=choice, probabilities=probabilities, confidence=confidence),
        )


def test_score_keeps_fractional_rubric_position():
    question = Score(instructions="Severity?", criteria=["Low", "High"])
    answer = ScoreAnswer(
        score=0.7, probabilities={0: 0.3, 1: 0.7}, legend={0: "Low", 1: "High"}, confidence=0.4
    )
    assert read_answer(question, answer)["value"] == 0.7


def test_score_rejects_drifted_rubric():
    with pytest.raises(ResponseError):
        read_answer(
            Score(instructions="Severity?", criteria=["Low", "High"]),
            ScoreAnswer(
                score=0.7,
                probabilities={0: 0.3, 1: 0.7},
                legend={0: "Calm", 1: "Angry"},
                confidence=0.4,
            ),
        )
