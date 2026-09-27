"""Several judgments settled into one value."""

import pytest

from decision_compose.rules import RuleError, compile_rules, render_rules
from decision_compose.runtime import DecisionProgram

SECOND_OPINION = """ask first = "Is this fraud?" [yes: "It is."] [no: "It is not."]
ask second = "Reading it again, is this fraud?" [yes: "It is."] [no: "It is not."]
vote checked = agree of [first, second]
when checked == "yes" and first == "yes": review
always: accept"""


def answers(provider, **said):
    provider.body["answers"] = {
        name: {
            "type": "choice",
            "choice": value,
            "probabilities": {"yes": 0.5, "no": 0.5},
            "confidence": 0.9,
        }
        for name, value in said.items()
    }


def test_two_judgements_that_agree_are_acted_on(provider, client):
    answers(provider, first="yes", second="yes")
    program = DecisionProgram(compile_rules(SECOND_OPINION), client, "fixture")
    assert program({"order": 1}).label == "review"


def test_a_second_opinion_that_disagrees_stops_the_action(provider, client):
    # The point of verifying: one model call said fraud, the other did not, so nothing happens.
    answers(provider, first="yes", second="no")
    program = DecisionProgram(compile_rules(SECOND_OPINION), client, "fixture")
    assert program({"order": 1}).label == "accept"


def test_a_majority_takes_the_answer_most_of_them_gave(provider, client):
    rules = (
        'ask a = "Fraud?" [yes: "y"] [no: "n"]\n'
        'ask b = "Fraud?" [yes: "y"] [no: "n"]\n'
        'ask c = "Fraud?" [yes: "y"] [no: "n"]\n'
        "vote most = majority of [a, b, c]\n"
        'when most == "yes": review\n'
        "always: accept"
    )
    answers(provider, a="yes", b="no", c="yes")
    assert DecisionProgram(compile_rules(rules), client, "fixture")({"order": 1}).label == "review"


def test_a_vote_round_trips():
    once = render_rules(compile_rules(SECOND_OPINION))
    assert "vote checked = agree of [first, second]" in once
    assert render_rules(compile_rules(once)) == once


@pytest.mark.parametrize(
    "line,message",
    [
        ("vote checked = sometimes of [first, second]", "majority"),
        ("vote checked = agree of [first]", "at least two"),
    ],
)
def test_a_vote_that_makes_no_sense_says_so(line, message):
    rules = (
        'ask first = "Fraud?" [yes: "y"] [no: "n"]\n'
        'ask second = "Fraud?" [yes: "y"] [no: "n"]\n'
        f"{line}\n"
        'when checked == "yes": review\n'
        "always: accept"
    )
    with pytest.raises(RuleError, match=message):
        compile_rules(rules)


@pytest.mark.parametrize(
    "question",
    [
        # A probability is a float; two of them agreeing exactly is a coincidence, not a verdict.
        {"type": "noul", "instructions": "Is this fraud?"},
        # A score is a position on a rubric, also a float.
        {"type": "score", "instructions": "How bad?", "criteria": ["minor", "major"]},
    ],
    ids=["probability", "score"],
)
def test_a_vote_counts_labels_so_it_refuses_sources_that_have_none(question):
    from pydantic import ValidationError

    from decision_compose.graph import Plan

    ask = {"type": "call", "question": question, "inputs": {"s": {"type": "state"}}}
    with pytest.raises(ValidationError, match="categorical"):
        Plan.model_validate(
            {
                "nodes": {
                    "a": ask,
                    "b": ask,
                    "both": {"type": "vote", "rule": "agree", "sources": ["a", "b"]},
                },
                "outputs": {"label": {"type": "node", "node": "both"}},
            }
        )
