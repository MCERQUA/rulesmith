"""How sure a judgement was, as something a rule can test."""

import pytest

from decision_compose.rules import compile_rules, render_rules
from decision_compose.runtime import DecisionProgram

GATED = """ask threat = "Fight or run?" [ATTACK: "Lined up."] [MOVE_BACKWARD: "Not lined up."]
when threat.confidence >= 0.7 and threat == "ATTACK": ATTACK
always: MOVE_FORWARD"""


def answering(provider, choice, confidence):
    provider.body["answers"] = {
        "threat": {
            "type": "choice",
            "choice": choice,
            "probabilities": {"ATTACK": 0.6, "MOVE_BACKWARD": 0.4},
            "confidence": confidence,
        }
    }


def test_a_sure_judgement_is_acted_on(provider, client):
    answering(provider, "ATTACK", 0.9)
    program = DecisionProgram(compile_rules(GATED), client, "fixture")
    assert program({"anything": 1}).label == "ATTACK"


def test_an_unsure_judgement_falls_through_to_the_cheap_rule(provider, client):
    # The point of the whole thing: the judge said ATTACK, but not confidently enough to act on.
    answering(provider, "ATTACK", 0.4)
    program = DecisionProgram(compile_rules(GATED), client, "fixture")
    assert program({"anything": 1}).label == "MOVE_FORWARD"


def test_confidence_survives_the_round_trip():
    once = render_rules(compile_rules(GATED))
    assert "threat.confidence >= 0.7" in once
    assert render_rules(compile_rules(once)) == once


def test_a_probability_question_has_no_separate_confidence(provider, client):
    # A noul answers with the probability itself, so there is nothing else to report and the
    # rule that tests its confidence simply never fires.
    rules = (
        'ask winning = "We are winning."\n'
        "when winning.confidence >= 0.5: ATTACK\n"
        "always: MOVE_FORWARD"
    )
    provider.body["answers"] = {"winning": {"type": "noul", "noul": 0.95}}
    program = DecisionProgram(compile_rules(rules), client, "fixture")
    assert program({"anything": 1}).label == "MOVE_FORWARD"


def test_a_confidence_on_something_that_was_never_asked_is_an_error():
    with pytest.raises(ValueError, match="nothing named"):
        compile_rules("read score in [0, 1]\nwhen score.confidence >= 0.5: high\nalways: low")
