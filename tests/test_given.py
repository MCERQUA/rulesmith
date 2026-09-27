"""Judgments asked about named values instead of the whole input."""

import pytest
from test_graph import call

from rulesmith.graph import Plan
from rulesmith.rules import RuleError, compile_rules, render_rules
from rulesmith.runtime import DecisionProgram

GIVEN = (
    "read health = variables.HEALTH in [0, 200]\n"
    "read enemy.distance in [0, 4096]\n"
    'ask threat = "Is the enemy a threat?" [fight: "Close and I am healthy."] '
    '[flee: "Otherwise."] given [health, enemy.distance]\n'
    'when threat == "fight": ATTACK\n'
    "always: MOVE_BACKWARD"
)
STATE = {"variables": {"HEALTH": 80}, "enemy": {"distance": 300}, "noise": "x" * 50}


def answer(provider, choice):
    provider.body["answers"] = {
        "threat": {
            "type": "choice",
            "choice": choice,
            "probabilities": {"fight": 0.5, "flee": 0.5},
            "confidence": 0.9,
        }
    }


def test_the_judge_sees_only_what_it_was_given(provider, client):
    answer(provider, "fight")
    run = DecisionProgram(compile_rules(GIVEN), client, "fixture")(STATE)
    assert run.label == "ATTACK"
    # Less to read is the point: the judge gets the two values, not the whole state.
    assert provider.requests[-1][1]["state"] == {"health": 80, "enemy.distance": 300}


def test_a_judgment_whose_inputs_are_missing_is_not_asked(provider, client):
    answer(provider, "fight")
    run = DecisionProgram(compile_rules(GIVEN), client, "fixture")({"variables": {"HEALTH": 80}})
    assert run.label == "MOVE_BACKWARD" and not provider.requests


def test_given_round_trips_and_may_include_the_whole_input():
    once = render_rules(compile_rules(GIVEN))
    assert "given [health, enemy.distance]" in once
    assert render_rules(compile_rules(once)) == once
    both = GIVEN.replace("given [health, enemy.distance]", "given [observation, health]")
    plan = compile_rules(both)
    assert plan.nodes["threat"].inputs["observation"].type == "state"
    assert render_rules(compile_rules(render_rules(plan))) == render_rules(plan)


def test_a_judgment_may_be_given_another_judgment():
    text = (
        'ask first = "Fraud?" [yes: "It is."] [no: "It is not."]\n'
        'ask second = "Is this verdict right?" [yes: "It is."] [no: "It is not."] '
        "given [observation, first]\n"
        'when second == "yes" and first == "yes": review\n'
        "always: accept"
    )
    plan = compile_rules(text)
    assert plan.dependencies()["second"] == {"first"}


def test_a_whole_input_under_another_key_is_refused_rather_than_renamed():
    # Writing it as 'observation' would change what the judge is shown.
    plan = Plan.model_validate(
        {
            "nodes": {
                "odds": call(ticket={"type": "state"}),
                "flag": {
                    "type": "value",
                    "value": "review",
                    "when": {"type": "compare", "node": "odds", "op": "ge", "value": 0.5},
                },
            },
            "outputs": {"label": {"type": "node", "node": "flag"}},
        }
    )
    with pytest.raises(RuleError, match="ticket"):
        render_rules(plan)
