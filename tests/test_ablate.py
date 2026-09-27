"""What each input is worth, measured by taking it away."""

from rulesmith.ablate import effects
from rulesmith.rules import compile_rules

DECIDER = """read score in [0, 1]
when score >= 0.5: high
always: low"""
# A reading no rule mentions is already dropped when the rules compile, so what is left to find
# is the other kind: a reading the graph asks for, and then decides the same way without.
DEAD_RULE = """read score in [0, 1]
always: high
when score >= 0.5: never_reached"""


def states(values):
    return [{"score": value} for value in values]


def test_the_reading_that_decides_is_the_one_that_changes_decisions():
    (effect,) = effects(compile_rules(DECIDER), states([0.1, 0.9, 0.4, 0.8]))
    # Without it the two high states fall through to the default; the low ones were low anyway.
    assert effect.changed == 2 and effect.share == 0.5
    assert effect.kind == "read" and effect.reason == ""


def test_a_rule_an_earlier_one_always_beats_changes_nothing():
    (effect,) = effects(compile_rules(DEAD_RULE), states([0.1, 0.9]))
    assert effect.changed == 0
    assert effect.reason == "present in 2 of 2 states and changes no decision"


def test_a_path_the_state_never_has_says_which_problem_it_is():
    # The Doom defect: every read path resolved to nothing, so the graph ran, decided badly, and
    # looked exactly like a graph that plays badly.
    plan = compile_rules(
        "read first_observation.score in [0, 1]\n"
        "when first_observation.score >= 0.5: high\n"
        "always: low"
    )
    (effect,) = effects(plan, states([0.1, 0.9]))
    assert effect.changed == 0
    assert effect.reason == "this path is never in the state"


def test_effects_are_ranked_and_counted_against_the_states_given():
    found = effects(compile_rules(DECIDER), states([0.3, 0.7, 0.9]))
    assert [effect.decided for effect in found] == [3]
    assert found == sorted(found, key=lambda effect: (-effect.changed, effect.node))


def test_the_judge_is_asked_once_per_state_however_many_inputs_are_ablated(provider, client):
    # Taking a reading away cannot change what the judge said about the same state, so only the
    # baseline pass may reach it. Every later pass must replay those answers.
    plan = compile_rules(
        "read score in [0, 1]\n"
        'ask team = "Which team?" [billing: "Charges."] [technical: "Bugs."]\n'
        'when score >= 0.5 and team == "billing": high\n'
        "always: low"
    )
    provider.body["answers"] = {
        "team": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.9, "technical": 0.1},
            "confidence": 0.9,
        }
    }
    effects(plan, states([0.2, 0.7, 0.9]), client, "fixture")
    asked = [path for path, _ in provider.requests if path == "/v1/systemone"]
    assert len(asked) == 3


def test_a_name_is_measured_like_an_input():
    # A named condition or a computed value is where a rewrite concentrates a decision, so it is
    # worth knowing whether the rules that test it ever fire.
    plan = compile_rules(
        "read a in [0, 9]\nread b in [0, 9]\n"
        "let ahead = a - b\n"
        "let close = ahead < 1 and ahead >= -1\n"
        "when close: tied\n"
        "always: apart"
    )
    found = {effect.node: effect for effect in effects(plan, [{"a": 1, "b": 1}, {"a": 9, "b": 0}])}
    assert found["close"].kind == "let" and found["close"].changed == 1
    assert found["ahead"].kind == "let" and found["ahead"].changed == 1


def test_switching_off_one_of_two_judgments_asked_together_asks_nothing_new(provider, client):
    # Judgments with the same inputs share one request, so taking one away changes the request;
    # the other's answer is still the one the baseline paid for.
    plan = compile_rules(
        'ask a = "First?" [yes: "y"] [no: "n"]\n'
        'ask b = "Second?" [yes: "y"] [no: "n"]\n'
        'when a == "yes" and b == "yes": both\n'
        "always: neither"
    )
    provider.body["answers"] = {
        name: {
            "type": "choice",
            "choice": "yes",
            "probabilities": {"yes": 1, "no": 0},
            "confidence": 1,
        }
        for name in ("a", "b")
    }
    found = effects(plan, states([0.1, 0.2]), client, "fixture")
    assert [path for path, _ in provider.requests].count("/v1/systemone") == 2
    assert all(effect.changed == 2 for effect in found)


def test_a_graph_without_a_label_output_is_measured_on_its_outputs():
    plan = compile_rules("read score in [0, 1]\noutput move:\nwhen score >= 0.5: go\nalways: stay")
    (effect,) = effects(plan, states([0.1, 0.9]))
    assert effect.changed == 1
