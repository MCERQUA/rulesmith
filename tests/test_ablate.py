"""What each input is worth, measured by taking it away."""

from decision_compose.ablate import effects
from decision_compose.rules import compile_rules

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
