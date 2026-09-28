"""A judgment that only a later rule would use is not asked once an earlier rule has answered."""

from rulesmith.rules import compile_rules
from rulesmith.runtime import DecisionProgram


def asked(provider) -> list[set[str]]:
    return [set(body["questions"]) for path, body in provider.requests if path == "/v1/systemone"]


def choice(value: str) -> dict:
    return {
        "type": "choice",
        "choice": value,
        "confidence": 0.9,
        "probabilities": {"yes": 0.1, "no": 0.1} | {value: 0.9},
    }


def test_a_rule_that_answers_saves_the_judgment_behind_it(provider, client):
    plan = compile_rules(
        "read rating in [1, 5]\n"
        "when rating >= 4: yes\n"
        'always: ask "Recommended?" [yes: "y"] [no: "n"]'
    )
    provider.answers = {"rule_2": choice("no")}
    program = DecisionProgram(plan, client, "fixture")
    assert program({"rating": 5}).label == "yes"
    assert asked(provider) == []
    assert program({"rating": 2}).label == "no"
    assert asked(provider) == [{"rule_2"}]


def test_asks_in_order_stop_at_the_first_that_answers(provider, client):
    plan = compile_rules(
        "read urgent in [0, 1]\n"
        'when urgent >= 1: ask "First?" [yes: "y"] [no: "n"]\n'
        'always: ask "Second?" [yes: "y"] [no: "n"]'
    )
    provider.answers = {"rule_1": choice("yes"), "rule_2": choice("no")}
    program = DecisionProgram(plan, client, "fixture")
    assert program({"urgent": 1}).label == "yes"
    assert asked(provider) == [{"rule_1"}]
    assert program({"urgent": 0}).label == "no"
    assert asked(provider)[-1] == {"rule_2"}


def test_a_judgment_something_else_reads_is_still_asked(provider, client):
    # 'sure' reads the judgment's confidence, so it is needed whatever the earlier rule decides.
    plan = compile_rules(
        "read rating in [1, 5]\n"
        'ask q = "Recommended?" [yes: "y"] [no: "n"]\n'
        "when rating >= 4: yes\n"
        'when q.confidence >= 0.5 and q == "no": no\n'
        "always: yes"
    )
    provider.answers = {"q": choice("no")}
    DecisionProgram(plan, client, "fixture")({"rating": 5})
    assert asked(provider) == [{"q"}]
