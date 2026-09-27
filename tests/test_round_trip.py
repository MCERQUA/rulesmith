"""Graphs whose rules, read back, must be the same graph: each case once broke the round trip."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rulesmith.graph import Plan
from rulesmith.rules import RuleError, compile_rules, parts, render_rules, whole
from rulesmith.runtime import DecisionProgram


def round_trips(text: str) -> Plan:
    plan = compile_rules(text)
    once = render_rules(plan)
    assert render_rules(compile_rules(once)) == once
    assert whole(parts(plan)) == compile_rules(once)
    return plan


def test_a_vote_over_unnamed_readings():
    round_trips(
        "read x one of [a, b]\nread y one of [a, b]\nvote v = majority of [x, y]\n"
        'when v == "a": A\nalways: B'
    )


@pytest.mark.parametrize("value", ["-x", "0 - x", "x + 0", "2 * x", "-2 * x + 3"])
def test_a_sum_of_any_shape_stays_a_number(value):
    plan = round_trips(f"read x in [0, 9]\nlet m = {value}\nwhen m >= 1: A\nalways: B")
    assert plan.nodes["m"].type == "sum"


@given(st.text(min_size=1).map(str.strip).filter(bool))
def test_any_label_text_comes_back_as_written(label):
    # A label is the task's own vocabulary, and escaping it changes what the graph answers.
    unwritable = len(label.splitlines()) != 1 or label.startswith("ask ")
    if unwritable or any(mark in label for mark in ":?#`") or "readings" in label:
        return
    plan = compile_rules(f"read x in [0, 9]\nwhen x >= 3: {label}\nalways: other")
    assert plan.nodes["rule_1"].value == label
    assert compile_rules(render_rules(plan)).nodes["rule_1"].value == label


@given(st.text(min_size=1).filter(lambda text: text.strip() == text and "\n" not in text))
def test_any_choice_description_comes_back_as_written(description):
    text = f'ask q = "Which?" [a: {_quoted(description)}] [b: "plain"]\nwhen q == "a": A\nalways: B'
    plan = round_trips(text)
    assert plan.nodes["q"].question.criteria["a"] == description


def _quoted(text: str) -> str:
    import json

    # Plain JSON escaping, \u included, is how a proposer writes quoted text.
    return json.dumps(text)


def test_an_instruction_may_contain_brackets():
    plan = round_trips('ask q = "Is [x] big?" [low] [high]\nwhen q >= 1: A\nalways: B')
    assert plan.nodes["q"].question.instructions == "Is [x] big?"


@pytest.mark.parametrize(
    "graph_part",
    [
        {"type": "read", "path": ["x"], "categories": ["a,b", "c"]},
        {
            "type": "call",
            "question": {
                "type": "score",
                "instructions": "How?",
                "criteria": ["1: bad", "2: good"],
            },
            "inputs": {"observation": {"type": "state"}},
        },
        {"type": "read", "path": ["years", "2024"], "bounds": [0, 9]},
    ],
)
def test_what_rules_cannot_write_exactly_is_refused(graph_part):
    test = (
        {"type": "compare", "node": "x", "op": "eq", "value": "c"}
        if graph_part.get("categories")
        else {"type": "present", "node": "x"}
    )
    plan = Plan.model_validate(
        {
            "nodes": {"x": graph_part, "flag": {"type": "value", "value": "A", "when": test}},
            "outputs": {"label": {"type": "node", "node": "flag"}},
        }
    )
    with pytest.raises(RuleError):
        render_rules(plan)


def test_a_digit_in_a_path_reads_an_object_key_as_well_as_a_list_index():
    plan = compile_rules("read years.2024 in [0, 9]\nwhen years.2024 >= 3: A\nalways: B")
    run = DecisionProgram(plan, None, "none")
    assert run({"years": {"2024": 5}}).label == "A"
    assert run({"years": [0] * 2024 + [5]}).label == "A"


def test_two_paths_that_flatten_to_one_name_stay_two_readings():
    plan = round_trips(
        'read a.b in [0, 10]\nread a_b one of [p, q]\nwhen a.b >= 3 and a_b == "p": A\nalways: B'
    )
    assert len([node for node in plan.nodes.values() if node.type == "read"]) == 2


def test_a_confidence_never_lands_on_a_name_already_taken():
    plan = round_trips(
        'read x in [0, 9]\nlet q_confidence = x >= 3\nask q = "Q?" [a: "A"] [b: "B"]\n'
        "when q.confidence >= 0.5 and q_confidence: A\nalways: B"
    )
    assert {node.type for node in plan.nodes.values()} >= {"confidence", "value"}


@pytest.mark.parametrize(
    "condition",
    ['x == "black or white"', 'x == "a:b"', 'x == "(odd"'],
)
def test_a_quoted_label_is_never_read_as_syntax(condition):
    label = condition.split(" == ")[1].strip('"')
    plan = round_trips(f"read x one of [{label}, other]\nwhen {condition}: A\nalways: B")
    assert DecisionProgram(plan, None, "none")({"x": label}).label == "A"


def test_a_path_segment_named_like_a_keyword_can_be_added():
    plan = round_trips("read flags.not in [0, 9]\nlet s = flags.not + 1\nwhen s >= 2: A\nalways: B")
    assert plan.nodes["s"].type == "sum"


@pytest.mark.parametrize("value", ["true", "false", "null"])
def test_a_comparison_with_true_false_or_null_is_refused(value):
    # A bool silently became 1.0, so 'x == true' compared a number the author never wrote.
    with pytest.raises(RuleError, match=value):
        compile_rules(f"read x in [0, 9]\nwhen x == {value}: A\nalways: B")


@pytest.mark.parametrize(
    "action",
    ['ask threat = "q" [A: "x"] [B: "y"]', "ask what is it [A] [B]", 'ask "q" [A: "x"] [B'],
)
def test_a_malformed_ask_names_its_line(action):
    # The repair loop hands this message to the proposer, so it has to point at the line.
    with pytest.raises(RuleError, match="line 2"):
        compile_rules(f"read a in [0, 9]\nwhen a >= 3: {action}\nalways: B")
