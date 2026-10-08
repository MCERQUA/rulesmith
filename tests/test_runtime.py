import hashlib
import json

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from test_graph import binary_plan
from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer

from rulesmith.graph import Choice, Noul, Plan, Score
from rulesmith.runtime import (
    DecisionProgram,
    ResponseError,
    read_answer,
)


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


@given(
    st.floats(min_value=0, max_value=1, allow_nan=False),
    st.floats(min_value=0, max_value=1, allow_nan=False),
)
def test_threshold_partition_after_serialization(value, cutoff):
    from rulesmith.graph import Plan
    from rulesmith.runtime import resolve_node

    plan = Plan.model_validate(binary_plan(cutoff))
    restored = Plan.model_validate_json(plan.model_dump_json())
    result = resolve_node(restored.nodes["decision"], {}, {"eligible": value}, {})
    assert result == ("accept" if value >= cutoff else "review")


def test_pure_graph_preserves_null_and_omits_only_skipped_optional_inputs():
    from test_graph import ref

    from rulesmith.graph import Plan
    from rulesmith.runtime import DecisionProgram

    plan = Plan.model_validate(
        {
            "nodes": {
                "null": {"type": "value", "value": None},
                "absent": {
                    "type": "value",
                    "value": "unused",
                    "when": {"type": "not", "condition": {"type": "present", "node": "null"}},
                },
                "required": {"type": "merge", "inputs": {"x": ref("absent")}},
                "optional": {
                    "type": "merge",
                    "inputs": {
                        "x": ref("absent", optional=True),
                        "y": ref("null"),
                        "state": {"type": "state"},
                    },
                },
                "joined": {"type": "coalesce", "sources": ["absent", "null"]},
            },
            "outputs": {
                "required": ref("required", optional=True),
                "optional": ref("optional"),
                "joined": ref("joined"),
                "literal": {"type": "literal", "value": False},
            },
        }
    )
    result = DecisionProgram(plan, None, "unused")(state=["original"])
    assert result.outputs == {
        "optional": {"y": None, "state": ["original"]},
        "joined": None,
        "literal": False,
    }
    assert set(result.skipped) == {"absent", "required"}
    assert result.calls == {} and result.input_tokens == result.output_tokens == 0


def test_a_skipped_required_output_leaves_the_decision_undecided():
    from test_graph import ref

    from rulesmith.graph import Plan
    from rulesmith.runtime import DecisionProgram

    plan = Plan.model_validate(
        {
            "nodes": {
                "base": {"type": "value", "value": "yes"},
                "absent": {
                    "type": "value",
                    "value": "x",
                    "when": {"type": "not", "condition": {"type": "present", "node": "base"}},
                },
            },
            "outputs": {"result": ref("absent")},
        }
    )
    result = DecisionProgram(plan, None, "unused")(state="input")
    assert result.undecided == "no_rule_applied"
    assert result.label is None and result.decided_by is None
    assert "result" not in result.outputs and result.skipped == ["absent"]


def test_execution_limits_reject_nonpositive_concurrency():
    from pydantic import ValidationError

    from rulesmith.runtime import ExecutionConfig

    with pytest.raises(ValidationError):
        ExecutionConfig(max_workers=0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_runtime_rejects_nonfinite_input_states(value):
    from pydantic import ValidationError

    from rulesmith.graph import Plan
    from rulesmith.runtime import DecisionProgram

    plan = Plan.model_validate({"nodes": {}, "outputs": {"original": {"type": "state"}}})
    with pytest.raises(ValidationError):
        DecisionProgram(plan, None, "unused")(state={"nested": [value]})


def test_mutating_results_cannot_change_graph_literals_or_later_runs():
    from test_graph import ref

    from rulesmith.graph import Plan
    from rulesmith.runtime import DecisionProgram

    plan = Plan.model_validate(
        {
            "nodes": {"data": {"type": "value", "value": {"items": [1]}}},
            "outputs": {
                "node": ref("data"),
                "literal": {"type": "literal", "value": {"items": [2]}},
            },
        }
    )
    original = plan.model_dump_json()
    program = DecisionProgram(plan, None, "unused")
    first = program(state="input")
    first.outputs["node"]["items"].append(3)
    first.outputs["literal"]["items"].append(4)
    assert plan.model_dump_json() == original
    assert program(state="input").outputs == {"node": {"items": [1]}, "literal": {"items": [2]}}


def read_plan(**read):
    return Plan.model_validate(
        {
            "nodes": {
                "ahead": {"type": "read", "path": ["rangefinder", 3], "bounds": [0, 2048]} | read,
                "move": {
                    "type": "threshold",
                    "source": "ahead",
                    "cutoff": 64,
                    "below": "TURN_LEFT",
                    "above": "MOVE_FORWARD",
                },
            },
            "outputs": {"label": {"type": "node", "node": "move"}},
        }
    )


def test_read_nodes_take_values_from_the_input_without_a_model_call():
    program = DecisionProgram(read_plan(), None, "none")
    assert program(state={"rangefinder": [9, 9, 9, 500, 9, 9, 9]}).label == "MOVE_FORWARD"
    assert program(state={"rangefinder": [900, 900, 900, 20, 900, 900, 900]}).label == "TURN_LEFT"


def test_a_missing_read_path_skips_the_node_and_the_decision_says_so():
    program = DecisionProgram(read_plan(), None, "none")
    result = program(state={"rangefinder": [1, 2]})
    assert result.undecided == "no_rule_applied" and result.label is None
    # A decision that was made says it was not undecided, so a caller tests one field.
    assert program(state={"rangefinder": [9, 9, 9, 500, 9, 9, 9]}).undecided is None


@pytest.mark.parametrize(
    "state", [{"rangefinder": [0, 0, 0, 9000]}, {"rangefinder": [0, 0, 0, "far"]}]
)
def test_values_outside_a_reads_declared_domain_skip_the_read(state):
    plan = read_plan().model_dump()
    plan["nodes"]["fallback"] = {"type": "value", "value": "TURN_LEFT"}
    plan["nodes"]["pick"] = {"type": "coalesce", "sources": ["move", "fallback"]}
    plan["outputs"]["label"] = {"type": "node", "node": "pick"}
    prediction = DecisionProgram(Plan.model_validate(plan), None, "none")(state=state)
    assert prediction.label == "TURN_LEFT"
    assert "ahead" in prediction.skipped


def test_categorical_reads_drive_switches():
    plan = Plan.model_validate(
        {
            "nodes": {
                "kind": {
                    "type": "read",
                    "path": ["objects", 0, "category"],
                    "categories": ["Monster", "Health"],
                },
                "act": {
                    "type": "switch",
                    "source": "kind",
                    "cases": {"Monster": "ATTACK", "Health": "MOVE_FORWARD"},
                },
            },
            "outputs": {"label": {"type": "node", "node": "act"}},
        }
    )
    program = DecisionProgram(plan, None, "none")
    assert program(state={"objects": [{"category": "Monster"}]}).label == "ATTACK"


@pytest.mark.parametrize("read", [{"bounds": None}, {"categories": ["a", "b"]}])
def test_reads_declare_exactly_one_domain(read):
    with pytest.raises(ValidationError):
        read_plan(**read)


def traced(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_trace_appends_one_record_per_decision_and_keeps_what_was_there(tmp_path):
    trace = tmp_path / "decisions.jsonl"
    trace.write_text('{"earlier": true}\n')
    plan = read_plan()
    program = DecisionProgram(plan, None, "none", trace=trace)
    program(state={"rangefinder": [9, 9, 9, 500, 9, 9, 9]})
    program(state={"rangefinder": [900, 900, 900, 20, 900, 900, 900]})
    earlier, first, second = traced(trace)
    assert earlier == {"earlier": True}
    assert [first["label"], second["label"]] == ["MOVE_FORWARD", "TURN_LEFT"]
    assert first["decided_by"] == "move" and first["undecided"] is None
    graph = hashlib.sha256(plan.model_dump_json().encode()).hexdigest()
    assert first["graph_sha256"] == second["graph_sha256"] == graph
    assert first["input_sha256"] != second["input_sha256"]


@given(st.dictionaries(st.text(max_size=5), st.integers(), max_size=6))
def test_an_input_is_traced_by_content_whatever_its_key_order(tmp_path_factory, state):
    trace = tmp_path_factory.mktemp("trace") / "decisions.jsonl"
    plan = Plan.model_validate(
        {"nodes": {}, "outputs": {"label": {"type": "literal", "value": "x"}}}
    )
    program = DecisionProgram(plan, None, "none", trace=trace)
    program(state=state)
    program(state=dict(reversed(list(state.items()))))
    first, second = traced(trace)
    assert first["input_sha256"] == second["input_sha256"]


def test_a_decision_the_graph_could_not_make_is_traced_as_undecided(tmp_path):
    trace = tmp_path / "decisions.jsonl"
    program = DecisionProgram(read_plan(), None, "none", trace=trace)
    program(state={"rangefinder": [1, 2]})
    program(state={"rangefinder": [9, 9, 9, 500, 9, 9, 9]})
    undecided, decided = traced(trace)
    assert undecided["label"] is None and undecided["decided_by"] is None
    assert undecided["undecided"] == "no_rule_applied"
    assert decided["label"] == "MOVE_FORWARD" and decided["undecided"] is None
    assert "error" not in undecided and "error" not in decided
