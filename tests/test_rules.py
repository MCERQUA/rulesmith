"""The rule surface a proposer writes, and the graph it compiles to."""

import json
import re
from pathlib import Path

import pytest
from test_graph import binary_plan, call, constant_plan, ref, task_data

from decision_compose.graph import Plan, Task
from decision_compose.rules import RuleError, compile_rules, render_rules

GRAPHS = sorted((Path(__file__).resolve().parents[1] / "examples/doom/graphs").glob("*.json"))


@pytest.mark.parametrize("path", GRAPHS, ids=lambda path: path.stem)
def test_every_repository_graph_survives_the_round_trip(path):
    plan = Plan.model_validate_json(path.read_text())
    rules = render_rules(plan)
    # Compiling picks its own node names, so the text is what must come back unchanged.
    assert render_rules(compile_rules(rules)) == rules


def test_rules_read_as_the_decisions_they_make():
    plan = Plan.model_validate_json((GRAPHS[0].parent / "duel-hunter.json").read_text())
    text = render_rules(plan)
    # One line per rule, in the order the graph tries them.
    assert "TURN_LEFT" in text and "compass.2.bearing >= 45" in text
    assert text.count("\n") + 1 < len(plan.model_dump_json()) // 200


def test_a_rule_may_ask_the_judge_and_fall_through_when_it_declines():
    text = "\n".join(
        [
            "read variables.HEALTH in [0, 200]",
            'when variables.HEALTH >= 25: ask "Fight or run?"'
            ' [ATTACK: "Opponent is lined up."] [MOVE_BACKWARD: "Opponent has the angle."]',
            "always: MOVE_FORWARD",
        ]
    )
    plan = compile_rules(text)
    kinds = [node.type for node in plan.nodes.values()]
    assert "call" in kinds and "value" in kinds
    assert render_rules(compile_rules(render_rules(plan))) == render_rules(plan)


def test_a_constant_policy_is_one_line():
    plan = Plan.model_validate(constant_plan("review"))
    assert render_rules(plan).strip() == "always: review"
    assert compile_rules("always: review") == plan


def test_a_rule_naming_an_unknown_reading_says_which_line():
    with pytest.raises(RuleError, match="line 2"):
        compile_rules("always: review\nwhen nonsense.path >= 1: accept")


def test_graphs_beyond_rules_stay_in_json():
    # Merges join several judgements into one value; rules have no way to say that.
    merged = {
        "nodes": {
            "left": call(),
            "right": call(),
            "both": {"type": "merge", "inputs": {"a": ref("left"), "b": ref("right")}},
        },
        "outputs": {"label": ref("both")},
    }
    with pytest.raises(RuleError, match="merge"):
        render_rules(Plan.model_validate(merged))
    # Several outputs are beyond rules too, and say so.
    pair = json.loads((Path(__file__).resolve().parents[1] / "examples/dag.json").read_text())
    with pytest.raises(RuleError, match="one label"):
        render_rules(Plan.model_validate(pair))


def test_the_compiler_accepts_a_whole_graph_as_json():
    # A proposal that needs the full language can still answer with it.
    graph = binary_plan()
    assert compile_rules(json.dumps(graph)) == Plan.model_validate(graph)


def test_rules_only_name_labels_the_task_allows():
    task = Task.model_validate(task_data())
    plan = compile_rules("always: review")
    plan.check(task.labels, 16, 8)


EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples/rules").glob("*.rules"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.stem)
def test_every_example_program_compiles_and_survives_the_round_trip(path):
    plan = compile_rules(path.read_text())
    assert render_rules(compile_rules(render_rules(plan))) == render_rules(plan)
    kinds = {node.question.type for node in plan.nodes.values() if node.type == "call"}
    # Each example shows one kind of judgement doing the work that kind is for, and the last
    # keeps its judgement behind readings, asked only when they cannot settle the case.
    shown = {"routing": "choice", "triage": "score", "grounding": "noul", "gating": "noul"}
    assert kinds == {shown[path.stem]}
    if path.stem == "gating":
        assert all(node.when is not None for node in plan.nodes.values() if node.type == "call")


def test_the_proposer_is_told_every_kind_of_judgement_it_may_ask():
    # Search can only use what its instructions describe, and the language has three kinds.
    from decision_compose.graph import Task
    from decision_compose.optimize import SearchConfig, proposer_prompt

    prompt = proposer_prompt(Task.model_validate(task_data()), SearchConfig())
    asked = re.findall(r"'(ask [^']+)'", prompt)
    kinds = set()
    for form in asked:
        name = form.split()[1]
        # A judgement nothing asks about is pruned, so give each one a rule that uses it.
        plan = compile_rules(f"{form}\nwhen {name}: review\nalways: accept")
        kinds |= {node.question.type for node in plan.nodes.values() if node.type == "call"}
    assert kinds == {"choice", "score", "noul"}
