import json

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from rulesmith.graph import Plan, Task, references
from rulesmith.optimize import assemble


def ref(node, **kwargs):
    return {"type": "node", "node": node, **kwargs}


def call(instructions="Is the request eligible?", **inputs):
    return {
        "type": "call",
        "question": {"type": "noul", "instructions": instructions},
        "inputs": inputs or {"state": {"type": "state"}},
    }


def binary_plan(cutoff=0.5):
    return {
        "nodes": {
            "eligible": call(),
            "decision": {
                "type": "threshold",
                "source": "eligible",
                "cutoff": cutoff,
                "below": "review",
                "above": "accept",
            },
        },
        "outputs": {"label": ref("decision")},
    }


def constant_plan(label="review"):
    return {"nodes": {}, "outputs": {"label": {"type": "literal", "value": label}}}


@given(
    st.recursive(
        st.none() | st.booleans() | st.integers() | st.text(),
        lambda values: st.lists(values) | st.dictionaries(st.text(), values),
    )
)
def test_literal_payloads_never_create_dependencies(payload):
    value = {"node": "unused", "source": "unused", "sources": payload}
    for kind in ("value", "literal"):
        assert references({"type": kind, "value": value}) == set()
    graph = {
        "nodes": {"data": {"type": "value", "value": value}},
        "outputs": {"data": ref("data"), "literal": {"type": "literal", "value": value}},
    }
    assert assemble({"graph": json.dumps(graph)}) == Plan.model_validate(graph)


@pytest.mark.parametrize(
    "part",
    [
        {"nodes": []},
        {"nodes": 7},
        {"nodes": {"bad": []}},
        {"nodes": {}, "outputs": 7},
        {"nodes": {}, "outputs": []},
    ],
)
def test_malformed_fragment_shapes_raise_validation_errors(part):
    with pytest.raises(ValueError):
        assemble({"policy": json.dumps(part)})


@given(st.lists(st.sets(st.integers(min_value=0, max_value=20)), max_size=20))
def test_arbitrary_dag_order_respects_every_edge_and_survives_serialization(edges):
    nodes = {}
    for i, candidates in enumerate(edges):
        parents = sorted(n for n in candidates if n < i)
        nodes[str(i)] = (
            {"type": "merge", "inputs": {str(n): ref(str(n)) for n in parents}}
            if parents
            else {"type": "value", "value": i}
        )
    outputs = {key: ref(key) for key in nodes} or {"constant": {"type": "literal", "value": 0}}
    plan = Plan.model_validate({"nodes": nodes, "outputs": outputs})
    restored = Plan.model_validate_json(plan.model_dump_json())
    assert restored == plan
    order = plan.order()
    assert len(order) == len(set(order)) == len(nodes)
    positions = {key: i for i, key in enumerate(order)}
    for child, parents in plan.dependencies().items():
        assert all(positions[parent] < positions[child] for parent in parents)
    data = plan.model_dump()
    data["nodes"] = dict(reversed(list(data["nodes"].items())))
    assert Plan.model_validate(data).order() == order


@given(st.integers(min_value=1, max_value=25))
def test_cycles_of_every_length_are_rejected(length):
    nodes = {
        str(i): {"type": "merge", "inputs": {"parent": ref(str((i + 1) % length))}}
        for i in range(length)
    }
    with pytest.raises(ValidationError, match="cycle"):
        Plan.model_validate({"nodes": nodes, "outputs": {"result": ref("0")}})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["nodes"]["decision"].update(source="missing"),
        lambda p: p["nodes"]["decision"].update(cutoff=1.1),
        lambda p: p["nodes"]["decision"].update(cutoff=float("nan")),
        lambda p: p["nodes"]["decision"].update(typo=True),
        lambda p: p["nodes"]["eligible"]["question"].update(instructions="   "),
        lambda p: p["nodes"].update(unused=call()),
        lambda p: p["nodes"]["eligible"]["question"].update(
            type="choice", criteria={"a": "A", "b": "B"}
        ),
        lambda p: p["outputs"].update(label=ref("missing")),
        lambda p: p["outputs"].update(label=ref("decision", view="answer")),
        lambda p: p["nodes"]["eligible"].update(inputs={"x": ref("missing", optional=True)}),
        lambda p: p["nodes"]["eligible"].update(when={"type": "present", "node": "decision"}),
        lambda p: p["nodes"]["eligible"].update(
            when={"type": "compare", "node": "missing", "op": "eq", "value": "x"}
        ),
    ],
)
def test_invalid_graphs_are_rejected_before_execution(mutation):
    data = binary_plan()
    mutation(data)
    with pytest.raises(ValidationError):
        Plan.model_validate(data)


def test_switch_requires_exact_coverage_and_conditions_are_typed():
    data = {
        "nodes": {
            "q": {
                "type": "call",
                "question": {
                    "type": "choice",
                    "instructions": "Choose.",
                    "criteria": {"a": "A", "b": "B"},
                },
                "inputs": {"state": {"type": "state"}},
            },
            "result": {"type": "switch", "source": "q", "cases": {"a": "ok"}},
        },
        "outputs": {"label": ref("result")},
    }
    with pytest.raises(ValidationError, match="cases"):
        Plan.model_validate(data)
    data["nodes"]["result"]["cases"]["b"] = "review"
    Plan.model_validate(data)
    data["nodes"]["result"]["when"] = {"type": "compare", "node": "q", "op": "ge", "value": 0.5}
    with pytest.raises(ValidationError, match="numeric"):
        Plan.model_validate(data)


@pytest.mark.parametrize(
    "labels,nodes,depth",
    [
        (["other"], 4, 4),
        (["accept", "review"], 1, 4),
        (["accept", "review"], 4, 1),
    ],
)
def test_search_constraints_are_enforced(labels, nodes, depth):
    with pytest.raises(ValueError):
        Plan.model_validate(binary_plan()).check(labels, nodes, depth)


def test_generic_graph_outputs_and_task_label_contract_are_separate():
    data = {
        "nodes": {"object": {"type": "value", "value": {"x": True}}},
        "outputs": {"object": ref("object"), "state": {"type": "state"}},
    }
    plan = Plan.model_validate(data)
    with pytest.raises(ValueError, match="label"):
        plan.check(["accept", "review"], 4, 4)
    data["outputs"]["label"] = ref("object")
    with pytest.raises(ValueError, match="label"):
        Plan.model_validate(data).check(["accept", "review"], 4, 4)
    Plan.model_validate(constant_plan()).check(["review"], 0, 0)


def test_coalesce_label_domain_covers_all_possible_branches():
    data = {
        "nodes": {
            "a": {"type": "value", "value": "accept"},
            "b": {"type": "value", "value": "outside"},
            "joined": {"type": "coalesce", "sources": ["a", "b"]},
        },
        "outputs": {"label": ref("joined")},
    }
    with pytest.raises(ValueError, match="label"):
        Plan.model_validate(data).check(["accept", "review"], 4, 4)


def task_data():
    return {
        "description": "Classify requests.",
        "labels": ["accept", "review"],
        "train": [{"id": "a", "state": "request A", "label": "accept"}],
        "validation": [{"id": "b", "state": "request B", "label": "review"}],
        "test": [{"id": "c", "state": "request C", "label": "accept"}],
    }


@pytest.mark.parametrize(
    "change",
    [
        lambda t: t["test"][0].update(id="a"),
        lambda t: t["test"][0].update(state="request A"),
        lambda t: t["test"][0].update(label="invented"),
        lambda t: t.update(validation=[]),
        lambda t: t.update(labels=["accept", "accept"]),
        lambda t: t.update(descriptions={"invented": "A label the task does not have."}),
    ],
)
def test_task_rejects_leakage_and_invalid_targets(change):
    data = task_data()
    change(data)
    with pytest.raises(ValidationError):
        Task.model_validate(data)


def test_task_detects_duplicate_json_states_regardless_of_key_order():
    data = task_data()
    data["train"][0]["state"] = {"a": 1, "b": 2}
    data["test"][0]["state"] = {"b": 2, "a": 1}
    with pytest.raises(ValidationError, match="state"):
        Task.model_validate_json(json.dumps(data))


def test_baseline_seed_predicts_the_majority_training_label_without_calls():
    data = task_data()
    data["train"] += [
        {"id": "d", "state": "request D", "label": "review"},
        {"id": "e", "state": "request E", "label": "review"},
    ]
    task = Task.model_validate(data)
    plan = task.baseline_plan()
    plan.check(task.labels, max_nodes=1, max_depth=1)
    assert plan.nodes == {}
    assert plan.outputs["label"].value == "review"


def unanswered(rules):
    from rulesmith.rules import compile_rules

    return compile_rules(rules).unanswered()


@pytest.mark.parametrize(
    "rules,expected",
    [
        ("when a >= 5: high\nalways: low", []),
        ("when a >= 5: high\nwhen a < 5: low", ["label"]),
        ("always: a >= 5 ? high : low", ["label"]),
        ('ask risky = "Is this risky?"\nwhen risky >= 0.5: high\nalways: low', []),
        ('ask risky = "Is this risky?"\nalways: risky >= 0.5 ? high : low', []),
        ('ask risky = "Is this risky?" when a >= 5\nalways: risky >= 0.5 ? high : low', ["label"]),
        ("when a >= 5: high\nwhen not (a >= 5): low", []),
        ("when a >= 5: high\nwhen not (a >= 5 or a < 2): low", ["label"]),
    ],
)
def test_a_graph_names_the_outputs_some_input_could_leave_unanswered(rules, expected):
    assert unanswered("read a in [0, 10]\n" + rules) == expected


def test_a_unit_that_can_go_unanswered_leaves_its_map_unanswered():
    from rulesmith.rules import compile_rules

    unit = compile_rules("read a in [0, 10]\nwhen a >= 5: high").model_dump()
    plan = Plan.model_validate(
        {
            "nodes": {"unit": {"type": "graph", "plan": unit}},
            "outputs": {"label": {"type": "node", "node": "unit"}},
        }
    )
    assert plan.unanswered() == ["label"]


CONDITIONS = st.sampled_from(
    [
        "a >= 5",
        "b < 3",
        "a",
        "not b",
        "a >= 2 and b",
        "a < 8 or b >= 1",
        "not (a >= 5 or b < 3)",
        "not (a >= 5 and b)",
        "not a or a >= 5",
    ]
)
ACTIONS = st.sampled_from(["high", "low", "a >= 5 ? high : low", "b >= 4 ? high : low"])


@given(
    st.lists(st.tuples(st.none() | CONDITIONS, ACTIONS), min_size=1, max_size=4),
    st.none() | st.integers(0, 10),
    st.none() | st.integers(0, 10),
)
def test_a_graph_said_to_always_answer_answers_every_input(rules, a, b):
    from rulesmith.rules import compile_rules
    from rulesmith.runtime import DecisionProgram

    lines = [f"when {cond}: {act}" if cond else f"always: {act}" for cond, act in rules]
    plan = compile_rules("read a in [0, 10]\nread b in [0, 10]\n" + "\n".join(lines))
    state = {key: value for key, value in {"a": a, "b": b}.items() if value is not None}
    if not plan.unanswered():
        assert DecisionProgram(plan, None, "none")(state=state).label in ("high", "low")


def split_plan(cases):
    def specialist(team):
        return {
            "type": "value",
            "value": f"{team} desk",
            "when": {"type": "compare", "node": "team", "op": "eq", "value": team},
        }

    return Plan.model_validate(
        {
            "nodes": {
                "team": {
                    "type": "call",
                    "question": {
                        "type": "choice",
                        "instructions": "Which team?",
                        "criteria": {
                            "billing": "Charges.",
                            "technical": "Errors.",
                            "sales": "Quotes.",
                        },
                    },
                    "inputs": {"ticket": {"type": "state"}},
                },
                **{f"{team}_desk": specialist(team) for team in cases},
                "desk": {"type": "coalesce", "sources": [f"{team}_desk" for team in cases]},
            },
            "outputs": {"label": {"type": "node", "node": "desk"}},
        }
    )


@pytest.mark.parametrize(
    "cases,expected",
    [(["billing", "technical", "sales"], []), (["billing", "technical"], ["label"])],
)
def test_a_join_of_one_case_per_answer_a_choice_can_give_always_answers(cases, expected):
    assert split_plan(cases).unanswered() == expected
