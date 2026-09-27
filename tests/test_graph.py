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
