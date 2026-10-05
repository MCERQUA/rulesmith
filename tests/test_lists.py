"""Lists: a rule can test whether any or all of an input's items meet a condition, or count the
ones that do, since real inputs carry line items, messages and past orders, not one value each."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from test_round_trip import round_trips

from rulesmith.diagram import mermaid
from rulesmith.rules import RuleError, compile_rules
from rulesmith.runtime import DecisionProgram

ITEMS = "read order.items each {price in [0, 100000], status one of [paid, open]}\n"
ORDERS = (
    ITEMS + "let big = count of order.items where price > 500\n"
    'when any order.items has (price > 500 and status == "open"): review\n'
    "when big >= 2: review\n"
    'when all order.items have status == "paid": ship\n'
    "always: hold"
)


def decide(rules: str, items) -> str:
    return DecisionProgram(compile_rules(rules), None, "none")(
        state={"order": {"items": items}}
    ).label


def test_quantifiers_read_back_as_written():
    plan = round_trips(ORDERS)
    assert {node.type for node in plan.nodes.values()} >= {"read", "quantify"}
    assert "any order.items" in mermaid(plan)


ITEM = st.one_of(
    st.fixed_dictionaries(
        {},
        optional={
            "price": st.one_of(st.integers(-10, 2000), st.text(max_size=2)),
            "status": st.sampled_from(["paid", "open", "lost"]),
        },
    ),
    st.integers(),
    st.text(max_size=3),
)


@given(st.lists(ITEM, max_size=6))
def test_any_all_and_count_agree_with_the_items(items):
    def price(item):
        value = item.get("price") if isinstance(item, dict) else None
        # A field outside its declared domain is absent, as for any reading.
        return value if isinstance(value, int) and 0 <= value <= 100000 else None

    over = [price(item) is not None and price(item) > 500 for item in items]
    rules = ITEMS + "let big = count of order.items where price > 500\n"
    assert decide(rules + "when any order.items has price > 500: yes\nalways: no", items) == (
        "yes" if any(over) else "no"
    )
    assert decide(rules + "when all order.items have price > 500: yes\nalways: no", items) == (
        "yes" if all(over) else "no"
    )
    assert decide(rules + "when big >= 2: yes\nalways: no", items) == (
        "yes" if sum(over) >= 2 else "no"
    )


def test_an_input_without_the_list_falls_through():
    assert DecisionProgram(compile_rules(ORDERS), None, "none")(state={"order": {}}).label == "hold"


@pytest.mark.parametrize(
    "condition, says",
    [
        ("any order.items has weight > 5", "weight"),
        ('any order.items has price == "paid"', "price"),
        ("any order.items has status > 5", "numeric"),
    ],
)
def test_a_condition_on_items_tests_only_the_fields_they_declare(condition, says):
    with pytest.raises(RuleError, match=says):
        compile_rules(f"{ITEMS}when {condition}: review\nalways: hold")


def test_a_quantifier_needs_its_list_declared():
    with pytest.raises(RuleError, match="order.items"):
        compile_rules("when any order.items has price > 500: review\nalways: hold")


def test_search_is_taught_lists_only_where_the_inputs_hold_them():
    from rulesmith.graph import Task
    from rulesmith.optimize import SearchConfig, proposer_prompt

    def task(states):
        return Task.model_validate(
            {
                "description": "Review orders.",
                "labels": ["review", "ship"],
                "train": [
                    {"id": f"t{i}", "state": s, "label": "ship"} for i, s in enumerate(states)
                ],
                "validation": [{"id": "v", "state": {"order": {"total": 9}}, "label": "ship"}],
                "test": [{"id": "e", "state": {"order": {"total": 8}}, "label": "ship"}],
            }
        )

    listed = task([{"order": {"items": [{"price": 5, "status": "paid"}]}}])
    told = proposer_prompt(listed, SearchConfig())
    assert "order.items (price, status)" in told and "any order.items has" in told
    assert "each {" not in proposer_prompt(task([{"order": {"total": 5}}]), SearchConfig())
