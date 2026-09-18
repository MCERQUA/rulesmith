"""Explicitly opt-in tests against the actual hosted services."""

import os
from pathlib import Path

import dspy
import pytest
from typesafe_sdk import RetryPolicy, TypeSafeClient

from decision_compose.graph import Plan, Task
from decision_compose.optimize import SearchConfig, optimize_plan
from decision_compose.runtime import DecisionProgram

pytestmark = pytest.mark.live


@pytest.fixture
def live_client():
    if not os.environ.get("TYPESAFE_API_KEY"):
        pytest.fail("--live requires TYPESAFE_API_KEY")
    with TypeSafeClient(retry=RetryPolicy(max_retries=0)) as client:
        yield client


def test_live_jev_returns_all_three_primitives(live_client, mixed_plan):
    model = os.environ.get("DECISION_COMPOSE_MODEL", "jev-latest")
    result = DecisionProgram(mixed_plan, live_client, model)(
        state="My card was charged twice. Please refund the duplicate charge immediately."
    )
    assert result.label in {"accept", "review"}
    assert set(result.answers) == {"team", "urgency", "refund"}
    assert result.model


def test_live_gepa_returns_a_valid_program(live_client):
    reflection_model = os.environ.get("DECISION_COMPOSE_REFLECTION_MODEL")
    if not reflection_model:
        pytest.fail("--live requires DECISION_COMPOSE_REFLECTION_MODEL and its provider credential")
    root = Path(__file__).resolve().parents[1]
    task = Task.model_validate_json((root / "examples/task.json").read_text())
    initial = Plan.model_validate(
        {
            "questions": {},
            "root": {"type": "return", "label": "sales"},
        }
    )
    lm = dspy.LM(reflection_model, temperature=1, max_tokens=4096, num_retries=0, cache=False)

    def reflect(prompt):
        return lm(messages=prompt)[0] if isinstance(prompt, list) else lm(prompt)[0]

    best, report = optimize_plan(
        task,
        initial,
        live_client,
        os.environ.get("DECISION_COMPOSE_MODEL", "jev-latest"),
        reflect,
        SearchConfig(max_metric_calls=12),
    )
    best.check(task.labels, max_questions=8, max_depth=6)
    assert 0 <= report["validation_accuracy"] <= 1
    assert report["candidates"] >= 2
