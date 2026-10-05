"""Named judges: a judgment may be asked of another judge than the graph's own, so a graph can
escalate the cases its small judge is unsure of to a larger one, and pay for that only then."""

import json

import pytest
from test_cli import invoke, write
from test_round_trip import round_trips
from typesafe_sdk import RetryPolicy, TypeSafeClient

from rulesmith.graph import Plan, Task
from rulesmith.optimize import GraphAdapter, SearchConfig
from rulesmith.rules import compile_rules, parts
from rulesmith.runtime import DecisionProgram, Judge, Models

TEAMS = '[billing: "Charges."] [technical: "Bugs."]'
# Sure answers settle by rule; anything else goes to the larger judge.
ESCALATING = (
    f'ask team = "Which team?" {TEAMS}\n'
    'when team.confidence >= 0.7 and team == "billing": billing\n'
    'when team.confidence >= 0.7 and team == "technical": technical\n'
    f'always: ask "Which team?" {TEAMS} using large'
)


def team(choice: str, sure: float) -> dict:
    other = "technical" if choice == "billing" else "billing"
    return {
        "type": "choice",
        "choice": choice,
        "probabilities": {choice: sure, other: 1 - sure},
        "confidence": sure,
    }


def asked(server) -> list[dict]:
    return [request for path, request in server.requests if path == "/v1/systemone"]


@pytest.fixture
def large(larger):
    with TypeSafeClient(
        base_url=larger.url, api_key="test-placeholder", retry=RetryPolicy(max_retries=0)
    ) as client:
        yield Judge(client, "large-model")


def test_a_judgment_names_its_judge_in_rules_and_reads_back_the_same():
    plan = round_trips(
        f'read amount in [0, 100]\nask team = "Which team?" {TEAMS} given [amount] using large '
        "when amount >= 5\n"
        "when team.confidence >= 0.7: sure\n"
        f'always: ask "Which team?" {TEAMS} using large'
    )
    assert {node.judge for node in plan.nodes.values() if node.type == "call"} == {"large"}
    assert plan.judges() == {"large"}


def test_the_judges_a_map_needs_include_its_units():
    unit = compile_rules(f'always: ask "Which team?" {TEAMS} using large').model_dump()
    plan = Plan.model_validate(
        {
            "nodes": {"route": {"type": "graph", "plan": unit}},
            "outputs": {"label": {"type": "node", "node": "route"}},
        }
    )
    assert plan.judges() == {"large"}


def test_the_larger_judge_is_asked_only_when_the_small_one_is_unsure(
    provider, client, larger, large
):
    provider.body["answers"] = {"team": team("billing", 0.95)}
    larger.body = {
        **larger.body,
        "model": "large-model",
        "answers": {"rule_3": team("technical", 0.9)},
    }
    program = DecisionProgram(
        compile_rules(ESCALATING), client, "small", models=Models({"large": large})
    )
    assert program(state="Refund my charge.").label == "billing"
    assert asked(larger) == []

    provider.body["answers"] = {"team": team("billing", 0.55)}
    decided = program(state="It crashes, or maybe I was charged?")
    assert decided.label == "technical"
    (escalated,) = asked(larger)
    assert escalated["model"] == "large-model" and set(escalated["questions"]) == {"rule_3"}
    # The scorecard says which judge each call went to, so the price of escalating is visible.
    assert decided.calls["rule_3"]["judge"] == "large"
    assert "judge" not in decided.calls["team"]


def test_a_graph_cannot_run_without_the_judges_it_names(client):
    with pytest.raises(ValueError, match="large"):
        DecisionProgram(compile_rules(ESCALATING), client, "small")


def test_search_prices_each_judge_by_its_own_cost(provider, client, larger, large):
    task = Task.model_validate(
        {
            "description": "Route tickets.",
            "labels": ["billing", "technical"],
            "train": [{"id": "a", "state": "It crashes.", "label": "technical"}],
            "validation": [{"id": "b", "state": "x", "label": "billing"}],
            "test": [{"id": "c", "state": "y", "label": "billing"}],
        }
    )
    provider.body["answers"] = {"team": team("technical", 0.55)}
    larger.body = {
        **larger.body,
        "model": "large-model",
        "answers": {"rule_3": team("technical", 0.9)},
    }
    config = SearchConfig(call_cost=0.01, call_costs={"large": 0.2})
    adapter = GraphAdapter(task, client, "small", config, models=Models({"large": large}))
    batch = adapter.evaluate(task.train, parts(compile_rules(ESCALATING)), capture_traces=True)
    # Right, less one small question at 0.01 and one large one at 0.2.
    assert batch.scores == [pytest.approx(1 - 0.01 - 0.2)]
    assert "large" in batch.trajectories[0]["Feedback"]


def test_the_command_line_sends_each_judgment_to_its_judge(tmp_path, provider, larger):
    provider.body["answers"] = {"team": team("billing", 0.55)}
    larger.body = {
        **larger.body,
        "model": "large-model",
        "answers": {"rule_3": team("technical", 0.9)},
    }
    rules = tmp_path / "graph.rules"
    rules.write_text(ESCALATING + "\n")
    state = write(tmp_path / "state.json", "It crashes, or maybe I was charged?")
    common = ["--backend", "ruling", "--base-url", provider.url, "--max-workers", 1]
    result = invoke(
        "run", rules, state, *common, "--judge", f"large=ruling:large-model@{larger.url}"
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["label"] == "technical"
    assert asked(larger)[0]["model"] == "large-model"

    missing = invoke("run", rules, state, *common)
    assert missing.returncode != 0
    assert "--judge large=" in missing.stderr


def test_calibrating_asks_each_judgment_of_its_own_judge(tmp_path, provider, larger):
    provider.body["answers"] = {"team": team("billing", 0.55)}
    larger.body = {
        **larger.body,
        "model": "large-model",
        "answers": {"rule_3": team("technical", 0.9)},
    }
    task = write(
        tmp_path / "task.json",
        {
            "description": "Route tickets.",
            "labels": ["billing", "technical"],
            "train": [{"id": "a", "state": "It crashes.", "label": "technical"}],
            "validation": [{"id": "b", "state": "It broke.", "label": "technical"}],
            "test": [{"id": "c", "state": "y", "label": "billing"}],
        },
    )
    rules = tmp_path / "graph.rules"
    rules.write_text(ESCALATING + "\n")
    output = tmp_path / "calibrated.rules"
    result = invoke(
        "calibrate",
        task,
        rules,
        "--output",
        output,
        "--backend",
        "ruling",
        "--base-url",
        provider.url,
        "--max-workers",
        1,
        "--judge",
        f"large=ruling:large-model@{larger.url}",
    )
    assert result.returncode == 0, result.stderr
    assert "using large" in output.read_text()
    assert asked(larger)


def test_search_is_told_which_judges_it_may_escalate_to_and_their_price():
    task = Task.model_validate(
        {
            "description": "Route tickets.",
            "labels": ["billing", "technical"],
            "train": [{"id": "a", "state": "x", "label": "billing"}],
            "validation": [{"id": "b", "state": "z", "label": "billing"}],
            "test": [{"id": "c", "state": "y", "label": "billing"}],
        }
    )
    from rulesmith.optimize import proposer_prompt

    config = SearchConfig(call_cost=0.01, call_costs={"large": 0.2})
    assert "'using NAME'" not in proposer_prompt(task, config)
    told = proposer_prompt(task, config, ["large"])
    assert "'using NAME'" in told and "large (0.2 a question)" in told


def confident_traffic(provider, larger, count=24):
    """A small judge right exactly when at least 0.6 sure, and a large one always right: the
    state carries each example's sureness and label."""

    def small(request):
        _, sure, label = request["state"]["observation"].split("|")
        sure = float(sure)
        wrong = "technical" if label == "billing" else "billing"
        provider.answers = {"team": team(label if sure >= 0.6 else wrong, sure)}

    def big(request):
        larger.answers = {"rule_3": team(request["state"]["observation"].split("|")[2], 0.9)}

    provider.on_request, larger.on_request = small, big
    larger.body = {**larger.body, "model": "large-model"}
    rows = [
        {
            "id": f"t{i}",
            "state": f"t{i}|{0.3 + 0.65 * i / count:.3f}|{'billing' if i % 2 else 'technical'}",
        }
        for i in range(count)
    ]
    rows = [row | {"label": row["state"].split("|")[2]} for row in rows]
    return Task.model_validate(
        {
            "description": "Route tickets.",
            "labels": ["billing", "technical"],
            "train": rows,
            "validation": [{"id": "v", "state": "v|0.9|billing", "label": "billing"}],
            "test": [{"id": "e", "state": "e|0.9|billing", "label": "billing"}],
        }
    )


def test_a_priced_judge_moves_the_escalation_line_down_to_where_it_pays(
    provider, client, larger, large
):
    from rulesmith.calibrate import fit_cutoffs
    from rulesmith.runtime import Prices

    task = confident_traffic(provider, larger)
    plan = compile_rules(ESCALATING.replace("0.7", "0.95"))
    models = Models({"large": large})
    # Free, escalating more is never worse, so the line stays where it was written.
    unpriced, _ = fit_cutoffs(plan, task, client, "small", models)
    assert unpriced == plan
    # Priced, every question the large judge need not answer is worth keeping.
    priced, report = fit_cutoffs(plan, task, client, "small", models, Prices(judges={"large": 0.2}))
    lines = [row["to"] for row in report if row["kept"]]
    assert lines and all(0.5 <= line <= 0.65 for line in lines)
    # It still escalates every case the small judge gets wrong.
    from rulesmith.calibrate import correct

    assert all(correct(priced, task.train, client, "small", models))


def test_calibrate_takes_the_prices_search_takes(tmp_path, provider, larger):
    task = confident_traffic(provider, larger)
    path = write(tmp_path / "task.json", task.model_dump(mode="json", exclude_none=True))
    rules = tmp_path / "graph.rules"
    rules.write_text(ESCALATING.replace("0.7", "0.95") + "\n")
    output = tmp_path / "calibrated.rules"
    result = invoke(
        "calibrate",
        path,
        rules,
        "--output",
        output,
        "--backend",
        "ruling",
        "--base-url",
        provider.url,
        "--max-workers",
        1,
        "--judge",
        f"large=ruling:large-model@{larger.url}",
        "--judge-cost",
        "large=0.2",
    )
    assert result.returncode == 0, result.stderr
    assert "0.95" not in output.read_text()
