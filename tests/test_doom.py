"""Real ViZDoom episodes; the judge is the local protocol fixture."""

import json
import os
from pathlib import Path

import pytest
import vizdoom as vzd
from conftest import chat_reply
from pydantic import ValidationError
from test_cli import invoke, write
from test_graph import constant_plan

from decision_compose.doom import DoomEnvironment, DoomTask, Episodes
from decision_compose.graph import Plan
from decision_compose.optimize import SearchConfig, optimize_plan
from decision_compose.runtime import DecisionProgram

ACTIONS = ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"]
# Half-width of the aiming window, as a fraction of the half-screen.
AIM_TOLERANCE = 0.05


def doom_data(**changes):
    data = {
        "description": "Kill the monster in front of you quickly. The label is your next action.",
        "scenario": "basic",
        "labels": ACTIONS,
        "reward_range": [-700, 100],
        "tics_per_action": 4,
        "train": [1, 2],
        "validation": [3],
        "test": [4],
    }
    return data | changes


def aiming_graph():
    """A one-call policy; the fixture below plays the judge that answers it."""
    return {
        "nodes": {
            "action": {
                "type": "call",
                "question": {
                    "type": "choice",
                    "instructions": "Which action lines up and kills the monster?",
                    "criteria": {
                        "MOVE_LEFT": "The monster is left of center.",
                        "MOVE_RIGHT": "The monster is right of center.",
                        "ATTACK": "The monster is centered.",
                    },
                },
                "inputs": {"observation": {"type": "state"}},
            }
        },
        "outputs": {"label": {"type": "node", "node": "action"}},
    }


def aim(provider):
    def answer(request):
        objects = request["state"]["observation"]["objects"]
        x = next(o["x"] for o in objects if o["category"] == "Monster")
        choice = "ATTACK" if abs(x) <= AIM_TOLERANCE else "MOVE_LEFT" if x < 0 else "MOVE_RIGHT"
        provider.body["answers"] = {
            "action": {
                "type": "choice",
                "choice": choice,
                "probabilities": {a: float(a == choice) for a in ACTIONS},
                "confidence": 1.0,
            }
        }

    provider.on_request = answer


@pytest.fixture
def task():
    return DoomTask.model_validate(doom_data())


@pytest.fixture
def env(task):
    with DoomEnvironment(task) as env:
        yield env


def reference_reward(action, seed, tics, scratch):
    """Play one fixed action directly through ViZDoom, bypassing this project."""
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
    game.set_doom_config_path(str(scratch / "_vizdoom.ini"))
    game.set_window_visible(False)
    game.set_seed(seed)
    game.init()
    buttons = [b.name for b in game.get_available_buttons()]
    total = 0.0
    while not game.is_episode_finished():
        total += game.make_action([b == action for b in buttons], tics)
    game.close()
    return total


@pytest.mark.parametrize(
    "change",
    [
        {"validation": [1]},
        {"reward_range": [100, -700]},
        {"labels": ["ATTACK"]},
    ],
)
def test_invalid_doom_tasks_are_rejected(change):
    with pytest.raises(ValidationError):
        DoomTask.model_validate(doom_data(**change))


@pytest.mark.parametrize(
    "change,message",
    [({"scenario": "no_such_map"}, "scenario"), ({"labels": ["ATTACK", "JUMP"]}, "JUMP")],
)
def test_environment_rejects_unknown_scenarios_and_actions(change, message):
    with pytest.raises(ValueError, match=message):
        DoomEnvironment(DoomTask.model_validate(doom_data(**change)))


def test_observations_locate_the_monster_and_repeat_for_a_seed(env):
    first = env.reset(7)
    assert env.reset(7) == first
    assert env.reset(8) != first
    (monster,) = first["objects"]
    assert monster["category"] == "Monster"
    assert -1 <= monster["x"] <= 1 and 0 < monster["width"] < 1
    assert set(first["variables"]) == {"AMMO2"}
    json.dumps(first)


@pytest.mark.parametrize("action", ["ATTACK", "MOVE_LEFT"])
def test_episode_reward_matches_playing_vizdoom_directly(task, env, action, client, tmp_path):
    program = DecisionProgram(Plan.model_validate(constant_plan(action)), client, "fixture")
    outcome = Episodes(task, env)(program, 7)
    expected = reference_reward(action, 7, task.tics_per_action, tmp_path)
    assert outcome.record["reward"] == expected
    low, high = task.reward_range
    assert outcome.score == pytest.approx((expected - low) / (high - low))
    assert outcome.error is None


def test_reward_outside_the_declared_range_stops_rather_than_clipping(env, client):
    task = DoomTask.model_validate(doom_data(reward_range=[-100, 100]))
    program = DecisionProgram(Plan.model_validate(constant_plan("ATTACK")), client, "fixture")
    with pytest.raises(ValueError, match="reward_range"):
        Episodes(task, env)(program, 7)


def test_graph_failure_ends_the_episode_with_the_lowest_score(task, env, provider, client):
    graph = aiming_graph()
    graph["nodes"]["gate"] = {"type": "value", "value": 0}
    graph["nodes"]["action"]["when"] = {"type": "compare", "node": "gate", "op": "ge", "value": 1}
    outcome = Episodes(task, env)(DecisionProgram(Plan.model_validate(graph), client, "f"), 7)
    assert outcome.score == 0.0
    assert "label" in outcome.error
    assert provider.requests == []


def test_aiming_policy_kills_the_monster_through_the_sdk(task, env, provider, client):
    aim(provider)
    program = DecisionProgram(Plan.model_validate(aiming_graph()), client, "fixture")
    outcome = Episodes(task, env)(program, 7)
    assert outcome.record["reward"] > 0
    assert outcome.record["steps"] == len(provider.requests)
    assert "ATTACK" in outcome.record["actions"]
    assert "reward" in outcome.trace["Feedback"]


def test_real_gepa_improves_a_doom_policy(task, env, provider, client):
    aim(provider)
    initial = task.baseline_plan()
    proposed = Plan.model_validate(aiming_graph())
    plan, report = optimize_plan(
        task,
        initial,
        client,
        "fixture",
        lambda prompt: proposed.model_dump_json(),
        SearchConfig(max_metric_calls=6, reflection_batch=2),
        score=Episodes(task, env),
    )
    assert plan == proposed
    assert report["validation_score"] > report["seed_validation_score"]


def test_cli_optimizes_and_evaluates_a_doom_task(tmp_path, provider, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    aim(provider)
    provider.completion = chat_reply(graph=aiming_graph())
    task = write(tmp_path / "doom.json", doom_data())
    output = tmp_path / "run"
    common = ["--base-url", provider.url, "--model", "fixture", "--max-workers", 1]
    result = invoke(
        "optimize",
        task,
        "--output",
        output,
        "--reflection-model",
        "openai/fixture",
        "--reflection-base-url",
        provider.url + "/v1",
        "--max-metric-calls",
        6,
        "--reflection-batch",
        2,
        *common,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((output / "report.json").read_text())["validation_score"] > 0.5
    evaluation = tmp_path / "test.json"
    result = invoke("evaluate", task, output / "plan.json", "--output", evaluation, *common)
    assert result.returncode == 0, result.stderr
    report = json.loads(evaluation.read_text())
    assert [row["seed"] for row in report["examples"]] == [4]
    assert report["examples"][0]["reward"] > 0


EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples/doom").glob("*.json"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.stem)
def test_every_doom_example_plays_within_its_reward_range(path, client):
    task = DoomTask.model_validate_json(path.read_text())
    with DoomEnvironment(task) as env:
        for label in task.labels:
            program = DecisionProgram(Plan.constant(label), client, "fixture")
            assert 0 <= Episodes(task, env)(program, task.test[0]).score <= 1


def test_combined_labels_press_every_listed_button_together(client, tmp_path):
    task = DoomTask.model_validate(doom_data(labels=["MOVE_LEFT", "MOVE_RIGHT+ATTACK"]))
    with DoomEnvironment(task) as env:
        env.reset(7)
        reward, _ = env.step("MOVE_RIGHT+ATTACK")
        assert env.game.get_last_action() == [0.0, 1.0, 1.0]


def test_combined_labels_must_name_available_buttons():
    with pytest.raises(ValueError, match="JUMP"):
        DoomEnvironment(DoomTask.model_validate(doom_data(labels=["ATTACK", "MOVE_LEFT+JUMP"])))


def test_episode_timeout_caps_an_episode(client):
    task = DoomTask.model_validate(doom_data(episode_timeout=40))
    program = DecisionProgram(Plan.model_validate(constant_plan("MOVE_LEFT")), client, "fixture")
    with DoomEnvironment(task) as env:
        outcome = Episodes(task, env)(program, 7)
    assert outcome.record["steps"] == 40 // task.tics_per_action


def test_scenarios_without_game_variables_observe_an_empty_mapping():
    task = DoomTask.model_validate(doom_data(scenario="simpler_basic"))
    with DoomEnvironment(task) as env:
        assert env.reset(7)["variables"] == {}
