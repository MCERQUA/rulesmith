"""Full-level observations and rewards, checked against Freedoom's real E1M1 map."""

import math

import pytest
import vizdoom as vzd
from pydantic import ValidationError

from decision_compose.doom import DoomEnvironment, DoomTask, Episodes
from decision_compose.graph import Plan
from decision_compose.runtime import DecisionProgram

TARGETS = ["DoomImp", "ShotgunGuy", "Zombieman", "Demon", "Medikit", "BlueCard"]


def level(**changes):
    data = {
        "description": "Clear Freedoom E1M1.",
        "scenario": "freedoom1",
        "map": "E1M1",
        "labels": ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "ATTACK", "USE"],
        "reward_range": [-10000, 10000],
        "episode_timeout": 120,
        "observe": ["pose", "rangefinder", "compass", "memory"],
        "compass_targets": TARGETS,
        "train": [1],
        "validation": [2],
        "test": [3],
    }
    return DoomTask.model_validate(data | changes)


@pytest.fixture
def env():
    with DoomEnvironment(level()) as env:
        yield env


def play(task, action, seed=1):
    with DoomEnvironment(task) as env:
        outcome = Episodes(task, env)(DecisionProgram(Plan.constant(action), None, "none"), seed)
        return outcome, env


def test_level_tasks_load_the_requested_map():
    starts = {}
    for name in ("E1M1", "E1M2"):
        with DoomEnvironment(level(map=name)) as env:
            assert env.game.get_doom_map() == name
            starts[name] = env.reset(1)["pose"]
    assert starts["E1M1"] != starts["E1M2"]


def settled(env):
    """Freedoom levels ignore input on the first tics, so spend one action before measuring."""
    env.reset(1)
    return env.step("ATTACK")[1]


def test_pose_tracks_position_and_facing(env):
    start = settled(env)["pose"]
    _, turned = env.step("TURN_LEFT")
    assert turned["pose"]["angle"] > start["angle"]
    # Doom movement has momentum, so one short step from standing may not register yet.
    for _ in range(3):
        _, moved = env.step("MOVE_FORWARD")
    assert math.dist((moved["pose"]["x"], moved["pose"]["y"]), (start["x"], start["y"])) > 0


def test_compass_bearings_turn_with_the_player_and_distances_hold(env):
    before = settled(env)
    _, after = env.step("TURN_LEFT")
    turned = after["pose"]["angle"] - before["pose"]["angle"]
    assert turned > 0 and before["compass"]
    assert len(before["compass"]) <= level().compass_size
    assert {entry["name"] for entry in before["compass"]} <= set(TARGETS)
    distances = [entry["distance"] for entry in before["compass"]]
    assert distances == sorted(distances)
    for old, new in zip(before["compass"], after["compass"], strict=True):
        assert new["name"] == old["name"]
        assert new["distance"] == pytest.approx(old["distance"], abs=0.5)
        # Turning left (counterclockwise) moves every bearing clockwise by the same angle.
        assert (old["bearing"] - new["bearing"] - turned + 180) % 360 - 180 == pytest.approx(
            0, abs=0.5
        )


def test_rangefinder_measures_wall_distance_in_map_units(env):
    start = settled(env)
    rays = start["rangefinder"]
    assert len(rays) == level().rays
    center = len(rays) // 2
    for _ in range(3):
        _, walked = env.step("MOVE_FORWARD")
    moved = math.dist(
        (walked["pose"]["x"], walked["pose"]["y"]), (start["pose"]["x"], start["pose"]["y"])
    )
    assert walked["pose"]["angle"] == start["pose"]["angle"] and moved > 10
    # Walking straight ahead shortens the center ray by exactly the distance walked.
    assert start["rangefinder"][center] - walked["rangefinder"][center] == pytest.approx(
        moved, abs=1.5
    )


def test_memory_counts_revisits_and_recent_actions(env):
    env.reset(1)
    _, first = env.step("TURN_LEFT")
    _, second = env.step("TURN_LEFT")
    assert second["memory"]["visits_here"] == first["memory"]["visits_here"] + 1
    assert second["memory"]["explored_cells"] == 1
    assert second["memory"]["previous_actions"][-2:] == ["TURN_LEFT", "TURN_LEFT"]
    assert second["memory"]["distance_moved"] == 0
    for _ in range(4):
        _, walking = env.step("MOVE_FORWARD")
    assert walking["memory"]["explored_cells"] > 1
    assert walking["memory"]["distance_moved"] > 0
    assert len(walking["memory"]["previous_actions"]) == level().memory_steps


def test_explore_reward_pays_once_per_new_cell():
    plain, _ = play(level(), "MOVE_FORWARD")
    shaped, env = play(level(rewards={"explore": 5}), "MOVE_FORWARD")
    new_cells = len(env.visited) - 1
    assert new_cells > 0
    assert shaped.record["reward"] - plain.record["reward"] == 5 * new_cells
    idle, _ = play(level(rewards={"explore": 5}), "TURN_LEFT")
    assert idle.record["reward"] == play(level(), "TURN_LEFT")[0].record["reward"]


def test_native_rewards_are_applied_by_name():
    kwargs = {"scenario": "defend_the_center", "map": None, "observe": [], "episode_timeout": None}
    kwargs |= {"labels": ["ATTACK", "TURN_LEFT"], "reward_range": [-100, 100]}
    plain, _ = play(level(**kwargs), "ATTACK", seed=7)
    with DoomEnvironment(level(**kwargs, rewards={"kill_reward": 10})) as env:
        task = level(**kwargs, rewards={"kill_reward": 10})
        program = DecisionProgram(Plan.constant("ATTACK"), None, "none")
        shaped = Episodes(task, env)(program, 7)
        kills = env.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
    assert kills > 0
    assert shaped.record["reward"] - plain.record["reward"] == pytest.approx(10 * kills)


@pytest.mark.parametrize(
    "change",
    [
        {"rewards": {"not_a_reward": 1}},
        {"observe": ["compass"], "compass_targets": []},
    ],
)
def test_invalid_level_settings_are_rejected(change):
    with pytest.raises((ValidationError, ValueError)):
        DoomEnvironment(level(**change))


def test_tasks_can_add_game_variables_the_scenario_lacks():
    with DoomEnvironment(level(variables=["HEALTH", "ARMOR"])) as env:
        assert env.reset(1)["variables"] == {"HEALTH": 100.0, "ARMOR": 0.0}
    with pytest.raises(ValidationError):
        level(variables=["NOT_A_VARIABLE"])


def test_target_is_the_nearest_live_monster_on_screen():
    task = level(scenario="basic", map=None, labels=["ATTACK", "MOVE_LEFT"], observe=["target"])
    task = task.model_copy(update={"compass_targets": [], "episode_timeout": None})
    with DoomEnvironment(task) as env:
        observation = env.reset(7)
        (monster,) = [o for o in observation["objects"] if o["category"] == "Monster"]
        assert observation["target"] == {k: monster[k] for k in ("name", "x", "width")}
    with DoomEnvironment(level(observe=["target"], compass_targets=[])) as env:
        assert "target" not in env.reset(1)


def test_episode_records_report_game_statistics():
    kwargs = {"scenario": "defend_the_center", "map": None, "observe": [], "episode_timeout": None}
    task = level(**kwargs, labels=["ATTACK", "TURN_LEFT"], reward_range=[-100, 100])
    with DoomEnvironment(task) as env:
        program = DecisionProgram(Plan.constant("ATTACK"), None, "none")
        outcome = Episodes(task, env)(program, 7)
        stats = outcome.record["stats"]
        assert stats["kills"] == env.game.get_game_variable(vzd.GameVariable.KILLCOUNT) > 0
        assert stats["explored_cells"] == len(env.visited)
    assert "kills" in outcome.trace["Feedback"]
