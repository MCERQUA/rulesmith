"""Full-level observations and rewards, checked against Freedoom's real E1M1 map."""

import itertools
import math
from pathlib import Path

import pytest
import vizdoom as vzd
from pydantic import ValidationError

from decision_compose.doom import DoomEnvironment, DoomTask, Episodes
from decision_compose.graph import Plan
from decision_compose.level import Level
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
        "observe": ["pose", "rangefinder", "compass", "memory", "route"],
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
            starts[name] = env.reset(1)["pose"]
            assert env.game.get_doom_map() == name
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
    # The nearest of each name leads, then the rest of the room by distance.
    first = {}
    for index, entry in enumerate(before["compass"]):
        first.setdefault(entry["name"], index)
    assert sorted(first.values()) == list(range(len(first)))
    for block in (before["compass"][: len(first)], before["compass"][len(first) :]):
        distances = [entry["distance"] for entry in block]
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
        assert {k: observation["target"][k] for k in ("name", "x", "width")} == {
            k: monster[k] for k in ("name", "x", "width")
        }
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


def test_level_file_locates_the_exit_switch_and_routes_to_it():
    level_map = Level(Path(vzd.scenarios_path).parent / "freedoom1.wad", "E1M1")
    (exit_line,) = level_map.exits
    assert exit_line.special == 11 and exit_line.middle == (-400, 1296)
    # Walking distance can never be shorter than the straight line.
    assert level_map.remaining(-416, 256) >= math.dist((-416, 256), exit_line.middle)


def test_route_points_along_the_way_to_the_exit(env):
    route = settled(env)["route"]
    assert route["next"] in {"path", "door", "exit"}
    assert route["remaining"] >= route["distance"] > 0
    assert -180 <= route["bearing"] <= 180


def test_progress_reward_pays_only_for_shortening_the_route():
    walked, _ = play(level(rewards={"progress": 1}), "MOVE_FORWARD")
    plain, _ = play(level(), "MOVE_FORWARD")
    assert walked.record["reward"] > plain.record["reward"]
    spun, _ = play(level(rewards={"progress": 1}), "TURN_LEFT")
    assert spun.record["reward"] == play(level(), "TURN_LEFT")[0].record["reward"]


def test_every_position_along_a_walk_has_a_route(env):
    observation = settled(env)
    while observation is not None:
        assert "route" in observation
        bearing = observation["route"]["bearing"]
        turn = "TURN_LEFT" if bearing > 15 else "TURN_RIGHT" if bearing < -15 else None
        _, observation = env.step(turn or "MOVE_FORWARD")


def test_target_distance_matches_the_monsters_map_position():
    task = level(scenario="basic", map=None, labels=["ATTACK", "MOVE_LEFT"], observe=["target"])
    task = task.model_copy(update={"compass_targets": [], "episode_timeout": None})
    with DoomEnvironment(task) as env:
        target = env.reset(7)["target"]
        x, y, _ = env.pose()
        (monster,) = [o for o in env.game.get_state().objects if o.name == target["name"]]
        expected = math.hypot(monster.position_x - x, monster.position_y - y)
        assert target["distance"] == pytest.approx(expected, abs=0.5)


def test_route_waypoints_lead_through_openings_not_along_them():
    level_map = Level(Path(vzd.scenarios_path).parent / "freedoom1.wad", "E1M1")
    # Standing on the edge of a door recess, the next opening is the recess line itself.
    waypoint, _ = level_map.waypoint(576.0, 1035.4)
    # The way on is west through the door at x = 560, not south along the wall.
    assert waypoint[0] < 560


def test_following_the_route_reaches_the_exit():
    """With damage off and no monsters in the way, following the route finishes the level."""
    task = level(
        episode_timeout=35 * 240,
        labels=["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD+USE"],
    )
    task = task.model_copy(update={"reward_range": (-10000, 100000)})
    with DoomEnvironment(task) as env:
        observation = env.reset(5)
        # Live monsters body-block a walker that never fights; this test is about navigation.
        env.game.send_game_command("god")
        env.game.send_game_command("kill monsters")
        while observation is not None:
            bearing = observation["route"]["bearing"]
            if abs(bearing) > 12:
                action = "TURN_LEFT" if bearing > 0 else "TURN_RIGHT"
            elif observation["route"]["distance"] < 64 or observation["rangefinder"][3] < 64:
                action = "MOVE_FORWARD+USE"
            else:
                action = "MOVE_FORWARD"
            _, observation = env.step(action)
        assert env.game.is_episode_finished()
        assert env.game.get_episode_time() < task.episode_timeout


def test_routes_never_climb_more_than_a_doom_step():
    level_map = Level(Path(vzd.scenarios_path).parent / "freedoom1.wad", "E1M1")
    # Beside a thin 32-unit ledge that a route once stepped over between cell centers.
    x, y = 736.0, 464.0
    for _ in range(40):
        (next_x, next_y), ahead = level_map.waypoint(x, y)
        if ahead == "exit":
            break
        floors = [
            level_map.heights[
                level_map.locate(x + (next_x - x) * t / 16, y + (next_y - y) * t / 16)
            ][0]
            for t in range(17)
        ]
        assert max(b - a for a, b in itertools.pairwise(floors)) <= 24, (x, y, floors)
        x, y = next_x, next_y


def test_solid_decorations_block_the_route():
    level_map = Level(Path(vzd.scenarios_path).parent / "freedoom1.wad", "E1M1")
    radii = {name: radius for name, radius in level_map.solid_types.values()}
    assert radii["TechPillar"] == 16 and radii["BigTree"] == 32 and radii["ExplosiveBarrel"] == 10
    assert "Zombieman" not in radii
    for (x, y), radius in level_map.obstacles:
        assert not level_map.walkable((x - radius - 20, y), (x + radius + 20, y))


def test_episodes_can_name_their_map_and_scenario():
    task = level(
        train=[{"map": "E1M7", "seed": 1}, {"scenario": "freedoom2", "map": "MAP07", "seed": 1}]
    )
    with DoomEnvironment(task) as env:
        seen = {}
        for episode in [*task.train, 1]:
            observation = env.reset(episode)
            seen[env.game.get_doom_map()] = (observation["pose"], env.remaining)
        assert set(seen) == {"E1M7", "MAP07", "E1M1"}
        assert len({pose["x"] for pose, _ in seen.values()}) == 3
        assert all(remaining > 0 for _, remaining in seen.values())


def test_the_same_episode_cannot_appear_in_two_splits():
    with pytest.raises(ValidationError):
        level(train=[{"map": "E1M7", "seed": 1}], validation=[{"map": "E1M7", "seed": 1}])
