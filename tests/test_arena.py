"""Deathmatch duels between decision graphs, and the Elo ranking built from them."""

import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from conftest import chat_reply
from PIL import Image
from test_cli import invoke, write
from test_graph import binary_plan, constant_plan

from rulesmith.arena import Duels, Entrant, free_port, match, ratings
from rulesmith.doom import DEATHMATCH, DoomEnvironment, DoomTask
from rulesmith.graph import Plan
from rulesmith.runtime import DecisionProgram

LABELS = ["ATTACK", "TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"]
# On this seed the two spawn points face across open floor, so spinning players meet.
SEEN_SEED = 1


def duel_data(**changes):
    data = {
        "description": "Frag your opponent more often than they frag you.",
        "scenario": "cig",
        "map": "MAP03",
        "deathmatch": True,
        "labels": LABELS,
        "reward_range": [-60, 60],
        "episode_timeout": 1050,
        "rewards": {"frag_reward": 1, "death_penalty": 1},
        "observe": ["target"],
        "train": [1],
        "validation": [2],
        "test": [3, 4],
    }
    return data | changes


def entrant(name, label, client):
    return Entrant(name, DecisionProgram(Plan.model_validate(constant_plan(label)), client, "f"))


def result(a, b, score_a):
    return {"players": [a, b], "score": [score_a, 1 - score_a]}


def test_evenly_split_results_rate_everyone_equally():
    matches = [result("a", "b", 1), result("b", "a", 1), result("a", "c", 0.5)]
    matches += [result("b", "c", 1), result("c", "b", 1)]
    assert ratings(matches, ["a", "b", "c"]) == pytest.approx({"a": 1500, "b": 1500, "c": 1500})


def test_ratings_fit_the_observed_win_rate_on_the_elo_scale():
    # Three wins in four, plus the one virtual draw every pair shares: 3.5 to 1.5.
    matches = [result("a", "b", 1)] * 3 + [result("a", "b", 0)]
    rated = ratings(matches, ["a", "b"])
    expected = 1 / (1 + 10 ** ((rated["b"] - rated["a"]) / 400))
    assert expected == pytest.approx(3.5 / 5)
    assert (rated["a"] + rated["b"]) / 2 == pytest.approx(1500)


def test_ratings_ignore_match_order_and_stay_finite_for_an_unbeaten_player():
    matches = [result("a", "b", 1)] * 4 + [
        result("b", "c", 1),
        result("c", "a", 0),
        result("b", "c", 0.5),
    ]
    rated = ratings(matches, ["a", "b", "c"])
    for _ in range(5):
        shuffled = random.sample(matches, len(matches))
        assert ratings(shuffled, ["a", "b", "c"]) == pytest.approx(rated)
    assert all(math.isfinite(value) for value in rated.values())
    assert rated["a"] > rated["b"] > rated["c"]


def spin(task, game_args, seed):
    """Turn in place for a whole networked episode, collecting every target seen."""
    with DoomEnvironment(task, game_args=game_args) as env:
        observation, targets = env.reset(seed), set()
        while observation is not None:
            targets.add(observation.get("target", {}).get("name"))
            _, observation = env.step("TURN_LEFT")
        return targets


def test_opponents_are_targets():
    task = DoomTask.model_validate(duel_data())
    port = free_port()
    seats = [f"-host 2 -port {port} {DEATHMATCH}", f"-join 127.0.0.1 -port {port}"]
    with ThreadPoolExecutor(2) as pool:
        seen = [pool.submit(spin, task, args, SEEN_SEED) for args in seats]
    assert all("DoomPlayer" in future.result() for future in seen)


def test_two_graphs_duel_in_lockstep_and_agree_on_the_score(client):
    task = DoomTask.model_validate(duel_data())
    runner, spinner = (
        entrant("runner", "MOVE_FORWARD", client),
        entrant("spinner", "TURN_LEFT", client),
    )
    played = match(task, runner, spinner, SEEN_SEED, host_first=True)
    swapped = match(task, runner, spinner, SEEN_SEED, host_first=False)
    assert played["host"] == "runner" and swapped["host"] == "spinner"
    full = math.ceil(task.episode_timeout / task.tics_per_action)
    for record, host in ((played, 0), (swapped, 1)):
        assert record["players"] == ["runner", "spinner"]
        a, b = record["seats"]
        # The host acts on every action boundary until the timeout; the joiner, a tic behind
        # now and then, gets somewhat fewer turns but still plays the whole match.
        assert record["seats"][host]["steps"] == full
        assert 0.8 * full <= record["seats"][1 - host]["steps"] <= full
        assert record["frags"] == [a["stats"]["frags"], b["stats"]["frags"]]
        for seat, rival in ((a, b), (b, a)):
            assert seat["reward"] == seat["stats"]["frags"] - seat["stats"]["deaths"]
            assert rival["stats"]["frags"] <= seat["stats"]["deaths"]
        assert record["score"] in ([1, 0], [0, 1], [0.5, 0.5])
        assert record["score"][0] == (record["frags"][0] > record["frags"][1]) + 0.5 * (
            record["frags"][0] == record["frags"][1]
        )


def test_a_failing_graph_forfeits_without_stalling_its_opponent(client):
    task = DoomTask.model_validate(duel_data())
    broken = Entrant("broken", DecisionProgram(Plan.model_validate(skipping_plan()), client, "f"))
    record = match(task, broken, entrant("spinner", "TURN_LEFT", client), SEEN_SEED, True)
    assert record["score"] == [0, 1]
    assert record["seats"][0]["error"] and not record["seats"][1]["error"]
    assert record["seats"][1]["steps"] > 1


def skipping_plan():
    """Reads a path no observation has, so its only action is always skipped."""
    return {
        "nodes": {
            "health": {"type": "read", "path": ["no_such_field"], "bounds": [0, 100]},
            "action": {
                "type": "value",
                "value": "ATTACK",
                "when": {"type": "present", "node": "health"},
            },
        },
        "outputs": {"label": {"type": "node", "node": "action"}},
    }


def test_cli_ranks_every_entrant_from_a_round_robin(tmp_path):
    task = write(tmp_path / "duel.json", duel_data())
    runner = write(tmp_path / "runner.json", constant_plan("MOVE_FORWARD"))
    spinner = write(tmp_path / "spinner.json", constant_plan("TURN_LEFT"))
    output = tmp_path / "arena.json"
    shooter = write(tmp_path / "shooter.json", constant_plan("ATTACK"))
    completed = invoke("arena", task, runner, spinner, shooter, "--output", output)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text())
    names = {"runner", "spinner", "shooter"}
    assert set(report["ratings"]) == names
    # Every pair meets once per test episode.
    pairs = [frozenset(record["players"]) for record in report["matches"]]
    assert len(pairs) == 3 * 2 and len(set(pairs)) == 3
    assert sum(report["ratings"].values()) / 3 == pytest.approx(1500)
    points = {name: 0.0 for name in names}
    for record in report["matches"]:
        for name, score in zip(record["players"], record["score"], strict=True):
            points[name] += score
    # In a balanced round robin, Bradley-Terry ratings order players exactly as their points do.
    rated = report["ratings"]
    for x in names:
        for y in names:
            if points[x] > points[y]:
                assert rated[x] > rated[y]


def test_cli_rejects_entrants_that_share_a_name(tmp_path):
    task = write(tmp_path / "duel.json", duel_data())
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = write(tmp_path / "a" / "plan.json", constant_plan("MOVE_FORWARD"))
    second = write(tmp_path / "b" / "plan.json", constant_plan("TURN_LEFT"))
    completed = invoke("arena", task, first, second, "--output", tmp_path / "arena.json")
    assert completed.returncode != 0 and "plan" in completed.stderr


def test_the_repository_duel_task_survives_a_networked_match(client):
    # Sectors info, which the rangefinder needs, crashes ViZDoom outright on the cig maps.
    path = Path(__file__).resolve().parents[1] / "examples/doom/duel.json"
    task = DoomTask.model_validate_json(path.read_text()).model_copy(
        update={"episode_timeout": 200}
    )
    attacker = entrant("attacker", "ATTACK", client)
    record = match(task, attacker, entrant("walker", "MOVE_FORWARD", client), task.test[0], True)
    assert all(seat["steps"] > 0 and not seat["error"] for seat in record["seats"])


def test_a_deathmatch_task_refuses_single_player_play():
    with pytest.raises(ValueError, match="arena"):
        DoomEnvironment(DoomTask.model_validate(duel_data()))


def test_duels_score_the_candidate_seat_and_alternate_hosting(client):
    task = DoomTask.model_validate(duel_data())
    score = Duels(task, entrant("spinner", "TURN_LEFT", client).program)
    runner = entrant("runner", "MOVE_FORWARD", client).program
    outcomes = [score(runner, seed) for seed in (SEEN_SEED, SEEN_SEED + 1)]
    assert outcomes[0].record["hosted"] != outcomes[1].record["hosted"]
    low, high = task.reward_range
    for outcome in outcomes:
        record = outcome.record
        assert outcome.score == (record["reward"] - low) / (high - low)
        margin = record["stats"]["frags"] - record["opponent_frags"]
        assert record["result"] == ("won" if margin > 0 else "lost" if margin < 0 else "drew")


def test_cli_needs_an_opponent_for_a_deathmatch_task(tmp_path):
    task = write(tmp_path / "duel.json", duel_data())
    plan = write(tmp_path / "plan.json", constant_plan("ATTACK"))
    completed = invoke("evaluate", task, plan, "--output", tmp_path / "out.json")
    assert completed.returncode != 0 and "--opponent" in completed.stderr


def test_cli_searches_and_evaluates_a_graph_against_an_opponent(tmp_path, provider, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    provider.completion = chat_reply(graph=constant_plan("MOVE_FORWARD"))
    task = write(tmp_path / "duel.json", duel_data(train=[1], validation=[2], test=[3]))
    opponent = write(tmp_path / "spinner.json", constant_plan("TURN_LEFT"))
    common = ["--base-url", provider.url, "--opponent", opponent]
    output = tmp_path / "run"
    completed = invoke(
        "optimize",
        task,
        "--output",
        output,
        "--reflection-model",
        "openai/fixture",
        "--reflection-base-url",
        provider.url + "/v1",
        "--max-metric-calls",
        4,
        "--reflection-batch",
        1,
        *common,
    )
    assert completed.returncode == 0, completed.stderr
    evaluation = tmp_path / "test.json"
    completed = invoke(
        "evaluate",
        task,
        output / "plan.json",
        "--output",
        evaluation,
        "--parallel-examples",
        2,
        *common,
    )
    assert completed.returncode == 0, completed.stderr
    (row,) = json.loads(evaluation.read_text())["examples"]
    assert row["seed"] == 3 and row["result"] in ("won", "drew", "lost")


def test_a_killed_player_respawns_and_keeps_playing(client):
    task = DoomTask.model_validate(duel_data(episode_timeout=600))
    port = free_port()
    seats = [f"-host 2 -port {port} {DEATHMATCH}", f"-join 127.0.0.1 -port {port}"]
    with ThreadPoolExecutor(2) as pool:
        played = [pool.submit(die_once, task, args) for args in seats]
    full = math.ceil(task.episode_timeout / task.tics_per_action)
    for future in played:
        steps, stats = future.result()
        # Respawning costs tics, so a death shortens the seat's turns without ending it.
        assert 20 < steps < full
        assert stats["deaths"] == 1


def die_once(task, game_args):
    """Play a whole duel, taking one suicide part way through; the engine reports no state
    while a dead player waits to respawn."""
    with DoomEnvironment(task, game_args=game_args) as env:
        observation, steps = env.reset(SEEN_SEED), 0
        while observation is not None:
            if steps == 20:
                env.game.send_game_command("kill")
            _, observation = env.step("TURN_LEFT")
            steps += 1
        return steps, env.stats()


def test_duels_are_fought_without_monsters_and_with_a_compass_to_the_opponent(client):
    task = DoomTask.model_validate(
        duel_data(
            episode_timeout=300,
            observe=["target", "compass"],
            compass_targets=["DoomPlayer", "Zombieman"],
        )
    )
    port = free_port()
    seats = [f"-host 2 -port {port} {DEATHMATCH}", f"-join 127.0.0.1 -port {port}"]
    with ThreadPoolExecutor(2) as pool:
        seen = [pool.submit(sightings, task, args) for args in seats]
    for future in seen:
        categories, compass = future.result()
        # The compass sees the whole map, so a monster would show up wherever it stood.
        assert not [entry for entry in compass if entry["name"] == "Zombieman"]
        assert "Monster" not in categories
        # The opponent is tracked anywhere on the map, and never the player's own body.
        bearings = [entry for entry in compass if entry["name"] == "DoomPlayer"]
        assert bearings and all(entry["distance"] > 0 for entry in bearings)


def sightings(task, game_args):
    """Every object category seen, and every compass entry, over one duel."""
    with DoomEnvironment(task, game_args=game_args) as env:
        observation, categories, compass = env.reset(SEEN_SEED), set(), []
        while observation is not None:
            categories.update(thing["category"] for thing in observation["objects"])
            compass += observation["compass"]
            _, observation = env.step("TURN_LEFT")
        return categories, compass


def test_cli_records_a_duel_from_one_graph_s_seat(tmp_path):
    task = write(tmp_path / "duel.json", duel_data(episode_timeout=200))
    plan = write(tmp_path / "runner.json", constant_plan("MOVE_FORWARD"))
    opponent = write(tmp_path / "spinner.json", constant_plan("TURN_LEFT"))
    output = tmp_path / "duel.gif"
    completed = invoke(
        "record", task, plan, "--opponent", opponent, "--seed", SEEN_SEED, "--output", output
    )
    assert completed.returncode == 0, completed.stderr
    (episode,) = json.loads(completed.stdout)["episodes"]
    tics = DoomTask.model_validate(duel_data()).tics_per_action
    with Image.open(output) as gif:
        # Every tic the seat played, less any the timeout cut from its final action.
        assert episode["steps"] * tics - tics < gif.n_frames <= episode["steps"] * tics


@pytest.mark.parametrize("command", ["evaluate", "record"])
def test_cli_initializes_judge_for_a_model_opponent(tmp_path, provider, command):
    task = write(tmp_path / "duel.json", duel_data(episode_timeout=120, test=[3]))
    plan = write(tmp_path / "runner.json", constant_plan("MOVE_FORWARD"))
    opponent_graph = binary_plan()
    opponent_graph["nodes"]["decision"].update(below="TURN_LEFT", above="ATTACK")
    opponent = write(tmp_path / "opponent.json", opponent_graph)
    provider.body["answers"] = {"eligible": {"type": "noul", "noul": 0.9}}
    output = tmp_path / ("duel.gif" if command == "record" else "evaluation.json")
    extra = ["--seed", SEEN_SEED] if command == "record" else []
    completed = invoke(
        command,
        task,
        plan,
        "--opponent",
        opponent,
        "--backend",
        "ruling",
        "--base-url",
        provider.url,
        "--output",
        output,
        *extra,
    )
    assert completed.returncode == 0, completed.stderr
    assert len(provider.requests) > 1
    assert all(path == "/v1/systemone" for path, _ in provider.requests)
    if command == "record":
        (episode,) = json.loads(completed.stdout)["episodes"]
        with Image.open(output) as gif:
            assert gif.n_frames > 1
    else:
        (episode,) = json.loads(output.read_text())["examples"]
    assert episode["steps"] > 1 and episode["error"] is None


def test_a_distant_opponent_is_never_crowded_off_the_compass():
    # Deathmatch maps are littered with pickups, and an opponent starts across the map.
    path = Path(__file__).resolve().parents[1] / "examples/doom/duel.json"
    task = DoomTask.model_validate_json(path.read_text()).model_copy(
        update={"episode_timeout": 200}
    )
    port = free_port()
    seats = [f"-host 2 -port {port} {DEATHMATCH}", f"-join 127.0.0.1 -port {port}"]
    with ThreadPoolExecutor(2) as pool:
        opening = [pool.submit(first_compass, task, args) for args in seats]
    for future in opening:
        compass = future.result()
        assert len(compass) == task.compass_size
        assert [entry["name"] for entry in compass].count("DoomPlayer") == 1


def first_compass(task, game_args):
    with DoomEnvironment(task, game_args=game_args) as env:
        return env.reset(task.train[0])["compass"]
