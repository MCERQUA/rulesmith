"""Score decision graphs by playing ViZDoom episodes, one graph execution per action."""

import math
import os
import tempfile
from collections import Counter, deque
from pathlib import Path
from typing import Literal, Self

import numpy as np
import vizdoom as vzd
from pydantic import Field, model_validator

from decision_compose.graph import Plan, StrictModel, Text
from decision_compose.level import Level
from decision_compose.optimize import Outcome
from decision_compose.runtime import DecisionProgram, GraphExecutionError


class DoomTask(StrictModel):
    description: Text
    scenario: Text = Field(description="Name of a scenario bundled with ViZDoom, e.g. basic")
    labels: list[Text] = Field(
        min_length=2,
        description="Actions the graph may choose; join buttons with + to press them together",
    )
    reward_range: tuple[float, float] = Field(description="Lowest and highest episode reward")
    tics_per_action: int = Field(default=4, ge=1)
    episode_timeout: int | None = Field(
        default=None, ge=1, description="Tics before an episode ends; defaults to the scenario's"
    )
    feedback_steps: int = Field(default=8, ge=1)
    map: Text | None = Field(default=None, description="Map to load, e.g. E1M1")
    variables: list[Text] = Field(
        default_factory=list, description="Game variables to add, e.g. HEALTH or AMMO2"
    )
    observe: list[Literal["pose", "rangefinder", "compass", "memory", "target", "route"]] = Field(
        default_factory=list, description="Extra observations beyond visible objects"
    )
    rays: int = Field(default=7, ge=2, description="Rangefinder directions across the view")
    rangefinder_range: float = Field(
        default=2048, gt=0, description="Map units reported when no wall is within reach"
    )
    compass_targets: list[Text] = Field(
        default_factory=list, description="Object names the compass tracks anywhere on the map"
    )
    compass_size: int = Field(default=8, ge=1, description="Nearest compass targets reported")
    memory_steps: int = Field(default=4, ge=1, description="Recent actions and movement kept")
    cell_size: float = Field(default=64, gt=0, description="Map units per exploration cell")
    rewards: dict[Text, float] = Field(
        default_factory=dict,
        description="ViZDoom rewards by setter name, e.g. kill_reward, plus explore per new "
        "cell and progress per map unit the route to the exit shrinks",
    )
    train: list[int] = Field(min_length=1)
    validation: list[int] = Field(min_length=1)
    test: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be distinct")
        if self.reward_range[0] >= self.reward_range[1]:
            raise ValueError("reward_range must be increasing")
        seeds = self.train + self.validation + self.test
        if len(set(seeds)) != len(seeds):
            raise ValueError("episode seeds must be distinct across splits")
        missing = {name for name in self.variables if not hasattr(vzd.GameVariable, name)}
        if missing:
            raise ValueError(f"unknown ViZDoom game variables: {sorted(missing)}")
        if "compass" in self.observe and not self.compass_targets:
            raise ValueError("the compass needs compass_targets")
        unknown = {
            name
            for name in self.rewards
            if name not in CUSTOM_REWARDS
            and not (
                name.endswith(("_reward", "_penalty")) and hasattr(vzd.DoomGame, f"set_{name}")
            )
        }
        if unknown:
            raise ValueError(f"unknown ViZDoom rewards: {sorted(unknown)}")
        return self

    def baseline_plan(self) -> Plan:
        """A call-free seed that always presses the first listed button."""
        return Plan.constant(self.labels[0])


# Doom's horizontal field of view, and the smallest opening the player fits through.
FIELD_OF_VIEW = 90
PLAYER_HEIGHT = 56


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The z component of 2-D cross products; NumPy 2 no longer accepts 2-D vectors."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def walls(state) -> tuple[np.ndarray, np.ndarray]:
    """Segments that stop the player: solid walls, and gaps too low to pass such as closed doors."""
    spans: dict[tuple, list] = {}
    blocking: dict[tuple, bool] = {}
    for sector in state.sectors:
        for line in sector.lines:
            key = tuple(sorted([(line.x1, line.y1), (line.x2, line.y2)]))
            spans.setdefault(key, []).append(sector.ceiling_height - sector.floor_height)
            blocking[key] = blocking.get(key, False) or line.is_blocking
    solid = [key for key, gaps in spans.items() if blocking[key] or min(gaps) < PLAYER_HEIGHT]
    segments = np.array(solid, dtype=float).reshape(-1, 2, 2)
    return segments[:, 0], segments[:, 1]


CUSTOM_REWARDS = {"explore", "progress"}


def setting(config: str, key: str) -> str | None:
    """A ViZDoom config value; keys ignore case and underscores, as ViZDoom does."""
    wanted = key.replace("_", "").lower()
    for line in Path(config).read_text().splitlines():
        name, _, value = line.split("#", 1)[0].partition("=")
        if value and name.strip().replace("_", "").lower() == wanted:
            return value.strip()
    return None


def level_file(config: str, map_name: str) -> Path:
    """The WAD holding a scenario's map: its scenario WAD, else its game WAD."""
    here = Path(config).parent
    for key in ("doom_scenario_path", "doom_game_path"):
        name = setting(config, key)
        if name is None:
            continue
        for folder in (here, Path(vzd.__file__).parent):
            if (folder / name).is_file():
                return folder / name
    raise ValueError(f"no WAD found for {config}")


class DoomEnvironment:
    """A headless ViZDoom game whose observations are JSON, not pixels."""

    def __init__(self, task: DoomTask, visible: bool = False, record: bool = False):
        config = os.path.join(vzd.scenarios_path, f"{task.scenario}.cfg")
        if not os.path.isfile(config):
            raise ValueError(f"unknown ViZDoom scenario: {task.scenario}")
        # ViZDoom otherwise writes _vizdoom.ini into the working directory.
        self.scratch = tempfile.TemporaryDirectory()
        self.game = vzd.DoomGame()
        self.game.load_config(config)
        self.game.set_doom_config_path(os.path.join(self.scratch.name, "_vizdoom.ini"))
        self.game.set_window_visible(visible)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_audio_buffer_enabled(False)
        self.game.set_sectors_info_enabled("rangefinder" in task.observe)
        self.game.set_objects_info_enabled(bool({"compass", "target"} & set(task.observe)))
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        if task.map is not None:
            self.game.set_doom_map(task.map)
        for name in task.variables:
            self.game.add_available_game_variable(getattr(vzd.GameVariable, name))
        for name, value in task.rewards.items():
            if name not in CUSTOM_REWARDS:
                getattr(self.game, f"set_{name}")(value)
        self.level = None
        if "route" in task.observe or "progress" in task.rewards:
            map_name = (task.map or setting(config, "doom_map") or "").upper()
            if not map_name:
                raise ValueError(f"routing needs a map name for {task.scenario}")
            self.level = Level(level_file(config, map_name), map_name)
        # Every rendered tic while recording, as height x width x RGB arrays.
        self.frames: list | None = [] if record else None
        if task.episode_timeout is not None:
            self.game.set_episode_timeout(task.episode_timeout)
        self.game.init()
        buttons = [button.name for button in self.game.get_available_buttons()]
        pressed = {label: set(label.split("+")) for label in task.labels}
        unknown = set().union(*pressed.values()) - set(buttons)
        if unknown:
            self.close()
            raise ValueError(f"buttons not available in {task.scenario}: {sorted(unknown)}")
        self.buttons = {label: [b in pressed[label] for b in buttons] for label in task.labels}
        self.task = task
        self.tics = task.tics_per_action
        self.variables = [v.name for v in self.game.get_available_game_variables()]
        self.width = self.game.get_screen_width()
        self.height = self.game.get_screen_height()

    def reset(self, seed: int) -> dict:
        self.game.set_seed(seed)
        self.game.new_episode()
        if self.frames is not None:
            self.frames = [self.game.get_state().screen_buffer.copy()]
        self.visited = Counter()
        self.recent = deque(maxlen=self.task.memory_steps)
        self.trail = deque(maxlen=self.task.memory_steps + 1)
        self.visit()
        self.plot()
        self.best = self.remaining
        return self.observe()

    def plot(self) -> None:
        """Recompute the walking distance from the player's position to the exit."""
        self.remaining = None
        if self.level is not None:
            x, y, _ = self.pose()
            self.remaining = self.level.remaining(x, y)

    def pose(self) -> tuple[float, float, float]:
        variable = self.game.get_game_variable
        GV = vzd.GameVariable
        return variable(GV.POSITION_X), variable(GV.POSITION_Y), variable(GV.ANGLE)

    def visit(self) -> bool:
        """Count the player's current cell; True when it was never visited before."""
        x, y, _ = self.pose()
        self.trail.append((x, y))
        cell = (math.floor(x / self.task.cell_size), math.floor(y / self.task.cell_size))
        self.visited[cell] += 1
        self.cell = cell
        return self.visited[cell] == 1

    def step(self, action: str) -> tuple[float, dict | None]:
        if self.frames is None:
            reward = self.game.make_action(self.buttons[action], self.tics)
        else:
            # One tic at a time so every frame is captured; the game advances identically.
            self.game.set_action(self.buttons[action])
            reward = 0.0
            for _ in range(self.tics):
                self.game.advance_action(1)
                reward += self.game.get_last_reward()
                if self.game.is_episode_finished():
                    break
                self.frames.append(self.game.get_state().screen_buffer.copy())
        self.recent.append(action)
        if self.game.is_episode_finished():
            return reward, None
        if self.visit():
            reward += self.task.rewards.get("explore", 0.0)
        self.plot()
        if self.remaining is not None and (self.best is None or self.remaining < self.best):
            if self.best is not None:
                reward += self.task.rewards.get("progress", 0.0) * (self.best - self.remaining)
            self.best = self.remaining
        return reward, self.observe()

    def observe(self) -> dict:
        state = self.game.get_state()
        objects = [
            {
                "name": label.object_name,
                "category": label.object_category,
                # Screen-relative centers in [-1, 1], left/top negative; rounded for short prompts.
                "x": round((label.x + label.width / 2) / self.width * 2 - 1, 3),
                "y": round((label.y + label.height / 2) / self.height * 2 - 1, 3),
                "width": round(label.width / self.width, 3),
                "height": round(label.height / self.height, 3),
            }
            for label in state.labels
            # "Self" is the player's own weapon sprite.
            if label.object_category != "Self"
        ]
        # ViZDoom reports None, not an empty array, when a scenario has no variables.
        values = state.game_variables if self.variables else []
        variables = dict(zip(self.variables, map(float, values), strict=True))
        observation = {"objects": objects, "variables": variables}
        x, y, angle = self.pose()
        if "target" in self.task.observe:
            target = self.target(state, objects, x, y)
            if target is not None:
                observation["target"] = target
        if "pose" in self.task.observe:
            observation["pose"] = {"x": round(x, 1), "y": round(y, 1), "angle": round(angle, 1)}
        if "rangefinder" in self.task.observe:
            observation["rangefinder"] = self.rangefinder(state, x, y, angle)
        if "compass" in self.task.observe:
            targets = []
            for thing in state.objects:
                if thing.name not in self.task.compass_targets:
                    continue
                dx, dy = thing.position_x - x, thing.position_y - y
                # Degrees from the facing direction; positive is to the left (counterclockwise).
                bearing = (math.degrees(math.atan2(dy, dx)) - angle + 180) % 360 - 180
                targets.append(
                    {
                        "name": thing.name,
                        "bearing": round(bearing, 1),
                        "distance": round(math.hypot(dx, dy)),
                    }
                )
            targets.sort(key=lambda target: target["distance"])
            observation["compass"] = targets[: self.task.compass_size]
        found = self.level.waypoint(x, y) if "route" in self.task.observe else None
        if found is not None:
            (target_x, target_y), ahead = found
            dx, dy = target_x - x, target_y - y
            observation["route"] = {
                "next": ahead,
                "bearing": round((math.degrees(math.atan2(dy, dx)) - angle + 180) % 360 - 180, 1),
                "distance": round(math.hypot(dx, dy)),
                "remaining": round(self.remaining),
            }
        if "memory" in self.task.observe:
            observation["memory"] = {
                "previous_actions": list(self.recent),
                "distance_moved": round(math.dist(self.trail[0], self.trail[-1]), 1),
                "visits_here": self.visited[self.cell],
                "explored_cells": len(self.visited),
            }
        return observation

    def target(self, state, objects: list[dict], x: float, y: float) -> dict | None:
        """The nearest live monster on screen, with its map distance."""
        # Killed monsters leave the labels, so every on-screen Monster is alive. Labels share
        # IDs with map objects, whose positions give each visible monster's distance.
        places = {thing.id: thing for thing in state.objects}
        visible = [label for label in state.labels if label.object_category != "Self"]
        ranged = [
            (
                math.hypot(
                    places[label.object_id].position_x - x, places[label.object_id].position_y - y
                ),
                entry,
            )
            for label, entry in zip(visible, objects, strict=True)
            if entry["category"] == "Monster" and label.object_id in places
        ]
        if not ranged:
            return None
        distance, nearest = min(ranged, key=lambda pair: pair[0])
        return {
            "name": nearest["name"],
            "x": nearest["x"],
            "width": nearest["width"],
            "distance": round(distance),
        }

    def rangefinder(self, state, x: float, y: float, angle: float) -> list[int]:
        """Map-unit distance to the nearest wall along rays across the view, left to right."""
        starts, ends = walls(state)
        span = FIELD_OF_VIEW / 2
        step = FIELD_OF_VIEW / (self.task.rays - 1)
        readings = []
        for index in range(self.task.rays):
            heading = math.radians(angle + span - index * step)
            direction = np.array([math.cos(heading), math.sin(heading)])
            edges = ends - starts
            offset = starts - (x, y)
            denominator = cross(direction, edges)
            with np.errstate(divide="ignore", invalid="ignore"):
                distance = cross(offset, edges) / denominator
                along = cross(offset, direction) / denominator
            hits = distance[(denominator != 0) & (distance > 0) & (along >= 0) & (along <= 1)]
            nearest = min(
                hits.min(initial=self.task.rangefinder_range), self.task.rangefinder_range
            )
            readings.append(round(float(nearest)))
        return readings

    def stats(self) -> dict:
        """Level progress counters; ViZDoom keeps them readable after the episode ends."""
        variable = self.game.get_game_variable
        GV = vzd.GameVariable
        return {
            "kills": int(variable(GV.KILLCOUNT)),
            "items": int(variable(GV.ITEMCOUNT)),
            "secrets": int(variable(GV.SECRETCOUNT)),
            "explored_cells": len(self.visited),
        }

    def close(self) -> None:
        self.game.close()
        self.scratch.cleanup()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Episodes:
    """Scores a graph by the normalized reward of one episode per seed."""

    def __init__(self, task: DoomTask, env: DoomEnvironment):
        self.task = task
        self.env = env

    def __call__(self, program: DecisionProgram, seed: int) -> Outcome:
        observation, steps, models, error = self.env.reset(seed), [], set(), None
        while observation is not None:
            try:
                prediction = program(state=observation)
            except GraphExecutionError as failure:
                models.update(failure.prediction.models)
                error = f"Graph execution failed at step {len(steps)}: {failure}"
                break
            models.update(prediction.models)
            reward, following = self.env.step(prediction.label)
            steps.append({"observation": observation, "action": prediction.label, "reward": reward})
            observation = following

        total = sum(step["reward"] for step in steps)
        low, high = self.task.reward_range
        if error is None and not low <= total <= high:
            raise ValueError(f"episode reward {total} is outside reward_range {[low, high]}")
        score = 0.0 if error else (total - low) / (high - low)
        actions = Counter(step["action"] for step in steps)
        stats = self.env.stats()
        record = {
            "seed": seed,
            "reward": total,
            "score": score,
            "steps": len(steps),
            "actions": dict(actions),
            "stats": stats,
            "error": error,
        }
        feedback = error or (
            f"Episode reward {total} in range {[low, high]} (score {score:.3f}) "
            f"after {len(steps)} actions, with {stats}. Higher reward is better."
        )
        return Outcome(
            score=score,
            record=record,
            trace={
                "Inputs": {
                    "seed": seed,
                    "first_observation": steps[0]["observation"] if steps else observation,
                },
                "Generated Outputs": {
                    "actions": dict(actions),
                    "last_steps": steps[-self.task.feedback_steps :],
                },
                "Feedback": feedback,
            },
            models=sorted(models),
            error=error,
        )
