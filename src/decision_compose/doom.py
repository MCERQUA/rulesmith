"""Score decision graphs by playing ViZDoom episodes, one graph execution per action."""

import math
import os
import tempfile
from collections import Counter, deque
from itertools import pairwise
from typing import Literal, Self

import vizdoom as vzd
from pydantic import Field, model_validator

from decision_compose.graph import Plan, StrictModel, Text
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
    observe: list[Literal["pose", "rangefinder", "compass", "memory"]] = Field(
        default_factory=list, description="Extra observations beyond visible objects"
    )
    rays: int = Field(default=7, ge=1, description="Rangefinder directions across the view")
    compass_targets: list[Text] = Field(
        default_factory=list, description="Object names the compass tracks anywhere on the map"
    )
    compass_size: int = Field(default=8, ge=1, description="Nearest compass targets reported")
    memory_steps: int = Field(default=4, ge=1, description="Recent actions and movement kept")
    cell_size: float = Field(default=64, gt=0, description="Map units per exploration cell")
    rewards: dict[Text, float] = Field(
        default_factory=dict,
        description="ViZDoom rewards by setter name, e.g. kill_reward, plus explore per new cell",
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
            if name != "explore"
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
        self.game.set_depth_buffer_enabled("rangefinder" in task.observe)
        self.game.set_objects_info_enabled("compass" in task.observe)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        if task.map is not None:
            self.game.set_doom_map(task.map)
        for name in task.variables:
            self.game.add_available_game_variable(getattr(vzd.GameVariable, name))
        for name, value in task.rewards.items():
            if name != "explore":
                getattr(self.game, f"set_{name}")(value)
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
        return self.observe()

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
        if "pose" in self.task.observe:
            observation["pose"] = {"x": round(x, 1), "y": round(y, 1), "angle": round(angle, 1)}
        if "rangefinder" in self.task.observe:
            # A band above the weapon sprite; the nearest depth per column slice, left to right.
            depth = state.depth_buffer[int(self.height * 0.3) : int(self.height * 0.5)]
            edges = [self.width * ray // self.task.rays for ray in range(self.task.rays + 1)]
            observation["rangefinder"] = [
                int(depth[:, start:end].min()) for start, end in pairwise(edges)
            ]
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
        if "memory" in self.task.observe:
            observation["memory"] = {
                "previous_actions": list(self.recent),
                "distance_moved": round(math.dist(self.trail[0], self.trail[-1]), 1),
                "visits_here": self.visited[self.cell],
                "explored_cells": len(self.visited),
            }
        return observation

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
        record = {
            "seed": seed,
            "reward": total,
            "score": score,
            "steps": len(steps),
            "actions": dict(actions),
            "error": error,
        }
        feedback = error or (
            f"Episode reward {total} in range {[low, high]} (score {score:.3f}) "
            f"after {len(steps)} actions. Higher reward is better."
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
