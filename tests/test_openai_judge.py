"""Opt-in integration with a running OpenAI-compatible chat server and its actual model."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rulesmith.graph import Plan, Task

pytestmark = pytest.mark.openai
ROOT = Path(__file__).resolve().parents[1]


def test_a_real_chat_model_judges_a_graph_from_the_command_line(request, tmp_path):
    plan_path = ROOT / "examples/support-routing/dag.json"
    plan = Plan.model_validate_json(plan_path.read_text())
    task = Task.model_validate_json((ROOT / "examples/support-routing/task.json").read_text())
    state = tmp_path / "state.json"
    state.write_text(json.dumps(task.train[0].state))
    trace = tmp_path / "trace.jsonl"
    model = request.config.getoption("--openai-model")
    environment = dict(os.environ)
    environment.pop("TYPESAFE_API_KEY", None)
    # Prevent DSPy's transitive dotenv support from restoring a hosted credential.
    environment["PYTHON_DOTENV_DISABLED"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rulesmith.cli",
            "run",
            str(plan_path),
            str(state),
            "--backend",
            "openai",
            "--base-url",
            request.config.getoption("--openai-url"),
            *(["--model", model] if model else []),
            "--max-workers",
            "1",
            "--trace",
            str(trace),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    decided = json.loads(result.stdout)
    department = decided["outputs"]["label"]
    assert department in plan.nodes["department"].question.criteria
    assert set(decided["answers"]) == {"department", "urgency", f"{department}_review"}
    for answer in decided["answers"].values():
        if answer["type"] != "noul":
            assert sum(answer["probabilities"].values()) == pytest.approx(1)
            assert 0 <= answer["confidence"] <= 1
    assert decided["input_tokens"] > 0
    # One single-token reply per question.
    assert decided["output_tokens"] == len(decided["answers"])
    assert json.loads(trace.read_text())["called"]
