"""The examples as kept: one folder per decision, whose graph fits its task."""

import json
from pathlib import Path

import pytest
from test_cli import invoke

ROOT = Path(__file__).resolve().parents[1]
GAMES = {"doom", "chess"}
DECISIONS = sorted(p for p in (ROOT / "examples").iterdir() if p.is_dir() and p.name not in GAMES)


@pytest.mark.parametrize("folder", DECISIONS, ids=lambda p: p.name)
def test_each_decision_has_one_graph_that_fits_its_task_and_scorecard(folder):
    # A decision is its graph, or its map; with a task, the graph must validate against it, and a
    # scorecard must have been written for this task's test split, so a move or an edit that
    # leaves the folder inconsistent fails here rather than in a reader's hands.
    graphs = [p for p in (folder / "graph.rules", folder / "map.json") if p.exists()]
    assert len(graphs) == 1, f"{folder.name} needs exactly one of graph.rules and map.json"
    task = folder / "task.json"
    if not task.exists():
        assert not (folder / "evaluation.json").exists()
        return
    result = invoke("validate", task, graphs[0], "--max-nodes", 40, "--max-depth", 16)
    assert result.returncode == 0, result.stderr
    scorecard = folder / "evaluation.json"
    assert scorecard.exists(), f"{folder.name} has a task but no evaluation.json"
    tested = {example["id"] for example in json.loads(task.read_text())["test"]}
    assert {row["id"] for row in json.loads(scorecard.read_text())["examples"]} == tested
