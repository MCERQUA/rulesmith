"""The Doom search chart, redrawn from the searches' own reports."""

import importlib.util
import json
import re
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "fitness.py"
spec = importlib.util.spec_from_file_location("fitness", SCRIPT)
fitness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fitness)


def report(scores, found_at, budget=200):
    return {
        "search": {"max_metric_calls": budget},
        "metric_calls": budget,
        "validation_scores": scores,
        "discovered_at": found_at,
        "seed_validation_score": scores[0],
        "validation_score": max(scores),
        "reflection_model": "openai/fixture",
        "backend": "ruling",
        "resolved_models": [],
    }


def write_run(runs: Path, name: str, scores, found_at, test=None):
    run = runs / name
    run.mkdir(parents=True)
    (run / "report.json").write_text(json.dumps(report(scores, found_at)))
    if test is not None:
        (run / "test.json").write_text(json.dumps({"score": test}))


def test_the_chart_and_results_come_from_the_reports(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "basic-0", [0.33, 0.9, 0.7], [0, 40, 90], test=0.94)
    write_run(runs, "basic-1", [0.33], [0])
    write_run(runs, "chess-search", [0.2], [0])
    fitness.main(runs, tmp_path / "doom.json", tmp_path / "fitness.svg")
    results = json.loads((tmp_path / "doom.json").read_text())
    # Only the Doom searches, with what was found and where the winner scored on test.
    assert [(r["task"], r["seed"], r["test"]) for r in results["runs"]] == [
        ("basic", 0, 0.94),
        ("basic", 1, None),
    ]
    svg = (tmp_path / "fitness.svg").read_text()
    assert svg.count("<polyline") == 2
    # A dot per improvement: the seed, then the candidate that beat it, not the one that did not.
    assert svg.count("<circle") == 3
    assert "seed 0 0.90, test 0.94" in svg and "seed 1 0.33" in svg
    assert re.search(r"<text[^>]*>basic: kill one monster</text>", svg)


@given(st.lists(st.floats(0, 1), min_size=1, max_size=20))
def test_the_best_so_far_only_ever_rises(scores):
    run = {"validation_scores": scores, "discovered_at": list(range(len(scores)))}
    bests = [score for _, score in fitness.best_so_far(run)]
    assert bests == sorted(bests) and bests[-1] == max(scores) and bests[0] == scores[0]
