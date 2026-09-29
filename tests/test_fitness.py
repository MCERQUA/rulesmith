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
    write_run(runs, "old-search", [0.2], [0])
    games = tmp_path / "games"
    write_run(games, "chess-0", [0.25, 0.3], [0, 500], test=0.22)
    fitness.main(runs, games, tmp_path / "games.json", tmp_path / "fitness.svg")
    results = json.loads((tmp_path / "games.json").read_text())
    # Every game search, from every run directory, with what was found and its test score.
    assert [(r["task"], r["seed"], r["test"]) for r in results["runs"]] == [
        ("basic", 0, 0.94),
        ("basic", 1, None),
        ("chess", 0, 0.22),
    ]
    svg = (tmp_path / "fitness.svg").read_text()
    assert svg.count("<polyline") == 3 and "chess: survive Stockfish" in svg
    # Each game's page gets its own chart, drawn from the same results.
    fitness.main(runs, games, tmp_path / "games.json", tmp_path / "chess.svg", tasks=["chess"])
    chess = (tmp_path / "chess.svg").read_text()
    assert chess.count("<polyline") == 1 and "kill one monster" not in chess
    assert ">games</text>" in chess and ">episodes</text>" in svg
    # The chart follows the viewer's theme: every color is a class the dark rule redefines.
    assert "@media (prefers-color-scheme: dark)" in svg and 'fill="#' not in svg
    # A dot per improvement: the seed, then the candidate that beat it, not the one that did not.
    assert svg.count("<circle") == 5
    assert "seed 0 0.90, test 0.94" in svg and "seed 1 0.33" in svg
    assert re.search(r"<text[^>]*>basic: kill one monster</text>", svg)


def test_a_continuation_is_drawn_as_the_same_search_running_on(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "basic-0", [0.33, 0.9], [0, 40], test=0.94)
    # Seeded from the first winner, so its seed scores what the first ended on.
    write_run(runs, "basic-0-more", [0.9, 0.95], [0, 100], test=0.96)
    (found,) = fitness.collect([runs])["runs"]
    assert found["stages"] == 2 and found["budget"] == 400 and found["metric_calls"] == 400
    assert found["discovered_at"] == [0, 40, 200, 300]
    assert found["validation_score"] == 0.95 and found["test"] == 0.96
    svg = fitness.chart({"runs": [found]})
    assert "seed 0 0.95, test 0.96" in svg and svg.count("<polyline") == 1


@given(st.lists(st.floats(0, 1), min_size=2, max_size=8), st.floats(1, 20))
def test_labels_are_spread_apart_without_reordering(positions, step):
    placed = fitness.spread(positions, step)
    order = sorted(range(len(positions)), key=positions.__getitem__)
    gaps = [placed[b] - placed[a] for a, b in zip(order, order[1:], strict=False)]
    assert all(gap >= step - 1e-9 for gap in gaps)
    assert all(placed[i] >= positions[i] for i in range(len(positions)))


def test_a_panel_is_fitted_to_the_scores_it_shows():
    runs = [{"validation_scores": [0.17, 0.26]}, {"validation_scores": [0.22, 0.27]}]
    assert fitness.y_range(runs) == (0.1, 0.35)
    assert fitness.y_range([{"validation_scores": [0.33, 0.97]}]) == (0.25, 1.0)


@given(st.lists(st.floats(0, 1), min_size=1, max_size=20))
def test_the_best_so_far_only_ever_rises(scores):
    run = {"validation_scores": scores, "discovered_at": list(range(len(scores)))}
    bests = [score for _, score in fitness.best_so_far(run)]
    assert bests == sorted(bests) and bests[-1] == max(scores) and bests[0] == scores[0]
