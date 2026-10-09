"""The benchmark charts in the docs are drawn from the results files, so a chart cannot claim a
number the results do not hold."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("charts", ROOT / "benchmarks" / "charts.py")
charts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(charts)


def test_search_is_shown_beside_the_best_baseline_that_asks_on_every_row():
    found = {
        "tfidf": (0.8, 0.0),
        "direct": (0.7, 1.0),
        "dspy": (0.9, 1.0),
        "dspy-typed": (0.85, 1.0),
        "majority": (0.95, 0.0),
        "search": (0.88, 0.2),
    }
    shown = charts.comparison(found)
    assert [(name, calls, accuracy) for name, calls, accuracy, _ in shown] == [
        ("TF-IDF", 0.0, 0.8),
        ("DSPy", 1.0, 0.9),
        ("rulesmith", 0.2, 0.88),
    ]
    assert [ours for *_, ours in shown] == [False, False, True]


def test_the_charts_in_the_docs_are_the_ones_the_results_draw():
    assert (ROOT / "docs/assets/benchmarks.svg").read_text() == charts.benchmarks_chart()
    assert (ROOT / "docs/assets/escalation.svg").read_text() == charts.escalation_chart()
