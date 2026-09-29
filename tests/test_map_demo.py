"""The demonstration map's summary: the whole, and unit by unit."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "map.py"
spec = importlib.util.spec_from_file_location("map_demo", SCRIPT)
map_demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(map_demo)


def test_the_summary_credits_each_example_to_the_unit_that_decided_it():
    evaluation = {
        "score": 0.75,
        "examples": [
            {"id": "jobs-1", "correct": True, "decided_by": "postings.rule_1", "calls": {}},
            {
                "id": "jobs-2",
                "correct": False,
                "decided_by": "postings.rule_2",
                "calls": {"postings.rule_2": {}},
            },
            {"id": "clothing-1", "correct": True, "decided_by": "reviews.rule_1", "calls": {}},
            {
                "id": "banking77-1",
                "correct": True,
                "decided_by": "intents.rule_1",
                "calls": {"intents.rule_1": {}},
            },
        ],
    }
    summary = map_demo.summarize(evaluation)
    assert summary["accuracy"] == 0.75 and summary["calls_per_example"] == 0.5
    assert summary["units"]["postings"] == {
        "examples": 2,
        "correct": 1,
        "calls": 1,
        "accuracy": 0.5,
        "calls_per_example": 0.5,
    }
    assert summary["units"]["reviews"]["calls_per_example"] == 0
    assert set(summary["units"]) == {"postings", "reviews", "intents"}
