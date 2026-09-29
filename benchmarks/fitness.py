"""The game search chart, drawn from the searches' own reports.

    python3 benchmarks/fitness.py runs/doom runs/chess \
        benchmarks/results/games.json docs/assets/fitness.svg

Every run directory `<task>-<seed>` under the run paths holds a `report.json` from
`rulesmith optimize` and, once the winner was scored, a `test.json`. The results file, the
second to last argument, keeps what the chart needs from each, so the chart can be redrawn
without the runs; the chart, the last argument, shows the best validation score found so far
against the metric calls spent finding it: episodes for Doom, games for chess.
"""

import json
import math
import sys
from pathlib import Path

TASKS = {
    "basic": "basic: kill one monster",
    "e1m1": "E1M1: play a full level",
    "chess": "chess: survive Stockfish at skill 0",
}
COLORS = ["#1d4ed8", "#b45309", "#15803d", "#7e22ce"]
PANEL, HEIGHT, LEFT, TOP, GAP = 460, 300, 48, 56, 40
PLOT_HEIGHT = HEIGHT - TOP - 64
# Room to the right of each plot for the end labels, and how far apart they must sit.
LABEL_WIDTH, LABEL_STEP = 170, 14


def collect(runs: list[Path]) -> dict:
    """What the chart and the write-up need from every finished search."""
    found = []
    for report_path in sorted(
        path for runs_dir in runs for path in runs_dir.glob("*-*/report.json")
    ):
        task, _, seed = report_path.parent.name.rpartition("-")
        if task not in TASKS or not seed.isdigit():
            continue
        report = json.loads(report_path.read_text())
        test_path = report_path.parent / "test.json"
        found.append(
            {
                "task": task,
                "seed": int(seed),
                "budget": report["search"]["max_metric_calls"],
                "metric_calls": report["metric_calls"],
                "validation_scores": report["validation_scores"],
                "discovered_at": report["discovered_at"],
                "seed_validation_score": report["seed_validation_score"],
                "validation_score": report["validation_score"],
                "test": json.loads(test_path.read_text())["score"] if test_path.exists() else None,
                "reflection_model": report["reflection_model"],
                "backend": report["backend"],
                "resolved_models": report["resolved_models"],
            }
        )
    # By task, in the order the chart draws them, then seed, whichever directory a run is in.
    found.sort(key=lambda run: (list(TASKS).index(run["task"]), run["seed"]))
    return {"runs": found}


def best_so_far(run: dict) -> list[tuple[int, float]]:
    """The score to beat after each metric call at which a candidate was found."""
    points, best = [], None
    for at, score in sorted(zip(run["discovered_at"], run["validation_scores"], strict=True)):
        if best is None or score > best:
            best = score
            points.append((at, best))
    return points


def chart(results: dict) -> str:
    runs = results["runs"]
    tasks = [task for task in TASKS if any(run["task"] == task for run in runs)]
    width = LEFT + len(tasks) * (PANEL + GAP)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{HEIGHT}" '
        f'viewBox="0 0 {width} {HEIGHT}" font-family="system-ui, sans-serif" font-size="12">',
        f'<rect width="{width}" height="{HEIGHT}" fill="#ffffff"/>',
        '<text x="16" y="22" font-size="14" font-weight="600" fill="#1c1917">Search fitness: '
        "best validation score after each candidate, by metric calls spent</text>",
        '<text x="16" y="40" fill="#57534e">Dots mark candidates that beat everything before '
        "them; a line ends where its search ended. Scores are per task, not comparable across "
        "panels.</text>",
    ]
    for index, task in enumerate(tasks):
        x0 = LEFT + index * (PANEL + GAP)
        out += panel(x0, task, [run for run in runs if run["task"] == task])
    out.append("</svg>")
    return "\n".join(out) + "\n"


def y_range(runs: list[dict]) -> tuple[float, float]:
    """The scores a panel shows, fitted to its runs in steps of 0.05 with a step of room, so a
    task scored in a narrow band still shows its moves; the ticks say what the band is."""
    scores = [score for run in runs for score in run["validation_scores"]]
    low = max(0.0, math.floor(min(scores) / 0.05 - 1) * 0.05)
    high = min(1.0, math.ceil(max(scores) / 0.05 + 1) * 0.05)
    return round(low, 2), round(high, 2)


def spread(positions: list[float], step: float) -> list[float]:
    """The positions, each at least `step` below the one above it, so labels never overlap."""
    order = sorted(range(len(positions)), key=positions.__getitem__)
    placed, last = list(positions), None
    for index in order:
        placed[index] = positions[index] if last is None else max(positions[index], last + step)
        last = placed[index]
    return placed


def panel(x0: int, task: str, runs: list[dict]) -> list[str]:
    plot_width = PANEL - LABEL_WIDTH
    budget = max(run["budget"] for run in runs)
    span = max(budget, max(run["metric_calls"] for run in runs))
    low, high = y_range(runs)
    y_bottom = TOP + 20 + PLOT_HEIGHT

    def x(calls: float) -> float:
        return x0 + plot_width * calls / span

    def y(score: float) -> float:
        return y_bottom - PLOT_HEIGHT * (score - low) / (high - low)

    out = [f'<text x="{x0}" y="{TOP + 4}" font-weight="600" fill="#1c1917">{TASKS[task]}</text>']
    ticks = [low + (high - low) * k / 4 for k in range(5)]
    for tick in ticks:
        out.append(
            f'<line x1="{x0}" y1="{y(tick):.1f}" x2="{x0 + plot_width}" y2="{y(tick):.1f}" '
            'stroke="#e7e5e4"/>'
        )
        out.append(
            f'<text x="{x0 - 6}" y="{y(tick) + 4:.1f}" text-anchor="end" fill="#57534e">'
            f"{tick:.2f}</text>"
        )
    # Ticks on the budget the runs were given; a run may overshoot it by one round.
    for tick in range(0, budget + 1, max(1, budget // 4)):
        out.append(
            f'<text x="{x(tick):.1f}" y="{y_bottom + 16}" text-anchor="middle" fill="#57534e">'
            f"{tick}</text>"
        )
    out.append(
        f'<text x="{x0 + plot_width / 2:.1f}" y="{y_bottom + 34}" text-anchor="middle" '
        'fill="#57534e">metric calls</text>'
    )
    ordered = sorted(runs, key=lambda run: run["seed"])
    finals = [best_so_far(run)[-1][1] for run in ordered]
    label_ys = spread([y(final) + 4 for final in finals], LABEL_STEP)
    for run, label_y in zip(ordered, label_ys, strict=True):
        color = COLORS[run["seed"] % len(COLORS)]
        points = best_so_far(run)
        # A step line: the best stays until the next candidate beats it, then jumps.
        path = [(x(points[0][0]), y(points[0][1]))]
        for (_, previous), (at, score) in zip(points, points[1:], strict=False):
            path += [(x(at), y(previous)), (x(at), y(score))]
        path.append((x(run["metric_calls"]), y(points[-1][1])))
        polyline = " ".join(f"{px:.1f},{py:.1f}" for px, py in path)
        out.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2"/>')
        out += [
            f'<circle cx="{x(at):.1f}" cy="{y(score):.1f}" r="3.5" fill="{color}"/>'
            for at, score in points
        ]
        final = points[-1][1]
        tested = f", test {run['test']:.2f}" if run["test"] is not None else ""
        out.append(
            f'<text x="{x(run["metric_calls"]) + 6:.1f}" y="{label_y:.1f}" fill="{color}">'
            f"seed {run['seed']} {final:.2f}{tested}</text>"
        )
    return out


def main(*paths: Path) -> None:
    *runs, results, svg = paths
    found = collect(runs)
    if not found["runs"]:
        raise SystemExit(f"no finished searches under {', '.join(map(str, runs))}")
    results.write_text(json.dumps(found, indent=2) + "\n")
    svg.write_text(chart(found))


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))
