"""The Doom search chart, drawn from the searches' own reports.

    python3 benchmarks/fitness.py runs/doom benchmarks/results/doom.json docs/assets/fitness.svg

Every run directory `<task>-<seed>` under the first path holds a `report.json` from
`rulesmith optimize` and, once the winner was scored, a `test.json`. The results file keeps
what the chart needs from each, so the chart can be redrawn without the runs, and the chart
shows the best validation score found so far against the metric calls spent finding it.
"""

import json
import sys
from pathlib import Path

TASKS = {"basic": "basic: kill one monster", "e1m1": "E1M1: play a full level"}
COLORS = ["#1d4ed8", "#b45309", "#15803d", "#7e22ce"]
PANEL, HEIGHT, LEFT, TOP, GAP = 420, 300, 48, 56, 40
PLOT_HEIGHT = HEIGHT - TOP - 64


def collect(runs: Path) -> dict:
    """What the chart and the write-up need from every finished search."""
    found = []
    for report_path in sorted(runs.glob("*-*/report.json")):
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


def panel(x0: int, task: str, runs: list[dict]) -> list[str]:
    plot_width = PANEL - 90
    budget = max(max(run["budget"], run["metric_calls"]) for run in runs)
    y_bottom = TOP + 20 + PLOT_HEIGHT

    def x(calls: float) -> float:
        return x0 + plot_width * calls / budget

    def y(score: float) -> float:
        return y_bottom - PLOT_HEIGHT * score

    out = [f'<text x="{x0}" y="{TOP + 4}" font-weight="600" fill="#1c1917">{TASKS[task]}</text>']
    for tick in (0, 0.25, 0.5, 0.75, 1):
        out.append(
            f'<line x1="{x0}" y1="{y(tick):.1f}" x2="{x0 + plot_width}" y2="{y(tick):.1f}" '
            'stroke="#e7e5e4"/>'
        )
        out.append(
            f'<text x="{x0 - 6}" y="{y(tick) + 4:.1f}" text-anchor="end" fill="#57534e">'
            f"{tick:.2f}</text>"
        )
    for tick in range(0, budget + 1, max(1, budget // 4)):
        out.append(
            f'<text x="{x(tick):.1f}" y="{y_bottom + 16}" text-anchor="middle" fill="#57534e">'
            f"{tick}</text>"
        )
    out.append(
        f'<text x="{x0 + plot_width / 2:.1f}" y="{y_bottom + 34}" text-anchor="middle" '
        'fill="#57534e">metric calls</text>'
    )
    for run in sorted(runs, key=lambda run: run["seed"]):
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
            f'<text x="{x(run["metric_calls"]) + 6:.1f}" y="{y(final) + 4:.1f}" fill="{color}">'
            f"seed {run['seed']} {final:.2f}{tested}</text>"
        )
    return out


def main(runs: Path, results: Path, svg: Path) -> None:
    found = collect(runs)
    if not found["runs"]:
        raise SystemExit(f"no finished searches under {runs}")
    results.write_text(json.dumps(found, indent=2) + "\n")
    svg.write_text(chart(found))


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:4]))
