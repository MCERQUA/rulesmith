"""The benchmark charts in the docs, drawn from the results files.

    python3 benchmarks/charts.py

Writes docs/assets/benchmarks.svg, accuracy against judge calls per row for rulesmith's search
and the baselines on each dataset, and docs/assets/escalation.svg, accuracy against the share of
rows sent to the larger judge. Every number is a mean over the seeds in benchmarks/results; the
tables in docs/benchmarks.md carry the spread and the tests of significance.
"""

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
ASSETS = ROOT / "docs" / "assets"

# Where rules over fields help first, then where the input is one piece of free text.
DATASETS = ("clothing", "jobs", "news", "trec", "banking77")
# The baselines that ask a language model on every row; the best of them per dataset is the
# fair comparison, so search is never shown beside a weak one.
ASKING = {
    "direct": "direct call",
    "dspy": "DSPy",
    "dspy-cot": "DSPy, reasoning",
    "dspy-typed": "DSPy typed",
}
ESCALATION = (("small", "4B alone"), ("large", "Clef-flash alone"), ("escalate", "escalating"))

# rulesmith is the one hue; every baseline is context, in the neutral, and named beside its mark.
STYLE = """
.bg { fill: #fcfcfb } .ink { fill: #0b0b0b } .muted { fill: #52514e }
.grid { stroke: #e4e3df } .axis { stroke: #b5b4ae }
.us { fill: #2a78d6 } .them { fill: #6b6a66 } .ring { stroke: #fcfcfb }
.mix { stroke: #6b6a66 }
.halo { stroke: #fcfcfb; stroke-width: 4px; paint-order: stroke; stroke-linejoin: round }
@media (prefers-color-scheme: dark) {
  .bg { fill: #1a1a19 } .ink { fill: #ffffff } .muted { fill: #c3c2b7 }
  .grid { stroke: #33322f } .axis { stroke: #57564f }
  .us { fill: #3987e5 } .them { fill: #8f8e89 } .ring { stroke: #1a1a19 }
  .mix { stroke: #8f8e89 }
  .halo { stroke: #1a1a19 }
}
"""

PANEL, WIDE, GAP, LEFT, TOP, PLOT = 176, 300, 62, 52, 74, 190
FONT = 'font-family="system-ui, sans-serif" font-size="12"'


def by_method(dataset: str) -> dict[str, tuple[float, float]]:
    """Mean accuracy and judge calls per row of each method on a dataset, over its seeds."""
    runs = json.loads((RESULTS / f"{dataset}.json").read_text())["runs"]
    methods = {r["method"] for r in runs}
    return {
        m: (
            mean(r["accuracy"] for r in runs if r["method"] == m),
            mean(r["calls_per_example"] for r in runs if r["method"] == m),
        )
        for m in methods
    }


def comparison(found: dict[str, tuple[float, float]]) -> list[tuple[str, float, float, bool]]:
    """What a panel shows: search, the best baseline that asks the model on every row, and the
    trained classifier that never asks, as (name, calls, accuracy, is rulesmith)."""
    best = max((m for m in ASKING if m in found), key=lambda m: found[m][0])
    return [
        ("TF-IDF", found["tfidf"][1], found["tfidf"][0], False),
        (ASKING[best], found[best][1], found[best][0], False),
        ("rulesmith", found["search"][1], found["search"][0], True),
    ]


def escalated() -> dict[str, list[tuple[str, float, float]]]:
    """Per dataset, each way of judging as (name, larger-judge calls per row, accuracy)."""
    runs = json.loads((RESULTS / "escalation.json").read_text())["runs"]
    found = {}
    for dataset in sorted({r["dataset"] for r in runs}):
        found[dataset] = []
        for graph, name in ESCALATION:
            mine = [r for r in runs if r["dataset"] == dataset and r["graph"] == graph]
            large = mean(r["calls_per_row"]["large"] for r in mine)
            found[dataset].append((name, large, mean(r["accuracy"] for r in mine)))
    return found


def ticks(low: float, high: float) -> tuple[float, float, list[float]]:
    """A y range padded to the nearest 0.05 around the values, and its gridlines."""
    step = 0.05 if high - low < 0.25 else 0.1
    bottom = max(0.0, (int(low / step) - (0 if low % step else 1)) * step)
    top = min(1.0, (int(high / step) + 1) * step)
    count = round((top - bottom) / step)
    return bottom, top, [round(bottom + i * step, 2) for i in range(count + 1)]


def labels(points: list[tuple[float, float]]) -> list[int]:
    """Which way each point's label goes: up, unless a higher point sits near enough that the
    labels would meet, in which case the lower one is labeled below."""
    sides = [-1] * len(points)
    for i, (x, y) in enumerate(points):
        for j, (x2, y2) in enumerate(points):
            if i != j and abs(x - x2) < 70 and 0 <= y - y2 < 26:
                sides[i] = 1
    return sides


def panel(left: float, title: str, points, x_label: str, segment=None, size=PANEL) -> list[str]:
    """One small multiple: its own y scale, x from 0 to 1, each point named beside it."""
    values = [y for _, _, y, *_ in points]
    bottom, top, grid = ticks(min(values), max(values))
    x = lambda v: left + 10 + v * (size - 20)  # noqa: E731
    y = lambda v: TOP + PLOT - (v - bottom) / (top - bottom) * PLOT  # noqa: E731
    out = [f'<text x="{left}" y="{TOP - 14}" font-weight="600" class="ink">{title}</text>']
    for g in grid:
        out.append(
            f'<line x1="{left}" y1="{y(g):.1f}" x2="{left + size}" y2="{y(g):.1f}" class="grid"/>'
        )
        out.append(
            f'<text x="{left - 6}" y="{y(g) + 4:.1f}" text-anchor="end" class="muted">'
            f"{g:.2f}</text>"
        )
    base = TOP + PLOT
    out.append(f'<line x1="{left}" y1="{base}" x2="{left + size}" y2="{base}" class="axis"/>')
    for v in (0, 0.5, 1):
        out.append(
            f'<text x="{x(v):.1f}" y="{base + 16}" text-anchor="middle" class="muted">{v:g}</text>'
        )
    out.append(
        f'<text x="{left + size / 2}" y="{base + 34}" text-anchor="middle" class="muted">'
        f"{x_label}</text>"
    )
    if segment:
        (x1, y1), (x2, y2) = segment
        out.append(
            f'<line x1="{x(x1):.1f}" y1="{y(y1):.1f}" x2="{x(x2):.1f}" y2="{y(y2):.1f}" '
            'class="mix" stroke-width="1.5" stroke-dasharray="4 4"/>'
        )
    placed = [(x(c), y(a)) for _, c, a, *_ in points]
    for (name, calls, accuracy, *ours), (px, py), side in zip(
        points, placed, labels(placed), strict=True
    ):
        kind = "us" if ours and ours[0] else "them"
        anchor = "start" if px < left + 30 else "end" if px > left + size - 30 else "middle"
        weight = ' font-weight="600"' if kind == "us" else ""
        out.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" class="{kind} ring" stroke-width="2">'
            f"<title>{name}: {accuracy:.3f} accuracy, {calls:.2f} calls per row</title></circle>"
        )
        ly = py - 10 if side < 0 else py + 19
        out.append(
            f'<text x="{px:.1f}" y="{ly:.1f}" text-anchor="{anchor}" class="ink halo"{weight}>'
            f"{name} {accuracy:.3f}</text>"
        )
    return out


def svg(width: int, height: int, title: str, subtitle: str, body: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" {FONT}>',
            f"<style>{STYLE}</style>",
            f'<rect width="{width}" height="{height}" class="bg"/>',
            f'<text x="16" y="24" font-size="14" font-weight="600" class="ink">{title}</text>',
            f'<text x="16" y="42" class="muted">{subtitle}</text>',
            *body,
            "</svg>",
            "",
        ]
    )


def benchmarks_chart() -> str:
    body = []
    for i, dataset in enumerate(DATASETS):
        shown = comparison(by_method(dataset))
        body += panel(LEFT + i * (PANEL + GAP), dataset, shown, "judge calls per row")
    width = LEFT + len(DATASETS) * (PANEL + GAP) - GAP + 24
    return svg(
        width,
        TOP + PLOT + 52,
        "Accuracy against judge calls per row, on 300 held-out rows",
        "rulesmith's searched rules against the best baseline that asks on every row, and a "
        "classifier that never asks. Means of three seeds; each panel has its own scale.",
        body,
    )


def escalation_chart() -> str:
    body = []
    for i, (dataset, points) in enumerate(escalated().items()):
        marked = [(name, calls, acc, name == "escalating") for name, calls, acc in points]
        ends = [(c, a) for n, c, a in points if n != "escalating"]
        body += panel(
            LEFT + i * (WIDE + GAP), dataset, marked, "larger-judge calls per row", ends, WIDE
        )
    width = LEFT + 2 * (WIDE + GAP) - GAP + 24
    return svg(
        width,
        TOP + PLOT + 52,
        "Escalating to a larger judge only where the small one is unsure",
        "Above the dashed line is better than sending that share of rows to the larger judge at "
        "random. Means of three seeds.",
        body,
    )


def main() -> None:
    (ASSETS / "benchmarks.svg").write_text(benchmarks_chart())
    (ASSETS / "escalation.svg").write_text(escalation_chart())


if __name__ == "__main__":
    main()
