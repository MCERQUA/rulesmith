# decision-compose

![decision-compose — Design and optimize decision graphs](image.png)

**decision-compose lets DSPy build decision graphs for you.** Give it a
classification task and labeled examples. A generative model proposes graphs of
small typed judgments (Choice, Score, Noul), [GEPA](https://github.com/gepa-ai/gepa)
scores them against a local Ruling or hosted Jev judge, and the best graph
is saved as plain JSON you can run, inspect, and edit.

This is an independent research prototype, not affiliated with TypeSafe. It
optimizes graphs and question instructions, not model weights.

## Installation

Requires Python 3.12–3.14 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
```

## Getting started

You need example inputs. Expected outputs (labels) are optional: if you don't
have them, a strong model drafts them for you to review.

```mermaid
flowchart LR
  I["inputs.json<br/>inputs only"] -->|"draft-labels"| R["task.json<br/>review the labels"]
  L["task.json<br/>inputs + labels"] --> O
  R --> O["optimize"]
  O --> P["plan.json<br/>your graph"]
  P --> RUN["run / evaluate"]
```

**1. Start a judge.** Ruling runs locally and needs no account:

```sh
cd ../ruling && uv sync && uv run ruling serve
```

**2. Configure a proposer.** This is the generative model that writes graphs.
Any [DSPy model identifier](https://dspy.ai/learn/programming/language_models/)
works. For a local OpenAI-compatible server:

```sh
export REFLECTION_MODEL=openai/<served-model-id>
export REFLECTION_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=local-no-auth
```

**3. Describe the task.** Pick the case that matches your data.

<details open>
<summary><b>I only have inputs</b></summary>

List the allowed labels and your raw inputs in `inputs.json`:

```json
{
  "description": "Route each support ticket to billing, technical, or sales.",
  "labels": ["billing", "technical", "sales"],
  "inputs": ["I was charged twice this month.", "The export button returns a 500 error."]
}
```

Then have the proposer model label them. `--validation` and `--test` set how
many inputs are held out for selection and final scoring:

```sh
uv run decision-compose draft-labels inputs.json --output task.json \
  --labeler-model "$REFLECTION_MODEL" --labeler-base-url "$REFLECTION_BASE_URL" \
  --validation 10 --test 10
```

> [!IMPORTANT]
> Review and correct `task.json` before optimizing. Search learns to agree with
> these labels, including their mistakes.

This is DSPy's teacher–student pattern: a large model labels, and the small
judge is fit to its judgment.

</details>

<details>
<summary><b>I have labeled examples</b></summary>

Write `task.json` directly, like [examples/task.json](examples/task.json):

```json
{
  "description": "Route each support ticket to billing, technical, or sales.",
  "labels": ["billing", "technical", "sales"],
  "train": [{"id": "t1", "state": "I was charged twice.", "label": "billing"}],
  "validation": [...],
  "test": [...]
}
```

`train` drives proposals, `validation` picks the winner, and `test` is scored once at the end.

</details>

**4. Build the graph.**

```sh
uv run decision-compose optimize task.json \
  --backend ruling --max-workers 1 \
  --reflection-model "$REFLECTION_MODEL" \
  --reflection-base-url "$REFLECTION_BASE_URL" \
  --output runs/my-task
```

This writes `runs/my-task/plan.json` (the graph) and `report.json` (scores).
Compare `validation_score` with `seed_validation_score`.

**5. Evaluate and run it.** The proposer is not needed from here on.

```sh
uv run decision-compose evaluate task.json runs/my-task/plan.json \
  --backend ruling --max-workers 1 --split test --output runs/my-task/test.json

uv run decision-compose run runs/my-task/plan.json examples/state.json \
  --backend ruling --max-workers 1
```

## How it works

```mermaid
flowchart LR
  T["task.json<br/>labels + examples"] --> S["Starting graph<br/>majority label"]
  S --> E["Run on training examples<br/>via Ruling or Jev"]
  E -->|"errors + expected labels"| P["DSPy proposer<br/>writes new graph JSON"]
  P -->|"schema check"| E
  E -->|"better on the batch"| V["Score on validation examples"]
  V --> B["Best graph<br/>plan.json"]
```

The proposer sees where the current graph answered wrong and writes a complete
replacement: new calls, rewritten questions, branches, and joins. A proposal is
kept only if it beats its parent on the training batch, and the graph with the
best validation score is saved. The test split never reaches the optimizer.

The proposer can use the full DAG language. This hand-written example
([examples/dag.json](examples/dag.json)), not a search result, asks two questions
in parallel, then runs only the specialist that matches:

```mermaid
flowchart LR
  IN(["ticket"]) --> D["department<br/>choice"]
  IN --> U["urgency<br/>score"]
  D --> C["context<br/>merge"]
  U --> C
  IN --> C
  C -.->|"if billing"| B["billing_review"]
  C -.->|"if technical"| T["technical_review"]
  C -.->|"if sales"| S["sales_review"]
  B --> CO["specialist_result<br/>coalesce"]
  T --> CO
  S --> CO
  D --> A["assessment<br/>merge"]
  U --> A
  CO --> A
  D ==> LBL(["label"])
  A ==> AS(["assessment"])
```

See [Graph format](docs/graphs.md) for the JSON behind each node.

## Play Doom

The same search can build a graph that plays [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)
scenarios. Each step, the graph gets the visible objects and game variables as
JSON and outputs the next button. Its score is the episode's reward, not a label
match, so no labeled examples are needed.

```sh
uv run decision-compose optimize examples/doom/basic.json \
  --backend ruling --max-workers 1 \
  --reflection-model "$REFLECTION_MODEL" \
  --reflection-base-url "$REFLECTION_BASE_URL" \
  --output runs/doom

uv run decision-compose evaluate examples/doom/basic.json runs/doom/plan.json \
  --backend ruling --max-workers 1 --split test --visible --output runs/doom/test.json
```

`--visible` opens the game window so you can watch it play. See [Doom](docs/doom.md)
for the task format and scenarios.

## Results

![Search fitness: best validation score after each search round](docs/assets/fitness.svg)

**`basic`**: search started from a graph that only strafes left. Within three
rounds it found a graph that strafes toward the monster and fires once it is
centered. On held-out test seeds it killed every monster:

| Judge | Validation | Test | Test rewards (best possible ≈ 97) |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B via Ruling (local) | 0.33 → 0.97 | 0.94 | 79, 38, 83 |
| Jev (`jev-1.13.0`, hosted) | 0.33 → 0.97 | 0.94 | 75, 38, 83 |

![The found graph killing the monster on test seed 12](docs/assets/doom/basic-seed12.gif)

[All 12 seeds back to back](docs/assets/doom/basic-reel.mp4): 12 of 12 kills.

**E1M1**: search started from a graph that only walks forward, with Claude Sonnet 5
proposing graphs and the local 4B judge. Its best graph so far
([examples/doom/graphs/e1m1-searched.json](examples/doom/graphs/e1m1-searched.json))
is 28 nodes of reads and routing with no judge calls. On an unseen test seed it
survived the full 90 seconds with 6 kills, 4 items, and 50 map cells explored,
for a reward of 2,090 against about 960 for walking forward. It has not reached
the exit yet. [Watch it](docs/assets/doom/e1m1-searched.mp4).

## Use from Python

A graph runs as a `dspy.Module`:

```python
from pathlib import Path

from typesafe_sdk import TypeSafeClient

from decision_compose.graph import Plan
from decision_compose.runtime import DecisionProgram

plan = Plan.model_validate_json(Path("runs/my-task/plan.json").read_text())
with TypeSafeClient(base_url="http://127.0.0.1:8010", api_key="local-no-auth") as client:
    program = DecisionProgram(plan, client, model="default")
    result = program(state="Please correct the duplicate charge on my invoice.")
    print(result.label, result.answers)
```

## Backends

| Backend | Flag | Endpoint | Credential |
| --- | --- | --- | --- |
| Ruling (local) | `--backend ruling` | `RULING_BASE_URL`, default `http://127.0.0.1:8010` | `RULING_API_KEY` if auth is on |
| Jev (hosted) | `--backend typesafe` (default) | `TYPESAFE_BASE_URL` | `TYPESAFE_API_KEY` |

Ruling serializes requests, so use `--max-workers 1`. Hosted calls may incur
charges; the search budget counts examples, not requests or dollars.

## Learn more

- [Graph format](docs/graphs.md): nodes, inputs, conditions, execution, and writing graphs by hand.
- [Optimization](docs/optimization.md): search settings, reports, testing, and limits.
- [Doom](docs/doom.md): task format, observations, scoring, and watching a graph play.

## Development

```sh
uv run pytest -q
uv run ruff check src tests && uv run ruff format --check src tests
```
