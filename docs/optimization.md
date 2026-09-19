# Optimization

`decision-compose optimize` searches for the graph with the best validation
accuracy. DSPy writes candidate graphs, and [GEPA](https://gepa-ai.github.io/gepa/guides/adapters/)
decides which to evaluate and keep.

## The search

1. **Seed.** Search starts from the graph you pass, or from a call-free graph
   that always answers the most common training label.
2. **Reflect.** When the current graph misses a training example, `GraphProposer`
   (a `dspy.Predict` with `JSONAdapter`) gets the graph, the inputs, its outputs,
   and the expected labels, then writes a complete replacement graph.
3. **Check.** The proposal is parsed and validated; no generated code runs.
   Invalid graphs score zero, and the error goes to the next proposal.
4. **Accept.** A proposal that beats its parent on the training batch is scored
   on validation.
5. **Select.** The graph with the best validation accuracy is saved. The seed
   stays eligible, so a graph that overfits training will not replace it.

The test split never reaches the optimizer. Score it once, after selection:

```sh
uv run decision-compose evaluate task.json runs/my-task/plan.json --split test \
  --output runs/my-task/test.json
```

## Settings

| Flag | Default | Meaning |
| --- | --- | --- |
| `--reflection-model` | required | DSPy model ID for the proposer |
| `--reflection-base-url` | — | Proposer endpoint, e.g. a local server's `/v1` |
| `--max-metric-calls` | 100 | Budget in example evaluations |
| `--reflection-batch` | 3 | Training examples per reflection |
| `--max-nodes` / `--max-depth` | 16 / 8 | Limits on generated graphs |
| `--seed` | 0 | Search seed |

All network commands also take `--backend`, `--model`, `--base-url`, `--timeout`,
and `--max-workers`. Retries are disabled. With `--backend typesafe`, the model
defaults to `jev-latest`; pin a version for reproducible runs.

> [!TIP]
> Reflection only runs when the graph misses something in a training batch. If
> one label dominates your data, raise `--reflection-batch` so each batch
> includes the rarer labels.

## Reports

Each run writes a new directory with `plan.json` and `report.json`. Output paths
that already exist are refused before any request is sent.

| Field | Meaning |
| --- | --- |
| `validation_accuracy` | Score of the saved graph |
| `seed_validation_accuracy` | Score of the starting graph |
| `candidates` | Graphs that reached validation, including the seed. `1` means nothing new did. |
| `reflection_calls` | Proposer calls |
| `invalid_candidate_evaluations` | Evaluations rejected by schema or graph checks |
| `metric_calls` | Example evaluations used |

Reports also record resolved model IDs, dependency versions, settings, and
task/seed hashes.

## From Python

```python
import dspy
from typesafe_sdk import TypeSafeClient

from decision_compose.graph import Task
from decision_compose.optimize import GraphProposer, SearchConfig, optimize_plan

task = Task.model_validate_json(open("task.json").read())
proposer = GraphProposer(dspy.LM("openai/<served-model-id>", api_base="http://127.0.0.1:8080/v1"))
with TypeSafeClient(base_url="http://127.0.0.1:8010", api_key="local-no-auth") as client:
    plan, report = optimize_plan(
        task, task.baseline_plan(), client, "default", proposer, SearchConfig()
    )
```

## Testing

Default tests need no credentials. They use a local HTTP fixture and fixed
proposals, so they verify wiring, not model quality.

```sh
uv run pytest -q
```

Real inference against a running Ruling server:

```sh
uv run pytest tests/test_ruling.py --ruling-url http://127.0.0.1:8010 -q
```

Add real DSPy graph generation. This fails unless a generated graph reaches
validation and at least matches the seed:

```sh
uv run pytest tests/test_ruling.py --ruling-url http://127.0.0.1:8010 \
  --ruling-reflection-model "$REFLECTION_MODEL" \
  --ruling-reflection-url "$REFLECTION_BASE_URL" -q
```

Hosted providers (paid):

```sh
export DECISION_COMPOSE_REFLECTION_MODEL="$REFLECTION_MODEL"
uv run pytest --live tests/test_live.py -q
```

A local smoke run with a `Qwen3.5-9B-4bit` proposer and Ruling's `Qwen3.5-4B-4bit`
judge went from an always-`sales` seed to a generated one-Choice graph: 1/3 → 3/3
on validation and on test, in 48 metric calls. That's 12 authored examples, which
shows the pipeline works end to end, not a benchmark result.

## Limits

- Search optimizes classification only, with exact-match accuracy on a `label` output.
- Graphs are finite and static: no loops, tools, or dynamic nodes.
- The budget counts example evaluations, not requests or dollars, and excludes
  proposer calls. It can overshoot by one iteration.
- Provider errors stop the search rather than counting as wrong answers.
  In-flight requests are not canceled.
- Node inputs go to the judge; training examples and feedback go to the proposer.
- Extra structure is kept only when it raises the score. Small validation sets
  make selection noisy.
- DSPy's LiteLLM dependency may load `.env` files, so shell variables may not be
  the only configuration source.
