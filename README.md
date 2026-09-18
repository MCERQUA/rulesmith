# jev-compose

Learn small, typed decision trees with GEPA. Execute their judgments with Jev,
through a DSPy module and TypeSafe's official Python SDK.

You provide a task, labeled examples, and a seed tree. GEPA proposes changes to
the questions and branching structure, evaluates them, and returns a selected
tree. The saved tree runs without the reflection model.

This is an independent research prototype, not affiliated with TypeSafe. It
optimizes programs; it does not train Jev or reproduce its weights.

## Design

```text
task + seed tree -> GEPA -> candidate JSON -> schema and type checks
                     ^                              |
                     |                              v
                 feedback <- predictions <- DSPy module -> Jev
```

Each program contains a question bank and a tree:

| Primitive | Result | Tree operation |
| --- | --- | --- |
| `choice` | One declared category | `switch` over every option |
| `noul` | Probability of yes | `threshold` in [0, 1] |
| `score` | Expected rubric position | `threshold` in [0, number of levels − 1] |
| `return` | One task label | Terminal node |

All questions evaluate the original state together in one Jev request. The
Python interpreter then follows the tree. Questions do not see sibling answers,
and questions on unused branches are still evaluated. A question can be reused
at several positions without another request. Constant trees make no request.

DSPy provides the callable module interface and reflection-model access. A custom
GEPA adapter exposes the **whole JSON program** as the mutable component. Wrapping
an HTTP call in a DSPy module alone would not expose its graph to optimization.

## Install

Requires Python 3.12–3.14 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
uv run jev-compose validate examples/task.json examples/seed.json
```

The dependency lock pins DSPy, GEPA, and the TypeSafe SDK. No local model download
or GPU is required. `validate` and the default tests need no API credentials.

## Run the example

`examples/task.json` contains 12 authored support tickets: six training, three
validation, and three test examples. It is a small wiring example, not a benchmark
or evidence of improvement. The seed routes tickets with one Choice question.

Set `TYPESAFE_API_KEY` in your shell using your own credential. The SDK reads it
from the environment; this project does not load `.env` files. The following
commands make billable API calls.

Evaluate the seed on validation examples:

```sh
uv run jev-compose evaluate examples/task.json examples/seed.json \
  --split validation --output runs/seed-validation.json
```

Choose a valid [DSPy model identifier](https://dspy.ai/learn/programming/language_models/)
for a generative reflection model and set `REFLECTION_MODEL` to that identifier.
Configure that provider's API credential as required by DSPy. Jev executes the
judgments; the reflection model generates revised programs.

```sh
uv run jev-compose optimize examples/task.json examples/seed.json \
  --reflection-model "$REFLECTION_MODEL" \
  --max-metric-calls 100 --seed 0 --output runs/search
```

The new directory contains `plan.json` and `report.json`. The report includes
seed and selected validation accuracy, candidate scores, search settings,
evaluation count, resolved Jev model IDs, dependency versions, and fingerprints
of the task and seed. Keep the original inputs alongside your experiment records.
Existing output paths are refused before requests are sent.
An already successful seed may remain unchanged; improvement is not guaranteed.

After selecting your program, evaluate it on the untouched test split:

```sh
uv run jev-compose evaluate examples/task.json runs/search/plan.json \
  --split test --output runs/test.json
```

Evaluation saves each example's expected and predicted labels, branch path,
answer distributions, resolved model, and reported token usage. The optimizer
uses exact-match accuracy and never evaluates the test split. Do not repeatedly
use test results to choose programs; that would make it another validation set.

To execute one JSON state, save it to a file and run:

```sh
uv run jev-compose run runs/search/plan.json path/to/state.json
```

Every network command accepts `--model`, `--base-url`, and `--timeout`. The default
model is `jev-latest`; pin a provider-supported model version for experiments.
`--base-url` overrides the SDK's `TYPESAFE_BASE_URL`. Reflection endpoints can be
overridden with `--reflection-base-url`.

## Tree format

This example returns `review` below the cutoff and `accept` at or above it:

```json
{
  "questions": {
    "eligible": {
      "type": "noul",
      "instructions": "Does the supplied policy explicitly make this request eligible?"
    }
  },
  "root": {
    "type": "threshold",
    "question": "eligible",
    "cutoff": 0.8,
    "below": {"type": "return", "label": "review"},
    "above": {"type": "return", "label": "accept"}
  }
}
```

The cutoff is illustrative, not calibrated. Choose thresholds using your own
data. A Score's fractional rubric position is not a probability of success, and
the returned confidence is not the probability that the whole tree is correct.

Question instructions and descriptions are nonempty strings. Choice supports
2–255 options; Score supports 2–10 distinct levels. The first version deliberately
supports a subset of the broader TypeSafe question schema.

Validation rejects unknown fields, unused or missing questions, incompatible
branch types, incomplete Choice cases, invalid cutoffs, and labels outside the
task. Search defaults to at most eight questions and six decision levels;
`--max-questions` and `--max-depth` change these limits. Duplicate example IDs or
JSON-equivalent states are rejected across the task to catch obvious leakage.

## Python

```python
from pathlib import Path

from typesafe_sdk import TypeSafeClient

from jev_compose.graph import Plan
from jev_compose.runtime import JevProgram

plan = Plan.model_validate_json(Path("examples/seed.json").read_text())
with TypeSafeClient() as client:
    program = JevProgram(plan, client, model="jev-latest")
    prediction = program(state="Please correct the duplicate charge on my invoice.")
    print(prediction.label, prediction.path, prediction.answers)
```

`JevProgram` is a `dspy.Module`. `optimize_plan` accepts a `Task`, seed `Plan`,
TypeSafe client, model ID, reflection callable, and `SearchConfig`; the reflection
callable returns a string containing the proposed JSON tree. No arbitrary Python
generated by the optimizer is executed.

## Verification

```sh
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv build
```

Unit tests cover tree semantics and provider response validation, including
property tests for threshold partitions and serialization. Integration tests
run the official SDK over local HTTP, the actual DSPy module and LM interfaces,
and the real GEPA search engine. Protocol fixtures and fixed reflection proposals
make these tests deterministic; they are not evidence of live model quality.

Paid live checks are opt-in:

```sh
# Configure TYPESAFE_API_KEY and your reflection provider's credential first.
export JEV_COMPOSE_REFLECTION_MODEL="$REFLECTION_MODEL"
uv run pytest --live tests/test_live.py -q
```

These check hosted execution and optimization, not a statistically significant
quality improvement. No live benchmark result is claimed in this repository.

## Limits

- A single shared-state batch followed by branching; no dependent model stages,
  tools, loops, generated code, or UI.
- GEPA's metric-call budget counts example evaluations, can exceed its target by
  an iteration, and excludes reflection calls. It is **not a dollar cap**. Client
  retries are disabled in the CLI; enforce spending limits with your providers.
- Invalid candidate trees receive validation feedback. Provider failures stop
  the run instead of being scored as bad task performance.
- Train examples and execution feedback are sent to the reflection provider;
  input states and questions are sent to Jev. Use appropriate data for those
  services. Saved evaluation reports contain model answers.
- The seed controls optimizer sampling, not hosted model determinism. A small
  validation set can be overfit; use representative data and a final test split.

## Sources

- [TypeSafe primitives](https://docs.typesafe.ai/primitives/choice)
- [Parallel question semantics](https://docs.typesafe.ai/cookbooks/parallel_questions)
- [DSPy modules](https://dspy.ai/learn/programming/modules/)
- [GEPA custom adapters](https://gepa-ai.github.io/gepa/guides/adapters/)
- [GEPA full-program evolution](https://gepa-ai.github.io/gepa/tutorials/dspy_full_program_evolution/)
