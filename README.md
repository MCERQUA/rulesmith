# decision-compose

Build and optimize typed directed acyclic graphs of Jev calls. DSPy provides the
execution module; GEPA evolves the complete graph; TypeSafe's official SDK makes
the model requests.

Graphs support arbitrary finite acyclic topology over the supported node types:
sequential calls, parallel fan-out, shared dependencies, conditional branches,
joins, and multiple outputs. Each call can consume the original input and earlier
answers. A saved graph runs without the reflection model.

This independent research prototype is not affiliated with TypeSafe. It optimizes
programs, not Jev's weights. No live benchmark improvement is claimed.

## Install and run

Requires Python 3.12–3.14 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
uv run decision-compose validate examples/task.json examples/seed.json
uv run decision-compose validate examples/task.json examples/dag.json
```

Validation and default tests need no credentials. For model calls, set
`TYPESAFE_API_KEY` in your shell. The SDK reads the environment; this project does
not load `.env` files. Commands that execute calls are billable.

`examples/task.json` contains 12 authored support tickets: six training, three
validation, and three test examples. This is a wiring example, not a benchmark.
`seed.json` uses one Choice call. `dag.json` demonstrates concurrent department
and urgency calls, a shared context join, a conditional specialist call that sees
the earlier answers, and a final join:

```text
                department ─┐                 billing_review ─┐
input ────────<             ├─ context ──────< technical_review ├─ coalesce ─┐
                urgency ────┘                 sales_review ────┘            ├─ assessment
                department + urgency ──────────────────────────────────────┘
```

Only the specialist matching the department runs. The label and assessment are
separate outputs; the classifier's label is the department.

To execute any graph, put its input state in a JSON file:

```sh
uv run decision-compose run examples/dag.json path/to/state.json --max-workers 4
```

To evaluate or optimize a classification task:

```sh
uv run decision-compose evaluate examples/task.json examples/seed.json \
  --split validation --output runs/seed-validation.json

uv run decision-compose optimize examples/task.json examples/seed.json \
  --reflection-model "$REFLECTION_MODEL" \
  --max-nodes 16 --max-depth 8 --max-workers 4 \
  --max-metric-calls 100 --seed 0 --output runs/search

uv run decision-compose evaluate examples/task.json runs/search/plan.json \
  --split test --output runs/test.json
```

Set `REFLECTION_MODEL` to a valid [DSPy model identifier](https://dspy.ai/learn/programming/language_models/)
and configure its provider credential. The generative reflection model proposes
JSON graphs. Jev executes their typed judgments. GEPA can change nodes,
instructions, connections, input mappings, conditions, and outputs.

The new search directory contains `plan.json` and `report.json`. Reports include
validation scores, search and execution settings, example evaluation counts,
resolved provider model IDs, dependency versions, and task/seed fingerprints.
Existing output paths are refused before requests are sent. Selection uses
validation accuracy; improvement is not guaranteed. The test split is never sent
to the optimizer. Evaluate it after selecting the graph.

All network commands accept `--model`, `--base-url`, `--timeout`, and
`--max-workers`. The default model is `jev-latest`; pin a provider-supported model
version for reproducible experiments. `--base-url` overrides `TYPESAFE_BASE_URL`;
`--reflection-base-url` configures the reflection endpoint. CLI retries are disabled.

## Graph format

A plan contains `nodes`, keyed by unique IDs, and `outputs`, keyed by output names.
Earlier `questions`/`root` plans must be rewritten in this schema. The order of
JSON keys does not determine execution. For example:

```json
{
  "nodes": {
    "eligible": {
      "type": "call",
      "question": {
        "type": "noul",
        "instructions": "Does the supplied policy explicitly make this request eligible?"
      },
      "inputs": {"request": {"type": "state"}}
    },
    "decision": {
      "type": "threshold",
      "source": "eligible",
      "cutoff": 0.8,
      "below": "review",
      "above": "accept"
    }
  },
  "outputs": {"label": {"type": "node", "node": "decision"}}
}
```

The cutoff is illustrative, not calibrated. Every `call` sends its mapped input
object as the SDK request's `state` and its own question in a separate request.
Independent ready calls execute concurrently, bounded by `max_workers` (default
4). Dependencies become available as requests finish; there is no layer-wide
barrier. A shared node runs once per execution, even when several nodes use it.

| Node | Behavior |
| --- | --- |
| `call` | Run a `choice`, `score`, or `noul` question on mapped `inputs`. |
| `switch` | Map a categorical `source` value through exhaustive string `cases`. |
| `threshold` | Return string `below` or `above`; `>= cutoff` selects `above`. |
| `merge` | Assemble mapped `inputs` into an object without a model call. |
| `coalesce` | After all `sources` settle, choose the first available value in declared order. |
| `value` | Produce a literal JSON `value`. |

A Choice value is its selected category; a Noul value is its yes probability; a
Score value is its expected rubric position. Score is not a probability of
success. Choice supports 2–255 options; Score supports 2–10 distinct levels. This
project exposes a subset of the provider's question schema.

Input mappings and graph outputs use these sources:

| Source | JSON |
| --- | --- |
| Original input | `{"type": "state"}` |
| Constant | `{"type": "literal", "value": {"policy": "example"}}` |
| Earlier node value | `{"type": "node", "node": "eligible"}` |
| Full validated call answer | `{"type": "node", "node": "eligible", "view": "answer"}` |
| Optional earlier value | `{"type": "node", "node": "eligible", "optional": true}` |

`view: "answer"` is allowed only for call nodes. It passes the validated answer
record, including its distribution where supplied, instead of the scalar value.

Every node can have a `when` condition. Comparisons use a node's scalar value:

```json
{"type": "compare", "node": "department", "op": "eq", "value": "billing"}
```

Numeric comparisons also support `lt` and `ge`. Compose conditions using
`{"type": "all", "conditions": [...]}`, `{"type": "any", "conditions": [...]}`,
`{"type": "not", "condition": ...}`, or `{"type": "present", "node": "id"}`.
A comparison against a skipped node is false; `present` tests availability.
Negation follows ordinary Boolean rules, so negating a skipped comparison is true.

A node waits for all data and condition dependencies to settle. A false condition
skips it. A skipped required input skips its consumer; an optional skipped input
is omitted from the input object. Optional skipped outputs are omitted from the
output object. A required skipped graph output raises `GraphExecutionError`.
`coalesce` skips when all its sources skip. Its declared source order, not request
completion order, determines which available value it returns.

Validation rejects cycles, missing references, unused nodes, incompatible types,
nonexhaustive switches, invalid cutoffs, and unknown fields. Disconnected
components are allowed when they feed separate outputs. Search limits count all
nodes and the longest dependency/control path; roots have depth one. Literal-only
outputs need no nodes or model requests.

## Python and optimization

```python
from pathlib import Path

from typesafe_sdk import TypeSafeClient

from decision_compose.graph import Plan
from decision_compose.runtime import DecisionProgram, ExecutionConfig

plan = Plan.model_validate_json(Path("examples/dag.json").read_text())
with TypeSafeClient() as client:
    program = DecisionProgram(
        plan, client, model="jev-latest", execution=ExecutionConfig(max_workers=4)
    )
    result = program(state="Please correct the duplicate charge on my invoice.")
    print(result.outputs, result.calls, result.skipped)
```

`DecisionProgram` is a `dspy.Module`. Predictions contain named `outputs`, executed
node `node_values`, validated call `answers`, skipped node IDs, per-call model/token
metadata, resolved `models`, and token totals. A `label` attribute is included
when the outputs contain `label`. Execution traces use a stable graph order,
independent of concurrent completion order.

Generic execution supports arbitrary named JSON outputs. The current task,
evaluation, and optimization interface remains supervised classification: it
requires a `label` output with a statically known categorical domain within the
task's labels and scores exact-match accuracy. Duplicate example IDs and
JSON-equivalent states are rejected across splits to catch obvious leakage.

`optimize_plan` accepts the task, seed plan, TypeSafe client, model ID, reflection
callable, `SearchConfig`, and optional `ExecutionConfig`. The reflection callable
returns a JSON graph. A custom `GraphAdapter` exposes that complete graph to
GEPA; no generated Python is executed. Invalid graphs and unavailable required
outputs get zero scores and feedback, including any partial execution trace.
Provider or malformed-response failures
stop optimization instead of becoming task-performance scores.

## Verification and limits

```sh
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv build
```

Unit tests cover graph validation and execution semantics. Integration tests
exercise the official SDK over local HTTP, DSPy's module and LM interfaces, the
real GEPA search engine, and CLI artifacts. Protocol fixtures and fixed proposals
verify wiring, not live model quality. Paid hosted checks are opt-in:

```sh
# Configure TYPESAFE_API_KEY and your reflection provider's credential first.
export DECISION_COMPOSE_REFLECTION_MODEL="$REFLECTION_MODEL"
uv run pytest --live tests/test_live.py -q
```

- Graphs are finite and static. No cycles, dynamic node expansion, arbitrary
  tools, generated code, durable execution, or UI are implemented.
- GEPA's soft metric-call budget counts example evaluations, not Jev requests.
  Each example can make many node calls. The budget may exceed its target by an
  iteration and excludes reflection calls; it is **not a dollar cap**.
- A failed request cannot undo independent requests already in flight. Provider
  failures abort execution; skipped branches are an explicit control-flow result.
- Input mappings and question instructions go to Jev. Training examples and
  execution feedback go to the reflection provider. Reports contain model answers.
- Optimizer seeds do not make hosted models deterministic. Use representative
  data, provider spending limits, and a final untouched test split.

## Sources

- [TypeSafe primitives](https://docs.typesafe.ai/primitives/choice)
- [TypeSafe question semantics](https://docs.typesafe.ai/cookbooks/parallel_questions)
- [DSPy modules](https://dspy.ai/learn/programming/modules/)
- [GEPA custom adapters](https://gepa-ai.github.io/gepa/guides/adapters/)
- [GEPA full-program evolution](https://gepa-ai.github.io/gepa/tutorials/dspy_full_program_evolution/)
