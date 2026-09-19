# decision-compose

![decision-compose — Design and optimize decision graphs](image.png)

Build and optimize typed directed acyclic graphs of Ruling or hosted Jev calls.
DSPy proposes complete graphs, GEPA searches and selects them, and a DSPy module
executes them through TypeSafe's official SDK against either compatible backend.

Graphs support arbitrary finite acyclic topology over the supported node types:
sequential calls, parallel fan-out, shared dependencies, conditional branches,
joins, and multiple outputs. Each call can consume the original input and earlier
answers. A saved graph runs without the reflection model. Execution supports
named JSON outputs; optimization currently scores classification accuracy against
a required `label` output.

This independent research prototype is not affiliated with TypeSafe. It optimizes
graphs and question instructions; it does not train model weights. No benchmark
improvement is claimed.

## Install and run

Requires Python 3.12–3.14 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
uv run decision-compose validate examples/task.json examples/seed.json
uv run decision-compose validate examples/task.json examples/dag.json
```

Validation and default tests need no credentials. Local Ruling calls need no
TypeSafe account. Hosted Jev calls require `TYPESAFE_API_KEY` in your shell.
The SDK reads the environment. DSPy's LiteLLM dependency can also load `.env`
files on import (its default `DEV` mode does), so do not assume shell variables
are the only configuration source. Commands that execute hosted model calls may
incur provider charges.

### Use the local Ruling repo

With `ruling` and `decision-compose` checked out as sibling directories, start
Ruling in another terminal:

```sh
cd ../ruling
uv sync
uv run ruling serve
```

From the `decision-compose` checkout, execute the bundled graph and input:

```sh
uv run decision-compose run examples/dag.json examples/state.json \
  --backend ruling --max-workers 1

uv run decision-compose evaluate examples/task.json examples/seed.json \
  --backend ruling --max-workers 1 --split validation \
  --output runs/ruling-validation.json
```

`--backend ruling` uses `RULING_BASE_URL` (default `http://127.0.0.1:8010`),
`RULING_API_KEY` if server authentication is enabled, and the `default` model
alias. `--base-url` and `--model` override the endpoint and requested model.
Ruling's `RULING_MODEL` selects the model loaded by the server; changing the graph
CLI's `--model` does not load another model. The response records the actual
loaded model ID.

No TypeSafe credential is required. For a server with authentication disabled,
the SDK receives an unused placeholder key because its constructor requires one.
The Ruling backend does not reuse `TYPESAFE_API_KEY` or `TYPESAFE_BASE_URL`.

The current Ruling MLX server serializes requests, so `--max-workers 1` avoids
queuing concurrent requests. Its within-request question batching is not used by
this executor: each executed call node makes a separate request.

GEPA can optimize the same graph with Ruling evaluating candidates:

```sh
uv run decision-compose optimize examples/task.json examples/seed.json \
  --backend ruling --max-workers 1 \
  --reflection-model "$REFLECTION_MODEL" \
  --max-metric-calls 100 --output runs/ruling-search
```

Ruling supplies typed decisions, not generated graph JSON. Reflection needs a
separate generative model, configured through DSPy. That can be a local
OpenAI-compatible text-generation server via `--reflection-base-url`, or a
hosted provider with its own credential and charges.

### Examples and hosted Jev

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
uv run decision-compose run examples/dag.json examples/state.json --backend typesafe
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
and configure its provider credential. `GraphProposer` uses `dspy.Predict` with
typed `nodes` and `outputs` dictionaries and `JSONAdapter`. Open dictionaries
allow new node IDs and wiring; graph validation enforces the complete schema.
The selected backend executes the generated graphs' typed judgments. GEPA can
change nodes, instructions, connections, input mappings, conditions, and outputs.

The new search directory contains `plan.json` and `report.json`. Reports include
validation scores, search and execution settings, example evaluation counts,
resolved provider model IDs, dependency versions, and task/seed fingerprints.
`reflection_calls` counts proposer invocations; `invalid_candidate_evaluations`
counts adapter evaluations rejected by schema or graph constraints. `candidates`
counts graphs admitted to validation evaluation, including the seed, rather than
every proposal.
Existing output paths are refused before requests are sent. Selection uses
validation accuracy; improvement is not guaranteed. The test split is never sent
to the optimizer. Evaluate it after selecting the graph.

All network commands accept `--backend ruling|typesafe`, `--model`, `--base-url`,
`--timeout`, and `--max-workers`. The backend defaults to `typesafe`, whose model
alias defaults to `jev-latest`; `--base-url` overrides its `TYPESAFE_BASE_URL`.
Pin a supported model version for reproducible experiments.
`--reflection-base-url` configures the separate reflection endpoint. CLI retries
are disabled.

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
barrier. An executed shared node runs once per execution, even when several
nodes use it.
Independent questions are not batched into one provider request. All call nodes
use the model selected for the program.

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
skips it. The condition gates that node, not its upstream dependencies, which
may already have run. A skipped required input skips its consumer; an optional
skipped input is omitted from the input object. Optional skipped outputs are omitted from the
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
task's labels and scores exact-match accuracy. Duplicate example IDs and states
with identical JSON serialization after sorting object keys are rejected across
splits. This catches exact duplicates, not semantic equivalence: `1` and `1.0`,
for example, remain distinct.

`optimize_plan` accepts the task, seed plan, TypeSafe-protocol client, model ID, reflection
callable, `SearchConfig`, and optional `ExecutionConfig`. The reflection callable
returns a JSON graph. A custom `GraphAdapter` exposes that complete graph to
GEPA; no generated Python is executed. Invalid graphs and unavailable required
outputs get zero scores and feedback, including any partial execution trace.
A failed training proposal and its error also reach the next reflection, even
when GEPA rejects that graph. This uses the existing search budget; it does not
add retry calls. Validation and test examples are not included in this feedback.
Malformed generated graphs receive zero scores and their parse errors become
feedback. Provider failures and malformed decision responses stop optimization
instead of becoming task-performance scores. Optimization reports save scores
and metadata, not complete execution traces; evaluation reports include
per-example node answers.

The seed remains eligible for selection. A graph that improves training but
regresses on validation does not displace a better seed. Perfect training
minibatches can skip reflection, and equal training scores can reject a proposed
structure. More elaborate DAGs are useful only when they improve the task score;
structural complexity alone is not rewarded.

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
verify wiring, not hosted-service compatibility or live model quality. Paid
hosted checks are opt-in. A separate suite runs actual inference against your
Ruling server, including dependent DAG calls, the CLI, and GEPA evaluation:

```sh
uv run pytest tests/test_ruling.py --ruling-url http://127.0.0.1:8010 -q
```

These tests use the server's loaded model. The GEPA test supplies a fixed graph
proposal to verify optimizer/executor wiring; it does not validate a generative
reflection model or demonstrate improved accuracy. They ran successfully against
Ruling with `mlx-community/Qwen3.5-0.8B-4bit`. An explicit `--ruling-url` is required;
default tests do not contact a local model server.

To exercise actual DSPy graph generation as well, pass a local generative model
and endpoint explicitly:

```sh
uv run pytest tests/test_ruling.py --ruling-url http://127.0.0.1:8010 \
  --ruling-reflection-model "$REFLECTION_MODEL" \
  --ruling-reflection-url "$REFLECTION_BASE_URL" -q
```

That test requires a generated candidate to reach validation, call the real
Ruling model, and preserve or improve the seed's validation score. It fails if
the proposer only emits invalid graphs; it does not silently count retaining the
seed as successful generation.

A local smoke run used `mlx-community/Qwen3.5-9B-4bit` for DSPy generation and
Ruling's default `mlx-community/Qwen3.5-4B-4bit` judge (three answer orderings,
no prior debiasing). Starting from an always-`sales` graph, GEPA selected a
generated one-Choice graph: validation improved from 1/3 to 3/3, and subsequent
test evaluation improved from 1/3 to 3/3. The run used all six training examples
per reflection batch and 48 metric calls, with one proposal and no invalid
candidates. The test split was excluded from search. This demonstrates the
complete local path on 12 authored examples, not a benchmark gain, superiority
over a hand-written classifier, or evidence that a complex DAG is necessary.
The smaller 0.8B judge did not improve over the constant seed in an earlier run.

To check the hosted providers:

```sh
# Configure TYPESAFE_API_KEY and your reflection provider's credential first.
export DECISION_COMPOSE_REFLECTION_MODEL="$REFLECTION_MODEL"
uv run pytest --live tests/test_live.py -q
```

- Graphs are finite and static. No cycles, dynamic node expansion, arbitrary
  tools, generated code, durable execution, or UI are implemented.
- GEPA's soft metric-call budget counts example evaluations, not backend requests.
  Each example can make many requests. The budget may exceed its target by an
  iteration and excludes reflection calls; it is **not a dollar cap**.
- A failed request cannot undo independent requests already in flight. Provider
  failures abort execution; skipped branches are an explicit control-flow result.
- Resolved node inputs and question instructions go to the selected backend.
  Training examples and execution feedback go to the reflection provider.
  Evaluation reports contain model answers.
- Optimizer seeds do not make hosted models deterministic. Use representative
  data, provider spending limits, and a final untouched test split.

## Sources

- [TypeSafe Choice](https://docs.typesafe.ai/primitives/choice)
- [TypeSafe Score semantics](https://docs.typesafe.ai/primitives/score)
- [TypeSafe question semantics](https://docs.typesafe.ai/cookbooks/parallel_questions)
- [DSPy modules](https://dspy.ai/learn/programming/modules/)
- [DSPy JSONAdapter](https://dspy.ai/api/adapters/JSONAdapter/)
- [GEPA custom adapters](https://gepa-ai.github.io/gepa/guides/adapters/)
- [GEPA full-program evolution](https://gepa-ai.github.io/gepa/tutorials/dspy_full_program_evolution/)
