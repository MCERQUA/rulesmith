# Optimization

`rulesmith optimize` searches for the graph with the best validation
score: agreement with the labels for classification (exact match, or weighted
by the task's `costs` and `accepted` labels), a grader model's score with
`--grader-model`, and normalized reward for [Doom](doom.md). DSPy writes candidate graphs, and
[GEPA](https://gepa-ai.github.io/gepa/guides/adapters/) decides which to
evaluate and keep.

## The search

1. **Seed.** Search starts from the graph you pass, or from a call-free graph
   that always answers the most common training label (the first listed button
   for Doom).
2. **Reflect.** When the current graph misses a training example, `GraphProposer`
   (a `dspy.Predict` with plain-text `ChatAdapter` fields, which any chat
   endpoint accepts) gets the graph, the inputs, its outputs, and the expected
   labels or episode rewards. Search revises one part at a time:
   **readings** holds reads and judgments; **rules** holds names, votes and the
   rules that decide. The proposer returns the replacement component, preserving
   references to nodes owned by the unchanged component.
3. **Check.** The replacement is assembled with the unchanged component and
   validated as a complete graph; no generated code runs. Validation errors
   go back to the proposer for up to `--repair-attempts` repairs. Candidates
   that remain invalid score zero, and their errors inform later proposals.
4. **Accept.** A proposal that beats its parent on the training batch is scored
   on validation.
5. **Select.** The graph with the best validation score is saved. The seed
   stays eligible, so a graph that overfits training will not replace it.

The test split never reaches the optimizer. Score it once, after selection:

```sh
uv run rulesmith evaluate task.json runs/my-task/plan.json --split test \
  --backend ruling --max-workers 1 --output runs/my-task/test.json
```

## Settings

| Flag | Default | Meaning |
| --- | --- | --- |
| `--reflection-model` | required | DSPy model ID for the proposer |
| `--reflection-base-url` | — | Proposer endpoint, e.g. a local server's `/v1` |
| `--max-metric-calls` | 100 | Budget in example evaluations |
| `--reflection-batch` | 3 | Training examples per reflection |
| `--repair-attempts` | 2 | Times an invalid proposal goes back to the proposer with its validation error before search sees it |
| `--reflection-max-tokens` | 4096 | Token limit on each proposal |
| `--reflection-timeout` | none | Seconds before a proposal request fails, so a hung proposer stops the search instead of stalling it |
| `--retries` | 0 | Retries for judge requests that fail or time out |
| `--parallel-examples` | 1 | Examples scored at once; Doom allows more than 1 only for deathmatch tasks, whose duels each run their own games |
| `--call-cost` | 0 | Score taken off an example for each judge question it asks, so search prefers the graph that asks less when two answer alike |
| `--merges` | 5 | Times search may combine two candidates, taking each part from whichever graph scored better |
| `--max-nodes` / `--max-depth` | 16 / 8 | Limits on generated graphs |
| `--seed` | 0 | Search seed |
| `--grader-model` / `--grader-base-url` | — | DSPy model that grades answers instead of the labels; examples may then omit `label`. Must differ from the judge the graph asks. Also on `evaluate`. |

All commands that ask a judge also take `--backend`, `--model`, `--base-url`,
`--timeout`, `--retries` and `--max-workers`. Retries default to 0, so a failed
judge request stops the run unless you raise them. With `--backend typesafe`,
the model defaults to `jev-latest`; pin a version for reproducible runs.

> [!TIP]
> Reflection only runs when the graph misses something in a training batch. If
> one label dominates your data, raise `--reflection-batch` so each batch
> includes the rarer labels.

## Reports

Each run writes a new directory with `plan.json` and `report.json`. Output paths
that already exist are refused before any request is sent.

| Field | Meaning |
| --- | --- |
| `validation_score` | Score of the saved graph, 0–1 |
| `seed_validation_score` | Score of the starting graph |
| `candidates` | Graphs that reached validation, including the seed. `1` means nothing new did. |
| `reflection_calls` | Proposer calls |
| `invalid_candidate_evaluations` | Evaluations rejected by schema or graph checks |
| `metric_calls` | Example evaluations used |
| `scored_by` | `labels`, `grader <model>`, or `games`: a graded score is not accuracy |

Reports also record resolved model IDs, dependency versions, settings, and
task/seed hashes.

## From Python

```python
import dspy
from typesafe_sdk import TypeSafeClient

from rulesmith.graph import Task
from rulesmith.optimize import GraphProposer, SearchConfig, optimize_plan

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
export RULESMITH_REFLECTION_MODEL="$REFLECTION_MODEL"
uv run pytest --live tests/test_live.py -q
```

A past local smoke run, whose output is not kept in the repo, with a `Qwen3.5-9B-4bit` proposer and Ruling's `Qwen3.5-4B-4bit`
judge went from an always-`sales` seed to a generated one-Choice graph: 1/3 → 3/3
on validation and on test, in 48 metric calls. That's 12 authored examples, which
shows the pipeline works end to end, not a benchmark result.

## Limits

- Search optimizes a `label` output: a class for classification, a button for Doom.
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

## What search writes

A candidate is a rule list, not graph JSON. It compiles to the graph language and
renders back from it, so the runtime, the validators and the saved `plan.json` are
unchanged: only the surface a proposer writes is different.

```
read variables.HEALTH in [0, 200]
read compass.2.bearing in [-180, 180]
ask threat = "Fight or fall back?" [fight: "Ammo and health allow it."] [flee: "They do not."] when variables.HEALTH < 40
when variables.HEALTH >= 25 and threat == "fight": ATTACK
when compass.2.bearing >= 45: TURN_LEFT
always: MOVE_FORWARD
```

- **Readings** declare what the program may look at: `read <path> in [low, high]`,
  `read <path> one of [a, b]`, `read <name> = <path> in [low, high]` to give a
  reading a name the rules use instead of its path, or `ask <name> = "question" [label: "when it applies"] ...`
  for a judgment, which may end `when <condition>` to be asked only then. A judgment
  reads the whole input unless it names what it is shown with `given [a, b]`;
  `observation` in that list is the whole input.
- **Rules** are `when <condition>: <action>` or `always: <action>`, tried in order.
  An action is a label, a conditional such as
  `route.bearing >= 0 ? TURN_LEFT : TURN_RIGHT`, or an ask without a name, whose
  answer is the label: `always: ask "Fight or fall back?" [ATTACK: "Ammo and health allow it."] [MOVE_BACKWARD: "They do not."]`.
- **Conditions** join with `and` and `or`, group with parentheses, invert with `not`,
  name a reading to test that it has a value, or compare one with `>=`, `>`, `<=`, `<`
  or `==`, against a number, a quoted label, or another reading
  (`capture.best_gain > hanging.worst`). The graph compares with three of those;
  `<=` and `>` compile to the same thing said longhand. `threat.confidence` is how
  sure the judgment `threat` was.
- **Names** stand for a condition the rules test more than once:
  `let armed = ammo >= 1 and health >= 25`, then `when armed: ATTACK`. A name is
  declared before the rules that use it and cannot reuse a reading's or judgment's
  name. It compiles to a node, so it comes back as `armed` when the graph is
  rendered, not as the condition written out again. `let` also computes a number:
  `let margin = 2 * gain - worst + 1` adds, subtracts and scales readings.
- **Votes** settle several choice judgments into one value:
  `vote verdict = agree of [first, second]` or `vote verdict = majority of [a, b, c]`.
- **Outputs** beyond the label each get a rule list of their own, after the
  label's, under an `output <name>:` heading. Search scores the label only.
- The two parts, readings and rules, are revised separately and may be merged
  between candidates. A rules part may declare any reading its rules need.
- A program beyond rules is proposed as whole-graph JSON, which is accepted as the
  complete program.

Why: in past runs, two fifths of proposals against a 36-node graph scored zero, not because the
graphs played badly but because the JSON came back malformed. Every searched graph
had the same shape anyway, so that shape became the surface.
