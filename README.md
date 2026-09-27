# rulesmith

![rulesmith: rules that ask a local model for typed decisions](image.png)

**rulesmith lets DSPy build decision graphs for you.** Give it a
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

**1. Start a judge.** [Ruling](https://github.com/bradAGI/ruling) runs locally
on an Apple Silicon Mac and needs no account. In a separate terminal, starting
from your `rulesmith` checkout:

```sh
git clone https://github.com/bradAGI/ruling.git ../ruling
uv sync --directory ../ruling
uv run --directory ../ruling ruling serve
```

Leave that server running. Run the remaining commands from `rulesmith`
in your original terminal. Ruling downloads its model on first use. On other
hardware, use a remote Ruling server or the [hosted Jev backend](#backends).

**2. Configure a proposer**, the model that writes graphs, using a
[DSPy model identifier](https://dspy.ai/learn/programming/language_models/).
The example below assumes you have separately started an OpenAI-compatible
model server at `http://127.0.0.1:8080/v1`. Replace `<served-model-id>` with its
model ID; Ruling provides the judge, not this generative endpoint:

```sh
export REFLECTION_MODEL="openai/<served-model-id>"
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
  "inputs": [
    "I was charged twice this month.",
    "The export button returns a 500 error.",
    "Can I get a quote for twenty seats?",
    "Please refund the charge after my cancellation.",
    "The app crashes when I open a project.",
    "Can we schedule a product demo before buying?"
  ]
}
```

Then have the proposer model label them. `--validation` and `--test` set how
many inputs are held out for selection and final scoring:

```sh
uv run rulesmith draft-labels inputs.json --output task.json \
  --labeler-model "$REFLECTION_MODEL" --labeler-base-url "$REFLECTION_BASE_URL" \
  --validation 1 --test 1
```

This leaves four training inputs, one validation input, and one test input: a
workflow example, too small to measure model quality reliably. For real work,
use more examples and larger held-out splits.

> [!IMPORTANT]
> Review and correct `task.json` before optimizing. Search learns to agree with
> these labels, including their mistakes.

This is DSPy's teacher–student pattern: a large model labels, and the small
judge is fit to its judgment.

</details>

<details>
<summary><b>My inputs are more than text</b></summary>

An input may be a string, a JSON object, or an array. Structure is worth giving
it: a rule can only `read` what has a path, so on a bare string every read is
missing and the graph has nothing to go on but judgments.

```json
{
  "description": "Route each refund request.",
  "labels": ["auto_refund", "review", "decline"],
  "inputs": [
    {"charge": {"amount": 40, "days_ago": 2}, "account": {"plan": "pro", "chargebacks": 0},
     "message": "Charged twice for the same seat."},
    {"charge": {"amount": 900, "days_ago": 60}, "account": {"plan": "free", "chargebacks": 2},
     "message": "I want all of it back."},
    {"charge": {"amount": 12, "days_ago": 1}, "account": {"plan": "enterprise", "chargebacks": 0},
     "message": "Wrong seat count on this invoice."}
  ]
}
```

Rules then mix what is known with what must be judged:

```
read charge.amount in [0, 100000]
read account.chargebacks in [0, 100]
ask genuine = "The message describes a mistake the customer did not cause."
when account.chargebacks >= 1: review
when charge.amount < 50 and genuine >= 0.8: auto_refund
always: review
```

</details>

<details>
<summary><b>I have labeled examples</b></summary>

Start with the complete labeled sample:

```sh
cp examples/task.json task.json
```

Or write `task.json` directly. This minimal example shows the required shape:

```json
{
  "description": "Route each support ticket to billing, technical, or sales.",
  "labels": ["billing", "technical", "sales"],
  "train": [{"id": "t1", "state": "I was charged twice.", "label": "billing"}],
  "validation": [{"id": "v1", "state": "The app crashes.", "label": "technical"}],
  "test": [{"id": "e1", "state": "Can I get a price quote?", "label": "sales"}]
}
```

`train` drives proposals, `validation` picks the winner, and `test` is scored once at the end.

</details>

**4. Build the graph.**

```sh
uv run rulesmith optimize task.json \
  --backend ruling --max-workers 1 \
  --reflection-model "$REFLECTION_MODEL" \
  --reflection-base-url "$REFLECTION_BASE_URL" \
  --output runs/my-task
```

This writes `runs/my-task/plan.json` (the graph) and `report.json` (scores).
Compare `validation_score` with `seed_validation_score`.

**5. Evaluate and run it.** The proposer is not needed from here on.

```sh
uv run rulesmith evaluate task.json runs/my-task/plan.json \
  --backend ruling --max-workers 1 --split test --output runs/my-task/test.json

uv run rulesmith run runs/my-task/plan.json examples/state.json \
  --backend ruling --max-workers 1
```

### How answers are scored

By default an answer scores 1 when it matches the example's label and 0
otherwise. Two additions in `task.json` refine that, and neither calls a model.

**Some mistakes cost more.** `costs[expected][answered]` prices a wrong answer;
anything unlisted costs 1. The dearest mistake scores 0 and a cheaper one scores
in between, so search avoids the expensive ones first:

```json
"costs": {"fraud": {"approve": 10, "review": 1}}
```

Approving fraud is ten times worse than sending it to review. The feedback search
sees names what each mistake cost.

**Some examples have more than one right answer.** `accepted` lists labels that
count as fully right besides `label`:

```json
{"id": "t7", "state": "Refund my duplicate charge and fix the export bug.", "label": "billing", "accepted": ["technical"]}
```

**No labels at all.** A stronger model can grade the answers instead. Examples
may then leave out `label`; where one is present, the grader is shown it as a
reference:

```sh
uv run rulesmith optimize task.json \
  --backend ruling --max-workers 1 \
  --reflection-model "$REFLECTION_MODEL" --reflection-base-url "$REFLECTION_BASE_URL" \
  --grader-model "$GRADER_MODEL" --grader-base-url "$GRADER_BASE_URL" \
  --output runs/my-task
```

The grader scores each answer from 0 to 1 and says why, and search reads the
reason. Each answer to each input is graded once, so a search pays for at most
one grade per distinct answer per example. Use a different, stronger model from
the one your graph asks. Otherwise search finds the graph that asks the grader's
own question and scores perfectly for nothing. rulesmith stops when the grader
and the judge report the same model name, but it cannot tell that two services
serve the same weights under different names, so that part is on you. Reports record `scored_by`, because a graded score is a
model's opinion, not accuracy against labels.

## How it works

```mermaid
flowchart LR
  T["task.json<br/>labels + examples"] --> S["Starting graph<br/>majority label"]
  S --> E["Run on training examples<br/>via Ruling or Jev"]
  E -->|"errors + expected labels"| P["DSPy proposer<br/>revises one graph component"]
  P -->|"assemble + schema check"| E
  E -->|"better on the batch"| V["Score on validation examples"]
  V --> B["Best graph<br/>plan.json"]
```

The proposer sees where the graph answered wrong and revises one part: the
readings it may look at, or the rules that decide. The revision is compiled with
the part it did not touch and validated as a complete graph. A proposal is kept
only if it beats its parent on the training batch; the best validation score
wins. The test split never reaches the optimizer.

The full DAG language is available. This hand-written example
([examples/dag.json](examples/dag.json)) asks two questions in parallel, then
runs only the specialist that matches:

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

Search does not write that JSON. It writes rules, which compile to it:

```
read target.x in [-1, 1]
ask threat = "Is this worth fighting?" [fight: "A live opponent is in view."] [clear: "Nothing is."]
when target.x and threat == "fight": ATTACK
always: MOVE_FORWARD
```

Rules are tried in order and the first whose condition holds decides. A judgment
can sit behind a cheap read, so the judge is asked only where a number cannot
settle it. Conditions join with `and` and `or`, group with parentheses, invert
with `not`, name a reading to test that it has a value, or compare one with `>=`,
`>`, `<=`, `<` or `==` against a number, a label, or another reading.

`let` names a condition so the rules can test it by name, or computes a number
from readings with `+`, `-` and a constant weight:

```
read ammo in [0, 50]
read health in [0, 200]
read enemy.health in [0, 200]
let armed = ammo >= 1 and health >= 25
let lead = health - enemy.health
when armed and lead >= 20: ATTACK
when not armed or health < enemy.health: MOVE_BACKWARD
always: MOVE_FORWARD
```

A reading can be named the same way, `read health = variables.HEALTH in [0, 200]`,
and the rules then say `health`. A name is a node in the graph, so it survives the
round trip: search reads and rewrites `armed`, not the condition spelled out in
every rule, and `draw` shows it as a box of its own.

Rules decide the label, and `output <name>:` starts another list that decides
something else from the same readings and judgments, asked once for both:

```
read severity in [0, 3]
read account.vip one of [yes, no]
when severity >= 2: review
always: accept
output priority:
when severity >= 3 or account.vip == "yes": high
always: low
```

Programs that rules cannot express, such as two judgments merged into one input,
are proposed as whole-graph JSON instead.

### What a judgment can be

A judge answers three kinds of question, and only the first is classification:

| Kind | Answer | Written as | Tested with |
| --- | --- | --- | --- |
| `choice` | one of the labels you name | `ask route = "Which team?" [billing: "..."] [technical: "..."]` | `route == "billing"` |
| `score` | a position on the rubric, ordered | `ask severity = "How bad is this?" [minor] [major] [critical]` | `severity >= 2` |
| `noul` | the probability its statement is true | `ask grounded = "The reply follows from the context."` | `grounded >= 0.85` |

A judgment reads the whole input unless you say what it is shown. `given` hands
it named values instead, which is less to read and keeps it on the question:

```
read health = variables.HEALTH in [0, 200]
read enemy.distance in [0, 4096]
ask close = "The enemy is close enough to fight." given [enemy.distance, health]
```

`given [observation, first]` shows it the whole input and another judgment's
answer, which is how one judgment checks another's work.

Two things a rule can say about a judgment besides what it answered.

**How sure it was.** A choice or a score comes back with a confidence, and
`<name>.confidence` puts it where a condition can test it, so an answer the judge
hedged on falls through to the cheap rule instead of being acted on:

```
ask threat = "Fight or fall back?" [ATTACK: "Ammo and health allow it."] [MOVE_BACKWARD: "They do not."]
when threat.confidence >= 0.7 and threat == "ATTACK": ATTACK
always: MOVE_FORWARD
```

**What several of them agree on.** A vote settles two or more judgments into
one value: `agree` reports whether they all said the same, which is how a second
opinion verifies the first, and `majority` takes the answer most of them gave.

```
ask first = "Is this order fraudulent?" [yes: "It is."] [no: "It is not."]
ask second = "Reading it again, is this order fraudulent?" [yes: "It is."] [no: "It is not."]
vote checked = agree of [first, second]
when checked == "yes" and first == "yes": review
always: accept
```

Each judgment in a vote is a separate model call, so a second opinion costs a
second question. Everything measured in this repo says calls are the expensive
part and rarely the deciding one, so reach for this where being wrong is costly,
not by default.

A probability is a number, so it composes like any read:

```
ask grounded = "The reply follows from the retrieved context."
when grounded >= 0.85: send
when grounded < 0.4: escalate
always: ask_for_review
```

That covers uses a classifier has no form for: gating an action on confidence,
verifying a generated answer before it goes out, ranking candidates by asking
which is better, or serving as the metric a search optimizes against rather than
the policy it searches. What a judge will not do is extract or compute: if the
answer is already a value in the input, read it.

One worked example of each, runnable from
[examples/rules](examples/rules):

**Route a ticket** ([routing.rules](examples/rules/routing.rules)) — a choice is
a label, and the rules act on it:

```
# choice: pick one of the labels you named, the classic classification.
ask team = "Which team should answer this ticket?" [billing: "Charges, invoices, refunds."] [technical: "Errors, outages, anything broken."] [sales: "Pricing, plans, buying more."]
when team == "billing": billing
when team == "technical": technical
always: sales
```

**Triage by how blocked the customer is**
([triage.rules](examples/rules/triage.rules)) — a score is ordered, so one
judgment feeds several thresholds, and a read decides who jumps the queue:

```
# score: a position on a rubric, so it can be compared and thresholded.
read variables.customer_tier in [0, 3]
ask severity = "How badly is this customer blocked?" [annoyance] [degraded] [blocked] [outage]
when severity >= 3: page_oncall
when severity >= 2 and variables.customer_tier >= 2: escalate
when severity >= 2: queue_urgent
always: queue_normal
```

**Spend the judge only where it decides something**
([gating.rules](examples/rules/gating.rules)) — readings settle the clear cases,
and the question is asked only about orders big enough to matter:

```
# Cheap readings decide the clear cases; the judge is asked only about the rest.
read order.total in [0, 1000000]
read order.country one of [domestic, eu, rest]
read customer.orders in [0, 10000]
ask suspicious = "This order reads like card testing or resale fraud." when order.total >= 500
when customer.orders >= 5 and order.total < 500: ship
when order.total < 50: ship
when suspicious >= 0.7: hold_for_review
when suspicious >= 0.3 and order.country == "rest": hold_for_review
always: ship
```

**Check a draft reply before sending it**
([grounding.rules](examples/rules/grounding.rules)) — probabilities compose, so
two judgments set a floor and a ceiling with a human in between:

```
# noul: a probability, so the decision is a threshold on confidence.
ask grounded = "Every claim in the draft reply follows from the retrieved context."
ask harmful = "The draft reply would embarrass us if a customer read it."
when harmful >= 0.2: rewrite
when grounded >= 0.85: send
when grounded < 0.4: escalate
always: ask_for_review
```

See [Graph format](docs/graphs.md) for the JSON behind each node, and
[Optimization](docs/optimization.md) for the rule surface search writes.

## Games

Two environments where a graph plays a game instead of labeling an input, and is
scored by how it plays.

**[Doom](docs/doom.md).** Each step a graph reads the visible objects and game
variables and presses a button, scored by episode reward, so no labels are
needed. Search found a graph that kills the monster on 12 of 12 held-out seeds, one
that finishes E1M1, and a duelist that won 12-0 on the map it trained on and came
last across 32 maps. [Results](docs/doom.md#results) ·
[the league](docs/doom.md#the-league)

**[Chess](docs/chess.md).** A label names a policy, such as taking the best
capture or saving a hanging piece, and the board picks the move. A game costs
milliseconds, so graphs are ranked over hundreds of games: search beat every
hand-written graph and then converged by generation 7.
[Results](docs/chess.md#results)

## Look at what a graph does

Two commands for reading a graph rather than running it.

**`draw`** writes it as a Mermaid flowchart, which is the readable form for the
graphs rules cannot express: solid arrows carry values, dashed ones gate, and
each dashed arrow says the condition it carries.

```sh
uv run rulesmith draw examples/rules/routing.rules
```

**`ablate`** answers what each reading, judgment and name is worth. It replays the
decisions the graph already made with one input switched off and counts how many
change, so it needs no fresh model calls beyond the first pass:

```sh
uv run rulesmith ablate task.json plan.json --split validation \
  --backend ruling --output runs/ablation.json
```

The command reads its states from a task's examples. A Doom or chess task keeps
seeds instead, so for those, record the observations a graph saw and pass them to
`effects(plan, states)` in `rulesmith.ablate`. Run that way against the
best graph the chess league produced, over 116 positions from one game:

```
reading                       changed   share   why not
capture.best_gain                  99     85%
hanging.worst                       9      8%
king.in_check                       3      3%
check.available                     1      1%
king.can_castle                     0      0%   present in 116 of 116 states and changes no decision
```

That last line is a rule the graph carries and never uses: `king.can_castle` is
there in every position, and the two rules above it fire first every time
castling is legal. A zero row means one of three things and the report says
which, because they are different problems: a path the state never has is a
typo, a rule an earlier one always beats is dead weight, and a threshold real
data never trips is a number chosen badly.

Search spends its budget rewriting whatever is in the graph, so rules that
cannot matter are rules the proposer wastes proposals on.

## Commands

| Command | Does |
| --- | --- |
| `validate` | Check a graph against a task's labels and size limits, without running it |
| `run` | Execute a graph once on one input |
| `evaluate` | Score a graph on a task split |
| `optimize` | Search for a better graph, writing `plan.json` and `report.json` |
| `arena` | Rank Doom graphs by Elo from deathmatch duels |
| `record` | Play a Doom episode to `.mp4` or `.gif` |
| `draw` | Write a graph as a Mermaid flowchart |
| `ablate` | Measure what each reading, judgment and name in a graph is worth |
| `draft-labels` | Label raw inputs with a strong model, for you to review as a task |

Every command that reaches a model takes `--backend`, `--model`, `--base-url`,
`--timeout` and `--max-workers`; see [backends](#backends).

A `.rules` file is a graph wherever the CLI takes one, so the examples run as
they stand:

```sh
uv run rulesmith validate examples/task.json examples/rules/routing.rules
uv run rulesmith run examples/rules/routing.rules examples/state.json \
  --backend ruling --max-workers 1
```

## Use from Python

A graph runs as a `dspy.Module`:

```python
from pathlib import Path

from typesafe_sdk import TypeSafeClient

from rulesmith.graph import Plan
from rulesmith.runtime import DecisionProgram

plan = Plan.model_validate_json(Path("runs/my-task/plan.json").read_text())
with TypeSafeClient(base_url="http://127.0.0.1:8010", api_key="local-no-auth") as client:
    program = DecisionProgram(plan, client, model="default")
    result = program(state="Please correct the duplicate charge on my invoice.")
    print(result.label, result.answers)
```

`result.answers` holds what each judgment said, so a decision can be logged or
audited, not just acted on. Rules load the same way, and a graph of reads alone
needs no judge at all:

```python
from rulesmith.rules import compile_rules, render_rules

plan = compile_rules(Path("examples/rules/gating.rules").read_text())
print(render_rules(plan))  # and back again: the graph reads as the rules it came from

triage = compile_rules("""
read order.total in [0, 1000000]
when order.total >= 500: hold_for_review
always: ship
""")
print(DecisionProgram(triage, None, "none")(state={"order": {"total": 900}}).label)
```

## Backends

| Backend | Flag | Endpoint | Credential |
| --- | --- | --- | --- |
| Ruling (local) | `--backend ruling` | `RULING_BASE_URL`, default `http://127.0.0.1:8010` | `RULING_API_KEY` if auth is on |
| Jev (hosted) | `--backend typesafe` (default) | `TYPESAFE_BASE_URL` | `TYPESAFE_API_KEY` |

Ruling serializes requests, so use `--max-workers 1`. Hosted calls may incur
charges: the search budget counts examples, not requests or dollars.

## Related work

An LLM rewriting a symbolic policy that is then scored by playing it is not a new
idea, and a couple of these got there first.

- [PolicyEvolve](https://arxiv.org/abs/2509.06053) is the closest in concept: an
  LLM writes rule-based policy code for a two-player game, a pool is ranked by
  Elo from head-to-head matches, and a critic feeds execution failures into the
  next rewrite. Its rules are free-form Python, not a compiled graph; it
  runs on one toy environment, and no code was released.
- [Digital Red Queen](https://github.com/SakanaAI/drq) evolves Core War warrior
  programs by adversarial self-play, each round targeting the champions before
  it. The artifact is assembly, not a rule list, and it scores on win rates.
- [autoevolve](https://github.com/MrTsepa/autoevolve) runs a GEPA-shaped loop
  over free-form strategy code with a self-play league and a Pareto front, which
  is the same optimizer and the same league shape used here.
- [FMSP](https://github.com/aadharna/FMSP) has a foundation model write code
  policies refined by competitive self-play with quality-diversity.
- [From Code to Play](https://arxiv.org/abs/2412.04057) benchmarks eleven models
  on program-policy search across 29 game tasks.

On the runtime side, [decision-circuits](https://github.com/Barneyjm/decision-circuits)
pairs typed questions with deterministic gates, though you build its circuits
in Python instead of declaring them, and [GoRules ZEN](https://github.com/gorules/zen) executes
statically validated JSON decision models with no model in the loop at all.

What none of them do, as far as five sweeps could tell: a rule list that compiles
to a statically validated DAG, calibrated probabilities taken from logits inside
that graph, and a judge that can be called mid-execution and gated behind a cheap
reading. Read the absence claims as unconfirmed, not proven: two of the
"nothing found" calls in that search turned out to be wrong when checked.

## Learn more

- [Graph format](docs/graphs.md): nodes, inputs, conditions, execution, and writing graphs by hand.
- [Optimization](docs/optimization.md): search settings, reports, testing, and limits.
- [Doom](docs/doom.md): task format, observations, scoring, and watching a graph play.
- [Chess](docs/chess.md): policies, what a graph sees, scoring, and the baseline ladder.

## Development

```sh
uv run pytest -q
uv run ruff check src tests && uv run ruff format --check src tests
```

## License

[MIT](LICENSE).
