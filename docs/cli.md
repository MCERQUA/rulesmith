# Command line

`rulesmith` is one command with subcommands; `rulesmith <command> --help` lists
every flag.

| Command | Does |
| --- | --- |
| `validate` | Check a graph against a task's labels and size limits, without running it |
| `run` | Execute a graph once on one input |
| `evaluate` | Score a graph on a task split |
| `optimize` | Search for a better graph, writing `plan.json`, editable `plan.rules` and `report.json` |
| `arena` | Rank Doom graphs by Elo from deathmatch duels |
| `record` | Play a Doom episode to `.mp4` or `.gif` |
| `calibrate` | Fit how sure each judgment should read and where each rule draws its line on one, then count how often each rule is right, from the labeled examples |
| `draw` | Write a graph as a Mermaid flowchart; `--closed` draws a map's units as single boxes |
| `ablate` | Measure what each reading, judgment and name in a graph is worth |
| `draft-labels` | Label raw inputs with a strong model, for you to review as a task |
| `tune` | Fine-tune the judge on a task's examples with Ruling, test split held out |

Every command that asks a judge (`run`, `evaluate`, `optimize`, `calibrate`,
`arena`, `record`, `ablate`) takes `--backend`, `--model`, `--base-url`, `--timeout`,
`--retries` and `--max-workers`; see [backends](#backends). `draft-labels` calls
a generative model instead, set with `--labeler-model` and `--labeler-base-url`.

`optimize` has flags of its own for the search budget, the proposer, checkpoints
and `--resume`, listed under [settings](optimization.md#settings); its `--unit`
and `--map-task` flags, for improving one unit of a map, are on the
[Maps](maps.md#working-with-a-map) page.

A `.rules` file is a graph wherever the CLI takes one, so the examples run as
they stand:

```sh
uv run rulesmith validate examples/support-routing/task.json examples/support-routing/graph.rules
uv run rulesmith run examples/support-routing/graph.rules examples/support-routing/input.json \
  --backend ruling --max-workers 1
```

Output (trimmed):

```text
{
  "nodes": 5,
  "labels": ["billing", "technical", "sales"]
}
{
  "outputs": {"label": "billing"},
  ...
  "answers": {
    "team": {
      "type": "choice",
      "value": "billing",
      "probabilities": {"billing": 0.997, "technical": 0.003, "sales": 0.0003},
      "confidence": 0.995
    }
  },
  "skipped": ["rule_2"],
  ...
  "label": "billing",
  "decided_by": "rule_1",
  "confidence": 0.997
}
```

## Backends

| Backend | Flag | Model | Endpoint | Credential |
| --- | --- | --- | --- | --- |
| Jev (hosted) | `--backend typesafe` (default) | `jev-latest`, or `--model` | `TYPESAFE_BASE_URL`, default `https://api.typesafe.ai` | `TYPESAFE_API_KEY` |
| Ruling (local) | `--backend ruling` | whatever Ruling serves | `RULING_BASE_URL`, default `http://127.0.0.1:8010` | `RULING_API_KEY` if auth is on |

Jev answers requests in parallel, so the default `--max-workers 4` applies;
Ruling serializes them, so use `--max-workers 1`. A Ruling judge fine-tuned on
your labels is served (`RULING_ADAPTER`) and asked the same way, and
`rulesmith tune task.json --output tuning` makes one: it writes the records
Ruling's trainer reads, every training and validation example asked the direct
question with its label in `train.jsonl` and the test examples in
`holdout.jsonl`, which the trainer refuses to overlap, then runs `ruling train`
from the checkout at `--ruling-dir` (default `$RULING_DIR`, else `../ruling`)
with the recipe the benchmark validated, 200 steps with the best of ten
validation checkpoints kept, on `--base`, a pure-attention model Ruling trains
at full speed. Anything after `--` goes to `ruling train` and overrides the
recipe. It prints where the adapter is and the line that serves it; rulesmith
then asks that judge like any other. The
[benchmarks](benchmarks.md#a-judge-tuned-on-the-same-examples) measure what a
tune on a hundred examples changes. Jev calls are billed: the
search budget counts examples, not requests or dollars, and `--call-cost` makes
search prefer graphs that ask less. The same search against Jev:

```sh
export TYPESAFE_API_KEY=<your key>
uv run rulesmith optimize task.json \
  --reflection-model "$REFLECTION_MODEL" --reflection-base-url "$REFLECTION_BASE_URL" \
  --output runs/my-task-jev
```
