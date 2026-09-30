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
uv run rulesmith validate examples/task.json examples/rules/routing.rules
uv run rulesmith run examples/rules/routing.rules examples/state.json \
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
Ruling serializes them, so use `--max-workers 1`. Jev calls are billed: the
search budget counts examples, not requests or dollars, and `--call-cost` makes
search prefer graphs that ask less. The same search against Jev:

```sh
export TYPESAFE_API_KEY=<your key>
uv run rulesmith optimize task.json \
  --reflection-model "$REFLECTION_MODEL" --reflection-base-url "$REFLECTION_BASE_URL" \
  --output runs/my-task-jev
```
