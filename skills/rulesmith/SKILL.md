---
name: rulesmith
description: Build, check, measure and run a rulesmith decision graph, cheap rules over an input's fields plus typed questions to a local judge, for a classification task with a fixed set of labels. Use when the user wants to turn labeled examples into a decision they can read, edit and audit, or to search, calibrate, evaluate, explain or run an existing graph.
---

# rulesmith

A decision is a graph of rules: reads of the input's fields settle the cases
they can, and a judge (Ruling locally, or Jev hosted) is asked typed questions
about the rest. Search writes the rules; you keep them as text.

Run every command from the project with `uv run rulesmith`. Against a local
Ruling judge add `--backend ruling --max-workers 1` to each command that asks
the judge (`run`, `evaluate`, `optimize`, `calibrate`, `ablate`).

## Workflow

1. **Make the task.** `task.json` holds a `description`, the `labels`, and
   examples in `train`, `validation` and `test`. Structured inputs (JSON
   objects) are worth keeping: a rule can only read a field that has a path.
   With inputs but no labels, draft them, then have the user review the file,
   since search learns the labels' mistakes too:

   ```sh
   uv run rulesmith draft-labels inputs.json --output task.json \
     --labeler-model "$REFLECTION_MODEL" --labeler-base-url "$REFLECTION_BASE_URL" \
     --validation 20 --test 20
   ```

   Ask the user before adding `costs` (what each mistake costs),
   `prevalence` (each label's share of real traffic, when examples were
   sampled at another rate), or `abstain` (a label that hands a case to a
   person). These are business choices; do not guess them.

2. **Search.** The proposer is any OpenAI-compatible model:

   ```sh
   uv run rulesmith optimize task.json --reflection-model "$REFLECTION_MODEL" \
     --reflection-base-url "$REFLECTION_BASE_URL" --output runs/my-task
   ```

   It writes `plan.json`, the same graph as editable `plan.rules`, and
   `report.json`. Report `validation_score`, `improvement_p` and `kept_seed`
   from the report: a high p, or a kept seed, means search found nothing
   better than where it started.

3. **Check it.** `validate` checks labels and size without running anything.
   An `unanswered` list in its output names outputs some input could leave
   without a value; fix it by ending the rule list with `always:`.

   ```sh
   uv run rulesmith validate task.json runs/my-task/plan.rules
   ```

4. **Calibrate, then score once on the test split.** `calibrate` fits how sure
   each judgment reads and where rules draw their lines, and writes how often
   each rule was right beside it.

   ```sh
   uv run rulesmith calibrate task.json runs/my-task/plan.rules --output runs/my-task/calibrated.rules
   uv run rulesmith evaluate task.json runs/my-task/calibrated.rules --split test \
     --output runs/my-task/evaluation.json
   ```

   Never tune on the test split; it is scored once, at the end.

5. **Explain and run.** `draw` writes a Mermaid flowchart; `ablate` measures
   what each reading and judgment is worth; `run` decides one input and
   `--trace` appends the decision to an audit file.

   ```sh
   uv run rulesmith draw runs/my-task/calibrated.rules
   uv run rulesmith ablate task.json runs/my-task/calibrated.rules --split validation \
     --output runs/my-task/ablation.json
   uv run rulesmith run runs/my-task/calibrated.rules input.json --trace decisions.jsonl
   ```

6. **Keep it.** A kept decision is a folder in git: `task.json`, the graph as
   `graph.rules` (or `map.json` for a map of graphs), and `evaluation.json`.
   After changing the judge, run `calibrate` and `evaluate` again and commit
   both.

7. **Serve and watch it.** `serve` answers decisions over HTTP for a workflow to
   call, appending each to a trace; `drift` checks those traces against what
   the rules were calibrated on and exits 1 when a rule drifted.

   ```sh
   uv run rulesmith serve decisions/my-task --trace decisions.jsonl
   uv run rulesmith drift decisions/my-task decisions.jsonl
   ```

   For a small judge that escalates to a large one, `distill` trains the small
   judge on the large one's answers first:

   ```sh
   uv run rulesmith distill task.json --output runs/student --teacher clef
   ```

## Editing rules

Edit `.rules` files directly; they compile to the graph that runs.

```
read has_company_logo in [0, 1]
read length of company_profile in [0, 10000]
when has_company_logo == 0 and length of company_profile == 0: fraudulent
always: ask "Is this job posting a scam?" [fraudulent: "fraudulent"] [genuine: "genuine"]
```

Rules are tried in order and the first that fires decides. A comparison with a
field the input lacks is false. Validate after every edit, and evaluate before
claiming an edit helped. The language is in `docs/rules.md`.

## Reporting

Quote numbers from the files the commands wrote, never from memory. Say which
split a score is on, which judge it was measured with, and how many examples
it rests on: a hundred examples move a score by several points by chance.
