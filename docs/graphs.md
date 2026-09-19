# Graph format

A graph (`plan.json`) has `nodes`, keyed by unique IDs, and `outputs`, keyed by
output names. Nodes connect by referencing each other's IDs; there is no separate
edge list, and JSON key order does not affect execution.

```json
{
  "nodes": {
    "department": {
      "type": "call",
      "question": {
        "type": "choice",
        "instructions": "Which department should handle this ticket?",
        "criteria": {
          "billing": "Charges, invoices, refunds.",
          "technical": "Product failures and troubleshooting.",
          "sales": "Purchases, pricing, demos."
        }
      },
      "inputs": {"ticket": {"type": "state"}}
    }
  },
  "outputs": {"label": {"type": "node", "node": "department"}}
}
```

Examples: [seed.json](../examples/seed.json) (one call) and
[dag.json](../examples/dag.json) (parallel calls, a conditional specialist, and joins).

## Nodes

| Node | Behavior |
| --- | --- |
| `call` | Ask a `choice`, `score`, or `noul` question about mapped `inputs`. |
| `switch` | Map a categorical `source` value through exhaustive string `cases`. |
| `threshold` | Return `below` or `above`; a value `>= cutoff` selects `above`. |
| `merge` | Build an object from mapped `inputs`, with no model call. |
| `coalesce` | After all `sources` settle, return the first available one in listed order. |
| `value` | Return a literal JSON `value`. |
| `read` | Take one value from the original input by `path`, declaring `bounds` (numbers) or `categories` (strings). A missing path skips the node; a value outside the declared domain is an execution error. |

## Questions

| Type | Value | Limits |
| --- | --- | --- |
| `choice` | The selected category | 2–255 `criteria` options |
| `score` | Expected rubric position (not a probability) | 2–10 distinct `criteria` levels |
| `noul` | Probability of yes | — |

This is a subset of the [TypeSafe question schema](https://docs.typesafe.ai/primitives/choice).

## Inputs and outputs

`inputs` and `outputs` map names to sources:

| Source | JSON |
| --- | --- |
| Original input | `{"type": "state"}` |
| Constant | `{"type": "literal", "value": {"policy": "example"}}` |
| Earlier node value | `{"type": "node", "node": "eligible"}` |
| Full call answer | `{"type": "node", "node": "eligible", "view": "answer"}` |
| Optional value | `{"type": "node", "node": "eligible", "optional": true}` |

`view: "answer"` works only on call nodes and passes the whole validated answer,
including its probability distribution, instead of the scalar value.

## Conditions

Any node can have a `when` condition. It gates that node, not its dependencies.

```json
{"type": "compare", "node": "department", "op": "eq", "value": "billing"}
```

- `compare` supports `eq`, and `lt` / `ge` for numbers.
- Combine with `{"type": "all", "conditions": [...]}`, `{"type": "any", "conditions": [...]}`,
  and `{"type": "not", "condition": ...}`.
- `{"type": "present", "node": "id"}` tests whether a node ran.
- A comparison against a skipped node is false, so its negation is true.

## Execution

- Ready calls whose mapped inputs are identical are asked together in one request,
  so the judge reads the shared input once. The request's `state` is those inputs.
  Independent calls run concurrently, up to `max_workers` (default 4), and each
  node starts as soon as its own dependencies finish.
- A shared node runs once per execution, however many nodes use it.
- A node waits for all its data and condition dependencies, then runs or is skipped.
- A skipped required input skips its consumer; a skipped optional input is left out.
- `coalesce` skips only when every source skipped.
- A skipped optional output is omitted; a skipped required output raises
  `GraphExecutionError`.

`DecisionProgram` returns a `dspy.Prediction` with `outputs`, `node_values`,
`answers`, `skipped`, per-call `calls` metadata, `models`, token totals, and
`label` when there is a `label` output.

## Validation

```sh
uv run decision-compose validate task.json plan.json
```

Validation rejects cycles, unknown references, nodes that feed no output,
incompatible types, switches that miss a category, cutoffs outside the source
range, and unknown fields. For classification, the `label` output must be a
required categorical value whose possible values are all task labels.

## Writing a graph by hand

1. Start from [seed.json](../examples/seed.json) or [dag.json](../examples/dag.json).
2. Give each node an ID and connect nodes through `inputs`, `source`, `sources`, or `when`.
3. Point a `label` output at the final decision.
4. Run `validate`, then use the graph directly or as a seed:
   `decision-compose optimize task.json plan.json ...`.
