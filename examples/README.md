# Examples

Each decision has a folder of its own, holding what you need to read it, rerun it, and trust it:

| File | What it is |
| --- | --- |
| `task.json` | The labeled examples the graph was searched and measured on |
| `graph.rules` | The decision as it runs: rules, and the questions the judge is asked |
| `map.json` | Instead of `graph.rules`, for a decision made of other graphs; its units sit in `units/` |
| `evaluation.json` | The graph scored on the task's test split, written by `rulesmith evaluate`; it names the judge it was scored against (`backend`, `resolved_models`) |

| Folder | Decision | Shows |
| --- | --- | --- |
| [support-routing](support-routing) | Which team answers a ticket | a choice; `seed.json` and `dag.json` are the same task as graph JSON, `input.json` an input for `rulesmith run` |
| [ticket-triage](ticket-triage) | How urgently to handle a ticket | a score, and a threshold `calibrate` fits |
| [order-review](order-review) | Ship an order or hold it for review | readings that settle most orders before the judge is asked |
| [reply-check](reply-check) | Whether a drafted reply can go out | probabilities and their thresholds |
| [ticket-handling](ticket-handling) | A ticket routed by team, billing handled by a unit | a map with one unit |
| [trust-safety](trust-safety) | Scam postings, review recommendations, banking intents | a map of three searched units, scored end to end |

`order-review` and `reply-check` have no task yet, so nothing measures them. A graph's rule
counts and cutoffs hold for the judge they were measured on; after changing the judge, run
`rulesmith calibrate` and `rulesmith evaluate` again and commit both with the change.

`doom/` and `chess/` hold the game tasks and graphs.
