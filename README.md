# decision-compose

![decision-compose — Design and optimize decision graphs](image.png)

**decision-compose lets DSPy build decision graphs for you.** Give it a
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

**1. Start a judge.** Ruling runs locally and needs no account:

```sh
cd ../ruling && uv sync && uv run ruling serve
```

**2. Configure a proposer.** This is the generative model that writes graphs.
Any [DSPy model identifier](https://dspy.ai/learn/programming/language_models/)
works. For a local OpenAI-compatible server:

```sh
export REFLECTION_MODEL=openai/<served-model-id>
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
  "inputs": ["I was charged twice this month.", "The export button returns a 500 error."]
}
```

Then have the proposer model label them. `--validation` and `--test` set how
many inputs are held out for selection and final scoring:

```sh
uv run decision-compose draft-labels inputs.json --output task.json \
  --labeler-model "$REFLECTION_MODEL" --labeler-base-url "$REFLECTION_BASE_URL" \
  --validation 10 --test 10
```

> [!IMPORTANT]
> Review and correct `task.json` before optimizing. Search learns to agree with
> these labels, including their mistakes.

This is DSPy's teacher–student pattern: a large model labels, and the small
judge is fit to its judgment.

</details>

<details>
<summary><b>I have labeled examples</b></summary>

Write `task.json` directly, like [examples/task.json](examples/task.json):

```json
{
  "description": "Route each support ticket to billing, technical, or sales.",
  "labels": ["billing", "technical", "sales"],
  "train": [{"id": "t1", "state": "I was charged twice.", "label": "billing"}],
  "validation": [...],
  "test": [...]
}
```

`train` drives proposals, `validation` picks the winner, and `test` is scored once at the end.

</details>

**4. Build the graph.**

```sh
uv run decision-compose optimize task.json \
  --backend ruling --max-workers 1 \
  --reflection-model "$REFLECTION_MODEL" \
  --reflection-base-url "$REFLECTION_BASE_URL" \
  --output runs/my-task
```

This writes `runs/my-task/plan.json` (the graph) and `report.json` (scores).
Compare `validation_score` with `seed_validation_score`.

**5. Evaluate and run it.** The proposer is not needed from here on.

```sh
uv run decision-compose evaluate task.json runs/my-task/plan.json \
  --backend ruling --max-workers 1 --split test --output runs/my-task/test.json

uv run decision-compose run runs/my-task/plan.json examples/state.json \
  --backend ruling --max-workers 1
```

## How it works

```mermaid
flowchart LR
  T["task.json<br/>labels + examples"] --> S["Starting graph<br/>majority label"]
  S --> E["Run on training examples<br/>via Ruling or Jev"]
  E -->|"errors + expected labels"| P["DSPy proposer<br/>writes new graph JSON"]
  P -->|"schema check"| E
  E -->|"better on the batch"| V["Score on validation examples"]
  V --> B["Best graph<br/>plan.json"]
```

The proposer sees where the current graph answered wrong and writes a complete
replacement: new calls, rewritten questions, branches, and joins. A proposal is
kept only if it beats its parent on the training batch, and the graph with the
best validation score is saved. The test split never reaches the optimizer.

The proposer can use the full DAG language. This hand-written example
([examples/dag.json](examples/dag.json)), not a search result, asks two questions
in parallel, then runs only the specialist that matches:

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

See [Graph format](docs/graphs.md) for the JSON behind each node.

## Play Doom

The same search can build a graph that plays [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)
scenarios. Each step, the graph gets the visible objects and game variables as
JSON and outputs the next button. Its score is the episode's reward, not a label
match, so no labeled examples are needed.

```sh
uv run decision-compose optimize examples/doom/basic.json \
  --backend ruling --max-workers 1 \
  --reflection-model "$REFLECTION_MODEL" \
  --reflection-base-url "$REFLECTION_BASE_URL" \
  --output runs/doom

uv run decision-compose evaluate examples/doom/basic.json runs/doom/plan.json \
  --backend ruling --max-workers 1 --split test --visible --output runs/doom/test.json
```

`--visible` opens the game window so you can watch it play. See [Doom](docs/doom.md)
for the task format and scenarios.

## Results

![Search fitness: best validation score after each search round](docs/assets/fitness.svg)

**`basic`**: search started from a graph that only strafes left. Within three
rounds it found a graph that strafes toward the monster and fires once it is
centered. On held-out test seeds it killed every monster:

| Judge | Validation | Test | Test rewards (best possible ≈ 97) |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B via Ruling (local) | 0.33 → 0.97 | 0.94 | 79, 38, 83 |
| Jev (`jev-1.13.0`, hosted) | 0.33 → 0.97 | 0.94 | 75, 38, 83 |

![The found graph killing the monster on test seed 12](docs/assets/doom/basic-seed12.gif)

[All 12 seeds back to back](docs/assets/doom/basic-reel.mp4): 12 of 12 kills.

**E1M1**: search started from a graph that only walks forward, with Claude Sonnet 5
proposing graphs, the local 4B judge, and a route to the exit computed from the map.
The best graph ([examples/doom/graphs/e1m1-exit.json](examples/doom/graphs/e1m1-exit.json))
is 27 nodes of reads and routing with no judge calls, and it **finishes the level**:
[watch it exit in 75 seconds](docs/assets/doom/e1m1-exit.mp4) with 10 kills.
[All five clears back to back](docs/assets/doom/e1m1-reel.mp4).

| Seeds | Exited | Died | Best time |
| --- | --- | --- | --- |
| 1–6 (the task's train, validation, and test seeds) | 3 | 3 | 75 s |
| 7–12 (never used by search) | 2 | 4 | 89 s |

Episodes were capped at 6 minutes. Two of the deaths came within 260 units of the
exit switch; the rest came in the first minute's fights.

How the graph decides: each step it reads 10 values from the observation
(health · ammo · wall ahead · next on route · route bearing · waypoint distance · monster x · monster distance · recent movement · visits here), then checks 16 rules in order. The first rule whose
conditions all hold sets the action.

```mermaid
flowchart TD
  OBS(["Each step: read 10 values from the observation"])
  G0["<b>Survive</b><br/>rules 1–2"]
  OBS --> G0
  G1["<b>Fight</b><br/>rules 3–5"]
  G0 -.->|none fired| G1
  G2["<b>Get unstuck</b><br/>rules 6–7"]
  G1 -.->|none fired| G2
  G3["<b>Doors and exit</b><br/>rules 8–9"]
  G2 -.->|none fired| G3
  G4["<b>Wall ahead</b><br/>rules 10–11"]
  G3 -.->|none fired| G4
  G5["<b>Follow the route</b><br/>rules 12–15"]
  G4 -.->|none fired| G5
  G6["<b>Default</b><br/>rule 16"]
  G5 -.->|none fired| G6
  G0 & G1 & G2 & G3 & G4 & G5 & G6 ==> ACT(["first rule that fires sets the action"])
```

| # | Purpose | When | Action |
| ---: | --- | --- | --- |
| 1 | Survive | monster visible, ammo < 1, monster distance < 150 | back up |
| 2 | Survive | monster visible, health < 15, ammo < 5, monster distance < 150 | back up |
| 3 | Fight | monster visible, ammo ≥ 1, monster distance < 450, -0.15 ≤ monster x < 0.15 | fire |
| 4 | Fight | monster visible, ammo ≥ 1, monster distance < 450, monster x < -0.15 | turn left |
| 5 | Fight | monster visible, ammo ≥ 1, monster distance < 450, monster x ≥ 0.15 | turn right |
| 6 | Get unstuck | recent movement < 3, visits here ≥ 20 | back up |
| 7 | Get unstuck | recent movement < 3, visits here ≥ 6 | turn left if route bearing ≥ 0, else turn right |
| 8 | Doors and exit | next on route = exit, waypoint distance < 100, -20 ≤ route bearing < 20 | forward + use |
| 9 | Doors and exit | next on route = door, waypoint distance < 90, -20 ≤ route bearing < 20 | forward + use |
| 10 | Wall ahead | wall ahead < 64, route bearing ≥ 0 | turn left |
| 11 | Wall ahead | wall ahead < 64, route bearing < 0 | turn right |
| 12 | Follow the route | route bearing ≥ 35 | turn left |
| 13 | Follow the route | route bearing < -35 | turn right |
| 14 | Follow the route | 10 ≤ route bearing < 35 | forward + left |
| 15 | Follow the route | -35 ≤ route bearing < -10 | forward + right |
| 16 | Default | always | forward |

<details>
<summary>The graph as JSON</summary>

```json
{
  "nodes": {
    "read_health": {
      "type": "read",
      "path": [
        "variables",
        "HEALTH"
      ],
      "bounds": [
        0,
        200
      ]
    },
    "read_ammo": {
      "type": "read",
      "path": [
        "variables",
        "SELECTED_WEAPON_AMMO"
      ],
      "bounds": [
        0,
        400
      ]
    },
    "read_rf_mid": {
      "type": "read",
      "path": [
        "rangefinder",
        3
      ],
      "bounds": [
        0,
        2048
      ]
    },
    "read_route_next": {
      "type": "read",
      "path": [
        "route",
        "next"
      ],
      "categories": [
        "path",
        "door",
        "exit"
      ]
    },
    "read_route_bearing": {
      "type": "read",
      "path": [
        "route",
        "bearing"
      ],
      "bounds": [
        -180,
        180
      ]
    },
    "read_route_distance": {
      "type": "read",
      "path": [
        "route",
        "distance"
      ],
      "bounds": [
        0,
        6000
      ]
    },
    "read_target_x": {
      "type": "read",
      "path": [
        "target",
        "x"
      ],
      "bounds": [
        -1,
        1
      ]
    },
    "read_target_distance": {
      "type": "read",
      "path": [
        "target",
        "distance"
      ],
      "bounds": [
        0,
        6000
      ]
    },
    "read_memory_moved": {
      "type": "read",
      "path": [
        "memory",
        "distance_moved"
      ],
      "bounds": [
        0,
        600
      ]
    },
    "read_visits": {
      "type": "read",
      "path": [
        "memory",
        "visits_here"
      ],
      "bounds": [
        1,
        10000
      ]
    },
    "flee_no_ammo": {
      "type": "value",
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "present",
            "node": "read_target_x"
          },
          {
            "type": "compare",
            "node": "read_ammo",
            "op": "lt",
            "value": 1
          },
          {
            "type": "compare",
            "node": "read_target_distance",
            "op": "lt",
            "value": 150
          }
        ]
      },
      "value": "MOVE_BACKWARD"
    },
    "retreat_critical": {
      "type": "value",
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "present",
            "node": "read_target_x"
          },
          {
            "type": "compare",
            "node": "read_health",
            "op": "lt",
            "value": 15
          },
          {
            "type": "compare",
            "node": "read_ammo",
            "op": "lt",
            "value": 5
          },
          {
            "type": "compare",
            "node": "read_target_distance",
            "op": "lt",
            "value": 150
          }
        ]
      },
      "value": "MOVE_BACKWARD"
    },
    "combat_attack": {
      "type": "value",
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "present",
            "node": "read_target_x"
          },
          {
            "type": "compare",
            "node": "read_ammo",
            "op": "ge",
            "value": 1
          },
          {
            "type": "compare",
            "node": "read_target_distance",
            "op": "lt",
            "value": 450
          },
          {
            "type": "compare",
            "node": "read_target_x",
            "op": "ge",
            "value": -0.15
          },
          {
            "type": "compare",
            "node": "read_target_x",
            "op": "lt",
            "value": 0.15
          }
        ]
      },
      "value": "ATTACK"
    },
    "combat_aim_left": {
      "type": "value",
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "present",
            "node": "read_target_x"
          },
          {
            "type": "compare",
            "node": "read_ammo",
            "op": "ge",
            "value": 1
          },
          {
            "type": "compare",
            "node": "read_target_distance",
            "op": "lt",
            "value": 450
          },
          {
            "type": "compare",
            "node": "read_target_x",
            "op": "lt",
            "value": -0.15
          }
        ]
      },
      "value": "TURN_LEFT"
    },
    "combat_aim_right": {
      "type": "value",
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "present",
            "node": "read_target_x"
          },
          {
            "type": "compare",
            "node": "read_ammo",
            "op": "ge",
            "value": 1
          },
          {
            "type": "compare",
            "node": "read_target_distance",
            "op": "lt",
            "value": 450
          },
          {
            "type": "compare",
            "node": "read_target_x",
            "op": "ge",
            "value": 0.15
          }
        ]
      },
      "value": "TURN_RIGHT"
    },
    "very_stuck_backward": {
      "type": "value",
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "compare",
            "node": "read_memory_moved",
            "op": "lt",
            "value": 3
          },
          {
            "type": "compare",
            "node": "read_visits",
            "op": "ge",
            "value": 20
          }
        ]
      },
      "value": "MOVE_BACKWARD"
    },
    "stuck_turn": {
      "type": "threshold",
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "compare",
            "node": "read_memory_moved",
            "op": "lt",
            "value": 3
          },
          {
            "type": "compare",
            "node": "read_visits",
            "op": "ge",
            "value": 6
          }
        ]
      },
      "source": "read_route_bearing",
      "cutoff": 0,
      "below": "TURN_RIGHT",
      "above": "TURN_LEFT"
    },
    "route_exit_use": {
      "type": "value",
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "compare",
            "node": "read_route_next",
            "op": "eq",
            "value": "exit"
          },
          {
            "type": "compare",
            "node": "read_route_distance",
            "op": "lt",
            "value": 100
          },
          {
            "type": "compare",
            "node": "read_route_bearing",
            "op": "ge",
            "value": -20
          },
          {
            "type": "compare",
            "node": "read_route_bearing",
            "op": "lt",
            "value": 20
          }
        ]
      },
      "value": "MOVE_FORWARD+USE"
    },
    "route_door_use": {
      "type": "value",
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "compare",
            "node": "read_route_next",
            "op": "eq",
            "value": "door"
          },
          {
            "type": "compare",
            "node": "read_route_distance",
            "op": "lt",
            "value": 90
          },
          {
            "type": "compare",
            "node": "read_route_bearing",
            "op": "ge",
            "value": -20
          },
          {
            "type": "compare",
            "node": "read_route_bearing",
            "op": "lt",
            "value": 20
          }
        ]
      },
      "value": "MOVE_FORWARD+USE"
    },
    "blocked_turn_left": {
      "type": "value",
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "compare",
            "node": "read_rf_mid",
            "op": "lt",
            "value": 64
          },
          {
            "type": "compare",
            "node": "read_route_bearing",
            "op": "ge",
            "value": 0
          }
        ]
      },
      "value": "TURN_LEFT"
    },
    "blocked_turn_right": {
      "type": "value",
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "compare",
            "node": "read_rf_mid",
            "op": "lt",
            "value": 64
          },
          {
            "type": "compare",
            "node": "read_route_bearing",
            "op": "lt",
            "value": 0
          }
        ]
      },
      "value": "TURN_RIGHT"
    },
    "route_turn_left": {
      "type": "value",
      "when": {
        "type": "compare",
        "node": "read_route_bearing",
        "op": "ge",
        "value": 35
      },
      "value": "TURN_LEFT"
    },
    "route_turn_right": {
      "type": "value",
      "when": {
        "type": "compare",
        "node": "read_route_bearing",
        "op": "lt",
        "value": -35
      },
      "value": "TURN_RIGHT"
    },
    "route_forward_left": {
      "type": "value",
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "compare",
            "node": "read_route_bearing",
            "op": "ge",
            "value": 10
          },
          {
            "type": "compare",
            "node": "read_route_bearing",
            "op": "lt",
            "value": 35
          }
        ]
      },
      "value": "MOVE_FORWARD+TURN_LEFT"
    },
    "route_forward_right": {
      "type": "value",
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "compare",
            "node": "read_route_bearing",
            "op": "ge",
            "value": -35
          },
          {
            "type": "compare",
            "node": "read_route_bearing",
            "op": "lt",
            "value": -10
          }
        ]
      },
      "value": "MOVE_FORWARD+TURN_RIGHT"
    },
    "default_forward": {
      "type": "value",
      "value": "MOVE_FORWARD"
    },
    "decision": {
      "type": "coalesce",
      "sources": [
        "flee_no_ammo",
        "retreat_critical",
        "combat_attack",
        "combat_aim_left",
        "combat_aim_right",
        "very_stuck_backward",
        "stuck_turn",
        "route_exit_use",
        "route_door_use",
        "blocked_turn_left",
        "blocked_turn_right",
        "route_turn_left",
        "route_turn_right",
        "route_forward_left",
        "route_forward_right",
        "default_forward"
      ]
    }
  },
  "outputs": {
    "label": {
      "type": "node",
      "node": "decision",
      "view": "value"
    }
  }
}
```

</details>

**Arena**: graphs can also play each other. `arena` runs a round-robin of
one-on-one deathmatches on ViZDoom's `cig` competition map and ranks the
entrants by Elo:

```sh
uv run decision-compose arena examples/doom/duel.json \
  examples/doom/graphs/e1m1-exit.json examples/doom/graphs/e1m1-searched.json \
  examples/doom/graphs/duel-attack.json --max-nodes 40 --max-depth 16 --output runs/arena.json
```

Three call-free graphs, six seeds per pair across two maps (18 one-minute
matches, under a minute of wall clock):

| Graph | Elo | Won | Drawn | Lost | Frags for / against |
| --- | --- | --- | --- | --- | --- |
| e1m1-exit | 1551 | 3 | 9 | 0 | 3 / 0 |
| e1m1-searched | 1517 | 2 | 9 | 1 | 2 / 1 |
| duel-attack | 1432 | 0 | 8 | 4 | 0 / 4 |

[Watch the champion take a frag](docs/assets/doom/duel-e1m1-exit.mp4) on MAP03.
Neither E1M1 graph was searched for dueling: they walk the map looking for an
exit that a deathmatch has none of, so 13 of the 18 matches ended without a
frag and the ranking mostly rewards staying alive. Network timing shifts which
tics the joining player acts on, so results vary between runs.

What the three entrants do, in the order each one checks its rules (the first
rule that fires wins, through a `coalesce` node):

| Graph | Rules | In a duel it | Reads a duel does not have |
| --- | --- | --- | --- |
| [e1m1-exit](examples/doom/graphs/e1m1-exit.json) | 16 | Backs off at low health or no ammo, fires when a target is centered, turns onto it otherwise, then walks and unsticks itself | `route.*`, `rangefinder.3` |
| [e1m1-searched](examples/doom/graphs/e1m1-searched.json) | 17 | Fires when a target is centered, aims when it is not, then unsticks itself and explores forward | `route.*`, `rangefinder.0/3/6` |
| [duel-attack](examples/doom/graphs/duel-attack.json) | 0 | Holds the trigger down, forever | none |

The first two were searched for E1M1, so most of their structure is navigation
toward an exit. A deathmatch map has no exit and does not fill in `route`, so
those reads go missing and every rule that depends on one is skipped: what is
left is the fighting half plus a walk-forward default. That is the bar a
purpose-built duel graph has to beat.

Search can then aim at the arena rather than at a scenario's own reward. With
`--opponent`, every episode is a duel against a fixed graph, so the leader can
be used to breed its own challenger:

```sh
uv run decision-compose optimize examples/doom/duel.json examples/doom/graphs/e1m1-exit.json \
  --opponent examples/doom/graphs/e1m1-exit.json --max-nodes 40 --max-depth 16 \
  --reflection-model "$REFLECTION_MODEL" --output runs/challenger
```

See [Doom](docs/doom.md#arena) for how matches are played and scored.

## Use from Python

A graph runs as a `dspy.Module`:

```python
from pathlib import Path

from typesafe_sdk import TypeSafeClient

from decision_compose.graph import Plan
from decision_compose.runtime import DecisionProgram

plan = Plan.model_validate_json(Path("runs/my-task/plan.json").read_text())
with TypeSafeClient(base_url="http://127.0.0.1:8010", api_key="local-no-auth") as client:
    program = DecisionProgram(plan, client, model="default")
    result = program(state="Please correct the duplicate charge on my invoice.")
    print(result.label, result.answers)
```

## Backends

| Backend | Flag | Endpoint | Credential |
| --- | --- | --- | --- |
| Ruling (local) | `--backend ruling` | `RULING_BASE_URL`, default `http://127.0.0.1:8010` | `RULING_API_KEY` if auth is on |
| Jev (hosted) | `--backend typesafe` (default) | `TYPESAFE_BASE_URL` | `TYPESAFE_API_KEY` |

Ruling serializes requests, so use `--max-workers 1`. Hosted calls may incur
charges; the search budget counts examples, not requests or dollars.

## Learn more

- [Graph format](docs/graphs.md): nodes, inputs, conditions, execution, and writing graphs by hand.
- [Optimization](docs/optimization.md): search settings, reports, testing, and limits.
- [Doom](docs/doom.md): task format, observations, scoring, and watching a graph play.

## Development

```sh
uv run pytest -q
uv run ruff check src tests && uv run ruff format --check src tests
```
