# Doom

rulesmith can optimize graphs that play [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)
scenarios. The graph runs once per action, and search scores it by episode
reward instead of label accuracy.
Install the `doom` extra first: `pip install 'rulesmith[doom]'`.

```mermaid
flowchart LR
  G["ViZDoom"] -->|"objects + variables as JSON"| P["graph"]
  P -->|"label = button"| G
  G -->|"episode reward"| S["score for GEPA"]
```

## Results

![Search fitness: best validation score after each candidate, by metric calls spent](assets/fitness.svg)

The chart is drawn by `benchmarks/fitness.py` from the searches' own reports,
summarized in [benchmarks/results/games.json](../benchmarks/results/games.json).
Every search in it ran with the same proposer, `openrouter/qwen/qwen3.8-27b`,
and a local Ruling judge (`Qwen3.5-4B-4bit`, one option ordering per question);
Doom searches had 200 episodes each and chess 1,000 games. A dot is a candidate
that beat everything before it, a line ends where its search's budget ran out,
and the label carries the winner's test score. The panels are not comparable:
each is a different task, and each is scaled to its own scores.

**`basic`** — kill one monster. `benchmarks/doom.sh`, from the task's own
baseline, a graph that only strafes left. Three search seeds found the same
answer, strafe toward the monster and fire when it is centered, after 15, 15
and 39 episodes: 0.33 → 0.97 on validation and 0.94 on the three held-out test
seeds, every time. That is the task's ceiling, since every action costs reward.
[gif](assets/doom/basic-seed12.gif) · [all 12](assets/doom/basic-reel.mp4)

**E1M1** — play a full level. `benchmarks/doom.sh`, from
[e1m1-target.json](../examples/doom/graphs/e1m1-target.json), a hand-written
graph that follows the route and asks the judge only whether a blocked way ahead
is a door. Three search seeds, 200 episodes each:

| Search seed | Validation, seed → best | Test | Exited | Judge calls in the winner | Episode length |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.22 → 0.31 | 0.28 | 1 of 6 | none | 93–336 steps |
| 1 | 0.22 → 0.32 | 0.26 | 1 of 6 | the door question | 108–259 steps |
| 2 | 0.22 → 0.27 | 0.26 | 0 of 6 | the door question | 89–200 steps |

What search found is worth reading with the reward in mind. The starting graph
survives the whole four minutes and scores 0.22; every winner runs the route,
kills three to ten monsters and usually dies inside a minute, and scores more,
because progress along the route and kills pay better than staying alive. Two
of the three reached the exit on one held-out seed each. The seeds agree on
the shape of the answer and differ on the judge: seed 0's winner asks nothing,
the other two keep the door question. How long each search took, from an hour
and a half to three and a half hours, turned on how soon its candidates stopped
asking the judge on nearly every step: an episode of the starting graph makes
about 1,400 judge calls and takes ten minutes on a local judge, and one whose
graph asks rarely takes seconds.

The graph in [the E1M1 exit graph](#the-e1m1-exit-graph) below, which exits
on 5 of 12 seeds, came from an earlier run with Claude Sonnet 5 proposing,
whose logs are not in the repo; it is a past result, not one the driver
reproduces.

**Arena** — graphs play each other, ranked by Elo from round-robin deathmatches.

```sh
uv run rulesmith arena examples/doom/duel.json \
  examples/doom/graphs/duel-hunter.json examples/doom/graphs/e1m1-exit.json \
  examples/doom/graphs/duel-attack.json --max-nodes 40 --max-depth 16 --output runs/arena.json
```

## Task file

A Doom task replaces labeled examples with episode seeds. See
[examples/doom/basic.json](../examples/doom/basic.json):

```json
{
  "description": "Play ViZDoom's basic scenario ... label output is the next button to press ...",
  "scenario": "basic",
  "labels": ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"],
  "reward_range": [-500, 100],
  "tics_per_action": 4,
  "train": [1, 2, 3, 4, 5, 6],
  "validation": [7, 8, 9],
  "test": [10, 11, 12]
}
```

| Field | Meaning |
| --- | --- |
| `description` | What the proposer is told about the game |
| `scenario` | A scenario bundled with ViZDoom: `basic`, `defend_the_center`, `health_gathering`, `deadly_corridor`, and others |
| `labels` | Actions the graph may choose: a Doom button, or buttons joined with `+` to press together, like `TURN_LEFT+ATTACK`. A button the scenario's own config leaves out is added for the task |
| `reward_range` | Lowest and highest possible episode reward, used to scale scores to 0–1 |
| `tics_per_action` | Game tics each chosen button is held (default 4) |
| `episode_timeout` | Tics before an episode ends (default: the scenario's own). Cap scenarios that otherwise run until death |
| `feedback_steps` | Final steps shown to the proposer per episode (default 8) |
| `map` | Map to load, like `E1M1` (default: the scenario's own) |
| `deathmatch` | Episodes are networked duels between graphs; `evaluate` and `optimize` require `--opponent` (default false) |
| `variables` | ViZDoom game variables added to the observation, like `HEALTH` or `AMMO2` |
| `rewards` | Rewards by ViZDoom setter name, like `kill_reward` or `death_penalty`, plus `explore` per newly visited cell and `progress` per map unit the route to the exit shrinks |
| `observe` | Additional observation fields: `pose`, `rangefinder`, `compass`, `memory`, `target`, `route` |
| `rays` | Rangefinder directions across the view (default 7) |
| `rangefinder_range` | Map units a ray reports when no wall is within reach (default 2048) |
| `compass_targets` | Object names the compass tracks anywhere on the map; required with `compass` |
| `compass_size` | Nearest compass targets reported (default 8) |
| `memory_steps` | Recent actions and movement kept by `memory` (default 4) |
| `cell_size` | Map units per cell for visit counts and `explore` (default 64) |
| `train`, `validation`, `test` | Episodes, distinct across splits: a seed, or `{"seed": 7, "scenario": "cig", "map": "MAP32"}` to play that seed on another map or scenario than the task's own |

## Scenarios

[examples/doom](../examples/doom) has a task for each scenario that suits
reactive, object-based play. Reward ranges come from ViZDoom's scenario rules and
configs, and a test plays every example to check them.

| Task | Goal | Reward range |
| --- | --- | --- |
| `basic`, `simpler_basic`, `rocket_basic` | Kill one monster quickly | [-500, 100] / [-500, 110] |
| `defend_the_center` | Kill approaching monsters with 26 rounds | [-1, 26] |
| `defend_the_line` | Kill advancing monsters with 79 rounds, capped at 2100 tics | [-1, 79] |
| `health_gathering`, `health_gathering_supreme` | Survive acid by collecting medkits | [-100, 2100] |
| `take_cover` | Dodge fireballs, capped at 2100 tics | [0, 2100] |
| `predict_position` | Lead a moving target with one slow rocket | [-0.35, 1] |

Navigation tasks can add wall distances, movement history, and an exit route
through `observe`. The [E1M1 example](../examples/doom/e1m1.json) uses these
features for a campaign level; the [duel example](../examples/doom/duel.json)
uses networked multiplayer games. These expose game state beyond the screen's
labels buffer, so results depend on the task's observation settings.

A file with a `scenario` field is treated as a Doom task. The same commands
(`validate`, `evaluate`, `optimize`) work on it.

Write the `description` for the proposer: say what the observation contains and
how actions change it. That is all it knows about the game.

> [!WARNING]
> An episode reward outside `reward_range` stops the run instead of being clipped,
> because clipping would hide real differences between graphs.

## Observations

Each step, the graph's `{"type": "state"}` input is:

```json
{
  "objects": [
    {"name": "Cacodemon", "category": "Monster", "x": 0.203, "y": -0.013, "width": 0.078, "height": 0.129}
  ],
  "variables": {"AMMO2": 50.0}
}
```

`x` runs from -1 (left edge) to 1 (right edge) and `y` from -1 (top) to 1 (bottom),
measured at the object's center. Objects come from ViZDoom's labels buffer, which
lists what is on screen. The player's own weapon is left out. `variables` holds
the scenario's game variables.

`observe` adds more, each under its own key: `target` (the nearest live monster
or opponent on screen, with its map distance), `pose`, `rangefinder` (wall
distances along rays across the view), `memory` (recent actions and movement),
`route` (the way to a level's exit) and `compass`:

```json
{"compass": [{"name": "DoomPlayer", "bearing": -23.2, "distance": 244}]}
```

The compass reaches the whole map, not just the screen, and reports the objects
named in `compass_targets`. `bearing` is degrees from where the player faces,
positive to the left. It lists the nearest of each name first, then fills the
remaining `compass_size` slots by distance, so one distant opponent is never
crowded out by the pickups underfoot. The player's own body is never a bearing.

## Scoring

An episode's score is `(reward - low) / (high - low)`. A graph that fails to
produce a label mid-episode, or an invalid graph, scores 0, below any episode it
could have played. The proposer sees each episode's reward, its action counts,
and its last few steps.

## Watching

```sh
uv run rulesmith evaluate examples/doom/basic.json runs/doom/plan.json \
  --backend ruling --max-workers 1 --split test --visible --output runs/doom/test.json
```

The game waits for each decision, so play runs at the judge's speed.

## Arena

`arena` ranks graphs by playing them against each other:

```sh
uv run rulesmith arena examples/doom/duel.json a.json b.json c.json --output runs/arena.json
```

- Every pair plays one deathmatch per episode in the task's `--split` (default
  `test`). Each graph runs its own ViZDoom instance, joined over a local network
  game. The engines run in lockstep, so a graph that waits on judge calls slows
  the match rather than losing turns.
- The graph with more frags wins; equal frags is a draw. A graph that fails to
  produce an action forfeits. Some maps, ViZDoom's `multi_deathmatch` among
  them, end the round at the first kill, which makes every match first blood;
  others play on to the timeout.
- The joining player sometimes loses a tic to network latency, getting up to
  about 10% fewer turns. Hosting alternates between episodes, so each pair
  shares that handicap evenly when it plays an even number of episodes.
  The same timing makes matches vary slightly from run to run.
- Ratings are a Bradley–Terry fit to all results at once, reported on the Elo
  scale: a 400-point gap means 10:1 odds and the field averages 1500. The fit
  ignores match order. Every pair also shares one virtual draw, which keeps an
  unbeaten graph's rating finite.
- Entrants are named by file stem, so two entrants' files need different names.
- Screen seeds before trusting a ladder. On many of them two graphs never meet,
  and a match nobody turns up to scores the same whatever the graphs do.

Opponents appear in `objects` with category `Player`, and `target` reports the
nearest one on screen. [examples/doom/duel.json](../examples/doom/duel.json)
scores frags minus deaths, plus a smaller reward for damage dealt and taken so that
a graph that lands hits scores above one that never connects, and sets `deathmatch`: deathmatch maps have no
single-player start, so `evaluate` and `optimize` play each episode as a duel
against the graph given by `--opponent`, alternating hosts by seed. Searching
against the arena's current leader breeds a challenger:

```sh
uv run rulesmith optimize examples/doom/duel.json examples/doom/graphs/e1m1-exit.json \
  --opponent examples/doom/graphs/e1m1-exit.json --max-nodes 40 --max-depth 16 \
  --backend ruling --max-workers 1 \
  --reflection-model "$REFLECTION_MODEL" \
  --reflection-base-url "$REFLECTION_BASE_URL" --output runs/challenger
```

The proposer sees each duel's frags, deaths, the opponent's frags, and whether
the graph won. ZDoom's built-in bots are left out: under ViZDoom's
synchronous play they barely fire, even with perfect skill settings.

Split by map, not by seed. A task whose three splits are seeds of one level
measures how well a graph knows that level: a champion bred on cig MAP03 shut
out the hand-written always-attack graph
[duel-attack](../examples/doom/graphs/duel-attack.json) 9-0 there and lost to it
on every other map where the two met. [examples/doom/duel-maps.json](../examples/doom/duel-maps.json)
trains on four maps, validates on three others and tests on four more, so the
validation score falls when a graph has learned a level rather than dueling.
A map only earns a place if graphs reliably meet on it; the scenario's other
levels are too big, and matches there end 0-0 whatever the graphs do.

## Limits

- Graphs keep no internal memory between steps. With `observe: ["memory"]`,
  the environment supplies recent actions, movement, and visit counts.
- The judge reads JSON, not pixels. Optional `rangefinder` and `route` fields
  supply wall distances and map-derived navigation hints; `compass` can reveal
  opponents outside the field of view. This is not a screen-only benchmark.
- Graphs with only reads and rules need no judge calls. Active call nodes use
  the judge, with calls sharing identical inputs batched into one request.
- Single-player episodes are evaluated one at a time. Deathmatch searches can
  use `--parallel-examples` because each duel owns its game instances.
- `rangefinder` crashes ViZDoom 1.3.0 on some maps, including E1M6 and the
  `cig` deathmatch maps, because it turns on sectors info. Leave it out there.

## The league

`--opponent` turns a search into a duel: every episode is played against a fixed
graph, scored from the candidate's own seat. Run that in a loop and it becomes a
league. Search a challenger against the champion with the `optimize` command
[under Arena](#arena), play it against everyone in the pool with `arena`, and
hand over the title only when it wins the head-to-head.

This section is a record of past runs. The script that drove the league loop is
not in the repo; each step it ran is an `optimize` or `arena` command, and the
graphs it produced are in [examples/doom/graphs](../examples/doom/graphs).

### Making the measurement mean something

Most of the work was in the task, not the search.

- **Screen seeds for contact.** On roughly a third of them the players never
  meet, and a match nobody turns up to scores the same whatever the graph does.
- **Pick the map deliberately.** ViZDoom's larger `cig` layouts produced no
  contact on any seed tried; the tiny `multi_duel` arena is small enough that
  both E1M1 graphs rocket themselves thirty-one times a match; on
  `multi_deathmatch` the first kill ends the round.
- **Make a match worth several frags.** Two minutes on a map where play
  continues after a kill gives three to five exchanges. On one-frag matches the
  run-to-run spread (0.004) was larger than the differences between candidates.
- **Crown on the head-to-head, not the ladder.** Two identical graphs once
  finished 117 Elo apart over six matches, and a clone took a 6.5-of-12 series
  on one stray frag. The title needs two clear wins over twelve paired matches.

### What it found

| Proposer | Generations | Outcome |
| --- | --- | --- |
| Qwen3.6-35B-A3B (mixture, 3B active) | 8 across two runs | No crown. Every candidate landed 0.01-0.03 below the champion, and three searches in four handed back their own seed. |
| Qwen3.8-27B (dense) | 2 | [duel-hunter](../examples/doom/graphs/duel-hunter.json) beat the champion 12-0, 44 frags to none. |

The difference was the proposer, not the harness. The mixture model edited the
champion's existing navigation rules and never touched the compass; the dense
model added three rules that read `compass.2.bearing` and `compass.2.distance`
and turn onto an opponent nobody can see, so it hunts between fights instead of
walking toward an exit a deathmatch does not have.

![The searched duelist hunting down the graph it dethroned](assets/doom/duel-hunter.gif)

Its full rule list is [below](#the-duel-hunter), and the whole duel is
[here](assets/doom/duel-hunter.mp4). Note what it is not: no judge calls.
Every graph that won anything reads numbers out of the observation, because
`target.x < -0.12` is exact and free while a call costs a round trip each step.

### Across maps

That result does not travel. Played across all 32 maps the scenario will load, the
same graph finishes behind the hand-written `duel-attack` it shut out 9-0 at home:
its compass rules hunt across geometry it has never seen, and the level kills it
before an opponent does. A task whose three splits are seeds of one map measures
how well a graph knows that map, so
[duel-maps.json](../examples/doom/duel-maps.json) trains on four maps, validates on
three others and tests on four more. Twelve generations on it crowned nobody —
two thirds of matches between evenly matched graphs end 0-0, and a metric that
cannot separate them cannot select between them either.

[duel-jev.json](../examples/doom/graphs/duel-jev.json) is also included for
comparison. It uses a judge call to choose combat actions, so running it requires
a configured [backend](cli.md#backends).

## Graphs

Two graphs the search found, in full. Each checks its rules in order, and the
first whose conditions all hold sets the action.

### The E1M1 exit graph

[examples/doom/graphs/e1m1-exit.json](../examples/doom/graphs/e1m1-exit.json),
which exits E1M1 on 5 of 12 seeds overall: 3 of 6 seeds used by search
(1–6), and 2 of 6 held-out seeds (7–12). The five-clear reel combines both groups.

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


### The duel hunter

[examples/doom/graphs/duel-hunter.json](../examples/doom/graphs/duel-hunter.json),
which a self-play league bred from the graph above, and which then beat it 12-0.

How it decides: each step it reads 12 values from the observation, then checks
21 rules in order. The first rule whose conditions all hold sets the action.
Mirrored pairs, and the door and exit rules, share a row in the table below. Rules 8-10 are the ones no other
graph has: with nobody on screen, the compass still gives a bearing to the
opponent, and the graph turns onto it.

```mermaid
flowchart TD
  OBS(["Each step: read 12 values from the observation"])
  G0["<b>Break off</b><br/>rules 1-2"]
  OBS --> G0
  G1["<b>Fight what is on screen</b><br/>rules 3-7"]
  G0 -.->|none fired| G1
  G2["<b>Hunt by compass</b><br/>rules 8-10"]
  G1 -.->|none fired| G2
  G3["<b>Get unstuck</b><br/>rules 11-12"]
  G2 -.->|none fired| G3
  G4["<b>Wall ahead</b><br/>rules 13-14"]
  G3 -.->|none fired| G4
  G5["<b>Doors and exit</b><br/>rules 15-16"]
  G4 -.->|none fired| G5
  G6["<b>Follow the route</b><br/>rules 17-20"]
  G5 -.->|none fired| G6
  G7["<b>Default</b><br/>rule 21"]
  G6 -.->|none fired| G7
  G0 & G1 & G2 & G3 & G4 & G5 & G6 & G7 ==> ACT(["first rule that fires sets the action"])
```

| # | Purpose | When | Action |
| ---: | --- | --- | --- |
| 1 | Break off | opponent on screen, ammo < 1, distance < 300 | back up |
| 2 | Break off | opponent on screen, health < 20 | back up |
| 3 | Fight | on screen, ammo ≥ 1, health ≥ 25, distance < 250, -0.12 ≤ x < 0.12 | fire |
| 4 | Fight | same, x < -0.12 | turn left |
| 5 | Fight | same, x ≥ 0.12 | turn right |
| 6 | Close in | on screen, ammo ≥ 1, health ≥ 25, distance ≥ 250, x < -0.05 | forward + left |
| 7 | Close in | same, x ≥ 0.05 | forward + right |
| 8 | Hunt | nobody on screen, compass bearing ≥ 45° | turn left |
| 9 | Hunt | nobody on screen, compass bearing < -45° | turn right |
| 10 | Hunt | nobody on screen, bearing within 45°, compass distance ≥ 300 | forward |
| 11 | Get unstuck | recent movement < 3, visits here ≥ 20 | back up |
| 12 | Get unstuck | recent movement < 3, visits here ≥ 6 | turn toward the route |
| 13 | Wall ahead | wall ahead < 64, route bearing ≥ 0 | turn left |
| 14 | Wall ahead | wall ahead < 64, route bearing < 0 | turn right |
| 15-16 | Doors and exit | route says exit within 100 or door within 90, bearing within 20° | forward + use |
| 17-18 | Follow the route | route bearing beyond ±35° | turn that way |
| 19-20 | Follow the route | route bearing 10° to 35° either side | forward + that way |
| 21 | Default | always | forward |

Rules 13-20 read `route`, which a deathmatch never fills in, so they are skipped
every step: inherited scaffolding the search kept but the map never exercises.

<details>
<summary>The winning graph as JSON</summary>

```json
{
  "nodes": {
    "read_health": {
      "when": null,
      "type": "read",
      "path": [
        "variables",
        "HEALTH"
      ],
      "bounds": [
        0.0,
        200.0
      ],
      "categories": null
    },
    "read_ammo": {
      "when": null,
      "type": "read",
      "path": [
        "variables",
        "SELECTED_WEAPON_AMMO"
      ],
      "bounds": [
        0.0,
        400.0
      ],
      "categories": null
    },
    "read_rf_mid": {
      "when": null,
      "type": "read",
      "path": [
        "rangefinder",
        3
      ],
      "bounds": [
        0.0,
        2048.0
      ],
      "categories": null
    },
    "read_route_next": {
      "when": null,
      "type": "read",
      "path": [
        "route",
        "next"
      ],
      "bounds": null,
      "categories": [
        "path",
        "door",
        "exit"
      ]
    },
    "read_route_bearing": {
      "when": null,
      "type": "read",
      "path": [
        "route",
        "bearing"
      ],
      "bounds": [
        -180.0,
        180.0
      ],
      "categories": null
    },
    "read_route_distance": {
      "when": null,
      "type": "read",
      "path": [
        "route",
        "distance"
      ],
      "bounds": [
        0.0,
        6000.0
      ],
      "categories": null
    },
    "read_target_x": {
      "when": null,
      "type": "read",
      "path": [
        "target",
        "x"
      ],
      "bounds": [
        -1.0,
        1.0
      ],
      "categories": null
    },
    "read_target_distance": {
      "when": null,
      "type": "read",
      "path": [
        "target",
        "distance"
      ],
      "bounds": [
        0.0,
        6000.0
      ],
      "categories": null
    },
    "read_enemy_compass_bearing": {
      "when": null,
      "type": "read",
      "path": [
        "compass",
        2,
        "bearing"
      ],
      "bounds": [
        -180.0,
        180.0
      ],
      "categories": null
    },
    "read_enemy_compass_distance": {
      "when": null,
      "type": "read",
      "path": [
        "compass",
        2,
        "distance"
      ],
      "bounds": [
        0.0,
        6000.0
      ],
      "categories": null
    },
    "read_memory_moved": {
      "when": null,
      "type": "read",
      "path": [
        "memory",
        "distance_moved"
      ],
      "bounds": [
        0.0,
        600.0
      ],
      "categories": null
    },
    "read_visits": {
      "when": null,
      "type": "read",
      "path": [
        "memory",
        "visits_here"
      ],
      "bounds": [
        1.0,
        10000.0
      ],
      "categories": null
    },
    "flee_no_ammo": {
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
            "value": 1.0
          },
          {
            "type": "compare",
            "node": "read_target_distance",
            "op": "lt",
            "value": 300.0
          }
        ]
      },
      "type": "value",
      "value": "MOVE_BACKWARD"
    },
    "retreat_critical": {
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
            "value": 20.0
          }
        ]
      },
      "type": "value",
      "value": "MOVE_BACKWARD"
    },
    "combat_strafe_left": {
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
            "value": 1.0
          },
          {
            "type": "compare",
            "node": "read_health",
            "op": "ge",
            "value": 25.0
          },
          {
            "type": "compare",
            "node": "read_target_distance",
            "op": "ge",
            "value": 250.0
          },
          {
            "type": "compare",
            "node": "read_target_x",
            "op": "lt",
            "value": -0.05
          }
        ]
      },
      "type": "value",
      "value": "MOVE_FORWARD+TURN_LEFT"
    },
    "combat_strafe_right": {
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
            "value": 1.0
          },
          {
            "type": "compare",
            "node": "read_health",
            "op": "ge",
            "value": 25.0
          },
          {
            "type": "compare",
            "node": "read_target_distance",
            "op": "ge",
            "value": 250.0
          },
          {
            "type": "compare",
            "node": "read_target_x",
            "op": "ge",
            "value": 0.05
          }
        ]
      },
      "type": "value",
      "value": "MOVE_FORWARD+TURN_RIGHT"
    },
    "combat_aim_left": {
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
            "value": 1.0
          },
          {
            "type": "compare",
            "node": "read_health",
            "op": "ge",
            "value": 25.0
          },
          {
            "type": "compare",
            "node": "read_target_distance",
            "op": "lt",
            "value": 250.0
          },
          {
            "type": "compare",
            "node": "read_target_x",
            "op": "lt",
            "value": -0.12
          }
        ]
      },
      "type": "value",
      "value": "TURN_LEFT"
    },
    "combat_aim_right": {
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
            "value": 1.0
          },
          {
            "type": "compare",
            "node": "read_health",
            "op": "ge",
            "value": 25.0
          },
          {
            "type": "compare",
            "node": "read_target_distance",
            "op": "lt",
            "value": 250.0
          },
          {
            "type": "compare",
            "node": "read_target_x",
            "op": "ge",
            "value": 0.12
          }
        ]
      },
      "type": "value",
      "value": "TURN_RIGHT"
    },
    "combat_attack": {
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
            "value": 1.0
          },
          {
            "type": "compare",
            "node": "read_health",
            "op": "ge",
            "value": 25.0
          },
          {
            "type": "compare",
            "node": "read_target_distance",
            "op": "lt",
            "value": 250.0
          },
          {
            "type": "compare",
            "node": "read_target_x",
            "op": "ge",
            "value": -0.12
          },
          {
            "type": "compare",
            "node": "read_target_x",
            "op": "lt",
            "value": 0.12
          }
        ]
      },
      "type": "value",
      "value": "ATTACK"
    },
    "combat_combat_pick": {
      "when": {
        "type": "present",
        "node": "read_target_x"
      },
      "type": "coalesce",
      "sources": [
        "flee_no_ammo",
        "retreat_critical",
        "combat_attack",
        "combat_aim_left",
        "combat_aim_right",
        "combat_strafe_left",
        "combat_strafe_right"
      ]
    },
    "track_bearing_left": {
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "present",
            "node": "read_enemy_compass_bearing"
          },
          {
            "type": "not",
            "condition": {
              "type": "present",
              "node": "read_target_x"
            }
          },
          {
            "type": "compare",
            "node": "read_enemy_compass_bearing",
            "op": "ge",
            "value": 45.0
          }
        ]
      },
      "type": "value",
      "value": "TURN_LEFT"
    },
    "track_bearing_right": {
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "present",
            "node": "read_enemy_compass_bearing"
          },
          {
            "type": "not",
            "condition": {
              "type": "present",
              "node": "read_target_x"
            }
          },
          {
            "type": "compare",
            "node": "read_enemy_compass_bearing",
            "op": "lt",
            "value": -45.0
          }
        ]
      },
      "type": "value",
      "value": "TURN_RIGHT"
    },
    "track_bearing_forward": {
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "present",
            "node": "read_enemy_compass_bearing"
          },
          {
            "type": "not",
            "condition": {
              "type": "present",
              "node": "read_target_x"
            }
          },
          {
            "type": "compare",
            "node": "read_enemy_compass_bearing",
            "op": "ge",
            "value": -45.0
          },
          {
            "type": "compare",
            "node": "read_enemy_compass_bearing",
            "op": "lt",
            "value": 45.0
          },
          {
            "type": "compare",
            "node": "read_enemy_compass_distance",
            "op": "ge",
            "value": 300.0
          }
        ]
      },
      "type": "value",
      "value": "MOVE_FORWARD"
    },
    "tracking_combat_pick": {
      "when": {
        "type": "not",
        "condition": {
          "type": "present",
          "node": "read_target_x"
        }
      },
      "type": "coalesce",
      "sources": [
        "track_bearing_left",
        "track_bearing_right",
        "track_bearing_forward"
      ]
    },
    "blocked_turn_left": {
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "present",
            "node": "read_rf_mid"
          },
          {
            "type": "present",
            "node": "read_route_bearing"
          },
          {
            "type": "compare",
            "node": "read_rf_mid",
            "op": "lt",
            "value": 64.0
          },
          {
            "type": "compare",
            "node": "read_route_bearing",
            "op": "ge",
            "value": 0.0
          }
        ]
      },
      "type": "value",
      "value": "TURN_LEFT"
    },
    "blocked_turn_right": {
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "present",
            "node": "read_rf_mid"
          },
          {
            "type": "present",
            "node": "read_route_bearing"
          },
          {
            "type": "compare",
            "node": "read_rf_mid",
            "op": "lt",
            "value": 64.0
          },
          {
            "type": "compare",
            "node": "read_route_bearing",
            "op": "lt",
            "value": 0.0
          }
        ]
      },
      "type": "value",
      "value": "TURN_RIGHT"
    },
    "route_exit_use": {
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
            "value": 100.0
          },
          {
            "type": "compare",
            "node": "read_route_bearing",
            "op": "ge",
            "value": -20.0
          },
          {
            "type": "compare",
            "node": "read_route_bearing",
            "op": "lt",
            "value": 20.0
          }
        ]
      },
      "type": "value",
      "value": "MOVE_FORWARD+USE"
    },
    "route_door_use": {
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
            "value": 90.0
          },
          {
            "type": "compare",
            "node": "read_route_bearing",
            "op": "ge",
            "value": -20.0
          },
          {
            "type": "compare",
            "node": "read_route_bearing",
            "op": "lt",
            "value": 20.0
          }
        ]
      },
      "type": "value",
      "value": "MOVE_FORWARD+USE"
    },
    "route_turn_left": {
      "when": {
        "type": "compare",
        "node": "read_route_bearing",
        "op": "ge",
        "value": 35.0
      },
      "type": "value",
      "value": "TURN_LEFT"
    },
    "route_turn_right": {
      "when": {
        "type": "compare",
        "node": "read_route_bearing",
        "op": "lt",
        "value": -35.0
      },
      "type": "value",
      "value": "TURN_RIGHT"
    },
    "route_forward_left": {
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "compare",
            "node": "read_route_bearing",
            "op": "ge",
            "value": 10.0
          },
          {
            "type": "compare",
            "node": "read_route_bearing",
            "op": "lt",
            "value": 35.0
          }
        ]
      },
      "type": "value",
      "value": "MOVE_FORWARD+TURN_LEFT"
    },
    "route_forward_right": {
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "compare",
            "node": "read_route_bearing",
            "op": "ge",
            "value": -35.0
          },
          {
            "type": "compare",
            "node": "read_route_bearing",
            "op": "lt",
            "value": -10.0
          }
        ]
      },
      "type": "value",
      "value": "MOVE_FORWARD+TURN_RIGHT"
    },
    "very_stuck_backward": {
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "compare",
            "node": "read_memory_moved",
            "op": "lt",
            "value": 3.0
          },
          {
            "type": "compare",
            "node": "read_visits",
            "op": "ge",
            "value": 20.0
          }
        ]
      },
      "type": "value",
      "value": "MOVE_BACKWARD"
    },
    "stuck_turn": {
      "when": {
        "type": "all",
        "conditions": [
          {
            "type": "compare",
            "node": "read_memory_moved",
            "op": "lt",
            "value": 3.0
          },
          {
            "type": "compare",
            "node": "read_visits",
            "op": "ge",
            "value": 6.0
          }
        ]
      },
      "type": "threshold",
      "source": "read_route_bearing",
      "cutoff": 0.0,
      "below": "TURN_RIGHT",
      "above": "TURN_LEFT"
    },
    "default_forward": {
      "when": null,
      "type": "value",
      "value": "MOVE_FORWARD"
    },
    "decision": {
      "when": null,
      "type": "coalesce",
      "sources": [
        "combat_combat_pick",
        "tracking_combat_pick",
        "very_stuck_backward",
        "stuck_turn",
        "blocked_turn_left",
        "blocked_turn_right",
        "route_exit_use",
        "route_door_use",
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
      "view": "value",
      "optional": false
    }
  }
}
```

</details>
