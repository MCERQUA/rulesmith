# Chess

A graph plays chess by naming a policy, not a move. There are thousands of moves
and which ones are legal changes every ply, so the label says what kind of move
to make and the board picks it.

## Policies

Each label is a selector over the legal moves. Every one of them always returns a
move, so a graph can never stall.

| Label | Plays |
| --- | --- |
| `CAPTURE` | the capture that wins most material, counting one recapture |
| `CHECK` | a checking move, and mate in one when there is one |
| `SAVE` | the most valuable hanging piece, to a square nobody attacks |
| `CASTLE` | castles |
| `DEVELOP` | a knight or bishop off the back rank, towards the middle |
| `ADVANCE` | a pawn forward: into the middle early, up the board later |
| `QUIET` | a safe quiet move |

When a policy finds nothing of its kind, the board plays a safe quiet move. That
fallback matters more than it sounds: an earlier version fell back to the
highest-gain move, which made every policy behave like `CAPTURE` and flattened
the whole ladder.

## What a graph sees

```json
{
  "ply": 12,
  "phase": "middlegame",
  "material": {"mine": 39, "theirs": 36, "balance": 3},
  "king": {"in_check": 0, "attackers": 0, "can_castle": 1, "castled": 0},
  "capture": {"best_gain": 3, "count": 2},
  "check": {"available": 1, "gain": 0},
  "hanging": {"count": 1, "worst": 3},
  "development": {"minor_home": 2},
  "moves": {"count": 31}
}
```

Every leaf is a number or a string, so a `read` node takes any of them by path.
All of it is from the side to move's point of view.

## Scoring a game

One seed is one game: four random legal opening plies, then the graphs play. A
graph is white on even seeds and black on odd ones, so the same opening is
scored from both chairs.

A rule list cannot force mate from a won position, so games that reach the move
limit are adjudicated: a side a minor piece ahead has won, anything closer is a
draw. The score is mostly the result and partly the final position:

```
score = 0.8 x result + 0.2 x (material edge, clipped at +/- 15, mapped to 0..1)
```

Set `material_weight` to 0 for results only. The split exists because results
alone are nearly flat here: most games between rule lists end drawn, and the
board is what says which graph stood better. It is the opposite problem to
[Doom](doom.md), where the reward moved but the graphs never met.

## The baseline ladder

Six graphs, every pair, 30 seeds played from both sides. 900 games, 38 seconds
on a laptop.

| Graph | Elo | W | D | L | Material |
| --- | --- | --- | --- | --- | --- |
| hunter | 1548 | 60 | 230 | 10 | +1518 |
| always-check | 1535 | 52 | 233 | 15 | -2522 |
| greedy | 1483 | 2 | 278 | 20 | +1948 |
| developer | 1480 | 1 | 277 | 22 | -1591 |
| always-capture | 1478 | 1 | 275 | 24 | +2881 |
| always-quiet | 1476 | 0 | 275 | 25 | -2234 |

Material and results pull apart. `always-check` wins 52 games while shedding
2,522 points of material, because a policy that keeps giving check keeps finding
mate. `greedy` gains 1,948 and wins twice, because nothing in it converts a won
position. `hunter` takes material and switches to checks once it is ahead, and
tops both columns.

## Running it

```sh
uv run decision-compose evaluate examples/chess/duel.json \
  examples/chess/graphs/hunter.rules \
  --opponent examples/chess/graphs/greedy.rules \
  --split test --backend ruling --output runs/chess.json

uv run decision-compose optimize examples/chess/duel.json \
  examples/chess/graphs/hunter.rules \
  --opponent examples/chess/graphs/greedy.rules \
  --backend ruling --max-workers 1 \
  --reflection-model "$REFLECTION_MODEL" \
  --reflection-base-url "$REFLECTION_BASE_URL" --output runs/chess-search
```

`--opponent` is required, as it is for a Doom deathmatch: the task is a game
between two graphs, and a graph's score only means something against a named
opponent.

## Limits

- The policies are the ceiling. A graph chooses among seven kinds of move, so it
  cannot play a move no policy describes, and it cannot search ahead.
- `gain` looks one recapture deep. A real exchange search would value some
  captures differently.
- Draws dominate. Two rule lists that both avoid blunders will shuffle to the
  move limit, which is why the score carries a material term.
- The opening is four random plies, so a seed can hand one side a better start.
  Playing every seed from both colors cancels it across a ladder, not within a
  single game.
