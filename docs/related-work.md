# Related work

## rulesmith and DSPy

rulesmith is built on DSPy and solves a narrower problem. It fits when:

- the answer must be one of a fixed set of labels, not free text;
- the input is structured, so cheap rules over its fields settle many cases and
  the judge is asked only about the rest;
- a calibrated probability matters, or an unsure case should go to a person
  rather than be guessed;
- a small local judge should be enough, and a large model per decision is more
  than the task is worth.

Use DSPy directly for open-ended generation, programs of several steps,
retrieval, and tasks where worked examples in the prompt carry the accuracy.
rulesmith searches over rules and questions, not demonstrations, so it has
nothing to add there.

DSPy 3.4 also speaks to the judge itself: an experimental `TypeSafe` client
used as any other `lm`, `Noul`, `Choice` and `Score` output types, and a
`ReAnchor` optimizer that fits their thresholds, cuts and weights to a metric.
If the question belongs inside a DSPy program, that is the way to ask it, and
rulesmith's own optimizer and extraction nodes run on it. rulesmith is for when
the decision is a program of its own: rules over the input's fields that settle
cases before any question is asked, several judges at different prices with
escalation between them, calibration that charges for calls, a rules file a
reviewer reads in a pull request, and `serve` and `drift` once it is in
production. The `dspy-typed` row in the benchmarks is that native path on the
same judge, with GEPA and ReAnchor on the same budget; the graphs are measured
against it, and the gap is what being a graph is worth.

What that buys, measured on five public datasets, is in [Benchmarks](benchmarks.md).

## Program search

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

DSPy's experimental [decision types](https://dspy.ai/current/api/experimental/DecisionTypes/)
put the same typed questions, a probability, a score and a choice, on a DSPy
predictor's outputs, each with one threshold, one set of cuts or one set of
weights, and [ReAnchor](https://dspy.ai/current/api/experimental/ReAnchor/)
fits those settings from the probabilities the model produced, with a fold check.
`rulesmith calibrate` fits the lines a graph's rules draw the same way; the
difference is that here a judgment is a node and any number of rules can draw a
line on it, so the settings belong to the rules rather than to the question.

On the runtime side, [decision-circuits](https://github.com/Barneyjm/decision-circuits)
pairs typed questions with deterministic gates, though you build its circuits
in Python instead of declaring them, and [GoRules ZEN](https://github.com/gorules/zen) executes
statically validated JSON decision models with no model in the loop at all.

None of the projects above combines a rule list that compiles to a statically
validated DAG, calibrated probabilities taken from logits inside that graph, and
a judge that can be called mid-execution and gated behind a cheap reading. That
is a claim about the work listed here, not a survey of the field.
