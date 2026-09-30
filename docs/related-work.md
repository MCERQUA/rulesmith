# Related work

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
