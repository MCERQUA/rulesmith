# Benchmarks

rulesmith against the usual baselines on five public classification datasets, plus a chess
task against Stockfish. The short version:

- **Where a field in the data settles most cases, rulesmith matches the best language-model
  methods at a small fraction of their calls.** On clothing reviews it ties DSPy (0.916 vs
  0.918, p = 0.91) and DSPy's typed classifier on the same judge (0.929, p = 0.16) while asking
  the judge on 14% of examples instead of 100%.
- **Where the signal is spread across structured fields and text, the searched graph beats
  every language-model baseline**, on job postings 0.733 vs DSPy's 0.693 (p = 0.021) and 0.692
  for DSPy's typed classifier on the same judge (p = 0.013), though one seed in three found no
  rule at all, and a TF-IDF classifier trained on the same 100 examples does better still
  (0.857).
- **Where the task is one judgment over free text, search has little to add.** On Banking77 it
  keeps the single direct question and ties DSPy's typed classifier (0.657 vs 0.677, p = 0.24).
  On TREC it now rewrites the question and writes a briefing the judge reads before the input,
  0.838 against the direct call's 0.759, three points under DSPy's typed classifier (0.869,
  p = 0.015) and chain of thought (0.884).
- **On news, the dataset held back as a blind check, search scores highest** (0.283 vs 0.253
  for the direct call, 0.253 for DSPy's typed classifier and 0.246 for DSPy), but every method
  there is close to chance.
- **The fair DSPy baseline, `dspy-typed`, changes one verdict.** DSPy's typed classifier asks the
  same judge the same question the direct call asks, then GEPA rewrites the instructions the
  judge reads and ReAnchor fits the option weights. It ties the old text baseline on jobs,
  clothing and Banking77, and on TREC it turns the direct question's 0.759 into 0.869 by
  rewriting the instructions alone. What mattered was where they went: DSPy puts them in the
  state the judge reads, next to the input, and the same text in the question changes nothing.
  Search can now write such a briefing too, and with every proposal aimed at the question it
  reaches 0.838 on TREC, three points under DSPy and inside the spread between seeds.
- **Chess:** search found no graph that did better against Stockfish on test than its starting one.

The text tables are reproducible with `benchmarks/overnight.sh`, the 40-game Stockfish
table with `python3 -m rulesmith.bench chess`, and the three 1,000-game searches with
`benchmarks/chess.sh`; the raw results, one record per
method and seed with every test answer, are in [`benchmarks/results`](../benchmarks/results).

## Setup

Each dataset is sampled three times (seeds 0, 1, 2). Each sample has 50 training examples, 50
validation examples, and 300 test examples drawn from the dataset's own test split, which no
method sees before it is scored. Accuracy is the mean over the three seeds, with the standard
deviation across seeds after ±.

| Method | What it is |
| --- | --- |
| majority | The most common training label, always. |
| tfidf | TF-IDF features and logistic regression, trained on the 100 training and validation examples. |
| direct | One judge question over every label, with the label descriptions as criteria. |
| handwritten | Rules written by hand before any result, where the dataset has an obvious field to use. |
| search | `rulesmith optimize` from the direct graph: mined rules, then GEPA proposals, 200 scored calls, a call cost of 0.02 per judge call, and a significance guard at p < 0.1 before replacing the seed. |
| dspy | DSPy `Predict` optimized with GEPA over the same examples and budget. |
| dspy-cot | The same with `ChainOfThought`. |
| dspy-typed | DSPy's typed classifier (its experimental decision types) on the same judge as the graphs: one choice question over the labels, instructions rewritten by GEPA on the same budget, option weights fitted by ReAnchor. The fair comparison: what a graph gets from being a graph is what is left over this. |

Models, all local except one:

| Role | Model |
| --- | --- |
| Judge (direct, handwritten, search) | `mlx-community/Qwen3.5-4B-4bit` under Ruling, one option ordering per question (`RULING_ROTATIONS=1`) |
| DSPy student | `mlx-community/Qwen3.5-4B-4bit` under `mlx_lm.server`, thinking off, 2,048 reply tokens |
| Proposer (search) and reflection (DSPy) | Qwen3.8 27B: through LM Studio for early runs, then the same model through OpenRouter with reasoning capped at 2,048 tokens |
| Judge (dspy-typed) | the same Ruling server, through DSPy's experimental `TypeSafe` client |

The same proposer writes rulesmith's graphs and DSPy's instructions, so neither side gets a
stronger optimizer. Both answer with the same weights, `Qwen3.5-4B-4bit`, and differ in how
it is asked: Ruling scores each allowed label and returns probabilities, while DSPy has the
model write a reply and parses a label from it. The direct question shows the interface alone
is no advantage: on the same model it scores below DSPy on jobs and TREC. What search adds
over it is the rules. `dspy-typed` removes the last difference: it is a DSPy program on the
judge itself, tuned by DSPy's own GEPA and ReAnchor (DSPy 3.4.0, experimental), so it and search
differ only in what the optimizer may change, the instructions and weights of one question
against the graph around it.

**Columns.** *At 80%* and *At 50%* are accuracy on the 80% and 50% of test examples the method
was surest of, for methods that report confidence. *Calls* are judge or language-model calls
per example. *Seconds* are wall time per example with five datasets sharing one machine, so
compare them within a dataset only. *Invalid* is the share of answers that named no allowed
label; they count as wrong. The p-values are exact McNemar tests of search against each
method over the 900 pooled test answers.

## Results

### Clothing reviews: would the shopper recommend it?

[Women's e-commerce clothing reviews](https://huggingface.co/datasets/saattrupdan/womens-clothing-ecommerce-reviews):
review text, star rating, department. Two labels.

| Method | Accuracy | At 80% | At 50% | Calls | Input tokens | Output tokens | Seconds | Invalid | p vs search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| handwritten | 0.932 ± 0.005 | 0.979 | 0.978 | 0.15 | 40 | 0 | 0.13 | 0.0% | 0.091 |
| dspy-typed | 0.929 ± 0.012 | 0.985 | 1.000 | 1.00 | 838 | 0 | 0.20 | 0.0% | 0.16 |
| direct | 0.921 ± 0.005 | 0.983 | 1.000 | 1.00 | 242 | 0 | 0.73 | 0.0% | 0.58 |
| dspy | 0.918 ± 0.020 | n/a | n/a | 1.00 | 602 | 14 | 2.78 | 0.0% | 0.91 |
| **search** | **0.916 ± 0.016** | 0.969 | 0.982 | **0.14** | 41 | 0 | 0.01 | 0.0% | |
| dspy-cot | 0.916 ± 0.018 | n/a | n/a | 1.00 | 881 | 160 | 7.04 | 0.1% | 1 |
| majority | 0.813 ± 0.012 | n/a | n/a | 0 | 0 | 0 | 0.00 | 0.0% | 2e-15 |
| tfidf | 0.813 ± 0.012 | 0.875 | 0.949 | 0 | 0 | 0 | 0.00 | 0.0% | 2e-15 |

Search found the rating rule on every seed, and on two of three put the judge behind it for
the three-star reviews, with a question of its own writing (seed 1, counts from the examples
search saw):

```text
read rating in [1, 5]
when rating >= 4: recommended calibrated 80/80 of 100
when rating < 2 or rating == 2: not_recommended calibrated 6/7 of 100
always: ask "Would this shopper recommend the clothing item they reviewed? Focus on the overall sentiment: a 3-star review that mentions fit issues, defects, or returning the item is not_recommended. ..." [not_recommended: "..."] [recommended: "..."]
```

The language-model methods are statistically tied, DSPy's typed classifier included. What
separates them is cost: search asks the judge on 14% of examples, DSPy asks its model, or the
same judge, on all of them.

### Job postings: is it a scam?

[Real or fake job postings](https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction):
title, company profile, description, requirements, salary, and flags such as whether the post
has a company logo. Sampled with both labels equally common, since fewer than 5% are scams.

| Method | Accuracy | At 80% | At 50% | Calls | Input tokens | Output tokens | Seconds | Invalid | p vs search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tfidf | 0.857 ± 0.015 | 0.921 | 0.976 | 0 | 0 | 0 | 0.00 | 0.0% | 1.1e-12 |
| **search** | **0.733 ± 0.125** | 0.757 | 0.784 | 0.72 | 471 | 0 | 0.12 | 0.0% | |
| dspy | 0.693 ± 0.035 | n/a | n/a | 1.00 | 1515 | 15 | 1.29 | 0.0% | 0.021 |
| dspy-typed | 0.692 ± 0.007 | 0.700 | 0.764 | 1.00 | 1629 | 0 | 0.39 | 0.0% | 0.013 |
| dspy-cot | 0.669 ± 0.033 | n/a | n/a | 1.00 | 1901 | 381 | 6.45 | 0.3% | 0.00097 |
| direct | 0.593 ± 0.015 | 0.640 | 0.767 | 1.00 | 551 | 0 | 0.72 | 0.0% | 9.6e-21 |
| handwritten | 0.588 ± 0.014 | 0.642 | 0.764 | 0.45 | 207 | 0 | 0.31 | 0.0% | 2.3e-21 |
| majority | 0.500 ± 0.000 | n/a | n/a | 0 | 0 | 0 | 0.00 | 0.0% | 6.2e-19 |

The graph search found on seed 2, and on seed 1 with a salary test beside it; on seed 0 this run
kept the direct question, having found nothing that passed the guard:

```text
read has_logo = has_company_logo in [0, 1]
read profile_len = length of company_profile in [0, 10000]
when has_logo == 0 and profile_len == 0: fraudulent
always: ask "Is this job posting a scam? Look for: external links used as the primary application method (phishing), requests for payment or fees, ..." [fraudulent: "..."] [genuine: "..."]
```

**At the real scam rate, that rule is much worse than it looks here.** The benchmark samples
both labels equally; in the dataset fewer than 5% of postings are scams. On all 3,576 test
postings at the real rate, the rule fires on 14% of them with precision 0.24: three in four
postings it calls scams are genuine, though it catches 71% of scams. On the balanced samples
its precision is 0.85 to 0.89 on held-out test postings, as on the examples search saw, so it
did not overfit its examples; it fits the balanced rate it was searched on. For real use,
search on a sample at the real rate, with a cost matrix that prices flagging a genuine
posting.

Search beats every language-model method here: by 4 points over DSPy's typed classifier
(p = 0.013) and 14 over the direct call, with the seed that found no rule scoring 0.59 to the
others' 0.79 and 0.82. TF-IDF beats search by 12: scam postings share vocabulary that a
word-level classifier picks up from 100 examples and a judge reading one posting at a time does
not.

### TREC: what kind of answer does the question ask for?

[TREC question classification](https://huggingface.co/datasets/SetFit/TREC-QC), six coarse
labels. The input is one short question.

| Method | Accuracy | At 80% | At 50% | Calls | Input tokens | Output tokens | Seconds | Invalid | p vs search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dspy-cot | 0.884 ± 0.041 | n/a | n/a | 1.00 | 647 | 148 | 3.81 | 0.4% | 0.00064 |
| dspy-typed | 0.869 ± 0.030 | 0.942 | 0.976 | 1.00 | 804 | 0 | 0.20 | 0.0% | 0.015 |
| **search** | **0.838 ± 0.085** | 0.903 | 0.967 | 1.00 | 354 | 0 | 0.11 | 0.0% | |
| dspy | 0.794 ± 0.033 | n/a | n/a | 1.00 | 616 | 15 | 1.61 | 0.0% | 0.013 |
| direct | 0.759 ± 0.016 | 0.849 | 0.931 | 1.00 | 148 | 0 | 0.72 | 0.0% | 1.7e-9 |
| tfidf | 0.444 ± 0.229 | 0.514 | 0.556 | 0 | 0 | 0 | 0.00 | 0.0% | 3.4e-76 |
| majority | 0.170 ± 0.046 | n/a | n/a | 0 | 0 | 0 | 0.00 | 0.0% | 1.7e-147 |

With no fields to write rules over, the best graph is the one question. No proposal beat it on
validation by enough to pass the significance guard, so search kept it and scored exactly what
the direct call scores. DSPy's chain of thought wins by 5 points: reasoning about a question
before labeling it helps on this task, and a single judge question cannot do that. DSPy's typed
classifier wins by 3 with no reasoning at all, on the very judge search used: GEPA rewrote the
instructions the judge reads, and that alone took the question from 0.759 to 0.869. Where the
text went is what mattered. Placed in the question's instructions, DSPy's winning text scores
0.743 on this test split, the same as no text; placed first in what the judge is shown, ahead of
the input, it scores 0.883. A judgment can now carry such a
[briefing](rules.md#what-a-judgment-can-be). Over an input with no fields, every proposal now
goes to the question, and the proposer is told to try a briefing first. Seed 1 wrote one, 0.877
on test; seed 2 rewrote the question and its label descriptions instead, 0.897; seed 0 kept the
plain question for the third search running, 0.74. The mean, 0.838, is above the direct call
(p = 1.7e-9) and three points under DSPy's typed classifier (p = 0.015), inside the spread
between seeds. What remains is that DSPy's GEPA finds its rewrite on every seed and search on
two of three.

### Banking77: which intent does the customer's message express?

[Banking77](https://github.com/PolyAI-LDN/task-specific-datasets), 77 intents, one short
message each.

| Method | Accuracy | At 80% | At 50% | Calls | Input tokens | Output tokens | Seconds | Invalid | p vs search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dspy-typed | 0.677 ± 0.044 | 0.740 | 0.860 | 1.00 | 8120 | 0 | 2.13 | 0.0% | 0.24 |
| dspy | 0.669 ± 0.054 | n/a | n/a | 1.00 | 1853 | 19 | 2.10 | 1.7% | 0.5 |
| **search** | **0.657 ± 0.018** | 0.722 | 0.842 | 1.00 | 5001 | 0 | 1.42 | 0.0% | |
| direct | 0.657 ± 0.018 | 0.722 | 0.842 | 1.00 | 5001 | 0 | 2.88 | 0.0% | 1 |
| tfidf | 0.103 ± 0.022 | 0.119 | 0.171 | 0 | 0 | 0 | 0.00 | 0.0% | 7.4e-121 |
| majority | 0.016 ± 0.008 | n/a | n/a | 0 | 0 | 0 | 0.00 | 0.0% | 4.3e-170 |

Again search kept the direct question, which ties DSPy and DSPy's typed classifier (p = 0.24);
the briefings it tried for 77 intents passed on no seed. DSPy's chain of thought was not run
here: 77 labels make it the slowest dataset for it, and TREC already shows what reasoning adds
on free text.

### News, the blind check: which channel did the article run in?

[Online News Popularity](https://archive.ics.uci.edu/dataset/332/online+news+popularity) as the
[multimodal AutoML benchmark](https://github.com/sxjscience/automl_multimodal_benchmark)
prepared it: a Mashable article's title and sixteen numbers about its text (length, links,
images, sentiment). Six channels. This dataset was chosen before any method ran on it, and
nothing in rulesmith was changed after seeing it.

| Method | Accuracy | At 80% | At 50% | Calls | Input tokens | Output tokens | Seconds | Invalid | p vs search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **search** | **0.283 ± 0.059** | 0.292 | 0.316 | **0.63** | 284 | 0 | 0.08 | 0.0% | |
| direct | 0.253 ± 0.019 | 0.274 | 0.304 | 1.00 | 453 | 0 | 0.85 | 0.0% | 0.028 |
| dspy-typed | 0.253 ± 0.019 | 0.271 | 0.300 | 1.00 | 747 | 0 | 0.19 | 0.0% | 0.062 |
| dspy-cot | 0.247 ± 0.030 | n/a | n/a | 1.00 | 1350 | 279 | 4.85 | 1.6% | 0.047 |
| dspy | 0.246 ± 0.013 | n/a | n/a | 1.00 | 1326 | 14 | 3.03 | 0.2% | 0.037 |
| majority | 0.237 ± 0.058 | n/a | n/a | 0 | 0 | 0 | 0.00 | 0.0% | 0.026 |
| tfidf | 0.213 ± 0.015 | 0.221 | 0.229 | 0 | 0 | 0 | 0.00 | 0.0% | 0.00042 |

Every method is near chance: a title and sixteen summary numbers say little about the channel.
Search is the best of them on average, 0.33 and 0.31 on two seeds and 0.22 on the first, where it
kept nothing; it beats the direct call, DSPy and its chain of thought at p < 0.05 and misses
against the typed classifier (p = 0.062). Against six methods at once that is weaker than it
looks: with a Bonferroni correction (p < 0.01) only its win over TF-IDF holds. What search found
on seed 1, mined rules over the numbers ahead of the judge question:

```text
when num_imgs >= 16: entertainment
when global_rate_positive_words < 0.0186335403727: world
when n_unique_tokens < 0.487351190114 and average_token_length >= 4.57673267327: world
when n_tokens_content >= 303 and num_self_hrefs >= 6: tech
```

So the direction held on data rulesmith was not tuned on: rules over the structured fields
help where the fields carry signal. The size of the effect here is small, and the absolute
accuracy is too low for anyone to use.

## Chess against Stockfish

The chess task (`examples/chess/stockfish.json`) plays graphs against Stockfish at skill
level 0 (about 1320 Elo), White on even seeds and Black on odd, scoring each game half by
the result, then by how long the graph survives (0.3) and how much material it keeps (0.2). Before training, 40 games per graph from
the example graphs:

| Graph | Won | Drew | Lost | Mean material | Mean plies |
| --- | --- | --- | --- | --- | --- |
| league-best | 1 | 0 | 39 | -10.9 | 55.7 |
| greedy | 0 | 1 | 39 | -7.1 | 65.1 |
| hunter | 0 | 0 | 40 | -7.9 | 58.6 |
| always-capture | 0 | 0 | 40 | -6.8 | 46.4 |
| searched | 0 | 0 | 40 | -9.9 | 53.5 |
| developer | 0 | 0 | 40 | -13.3 | 52.5 |
| always-quiet | 0 | 0 | 40 | -13.5 | 43.3 |
| always-check | 0 | 0 | 40 | -16.5 | 44.7 |

Search from `greedy`, with a budget of 200 scored games and eight proposals, kept a graph identical to `greedy`:
no proposal scored better on validation. On the 40 test games the two scored 0.215 and 0.218
(no wins, one draw for the second), which is the same graph playing different games. A rules
graph that picks one move type per position is far below even the weakest Stockfish, and
nothing the proposer tried changed that. A later run with five times the budget, three seeds
of 1,000 games each, is in the [chess results](chess.md#against-stockfish): search raised
validation scores there and still lost every test game.

## Caveats

**Order of events.** The benchmark was not frozen before the library was tuned against it.
The first job-postings run lost to TF-IDF and DSPy; after seeing that, rule mining gained
tests on the length of text fields and on pairs of fields, the change that finds the logo and
empty-profile rule above. The job-postings numbers here come from the code after that change.
**The news dataset was chosen afterward, before any method was run on it, and nothing was
changed on its account:** it is the check on whether those changes generalize. The proposer
changed once more after DSPy's typed classifier beat search on TREC: it may now describe the
labels in its own words and write a briefing the judge reads before the input. Search was then
rerun on every dataset with that proposer, so one method stands behind every search row; the
earlier rows are in the git history.

**Search is high-variance.** A budget of 200 scored calls is about ten proposals. On jobs, the
run before this one found the scam rule on all three seeds (0.781); this one found it on two
(0.733), and the seed that found nothing scored 0.59 to the others' 0.79 and 0.82. Three seeds
show the spread but do not pin the mean, and a single-seed comparison between methods means
little.

**Scale.** 100 training and validation examples and 300 test examples per seed, three seeds.
This is the few-shot setting rulesmith is built for; with thousands of labeled examples a
trained classifier would be the baseline to beat, and would be harder to beat than TF-IDF on
100.

**One of each model.** One judge, one DSPy student, one proposer. Ruling ran with one option
ordering per question to fit the run on one machine overnight; with more orderings it is
slower and usually more accurate, which would move direct, handwritten and search together.
A larger model would lift both sides; whether the gaps hold there is untested.

**Proposer served two ways.** Early runs got the proposer from LM Studio; after it fell behind,
the rest got the same model from OpenRouter with its reasoning capped. The results files do not
record which run used which: the harness wrote model names once, when each file was created,
so every file names the LM Studio model. Each result now records its own models.

**ReAnchor's fit did not travel.** On the seeds where it kept a weight, DSPy's ReAnchor raised
training accuracy and lowered validation: jobs seeds 0 and 1 fell from 0.72 to 0.60 and 0.76 to
0.64, Banking77 seed 2 from 0.70 to 0.68. Its fold check runs on the training examples only,
which is also true of `rulesmith calibrate`'s cutoff fit; both report the validation change so a
fit that hurt can be seen.

**Failures count as wrong.** A DSPy example whose model call failed during GEPA scored 0 rather
than ending the run, and a test answer naming no allowed label is invalid and wrong.

**Timing.** Five datasets and chess shared one Mac, two judge servers and three student servers,
so seconds per example are loaded-machine numbers.

**Run history.** The overnight run was restarted a number of times after fixes: judge and
student timeouts, a mining crash on blank fields, proposer replies cut off, and the student
server running out of GPU memory. Results already recorded were kept across restarts, each
fix is its own commit, and none changed how a finished method was scored.

## Reproducing

```sh
uv sync --all-extras
OPENROUTER_API_KEY=... \
JUDGES="http://127.0.0.1:8010 http://127.0.0.1:8012" STUDENT_PORTS="8081 8082 8083" \
PROPOSER_MODEL=openrouter/qwen/qwen3.8-27b PROPOSER_URL= \
PROPOSER_OPTIONS='{"reasoning": {"max_tokens": 2048}}' \
benchmarks/overnight.sh
```

The driver starts Ruling and the student servers if they are down, resumes from whatever is in
`benchmarks/results`, stops at `HOURS` (default 10), and writes the tables to
`runs/overnight/report.md`. One dataset at a time, with the models above:

```sh
uv run python3 -m rulesmith.bench run clothing --seeds 0 1 2 \
  --methods majority tfidf direct handwritten search dspy dspy-cot dspy-typed \
  --train 50 --validation 50 --test 300 --budget 200 \
  --judge-base-url http://127.0.0.1:8010 \
  --proposer-model openrouter/qwen/qwen3.8-27b \
  --proposer-options '{"reasoning": {"max_tokens": 2048}}' \
  --student-model openai/mlx-community/Qwen3.5-4B-4bit \
  --student-base-url http://127.0.0.1:8081/v1 \
  --student-options '{"chat_template_kwargs": {"enable_thinking": false}}' \
  --output benchmarks/results/clothing.json
uv run python3 -m rulesmith.bench report benchmarks/results/clothing.json
```
