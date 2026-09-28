#!/bin/zsh
# Every public benchmark, run locally with a hard time limit. Results resume from their files in
# benchmarks/results, so running this again picks up where an interrupted run stopped.
#
#   benchmarks/overnight.sh                # 10 hours
#   HOURS=4 benchmarks/overnight.sh
#
# Needs: LM Studio serving $PROPOSER, a checkout of Ruling at $RULING_DIR, and Stockfish at
# $STOCKFISH for the chess run. Logs and a status file go to runs/overnight.
set -u
cd "${0:A:h}/.."
ROOT=$PWD

HOURS=${HOURS:-10}
RULING_DIR=${RULING_DIR:-../ruling}
STOCKFISH=${STOCKFISH:-$HOME/.local/bin/stockfish}
PROPOSER=${PROPOSER:-qwen3.8-27b-mlx}
PROPOSER_URL=${PROPOSER_URL:-http://127.0.0.1:1234/v1}
STUDENT=${STUDENT:-mlx-community/Qwen3.5-4B-4bit}
STUDENT_PORT=${STUDENT_PORT:-8081}
BUDGET=${BUDGET:-200}
LOGS=runs/overnight
mkdir -p $LOGS benchmarks/results
export OPENAI_API_KEY=${OPENAI_API_KEY:-local-no-auth}

deadline=$(( $(date +%s) + HOURS * 3600 ))
note() { print -r -- "$(date '+%F %T') $*" | tee -a $LOGS/status.txt; }
up() { curl -s -m 5 "$1" >/dev/null; }
ready() {
  # A model server takes a minute or two to load; give up after five.
  for _ in {1..150}; do up "$1" && return 0; sleep 2; done
  return 1
}

# The Mac must not sleep, or every model and game stops at the first idle timeout.
caffeinate -dimsu -w $$ &

if ! up http://127.0.0.1:8010/v1/models; then
  note "starting Ruling (one option ordering per question)"
  (cd $RULING_DIR && RULING_ROTATIONS=1 nohup uv run ruling serve >$ROOT/$LOGS/ruling.log 2>&1 &)
fi
if ! up http://127.0.0.1:$STUDENT_PORT/v1/models; then
  note "starting the DSPy student $STUDENT"
  (cd $RULING_DIR && nohup uv run mlx_lm.server --model $STUDENT --port $STUDENT_PORT \
    >$ROOT/$LOGS/student.log 2>&1 &)
fi
ready http://127.0.0.1:8010/v1/models || { note "Ruling did not start; see $LOGS/ruling.log"; exit 1; }
ready http://127.0.0.1:$STUDENT_PORT/v1/models || { note "student did not start"; exit 1; }
curl -s -m 10 $PROPOSER_URL/models | grep -q "\"$PROPOSER\"" || {
  note "LM Studio is not serving $PROPOSER at $PROPOSER_URL; load it and run again"
  exit 1
}

bench() {
  local dataset=$1; shift
  # A crash loses only the step it was on, so try again while there is time left.
  for attempt in 1 2 3; do
    local left=$(( (deadline - $(date +%s)) / 3600.0 ))
    (( left > 0 )) || return
    note "$dataset: attempt $attempt, ${left}h left"
    uv run python3 -m rulesmith.bench run $dataset --seeds 0 1 2 --methods "$@" \
      --train 50 --validation 50 --test 300 --budget $BUDGET --hours $left \
      --proposer-model openai/$PROPOSER --proposer-base-url $PROPOSER_URL \
      --student-model openai/$STUDENT --student-base-url http://127.0.0.1:$STUDENT_PORT/v1 \
      --student-options '{"chat_template_kwargs": {"enable_thinking": false}}' \
      --note "Ruling with RULING_ROTATIONS=1 (one option ordering per question), $STUDENT" \
      --output benchmarks/results/$dataset.json >>$LOGS/$dataset.log 2>&1 \
      && { note "$dataset: done"; return; }
    note "$dataset: exited with $?, see $LOGS/$dataset.log"
  done
}

chess() {
  note "chess: searching against Stockfish from greedy"
  uv run rulesmith optimize examples/chess/stockfish.json examples/chess/graphs/greedy.rules \
    --engine $STOCKFISH --backend ruling --max-workers 1 \
    --reflection-model openai/$PROPOSER --reflection-base-url $PROPOSER_URL \
    --reflection-max-tokens 16384 --reflection-timeout 3600 --max-metric-calls $BUDGET \
    --output $LOGS/chess-search >>$LOGS/chess.log 2>&1 || { note "chess: search failed"; return; }
  for graph in $LOGS/chess-search/plan.json examples/chess/graphs/greedy.rules; do
    uv run rulesmith evaluate examples/chess/stockfish.json $graph --engine $STOCKFISH \
      --backend ruling --split test --output $LOGS/chess-test-${graph:h:t}-${graph:t:r}.json \
      >>$LOGS/chess.log 2>&1
  done
  note "chess: done"
}

# The cheap methods first, then the ones that need the proposer. Banking77 skips DSPy's
# chain of thought: seventy-seven labels make it the slowest place to run it and the least
# telling.
runs=()
bench trec majority tfidf direct search dspy dspy-cot & runs+=($!)
bench clothing majority tfidf direct handwritten search dspy dspy-cot & runs+=($!)
bench jobs majority tfidf direct handwritten search dspy dspy-cot & runs+=($!)
bench banking77 majority tfidf direct search dspy & runs+=($!)
chess & runs+=($!)

# The hard stop: whatever is still running at the deadline ends, and what finished is kept.
( sleep $(( deadline - $(date +%s) )); note "deadline reached, stopping";
  pkill -f "rulesmith.bench run"; pkill -f "rulesmith optimize"; pkill -f "rulesmith evaluate" ) &
stopper=$!

# Waiting on these alone: caffeinate runs until this script exits, so a bare wait never returns.
wait $runs
kill $stopper 2>/dev/null
for dataset in trec clothing jobs banking77; do
  [[ -f benchmarks/results/$dataset.json ]] && uv run python3 -m rulesmith.bench report \
    benchmarks/results/$dataset.json >>$LOGS/report.md && print >>$LOGS/report.md
done
note "all finished; tables in $LOGS/report.md"
