#!/bin/zsh
# Every public benchmark, run locally with a hard time limit. Results resume from their files in
# benchmarks/results, so running this again picks up where an interrupted run stopped.
#
#   benchmarks/overnight.sh                # 10 hours, proposer served by LM Studio
#   HOURS=4 benchmarks/overnight.sh
#   PROPOSER_MODEL=openrouter/qwen/qwen3.8-27b PROPOSER_URL= benchmarks/overnight.sh
#   JUDGES="http://127.0.0.1:8010 http://127.0.0.1:8012" benchmarks/overnight.sh
#   kill $(cat runs/overnight/driver.pid)   # stop it, and only it
#
# One Ruling answers one request at a time, so a second on another port shares the datasets.
#
# Needs: the proposer (LM Studio at $PROPOSER_URL, or a hosted model with its key in the
# environment when PROPOSER_URL is empty), a checkout of Ruling at $RULING_DIR, and Stockfish at
# $STOCKFISH for the chess run. Logs and a status file go to runs/overnight.
set -u
cd "${0:A:h}/.."
ROOT=$PWD

HOURS=${HOURS:-10}
RULING_DIR=${RULING_DIR:-../ruling}
STOCKFISH=${STOCKFISH:-$HOME/.local/bin/stockfish}
JUDGES=(${=JUDGES:-http://127.0.0.1:8010})
PROPOSER_MODEL=${PROPOSER_MODEL:-openai/qwen3.8-27b-mlx}
PROPOSER_URL=${PROPOSER_URL-http://127.0.0.1:1234/v1}
proposer=(--proposer-model $PROPOSER_MODEL)
reflector=(--reflection-model $PROPOSER_MODEL)
if [[ -n $PROPOSER_URL ]]; then
  proposer+=(--proposer-base-url $PROPOSER_URL)
  reflector+=(--reflection-base-url $PROPOSER_URL)
fi
# Extra request JSON for the proposer; a hosted thinking model needs its reasoning capped, or it
# reasons past any reply limit: '{"reasoning": {"max_tokens": 2048}}'.
if [[ -n ${PROPOSER_OPTIONS:-} ]]; then
  proposer+=(--proposer-options $PROPOSER_OPTIONS)
  reflector+=(--reflection-options $PROPOSER_OPTIONS)
fi
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

# Stopping the run is 'kill $(cat runs/overnight/driver.pid)': the driver then ends its own work,
# found by the command lines only it starts, and nothing else on the machine.
print $$ >$LOGS/driver.pid
halt() {
  trap - TERM INT
  note "stopping"
  # The retry loops go first, or they would take their runs ending for a crash and start again.
  (( ${#runs} )) && kill $runs 2>/dev/null
  # Anchored to how the runs start, so a shell that merely mentions these words is left alone.
  pkill -f '^[^ ]*python3 -m rulesmith[.]bench run'
  pkill -f '^[^ ]*python3 [^ ]*/rulesmith (optimize|evaluate) '
  [[ -n ${stopper:-} ]] && kill $stopper 2>/dev/null
  rm -f $LOGS/driver.pid
  exit 143
}
runs=()
trap halt TERM INT

for judge in $JUDGES; do
  if ! up $judge/v1/models; then
    note "starting Ruling at $judge (one option ordering per question)"
    (cd $RULING_DIR && RULING_ROTATIONS=1 RULING_PORT=${judge##*:} nohup uv run ruling serve \
      >$ROOT/$LOGS/ruling-${judge##*:}.log 2>&1 &)
  fi
done
if ! up http://127.0.0.1:$STUDENT_PORT/v1/models; then
  note "starting the DSPy student $STUDENT"
  (cd $RULING_DIR && nohup uv run mlx_lm.server --model $STUDENT --port $STUDENT_PORT \
    >$ROOT/$LOGS/student.log 2>&1 &)
fi
for judge in $JUDGES; do
  ready $judge/v1/models || { note "Ruling did not start at $judge; see $LOGS"; exit 1; }
done
ready http://127.0.0.1:$STUDENT_PORT/v1/models || { note "student did not start"; exit 1; }
if [[ -n $PROPOSER_URL ]] && ! curl -s -m 10 $PROPOSER_URL/models | grep -q "\"${PROPOSER_MODEL#openai/}\""; then
  note "nothing serves $PROPOSER_MODEL at $PROPOSER_URL; load it and run again"
  exit 1
fi

bench() {
  local dataset=$1 judge=$2; shift 2
  # A crash loses only the step it was on, so try again while there is time left.
  for attempt in 1 2 3; do
    local left=$(( (deadline - $(date +%s)) / 3600.0 ))
    (( left > 0 )) || return
    note "$dataset: attempt $attempt, ${left}h left"
    uv run python3 -m rulesmith.bench run $dataset --seeds 0 1 2 --methods "$@" \
      --train 50 --validation 50 --test 300 --budget $BUDGET --hours $left --judge-base-url $judge \
      $proposer \
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
    $reflector \
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
# Banking77's seventy-seven options make it the heaviest on a judge, so it and news take the
# second judge when there is one.
first=$JUDGES[1] second=${JUDGES[2]:-$JUDGES[1]}
bench trec $first majority tfidf direct search dspy dspy-cot & runs+=($!)
bench clothing $first majority tfidf direct handwritten search dspy dspy-cot & runs+=($!)
bench jobs $first majority tfidf direct handwritten search dspy dspy-cot & runs+=($!)
bench banking77 $second majority tfidf direct search dspy & runs+=($!)
# Chosen before any result and used for nothing while building: the check on the rest.
bench news $second majority tfidf direct search dspy dspy-cot & runs+=($!)
chess & runs+=($!)

# The hard stop: whatever is still running at the deadline ends, and what finished is kept.
( sleep $(( deadline - $(date +%s) )); note "deadline reached"; kill -TERM $$ ) &
stopper=$!

# Waiting on these alone: caffeinate runs until this script exits, so a bare wait never returns.
wait $runs
kill $stopper 2>/dev/null
for dataset in trec clothing jobs banking77 news; do
  [[ -f benchmarks/results/$dataset.json ]] && uv run python3 -m rulesmith.bench report \
    benchmarks/results/$dataset.json >>$LOGS/report.md && print >>$LOGS/report.md
done
rm -f $LOGS/driver.pid
note "all finished; tables in $LOGS/report.md"
