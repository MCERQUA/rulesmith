#!/bin/zsh
# The chess searches against Stockfish behind docs/assets/fitness.svg: from the greedy graph,
# several search seeds at once, each with the same budget of games. A finished search is kept,
# so running this again picks up where an interrupted run stopped.
#
#   benchmarks/chess.sh                     # three seeds, 1000 games each
#   BUDGET=200 SEEDS="0" benchmarks/chess.sh
#   kill $(cat runs/chess/driver.pid)       # stop it, and only it
#
# Needs: Stockfish at $STOCKFISH, a proposer (a hosted model with its key in the environment,
# or a local server at $PROPOSER_URL), and Ruling at $JUDGE for any graph that asks it.
set -u
cd "${0:A:h}/.."

HOURS=${HOURS:-6}
BUDGET=${BUDGET:-1000}
SEEDS=(${=SEEDS:-0 1 2})
STOCKFISH=${STOCKFISH:-$HOME/.local/bin/stockfish}
JUDGE=${JUDGE:-http://127.0.0.1:8010}
PROPOSER_MODEL=${PROPOSER_MODEL:-openrouter/qwen/qwen3.8-27b}
PROPOSER_URL=${PROPOSER_URL-}
PROPOSER_OPTIONS=${PROPOSER_OPTIONS-'{"reasoning": {"max_tokens": 2048}}'}
TASK=examples/chess/stockfish.json
START=examples/chess/graphs/greedy.rules
LOGS=runs/chess
mkdir -p $LOGS benchmarks/results
export OPENAI_API_KEY=${OPENAI_API_KEY:-local-no-auth}

proposer=(--reflection-model $PROPOSER_MODEL --reflection-max-tokens 16384 --reflection-timeout 3600)
[[ -n $PROPOSER_URL ]] && proposer+=(--reflection-base-url $PROPOSER_URL)
[[ -n $PROPOSER_OPTIONS ]] && proposer+=(--reflection-options $PROPOSER_OPTIONS)
judge=(--backend ruling --max-workers 1 --base-url $JUDGE --engine $STOCKFISH)

deadline=$(( $(date +%s) + HOURS * 3600 ))
note() { print -r -- "$(date '+%F %T') $*" | tee -a $LOGS/status.txt; }

caffeinate -dimsu -w $$ &
print $$ >$LOGS/driver.pid
runs=()
halt() {
  trap - TERM INT
  # Each search stops after its current round and keeps its checkpoint; running this script
  # again resumes them. A second signal ends the searches at once.
  note "stopping after the current round"
  for search in $LOGS/*/search; do [[ -d $search ]] && touch $search/gepa.stop; done
  trap 'pkill -f "^[^ ]*python3 [^ ]*/rulesmith (optimize|evaluate) examples/chess/"' TERM INT
  [[ -n ${stopper:-} ]] && kill $stopper 2>/dev/null
  wait $runs
  rm -f $LOGS/driver.pid
  exit 143
}
trap halt TERM INT

search() {
  local seed=$1 run=$LOGS/chess-$1
  if [[ -f $run/report.json ]]; then
    note "seed $seed: already done"
  else
    # A search that was interrupted resumes from its checkpoint; one that left none is set aside.
    resume=()
    if [[ -f $run/search/gepa_state.bin ]]; then resume=(--resume)
    elif [[ -d $run ]]; then mv $run $run-failed-$(date +%s)
    fi
    note "seed $seed: searching, $BUDGET games"
    uv run rulesmith optimize $TASK $START $judge --max-metric-calls $BUDGET --seed $seed \
      $proposer $resume --output $run >>$LOGS/chess-$seed.log 2>&1 \
      || { note "seed $seed: search failed, see $LOGS/chess-$seed.log"; return; }
  fi
  if [[ ! -f $run/test.json ]]; then
    note "seed $seed: scoring the winner on test"
    uv run rulesmith evaluate $TASK $run/plan.json $judge --split test --output $run/test.json \
      >>$LOGS/chess-$seed.log 2>&1 || note "seed $seed: test failed, see $LOGS/chess-$seed.log"
  fi
  note "seed $seed: done"
}

# Games take a core each and share nothing, so the seeds run at once.
for seed in $SEEDS; do search $seed & runs+=($!); done
( sleep $(( deadline - $(date +%s) )); note "deadline reached"; kill -TERM $$ ) &
stopper=$!
wait $runs
kill $stopper 2>/dev/null
rm -f $LOGS/driver.pid
uv run python3 benchmarks/fitness.py runs/doom runs/chess benchmarks/results/games.json \
  docs/assets/chess-stockfish.svg --tasks chess \
  && note "all finished; chart rebuilt" || note "chart not rebuilt; see above"
