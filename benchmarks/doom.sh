#!/bin/zsh
# The Doom searches behind docs/assets/fitness.svg, run with one budget, one proposer and one
# judge each, so the chart compares like with like. Each search that finishes is kept, so
# running this again picks up where an interrupted run stopped.
#
#   benchmarks/doom.sh                          # basic and e1m1, three search seeds each
#   BUDGET=100 TASKS=basic benchmarks/doom.sh
#   PROPOSER_MODEL=openrouter/qwen/qwen3.8-27b PROPOSER_OPTIONS='{"reasoning": {"max_tokens": 2048}}' \
#     benchmarks/doom.sh
#   kill $(cat runs/doom/driver.pid)            # stop it, and only it
#
# Needs: a proposer (a hosted model with its key in the environment, or a local server at
# $PROPOSER_URL), and Ruling at $JUDGE for graphs that ask it, started from $RULING_DIR if down.
# When every search has run, the chart and the results file are rebuilt from the reports.
set -u
cd "${0:A:h}/.."
ROOT=$PWD

HOURS=${HOURS:-10}
BUDGET=${BUDGET:-200}
SEEDS=(${=SEEDS:-0 1 2})
TASKS=(${=TASKS:-basic e1m1})
JUDGE=${JUDGE:-http://127.0.0.1:8010}
RULING_DIR=${RULING_DIR:-../ruling}
PROPOSER_MODEL=${PROPOSER_MODEL:-openrouter/qwen/qwen3.8-27b}
PROPOSER_URL=${PROPOSER_URL-}
PROPOSER_OPTIONS=${PROPOSER_OPTIONS-'{"reasoning": {"max_tokens": 2048}}'}
# A search starts from a graph that already plays a little: basic from the task's own
# baseline, which only strafes, and E1M1 from the hand-written graph that reads the route and
# asks the judge only about a blocked way ahead.
typeset -A STARTS
STARTS=(basic "" e1m1 examples/doom/graphs/e1m1-target.json)
LOGS=runs/doom
mkdir -p $LOGS benchmarks/results
export OPENAI_API_KEY=${OPENAI_API_KEY:-local-no-auth}

proposer=(--reflection-model $PROPOSER_MODEL --reflection-max-tokens 16384 --reflection-timeout 3600)
[[ -n $PROPOSER_URL ]] && proposer+=(--reflection-base-url $PROPOSER_URL)
[[ -n $PROPOSER_OPTIONS ]] && proposer+=(--reflection-options $PROPOSER_OPTIONS)

deadline=$(( $(date +%s) + HOURS * 3600 ))
note() { print -r -- "$(date '+%F %T') $*" | tee -a $LOGS/status.txt; }
up() { curl -s -m 5 "$1" >/dev/null; }

caffeinate -dimsu -w $$ &
print $$ >$LOGS/driver.pid
halt() {
  trap - TERM INT
  note "stopping"
  pkill -f '^[^ ]*python3 [^ ]*/rulesmith (optimize|evaluate) examples/doom/'
  [[ -n ${stopper:-} ]] && kill $stopper 2>/dev/null
  rm -f $LOGS/driver.pid
  exit 143
}
trap halt TERM INT

if ! up $JUDGE/v1/models; then
  note "starting Ruling at $JUDGE (one option ordering per question)"
  (cd $RULING_DIR && RULING_ROTATIONS=1 RULING_PORT=${JUDGE##*:} nohup uv run ruling serve \
    >$ROOT/$LOGS/ruling.log 2>&1 &)
  for _ in {1..150}; do up $JUDGE/v1/models && break; sleep 2; done
  up $JUDGE/v1/models || { note "Ruling did not start at $JUDGE"; exit 1; }
fi

( sleep $(( deadline - $(date +%s) )); note "deadline reached"; kill -TERM $$ ) &
stopper=$!

for task in $TASKS; do
  for seed in $SEEDS; do
    run=$LOGS/$task-$seed
    (( $(date +%s) < deadline )) || break
    if [[ -f $run/report.json ]]; then
      note "$task seed $seed: already done"
    else
      # optimize refuses an output that exists, so a search that crashed is set aside, not lost.
      [[ -d $run ]] && mv $run $run-failed-$(date +%s)
      note "$task seed $seed: searching, $BUDGET metric calls"
      uv run rulesmith optimize examples/doom/$task.json ${STARTS[$task]} \
        --backend ruling --max-workers 1 --base-url $JUDGE \
        --max-nodes 40 --max-depth 16 --max-metric-calls $BUDGET --seed $seed \
        $proposer --output $run >>$LOGS/$task-$seed.log 2>&1 \
        || { note "$task seed $seed: search failed, see $LOGS/$task-$seed.log"; continue; }
    fi
    if [[ ! -f $run/test.json ]]; then
      note "$task seed $seed: scoring the winner on test"
      uv run rulesmith evaluate examples/doom/$task.json $run/plan.json --split test \
        --backend ruling --max-workers 1 --base-url $JUDGE --max-nodes 40 --max-depth 16 \
        --output $run/test.json >>$LOGS/$task-$seed.log 2>&1 \
        || note "$task seed $seed: test failed, see $LOGS/$task-$seed.log"
    fi
  done
done
kill $stopper 2>/dev/null
rm -f $LOGS/driver.pid
uv run python3 benchmarks/fitness.py $LOGS benchmarks/results/doom.json docs/assets/fitness.svg \
  && note "all finished; chart rebuilt" || note "chart not rebuilt; see above"
