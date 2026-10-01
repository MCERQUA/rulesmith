#!/bin/zsh
# The tuned-judge setting, end to end and resumable: an adapter per dataset and seed trained by
# Ruling on the records benchmarks/tuned_judge.py build wrote, then direct, search and dspy-typed
# on the stock Instruct base and on each adapter. Banking77 has no adapter: Ruling trains only
# questions that fit its answer alphabet, and a 77-way choice does not. Needs a Ruling checkout
# at $RULING_DIR and the proposer on OpenRouter.
#
#   benchmarks/tuned-judge.sh
set -u
cd "$(dirname "$0")/.."
L=$PWD/runs/lora
BASE=${BASE:-mlx-community/Qwen3-4B-Instruct-2507-4bit}
RULING_DIR=${RULING_DIR:-$HOME/Documents/ruling}
PROPOSER=${PROPOSER:-openrouter/qwen/qwen3.8-27b}
DATASETS=(jobs clothing trec banking77 news)
mkdir -p $L/adapters benchmarks/results/instruct benchmarks/results/tuned
note() { echo "$(date '+%F %T') $1" >> $L/status.txt }
up() { curl -sf "$1/v1/models" > /dev/null 2>&1 }
serve() {  # port, adapter directory or empty
  local env=(RULING_ROTATIONS=1 RULING_PORT=$1 RULING_MODEL=$BASE)
  [[ -n $2 ]] && env+=(RULING_ADAPTER=$2)
  (cd $RULING_DIR && env ${env[@]} nohup uv run ruling serve > $L/serve-$1.log 2>&1 & echo $! > $L/serve-$1.pid)
  for _ in {1..120}; do up http://127.0.0.1:$1 && return 0; sleep 3; done
  note "judge on $1 did not start"; return 1
}
stop() { [[ -f $L/serve-$1.pid ]] && pkill -TERM -P $(cat $L/serve-$1.pid) 2>/dev/null; kill $(cat $L/serve-$1.pid) 2>/dev/null; rm -f $L/serve-$1.pid; sleep 3 }
bench() {  # dataset, seeds, port, output, note
  uv run python3 -m rulesmith.bench run $1 --seeds ${=2} --methods direct search dspy-typed --train 50 --validation 50 --test 300 --budget 200 \
    --judge-base-url http://127.0.0.1:$3 --proposer-model $PROPOSER --proposer-options '{"reasoning": {"max_tokens": 2048}}' \
    --note "$5" --output $4 >> $L/bench-$1.log 2>&1
}
[[ -d $L/data ]] || uv run python3 benchmarks/tuned_judge.py build
for d in $DATASETS; do for s in 0 1 2; do
  [[ $d == banking77 ]] && continue
  [[ -f $L/adapters/$d-$s/adapters.safetensors ]] && grep -q '"step": 200' $L/adapters/$d-$s/training.json 2>/dev/null && continue
  note "training $d-$s"
  (cd $RULING_DIR && uv run ruling train --data $L/data/$d-$s-train.jsonl --holdout $L/data/$d-$s-holdout.jsonl --model $BASE \
    --steps 200 --eval-every 20 --valid-fraction 0.2 --output $L/adapters/$d-$s > $L/train-$d-$s.log 2>&1)
  note "training $d-$s done, exit $?"
done; done
note "serving stock $BASE on 8011"
serve 8011 "" || exit 1
for d in $DATASETS; do note "stock $d"; bench $d "0 1 2" 8011 benchmarks/results/instruct/$d.json "Ruling with RULING_ROTATIONS=1, $BASE, no adapter"; note "stock $d done, exit $?"; done
stop 8011
for d in $DATASETS; do for s in 0 1 2; do
  [[ -f $L/adapters/$d-$s/adapters.safetensors ]] || { note "no adapter for $d-$s, skipped"; continue }
  note "serving adapter $d-$s on 8012"
  serve 8012 $L/adapters/$d-$s || exit 1
  bench $d "$s" 8012 benchmarks/results/tuned/$d.json "Ruling with RULING_ROTATIONS=1, $BASE with a LoRA adapter trained on this seed's 100 search examples"
  note "tuned $d-$s done, exit $?"
  stop 8012
done; done
note "tuned-judge setting done"
