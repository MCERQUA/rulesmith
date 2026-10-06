#!/bin/zsh
# The escalation setting: for each tuned-judge seed of trec and jobs, the tuned 4B alone, Clef-flash
# alone, and the 4B escalating to Clef-flash where it is unsure, with the cutoff fitted against a
# price on the Clef call. Needs the adapters benchmarks/tuned-judge.sh trained under runs/lora, a
# Ruling checkout at $RULING_DIR, and Clef-flash's own server (the MLX port's clef_mlx.py) at $CLEF.
# Resumable; working files under runs/escalation, results in benchmarks/results/escalation.json.
#
#   benchmarks/escalation.sh
set -u
cd "$(dirname "$0")/.."
E=$PWD/runs/escalation
L=$PWD/runs/lora
BASE=mlx-community/Qwen3-4B-Instruct-2507-4bit
RULING_DIR=${RULING_DIR:-$HOME/Documents/ruling}
CLEF=${CLEF:-http://127.0.0.1:8013}
note() { echo "$(date '+%F %T') $1" >> $E/status.txt }
up() { curl -sf "$1/v1/models" > /dev/null 2>&1 }
serve() {  # adapter directory
  (cd $RULING_DIR && RULING_ROTATIONS=1 RULING_PORT=8011 RULING_MODEL=$BASE RULING_ADAPTER=$1 nohup uv run ruling serve > $E/serve-8011.log 2>&1 & echo $! > $E/serve-8011.pid)
  for _ in {1..120}; do up http://127.0.0.1:8011 && return 0; sleep 3; done
  note "judge on 8011 did not start"; return 1
}
stop() { [[ -f $E/serve-8011.pid ]] && pkill -TERM -P $(cat $E/serve-8011.pid) 2>/dev/null; kill $(cat $E/serve-8011.pid) 2>/dev/null; rm -f $E/serve-8011.pid; sleep 3 }
JUDGES=(--backend ruling --base-url http://127.0.0.1:8011 --max-workers 1 --judge large=clef@$CLEF)
for d in trec jobs; do for s in 0 1 2; do
  D=$E/$d-$s
  [[ -f $D/escalate.evaluation.json ]] && continue
  uv run python3 benchmarks/escalation.py build $d $s $D
  note "serving adapter $d-$s"
  serve $L/adapters/$d-$s || exit 1
  for g in small large; do
    [[ -f $D/$g.evaluation.json ]] || uv run rulesmith evaluate $D/task.json $D/$g.rules --split test --output $D/$g.evaluation.json ${JUDGES[@]} > $D/$g.log 2>&1
  done
  # The escalation line is placed where a Clef call, at a tenth of a right answer, still pays.
  [[ -f $D/escalate.calibrated.rules ]] || uv run rulesmith calibrate $D/task.json $D/escalate.rules --output $D/escalate.calibrated.rules ${JUDGES[@]} --judge-cost large=0.1 > $D/calibrate.log 2>&1
  uv run rulesmith evaluate $D/task.json $D/escalate.calibrated.rules --split test --output $D/escalate.evaluation.json ${JUDGES[@]} > $D/escalate.log 2>&1
  note "$d-$s done, exit $?"
  stop
done; done
uv run python3 benchmarks/escalation.py collect $E
note "escalation done"
