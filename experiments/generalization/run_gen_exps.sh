#!/usr/bin/env bash
# usage: run_gen_exps.sh STREAM_NAME exp1 exp2 ...   (runs inside /content/gpr1)
cd /content/gpr1; ST=$1; shift
for E in "$@"; do
  N=$(python - <<PY
import json;print(len(json.load(open("generalization_experiments.json"))["$E"]["train"]))
PY
)
  case $E in scale_*) EP=$(python -c "print(min(300,max(60,round(60*3500/$N))))");; *) EP=60;; esac
  echo "[$ST] $E n=$N epochs=$EP start $(date +%T)" >> /content/gen_status.log
  python scripts/train_detector.py --model yolo11m.pt --data /content/trees/$E/data.yaml --name gen_$E --epochs $EP --batch 32 --seed 0 --out /content/runs > /content/logs/train_$E.log 2>&1
  W=/content/runs/gen_$E/weights/best.pt
  python scripts/eval_detectors_test.py $W gen_$E /content /content/gen_out --device 0 --batch 64 --boot 1000 > /content/logs/eval_$E.log 2>&1
  echo "[$ST] $E done $(date +%T)" >> /content/gen_status.log
done
echo "[$ST] STREAMDONE" >> /content/gen_status.log
