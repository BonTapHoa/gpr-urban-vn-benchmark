#!/bin/bash
# Control: pre-train YOLO11-m on the simulated images with SHUFFLED labels (same recipe, no augmentation), then fine-tune on Mojahid and Rebar. log /content/ctrl.log
exec > /content/ctrl.log 2>&1
set -x
mkdir -p /content/data /content/logs /content/trees /content/runs /content/real_gpr_mojahid /content/rebar_flat
cd /content/up
tar xzf bundle.tgz -C /content
cat raw_0* > raw.tgz && tar xzf raw.tgz -C /content/real_gpr_mojahid
cat img_* > images.tgz && tar xzf images.tgz -C /content/data
tar xzf 03_labels.tgz -C /content/data
cat $(ls a_* | sort) $(ls b_* | sort) > rebar_named.tgz && tar xzf rebar_named.tgz -C /content/rebar_flat
echo "images $(ls /content/data/images | wc -l) rebar $(ls /content/rebar_flat/images | wc -l)"
cp /content/up/train_physaug.py /content/physaug/train_physaug.py
cd /content/gpr1
python /content/up/build_rebar_trees_vm.py /content/up/assign.json | tail -1
python scripts/build_real_trees.py /content/real | tail -1
python /content/up/make_shuffled_labels.py /content/data/labels_v31/uncorrected /content/data/labels_shuf 777
python /content/physaug/build_tree.py configs/splits/gpr_urban_vn_split_10k_base.json /content/data/images /content/data/labels_shuf /content/trees/shuf
echo "TRAIN_START $(date +%T)"
python /content/physaug/train_physaug.py --data /content/trees/shuf/data.yaml --name shuf_yolo11m_s0 --noaug --save_period 5 --workers 12 > /content/logs/train_shuf.log 2>&1
echo "TRAIN_END $(date +%T)"
ls /content/runs/shuf_yolo11m_s0/weights
sed 's#/content/runs_ft#/content/runs_ft_rebar#g' /content/gpr1/scripts/finetune_real.py > /content/gpr1/scripts/finetune_real_rebar.py
for tag in epoch25 last; do
  W=/content/runs/shuf_yolo11m_s0/weights/$tag.pt
  [ -f $W ] || { echo "MISSING $W"; continue; }
  cd /content
  python /content/gpr1/scripts/finetune_real.py /content/real $W /content/ft_shuf_${tag}_moj.jsonl --inits sim --seeds 0,1,2 --fracs 10,25,100 > /content/ft_shuf_${tag}_moj.log 2>&1 &
  python /content/gpr1/scripts/finetune_real_rebar.py /content/rebar $W /content/ft_shuf_${tag}_rebar.jsonl --inits sim --seeds 0,1,2 --fracs 10,25,100 > /content/ft_shuf_${tag}_rebar.log 2>&1 &
  wait
  echo "FT_DONE $tag $(date +%T)"
done
echo "CHAIN DONE $(date +%T)"
