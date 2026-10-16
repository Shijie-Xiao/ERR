#!/bin/bash
# MABe22 13-task linear readout for one or more embeddings: full features, PCA-128 (the challenge constraint),
# node memories only (h), graph features only, and + hand-crafted geometry.
# Usage (from the repository root):  bash eval/mabe22/run_probes.sh err_36kp [err_12kp ...]
# Expects embeddings/mabe22/<name>.npy|json (eval/mabe22/embed.py) and embeddings/mabe22/handcrafted.* (eval/mabe22/handcrafted.py).
PY=${PY:-python}
L=data/mabe22/mouse_triplets_test_labels.npy; S=data/mabe22/jax_split.json; E=embeddings/mabe22; O=results/mabe22
mkdir -p $O
run() { tag=$1; shift; echo "=== $tag $(date)"; $PY eval/mabe22/probe.py --labels $L --split $S --stride 4 --tag $tag --out $O/$tag.json "$@" 2>&1 | grep -v -E "Warning|linalg.solve|ill-conditioned"; }
for name in "$@"; do
  [ -f $E/$name.npy ] || { echo "skip $name (no embedding)"; continue; }
  run ${name}_all --emb $E/$name
  run ${name}_pca128 --emb $E/$name --pca 128
  run ${name}_h --emb $E/$name --feats h
  run ${name}_graph --emb $E/$name --feats graph
  [ -f $E/handcrafted.npy ] && run ${name}_plus_hand --emb $E/$name --emb $E/handcrafted
done
[ -f $E/handcrafted.npy ] && [ ! -f $O/handcrafted.json ] && run handcrafted --emb $E/handcrafted
echo "=== probes done $(date)"
