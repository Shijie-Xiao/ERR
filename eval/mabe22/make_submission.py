"""Turn a frame-embedding file (embed.py output) into an official MABe22 round-1 submission for the organisers' evaluator:
{"frame_number_map": {seq_id: (start, end)}, "embeddings": float32 (sum of clip lengths, <= 128)}.
Standardise on the SubmissionTrain frames, PCA to --dims (fit on SubmissionTrain), cast to float32.
Usage: python eval/mabe22/make_submission.py --emb embeddings/mabe22/err_36kp --labels data/mabe22/mouse_triplets_test_labels.npy \
           --split data/mabe22/jax_split.json --out results/mabe22/submission_err_36kp.npy [--dims 128]
Then (see README): cd <MABe2022 evaluator clone> && python round1_evaluator.py --task mouse --submission <out> --labels <labels.npy>
"""
import argparse
import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--emb", action="append", required=True); p.add_argument("--labels", required=True); p.add_argument("--split", required=True)
p.add_argument("--out", required=True); p.add_argument("--dims", type=int, default=128)
args = p.parse_args()
fm = np.load(args.labels, allow_pickle=True).item()["frame_number_map"]
split = json.load(open(args.split))
blocks = []
for prefix in args.emb:
    e = np.load(prefix + ".npy", mmap_mode="r"); j = json.load(open(prefix + ".json"))
    assert all(tuple(fm[k]) == tuple(o) for k, o in zip(j["ids"], j["offsets"]))
    blocks.append(e)
X = np.concatenate([np.asarray(e, dtype=np.float32) for e in blocks], 1)
X[~np.isfinite(X)] = 0
tr = np.concatenate([np.arange(*fm[k]) for k in split["SubmissionTrain"]])
sc = StandardScaler().fit(X[tr[::4]]); X = sc.transform(X).astype(np.float32)
if X.shape[1] > args.dims:
    pca = PCA(args.dims, random_state=0).fit(X[tr[::8]]); X = pca.transform(X).astype(np.float32)
    print(f"PCA {len(sc.mean_)} -> {args.dims} dims, explained variance {pca.explained_variance_ratio_.sum():.3f}")
sub = {"frame_number_map": {k: tuple(int(x) for x in v) for k, v in fm.items()}, "embeddings": np.ascontiguousarray(X, dtype=np.float32)}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.save(args.out, sub, allow_pickle=True)
print(f"wrote {args.out}: embeddings {X.shape} {X.dtype}, {len(fm)} clips")
