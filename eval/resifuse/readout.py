"""Frozen-feature readout on the ResiFuse Mouse Triplet fine-tuning protocol (linear and gradient-boosting heads).

Frames, labels and splits come from protocol_frames.py (exactly the frames ResiFuse scores).  Metric as in
resifuse.training.mabe: per-class F1 with a 0.5 decision threshold (multi-label, one decision per class), macro over the 4
classes, on the test clips; hyper-parameters chosen on the validation clips.  Readouts: one-vs-rest logistic regression
(class_weight balanced, C chosen on val) and one-vs-rest gradient boosting (class_weight balanced, n seeds).  Temporal context
(--ctx 1): the per-frame features plus their centred moving averages over +-15 and +-100 frames of the full clip.
Usage (from the repository root):
  python eval/resifuse/readout.py --frames data/resifuse/finetune_scored_frames.npz --emb embeddings/resifuse/err_36kp \
      [--emb embeddings/resifuse/handcrafted] [--feats all|h|graph] [--ctx 1] [--seeds 0 1 2 3 4] --out results/resifuse/err_36kp.json
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--frames", required=True); p.add_argument("--emb", action="append", required=True)
p.add_argument("--feats", default="all", choices=["all", "h", "graph"]); p.add_argument("--ctx", type=int, default=1)
p.add_argument("--C", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0]); p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
p.add_argument("--out", default=None); p.add_argument("--tag", default="")
args = p.parse_args()
Z = np.load(args.frames, allow_pickle=True)
clip_ids, split, clip, frame, Y, names = Z["clip_ids"], Z["split"], Z["clip"], Z["frame"], Z["labels"].astype(int), list(Z["label_names"])


def moving_mean(F, size):
    h = size // 2
    P = np.concatenate([np.repeat(F[:1], h, 0), F, np.repeat(F[-1:], h, 0)], 0).astype(np.float64)
    c = np.cumsum(np.concatenate([np.zeros((1, F.shape[1])), P], 0), 0)
    return ((c[size:] - c[:-size]) / size).astype(np.float32)


blocks = []
for prefix in args.emb:
    E = np.load(prefix + ".npy", mmap_mode="r"); J = json.load(open(prefix + ".json")); off = {k: o for k, o in zip(J["ids"], J["offsets"])}
    names_f = J["features"]
    if args.feats != "all" and names_f[0].startswith("h0_"):
        is_h = np.array([n.startswith("h") and n.split("_")[0][1:].isdigit() for n in names_f])
        cols = np.nonzero(is_h if args.feats == "h" else ~is_h)[0]
    else:
        cols = np.arange(len(names_f))
    X = np.zeros((len(frame), len(cols) * (3 if args.ctx else 1)), np.float32)
    for ci, cid in enumerate(clip_ids):
        a, b = off[str(cid)]; F = np.asarray(E[a:b][:, cols], dtype=np.float32); F[~np.isfinite(F)] = 0
        if args.ctx:
            F = np.concatenate([F, moving_mean(F, 31), moving_mean(F, 201)], 1)
        sel = clip == ci; X[sel] = F[np.minimum(frame[sel], len(F) - 1)]
    blocks.append(X)
X = np.concatenate(blocks, 1)
tr, va, te = split == 0, split == 1, split == 2
sc = StandardScaler().fit(X[tr]); X = sc.transform(X).astype(np.float32)
print(f"features {X.shape[1]} ({','.join(args.emb)} feats={args.feats} ctx={args.ctx}); scored frames train {tr.sum()} val {va.sum()} test {te.sum()}")


def macro(Yt, P):
    f = [f1_score(Yt[:, c], P[:, c], zero_division=0) for c in range(Yt.shape[1])]
    return float(np.mean(f)), f


res = {}
# logistic, C chosen on val macro F1
best = None
for C in args.C:
    P = np.stack([LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(X[tr], Y[tr, c]).predict_proba(X[va])[:, 1] >= 0.5 for c in range(4)], 1).astype(int)
    m, _ = macro(Y[va], P)
    if best is None or m > best[1]:
        best = (C, m)
P = np.stack([LogisticRegression(C=best[0], max_iter=2000, class_weight="balanced").fit(X[tr], Y[tr, c]).predict_proba(X[te])[:, 1] >= 0.5 for c in range(4)], 1).astype(int)
m, f = macro(Y[te], P); res["linear"] = {"macro_f1": m, "per_class": dict(zip(names, f)), "C": best[0], "val_macro": best[1]}
print(f"LINEAR  test macro F1 {m:.3f} | " + "  ".join(f"{n} {v:.3f}" for n, v in zip(names, f)) + f"  (C {best[0]}, val {best[1]:.3f})", flush=True)
# GBM, seeds
ms, fs = [], []
for s in args.seeds:
    t0 = time.time(); Pg = []
    for c in range(4):
        g = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, early_stopping=False, random_state=s, class_weight="balanced")
        g.fit(X[tr], Y[tr, c]); Pg.append(g.predict_proba(X[te])[:, 1] >= 0.5)
    m, f = macro(Y[te], np.stack(Pg, 1).astype(int)); ms.append(m); fs.append(f)
    print(f"  GBM seed {s}: macro {m:.3f} | " + " ".join(f"{v:.3f}" for v in f) + f" ({time.time() - t0:.0f}s)", flush=True)
fs = np.array(fs); res["gbm"] = {"macro_f1": float(np.mean(ms)), "macro_std": float(np.std(ms)), "per_class": dict(zip(names, fs.mean(0).tolist()))}
print(f"GBM     test macro F1 {np.mean(ms):.3f} +- {np.std(ms):.3f} | " + "  ".join(f"{n} {v:.3f}" for n, v in zip(names, fs.mean(0))) + f"   [{args.tag}]")
print("reference (ResiFuse paper, Mouse Triplet, BiGRU head, 5 seeds): pose coords .583, BAMS .599, hBehaveMAE .621, ResiFuse pose-only .654, hand-crafted .684, VideoPrism .722, ResiFuse .784")
if args.out:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"args": vars(args), "n_features": int(X.shape[1]), "results": res}, open(args.out, "w"), indent=1)
