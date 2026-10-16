"""CalMS21 task-1 readout on frozen ERR features (and the hand-crafted geometry block), official protocol:
train on the 70 training videos, test on the 19 test videos, frame-level 4-way prediction (argmax), F1 per class and
class-averaged F1 over attack / investigation / mount (the 'other' class is not scored) plus mean average precision.
Readouts: linear (multinomial logistic regression, C chosen on 1/5 of the training videos) and non-linear (histogram
gradient boosting).  Temporal context (--ctx 1): the frame features are concatenated with their centred moving averages over
+-15 and +-100 frames (1 s and 6.7 s), the context length the CalMS21 1D-conv baseline uses.
Usage (from the repository root):
  python eval/calms21/probe.py --emb embeddings/calms21/err_zeroshot [--feats all|err|geo|h|graph] [--seed 0] [--out results/calms21/err_zeroshot_all.json]
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--emb", required=True); p.add_argument("--feats", default="all", choices=["err", "geo", "all", "h", "graph"],
               help="err = all model features, geo = geometry block only, h = node memories, graph = pcontact/intent/surprise/speed")
p.add_argument("--ctx", type=int, default=1); p.add_argument("--stride", type=int, default=2)
p.add_argument("--models", nargs="+", default=["linear", "gbm"]); p.add_argument("--C", type=float, nargs="+", default=[0.01, 0.1, 1.0])
p.add_argument("--out", default=None); p.add_argument("--tag", default=""); p.add_argument("--seed", type=int, default=0, help="GBM random_state and validation-video split seed")
args = p.parse_args()
CLASSES = ["attack", "investigation", "mount"]


def moving_mean(F, size):
    h = size // 2
    P = np.concatenate([np.repeat(F[:1], h, 0), F, np.repeat(F[-1:], h, 0)], 0).astype(np.float64)
    c = np.cumsum(np.concatenate([np.zeros((1, F.shape[1])), P], 0), 0)
    return ((c[size:] - c[:-size]) / size).astype(np.float32)


def load(split):
    j = json.load(open(f"{args.emb}_{split}.json")); X = np.load(f"{args.emb}_{split}.npy", mmap_mode="r"); y = np.load(f"{args.emb}_{split}_labels.npy")
    names = j["features"]; is_geo = np.array([not (n.startswith("h") or n.startswith("pcontact") or n.startswith("intent") or n.startswith("surprise") or n.startswith("speed_hat")) for n in names])
    is_h = np.array([n.startswith("h") and n[1].isdigit() for n in names])
    keep = {"all": np.ones(len(names), bool), "err": ~is_geo, "geo": is_geo, "h": is_h, "graph": ~is_geo & ~is_h}[args.feats]
    cols = np.nonzero(keep)[0]
    Xs, ys, vid = [], [], []
    for i, (s, e) in enumerate(j["offsets"]):
        F = np.asarray(X[s:e][:, cols], dtype=np.float32); F[~np.isfinite(F)] = 0
        if args.ctx:
            F = np.concatenate([F, moving_mean(F, 31), moving_mean(F, 201)], 1)
        Xs.append(F); ys.append(y[s:e]); vid.append(np.full(e - s, i))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(vid), [names[c] for c in cols]


t0 = time.time()
Xtr, ytr, vtr, names = load("train"); Xte, yte, vte, _ = load("test")
print(f"train {Xtr.shape} test {Xte.shape} features={args.feats} ctx={args.ctx} ({time.time() - t0:.0f}s)")
sel = np.arange(0, len(ytr), args.stride)
sc = StandardScaler().fit(Xtr[sel]); Xtr_s = sc.transform(Xtr[sel]).astype(np.float32); ytr_s = ytr[sel]; vtr_s = vtr[sel]
Xte_s = sc.transform(Xte).astype(np.float32)
rng = np.random.default_rng(args.seed); vids = np.unique(vtr_s); val_v = rng.choice(vids, size=max(1, len(vids) // 5), replace=False)
m_val = np.isin(vtr_s, val_v); m_fit = ~m_val


def score(y, pred, proba):
    f1 = {c: float(f1_score(y == k, pred == k)) for k, c in enumerate(CLASSES)}
    ap = {c: float(average_precision_score(y == k, proba[:, k])) for k, c in enumerate(CLASSES)}
    return {"f1": f1, "ap": ap, "avg_f1": float(np.mean(list(f1.values()))), "map": float(np.mean(list(ap.values())))}


results = {}
if "linear" in args.models:
    best = None
    for C in args.C:
        mdl = LogisticRegression(C=C, max_iter=500, class_weight="balanced").fit(Xtr_s[m_fit], ytr_s[m_fit])
        pr = mdl.predict_proba(Xtr_s[m_val]); s = score(ytr_s[m_val], pr.argmax(1), pr)
        print(f"  linear C={C}: val avg F1 {s['avg_f1']:.3f} map {s['map']:.3f}", flush=True)
        if best is None or s["avg_f1"] > best[1]:
            best = (C, s["avg_f1"])
    mdl = LogisticRegression(C=best[0], max_iter=500, class_weight="balanced").fit(Xtr_s, ytr_s)
    pr = mdl.predict_proba(Xte_s); results["linear"] = score(yte, pr.argmax(1), pr); results["linear"]["C"] = best[0]
    r = results["linear"]; print(f"LINEAR  test: avg F1 {r['avg_f1']:.3f}  MAP {r['map']:.3f}  | " + "  ".join(f"{c} F1 {r['f1'][c]:.3f} AP {r['ap'][c]:.3f}" for c in CLASSES), flush=True)
if "gbm" in args.models:
    t1 = time.time()
    mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, early_stopping=True, validation_fraction=0.1, random_state=args.seed, class_weight="balanced")
    mdl.fit(Xtr_s, ytr_s)
    pr = mdl.predict_proba(Xte_s); results["gbm"] = score(yte, pr.argmax(1), pr); results["gbm"]["n_iter"] = int(mdl.n_iter_)
    r = results["gbm"]; print(f"GBM     test: avg F1 {r['avg_f1']:.3f}  MAP {r['map']:.3f}  | " + "  ".join(f"{c} F1 {r['f1'][c]:.3f} AP {r['ap'][c]:.3f}" for c in CLASSES) + f"  ({mdl.n_iter_} iters, {time.time() - t1:.0f}s)", flush=True)
print(f"[{args.tag or args.emb} feats={args.feats} ctx={args.ctx}] reference (CalMS21 paper, Task 1): 1D-conv keypoints avg F1 .793 / MAP .856; + task-programming .829 / .889; challenge top-1 .864 / .914")
if args.out:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"args": vars(args), "n_features": Xtr.shape[1], "results": results}, open(args.out, "w"), indent=1)
