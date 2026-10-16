"""MABe22 mouse-triplet linear readout on frozen frame embeddings (the official protocol, re-implemented).

Official evaluator (damaggu/MABe2022, round 1): per task a RidgeClassifier(class_weight='balanced') (binary tasks, F1) or
Ridge (continuous tasks, MSE) is trained on the frames of the 'SubmissionTrain' sequences and scored on 'publicTest' +
'privateTest'; alpha in {0.1, 0.5, 1, 2, 5} is chosen on a 10 % held-out split of SubmissionTrain; sequence-level tasks are
additionally scored after averaging the frame predictions over the sequence ('pooled').  Embeddings are limited to 128
dimensions in the challenge; --pca 128 reproduces that constraint (make_submission.py writes the file for the official code).

Usage (from the repository root):
  python eval/mabe22/probe.py --emb embeddings/mabe22/err_36kp --labels data/mabe22/mouse_triplets_test_labels.npy \
      --split data/mabe22/jax_split.json [--feats all|h|graph] [--pca 128] [--out results/mabe22/err_36kp.json]
--emb may be given several times: the feature blocks are concatenated (e.g. ERR + hand-crafted features).
jax_tasks.json (task list, sequence-level tasks) is expected next to the split file.
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--emb", action="append", required=True); p.add_argument("--labels", required=True); p.add_argument("--split", required=True)
p.add_argument("--feats", default="all", help="all | h | graph  (h = node memories + tracked flags; graph = share + pcontact + intent + surprise + speed)")
p.add_argument("--pca", type=int, default=0); p.add_argument("--stride", type=int, default=4, help="train-frame stride (test uses every frame)")
p.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0, 5.0]); p.add_argument("--seed", type=int, default=42)
p.add_argument("--tasks", nargs="+", default=None); p.add_argument("--out", default=None); p.add_argument("--tag", default="")
args = p.parse_args()

L = np.load(args.labels, allow_pickle=True).item()
fm, Y, vocab, ttype = L["frame_number_map"], L["label_array"], L["vocabulary"], L["task_type"]
tasks_json = json.load(open(os.path.join(os.path.dirname(args.split), "jax_tasks.json")))
seq_level = set(tasks_json["sequence_level_tasks"])
split = json.load(open(args.split))

# ---- features --------------------------------------------------------------------------------------------------
blocks, names = [], []
for prefix in args.emb:
    e = np.load(prefix + ".npy", mmap_mode="r"); j = json.load(open(prefix + ".json"))
    assert j["offsets"][-1][1] == e.shape[0] == Y.shape[1], f"{prefix}: rows {e.shape[0]} vs labels {Y.shape[1]}"
    assert all(tuple(fm[k]) == tuple(o) for k, o in zip(j["ids"], j["offsets"])), f"{prefix}: frame order differs from the label map"
    fn = j["features"]
    if args.feats != "all" and prefix == args.emb[0] and fn[0].startswith("h0_"):
        is_h = np.array([n.startswith("h") and "_" in n and n.split("_")[0][1:].isdigit() for n in fn])
        is_tr = np.array([n.startswith("tracked") for n in fn])
        keep = {"h": is_h | is_tr, "graph": ~is_h}[args.feats]
        cols = np.nonzero(keep)[0]
    else:
        cols = np.arange(len(fn))
    blocks.append((e, cols)); names += [f"{os.path.basename(prefix)}:{fn[c]}" for c in cols]
print(f"features: {len(names)} columns from {len(args.emb)} file(s); feats={args.feats} pca={args.pca}")


def rows_of(ids, stride=1):
    return np.concatenate([np.arange(fm[k][0], fm[k][1], stride) for k in ids])


def gather(idx):
    return np.concatenate([np.asarray(e[idx][:, cols], dtype=np.float32) for e, cols in blocks], 1)


def seq_of(ids, stride=1):
    return np.concatenate([np.full(len(range(fm[k][0], fm[k][1], stride)), i) for i, k in enumerate(ids)])


tr_ids, te_ids = split["SubmissionTrain"], split["publicTest"] + split["privateTest"]
t0 = time.time()
idx_tr = rows_of(tr_ids, args.stride); idx_te = rows_of(te_ids)
X_tr = gather(idx_tr); X_te = gather(idx_te); seq_te = seq_of(te_ids)
print(f"X_train {X_tr.shape} (stride {args.stride}), X_test {X_te.shape}; gathered in {time.time() - t0:.0f}s")
X_tr[~np.isfinite(X_tr)] = 0; X_te[~np.isfinite(X_te)] = 0
sc = StandardScaler().fit(X_tr); X_tr = sc.transform(X_tr).astype(np.float32); X_te = sc.transform(X_te).astype(np.float32)
if args.pca and args.pca < X_tr.shape[1]:
    pca = PCA(args.pca, random_state=0).fit(X_tr[::4]); X_tr = pca.transform(X_tr).astype(np.float32); X_te = pca.transform(X_te).astype(np.float32)
    print(f"PCA -> {args.pca} dims, explained {pca.explained_variance_ratio_.sum():.3f}")

# alpha-selection split inside SubmissionTrain (by sequence, 90/10, seed)
tr_sub, va_sub = train_test_split(tr_ids, test_size=0.1, random_state=args.seed)
pos_tr = {k: i for i, k in enumerate(tr_ids)}
seq_tr = seq_of(tr_ids, args.stride); in_va = np.isin(seq_tr, [pos_tr[k] for k in va_sub])

results = {}
task_list = args.tasks or vocab
for name in task_list:
    ti = vocab.index(name); cont = ttype[ti].lower().startswith("cont")
    y_tr, y_te = Y[ti, idx_tr], Y[ti, idx_te]
    ok_tr, ok_te = np.isfinite(y_tr), np.isfinite(y_te)
    t1 = time.time(); best = None
    for a in args.alphas:
        mdl = Ridge(alpha=a) if cont else RidgeClassifier(alpha=a, class_weight="balanced")
        m_fit = ok_tr & ~in_va; m_val = ok_tr & in_va
        mdl.fit(X_tr[m_fit], y_tr[m_fit]); pv = mdl.predict(X_tr[m_val])
        sc_v = mean_squared_error(y_tr[m_val], pv) if cont else f1_score(y_tr[m_val], pv, average="binary")
        if best is None or (sc_v < best[1] if cont else sc_v > best[1]):
            best = (a, sc_v)
    a = best[0]
    mdl = Ridge(alpha=a) if cont else RidgeClassifier(alpha=a, class_weight="balanced")
    mdl.fit(X_tr[ok_tr], y_tr[ok_tr])
    pt = mdl.predict(X_te[ok_te])
    score = mean_squared_error(y_te[ok_te], pt) if cont else f1_score(y_te[ok_te], pt, average="binary")
    r = {"metric": "mse" if cont else "f1", "score": float(score), "alpha": a, "val": float(best[1]), "pos_rate_test": float(np.nanmean(y_te)) if not cont else None}
    if name in seq_level or cont:                                                             # pooled: average the frame scores per sequence
        raw = mdl.predict(X_te) if cont else mdl.decision_function(X_te)
        pooled = np.array([raw[seq_te == i].mean() for i in range(len(te_ids))])
        y_seq = np.array([np.nanmean(y_te[seq_te == i]) for i in range(len(te_ids))])
        if cont:
            r["pooled"] = float(mean_squared_error(y_seq, pooled))
        else:
            r["pooled"] = float(f1_score((y_seq > 0.5).astype(int), (pooled > 0).astype(int), average="binary"))
    results[name] = r
    print(f"  {name:22s} {r['metric']} {score:.4f}  (alpha {a}, val {best[1]:.4f}{', pooled %.4f' % r['pooled'] if 'pooled' in r else ''}, {time.time() - t1:.0f}s)", flush=True)

f1s = [r["score"] for n, r in results.items() if r["metric"] == "f1"]
print(f"mean F1 over {len(f1s)} binary tasks: {np.mean(f1s):.4f}   [{args.tag or ','.join(os.path.basename(e) for e in args.emb)} feats={args.feats} pca={args.pca}]")
if args.out:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"args": vars(args), "n_features": len(names), "results": results, "mean_f1": float(np.mean(f1s))}, open(args.out, "w"), indent=1)
