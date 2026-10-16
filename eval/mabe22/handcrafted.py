"""Hand-crafted per-frame geometry features for the MABe22 mouse triplets (the 'PCA on hand-crafted features' style baseline).

Computed at 30 Hz from the imputed keypoints (same imputation as embed.py), in the same row order as the label
frame_number_map, so probe.py can read them like a model embedding or concatenate them with one.
Per mouse (3): cb speed, nose speed, body length (nose - tail base), head-body angle cosine, turning rate.
Per ordered pair (6): nose-nose, nose(m)-tail base(m'), cb-cb distances, closing speed of cb-cb, cos(heading of m, direction to m').
Scene: min nose-nose, min nose-tail, min cb-cb, number of mice tracked.
Each feature is given raw and smoothed over a 1-s centred window (31 frames)  -> 2 x (15 + 30 + 4) + 3 tracked = 101 dims.
Usage: python eval/mabe22/handcrafted.py --npy data/mabe22/mouse_triplet_test.npy --labels data/mabe22/mouse_triplets_test_labels.npy --out embeddings/mabe22/handcrafted
"""
import argparse
import json
import os
import time

import numpy as np

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--npy", required=True); p.add_argument("--labels", default=None); p.add_argument("--out", required=True)
p.add_argument("--arena_px", type=float, default=710.0); p.add_argument("--smooth", type=int, default=31)
args = p.parse_args()
NOSE, NECK, CB, TAIL = 0, 3, 6, 9


def uniform_filter1d(F, size):
    """centred moving average along axis 0 with edge replication (numpy only)"""
    h = size // 2
    P = np.concatenate([np.repeat(F[:1], h, 0), F, np.repeat(F[-1:], h, 0)], 0).astype(np.float64)
    c = np.cumsum(np.concatenate([np.zeros((1,) + P.shape[1:]), P], 0), 0)
    return ((c[size:] - c[:-size]) / size).astype(np.float32)


def impute(kp):
    x = kp.astype(np.float32) / args.arena_px
    miss = np.abs(kp).sum(-1) == 0; tracked = ~miss.all(2)
    T = kp.shape[0]; t_all = np.arange(T)
    for m in range(3):
        for k in range(12):
            bad = miss[:, m, k]
            if not bad.any():
                continue
            good = ~bad
            if good.any():
                for c in range(2):
                    x[bad, m, k, c] = np.interp(t_all[bad], t_all[good], x[good, m, k, c])
            else:
                x[:, m, k] = x[:, m, CB] if not miss[:, m, CB].all() else 0.5
    return x, tracked.astype(np.float32)


def feats(x, tracked):
    d = lambda a, b: np.linalg.norm(a - b, axis=-1)
    vel = np.gradient(x, axis=0)                                                            # T,3,12,2 per frame
    f = []
    head = x[:, :, NOSE] - x[:, :, TAIL]; head_n = head / (np.linalg.norm(head, axis=-1, keepdims=True) + 1e-6)
    neck = x[:, :, NOSE] - x[:, :, NECK]; body = x[:, :, NECK] - x[:, :, TAIL]
    ang = np.arctan2(head_n[..., 1], head_n[..., 0]); turn = np.abs(np.gradient(np.unwrap(ang, axis=0), axis=0))
    for m in range(3):
        f += [np.linalg.norm(vel[:, m, CB], axis=-1), np.linalg.norm(vel[:, m, NOSE], axis=-1), d(x[:, m, NOSE], x[:, m, TAIL]),
              (neck[:, m] * body[:, m]).sum(-1) / (np.linalg.norm(neck[:, m], axis=-1) * np.linalg.norm(body[:, m], axis=-1) + 1e-6), turn[:, m]]
    for a in range(3):
        for b in range(3):
            if a == b:
                continue
            dc = d(x[:, a, CB], x[:, b, CB]); to_b = x[:, b, CB] - x[:, a, CB]; to_b = to_b / (np.linalg.norm(to_b, axis=-1, keepdims=True) + 1e-6)
            f += [d(x[:, a, NOSE], x[:, b, NOSE]), d(x[:, a, NOSE], x[:, b, TAIL]), dc, np.gradient(dc), (head_n[:, a] * to_b).sum(-1)]
    nn = np.stack([d(x[:, a, NOSE], x[:, b, NOSE]) for a in range(3) for b in range(3) if a != b], 1).min(1)
    nt = np.stack([d(x[:, a, NOSE], x[:, b, TAIL]) for a in range(3) for b in range(3) if a != b], 1).min(1)
    cc = np.stack([d(x[:, a, CB], x[:, b, CB]) for a in range(3) for b in range(3) if a != b], 1).min(1)
    f += [nn, nt, cc, tracked.sum(1)]
    F = np.stack(f, 1).astype(np.float32)                                                   # T,49
    Fs = uniform_filter1d(F, size=args.smooth)
    return np.concatenate([F, Fs, tracked], 1)


raw = np.load(args.npy, allow_pickle=True).item()["sequences"]
if args.labels:
    fmap = np.load(args.labels, allow_pickle=True).item()["frame_number_map"]
    ids = sorted(fmap.keys(), key=lambda k: fmap[k][0]); offsets = [fmap[k] for k in ids]
else:
    ids = list(raw.keys()); offsets = []; o = 0
    for k in ids:
        n = len(raw[k]["keypoints"]); offsets.append((o, o + n)); o += n
names = [f"{g}_{m}" for m in range(3) for g in ("spd_cb", "spd_nose", "body_len", "head_cos", "turn")] + \
        [f"{g}_{a}{b}" for a in range(3) for b in range(3) if a != b for g in ("nn", "ntail", "cc", "closing", "facing")] + ["min_nn", "min_ntail", "min_cc", "n_tracked"]
names = names + [n + "_s" for n in names] + [f"tracked_{m}" for m in range(3)]
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
out = np.lib.format.open_memmap(args.out + ".npy", mode="w+", dtype=np.float16, shape=(offsets[-1][1], len(names)))
t0 = time.time()
for i, k in enumerate(ids):
    x, tr = impute(np.asarray(raw[k]["keypoints"]))
    s, e = offsets[i]; out[s:e] = feats(x, tr).astype(np.float16)
    if i % 500 == 0:
        print(f"  {i}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
out.flush()
json.dump({"ids": ids, "offsets": [list(map(int, o)) for o in offsets], "features": names, "mode": "handcrafted"}, open(args.out + ".json", "w"))
print(f"wrote {args.out}.npy {out.shape} in {time.time() - t0:.0f}s")
