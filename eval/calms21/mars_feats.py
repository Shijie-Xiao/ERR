"""Put the official CalMS21 MARS features (task1_MARS_features.zip: data_smooth (2 mice, T, 158) per video) into the embedding
file format used by probe.py, in the same video order / labels as an existing ERR embedding, and also write a MARS + ERR
concatenation (the complementary-features experiment).
Usage: python eval/calms21/mars_feats.py --mars_dir data/calms21/calms21_task1_features --ref embeddings/calms21/err_zeroshot --out embeddings/calms21/mars
"""
import argparse
import json
import os

import numpy as np

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--mars_dir", required=True); p.add_argument("--ref", required=True); p.add_argument("--out", required=True)
args = p.parse_args()
for split in ("train", "test"):
    j = json.load(open(f"{args.ref}_{split}.json")); E = np.load(f"{args.ref}_{split}.npy", mmap_mode="r"); y = np.load(f"{args.ref}_{split}_labels.npy")
    names = None; blocks = []; blocks_c = []
    for k, (s, e) in zip(j["ids"], j["offsets"]):
        z = np.load(os.path.join(args.mars_dir, split, os.path.basename(k) + ".npz"), allow_pickle=True)
        f = z["data_smooth"].astype(np.float32)                                             # 2, T, 158
        assert f.shape[1] == e - s, f"{k}: MARS T {f.shape[1]} vs {e - s}"
        if names is None:
            names = [f"mars_m{m}_{n}" for m in range(2) for n in z["features"]]
        F = np.concatenate([f[0], f[1]], 1)                                                  # T, 316
        blocks.append(F); blocks_c.append(np.concatenate([np.asarray(E[s:e], dtype=np.float32), F], 1))
    X = np.concatenate(blocks); Xc = np.concatenate(blocks_c)
    X[~np.isfinite(X)] = 0; Xc[~np.isfinite(Xc)] = 0
    np.save(f"{args.out}_{split}.npy", X); np.save(f"{args.out}_{split}_labels.npy", y)
    json.dump({"ids": j["ids"], "offsets": j["offsets"], "features": names, "source": "CalMS21 task1_MARS_features data_smooth"}, open(f"{args.out}_{split}.json", "w"))
    np.save(f"{args.out}_plus_err_{split}.npy", Xc); np.save(f"{args.out}_plus_err_{split}_labels.npy", y)
    json.dump({"ids": j["ids"], "offsets": j["offsets"], "features": j["features"] + names, "source": "ERR + MARS"}, open(f"{args.out}_plus_err_{split}.json", "w"))
    print(f"{split}: MARS {X.shape}, MARS+ERR {Xc.shape}, labels {np.bincount(y, minlength=4)}")
