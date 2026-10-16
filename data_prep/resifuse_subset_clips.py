"""Subset a MABe-format sequences npy to the clips of a ResiFuse split file (e.g. the 108 fine-tuning clips), so that the
frame embedding (eval/mabe22/embed.py) and the hand-crafted features (eval/mabe22/handcrafted.py) are computed only for the
clips the ResiFuse Mouse Triplet protocol scores.
Usage (from the repository root):
    python data_prep/resifuse_subset_clips.py --npy data/resifuse/mouse_submission_dlcrf.npy \
        --split data/resifuse/code/splits/mabe_finetune_split.json --out data/resifuse/finetune108_dlcrf.npy
"""
import argparse
import json

import numpy as np

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--npy", required=True); p.add_argument("--split", required=True); p.add_argument("--out", required=True)
args = p.parse_args()
d = np.load(args.npy, allow_pickle=True).item()
sp = json.load(open(args.split)); sp = sp.get("splits", sp)
keep = [c for s in ("train", "val", "test") for c in sp[s]]
missing = [c for c in keep if c not in d["sequences"]]
sub = {"vocabulary": d["vocabulary"], "sequences": {c: d["sequences"][c] for c in keep if c in d["sequences"]}, "meta": d.get("meta", {})}
np.save(args.out, sub, allow_pickle=True)
print(f"wrote {args.out}: {len(sub['sequences'])} clips" + (f" ({len(missing)} clips of the split are not in the source file)" if missing else ""))
