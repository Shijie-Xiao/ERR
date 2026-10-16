"""CalMS21 task-1 videos -> ERR window dataset (same npz layout as the MABe files, so err/train.py trains on it).

Input : task1_classic_classification/calms21_task1_{train,test}.json (CaltechDATA record s0vdx-0k302), 70 + 19 videos,
        2 mice x 7 keypoints (nose, left ear, right ear, neck, left hip, right hip, tail base), 1024x570 px, 30 fps.
Nodes : node = mouse * 7 + kp (14 nodes).  Frames are subsampled 30 -> 7.5 Hz (fs = 4), windows of 20 steps
        (10 observed + 10 predicted) with stride --stride steps, no event centring (natural sliding windows).
Scale : coordinates are divided by --scale px per unit; with --scale 0 the scale is set so that the median body length
        (nose - tail base) of the training videos equals --body_len (0.0992 = the MABe22 value), which keeps the step bound,
        the contact radius and all invariants on the scale the MABe models use.  The resulting px/unit is printed and stored
        in meta_arena_px: pass it to the trainer as --arena_px (1827.7 for the official files).
Splits: the 70 training videos are split by video into train / val (--val_videos), the 19 test videos are the test split.
Behaviour labels (attack 0, investigation 1, mount 2, other 3) are stored per window at every step ({split}_labels, int8)
for later analysis; the trainer never reads them.  Event metadata keys are written as 'no event' so the trainer's metrics run.
Usage (from the repository root):
    python data_prep/calms21_windows.py --json_dir data/calms21/task1_classic_classification --out data/windows/dataset_calms21_w20_fs4.npz
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--json_dir", default=os.path.join(ROOT, "data", "calms21", "task1_classic_classification"))
p.add_argument("--out", default=os.path.join(ROOT, "data", "windows", "dataset_calms21_w20_fs4.npz"))
p.add_argument("--fs", type=int, default=4); p.add_argument("--window", type=int, default=20); p.add_argument("--stride", type=int, default=5)
p.add_argument("--scale", type=float, default=0.0); p.add_argument("--body_len", type=float, default=70.4 / 710.0)
p.add_argument("--val_videos", type=int, default=8); p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
KPN = ["nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip", "tail_base"]


def load(split):
    d = json.load(open(os.path.join(args.json_dir, f"calms21_task1_{split}.json")))["annotator-id_0"]
    out = []
    for k, s in d.items():
        kp = np.transpose(np.asarray(s["keypoints"], np.float32), (0, 1, 3, 2))             # T,2,7,2 px
        out.append((k, kp, np.asarray(s["annotations"], np.int8)))
    return out


train_all, test = load("train"), load("test")
rng = np.random.default_rng(args.seed); perm = rng.permutation(len(train_all))
val = [train_all[i] for i in perm[:args.val_videos]]; train = [train_all[i] for i in perm[args.val_videos:]]
if args.scale <= 0:
    bl = np.concatenate([np.linalg.norm(kp[:, :, 0] - kp[:, :, 6], axis=-1).ravel() for _, kp, _ in train])
    args.scale = float(np.median(bl)) / args.body_len
print(f"videos: train {len(train)} val {len(val)} test {len(test)}; scale {args.scale:.1f} px per unit (use --arena_px {args.scale:.1f} in err/train.py)")

arrays = {}
for split, vids in (("train", train), ("val", val), ("test", test)):
    data, labels, seq_id, t_start = [], [], [], []
    for vi, (k, kp, y) in enumerate(vids):
        x = kp[::args.fs] / args.scale; yl = y[::args.fs]
        for s in range(0, len(x) - args.window + 1, args.stride):
            data.append(x[s:s + args.window]); labels.append(yl[s:s + args.window]); seq_id.append(vi); t_start.append(s)
    n = len(data)
    arrays[f"{split}_data"] = np.stack(data).astype(np.float32)                                 # N,20,2,7,2
    arrays[f"{split}_labels"] = np.stack(labels).astype(np.int8)
    arrays[f"{split}_seq_id"] = np.array(seq_id, np.int32); arrays[f"{split}_t_start"] = np.array(t_start, np.int32)
    for key, val_ in (("event_type", 0), ("focal", -1), ("partner", -1), ("event_step", -1), ("lights", 0), ("chase", 0)):
        arrays[f"{split}_{key}"] = np.full(n, val_, np.int32)
    arrays[f"{split}_activity"] = np.linalg.norm(np.diff(arrays[f"{split}_data"][:, :, :, 3], axis=1), axis=-1).mean((1, 2)).astype(np.float32)
    arrays[f"meta_split_seq_keys_{split}"] = np.array([k for k, _, _ in vids])
    lab = arrays[f"{split}_labels"][:, 9]
    print(f"  {split}: {n} windows from {len(vids)} videos; label at the last observed step: " + ", ".join(f"{c} {np.mean(lab == i):.3f}" for i, c in enumerate(["attack", "investigation", "mount", "other"])))
arrays.update(meta_window_size=args.window, meta_stride=args.stride, meta_fs=args.fs, meta_hz=30.0 / args.fs, meta_n_nodes=14, meta_n_mice=2,
              meta_kp_indices=np.arange(7), meta_kp_names=np.array(KPN), meta_arena_px=args.scale, meta_event_bits=np.array([]),
              meta_args=np.array(vars(args), dtype=object))
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez_compressed(args.out, **arrays)
print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB)")
