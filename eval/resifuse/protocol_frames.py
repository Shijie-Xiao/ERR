"""Reproduce, with the ResiFuse code itself, the set of frames and 4-class labels that its Mouse Triplet fine-tuning evaluation
scores, so that any frozen per-frame representation can be read out on exactly the same frames.

ResiFuse (github.com/BRAINML-GT/ResiFuse) protocol: 4 label groups = chases, huddles, oral_contact (oral_ear | oral_oral),
oral_genital_contact; fine-tuning split of 108 clips (54 / 11 / 43); 200-frame windows (stride 200, conf_threshold 0.2, max_gap 60,
min_valid_len 30) shared by all three mice; labels sub-sampled with ds_stride 4 from the window start; a frame is scored only
when at least one of the 4 classes is positive; majority vote over mice = the clip label; each frame is counted once per clip.
Uses resifuse.preprocessing.mabe (FINETUNE_LABEL_GROUPS, extract_mouse_data, find_valid_segments, generate_labeled_windows,
build_frame_labels, _segments_to_mask, ProcessingConfig) from the cloned repository (--code).

Output npz: clip_ids, split (0 train / 1 val / 2 test), clip index, frame index (30 Hz), labels (N, 4) {0, 1}, label names,
            plus his fine-tuning windows (win_clip, win_split, win_start, win_labels (W, 50, 4) with -1 = unscored, win_mask).
Usage (from the repository root):
  python eval/resifuse/protocol_frames.py --code data/resifuse/code --h5_dir data/resifuse/deeplabcut_filtered \
      --labels data/mabe22/mouse_submission_labels.npy --split data/resifuse/code/splits/mabe_finetune_split.json \
      --out data/resifuse/finetune_scored_frames.npz
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--code", required=True); p.add_argument("--h5_dir", required=True); p.add_argument("--labels", required=True)
p.add_argument("--split", required=True); p.add_argument("--out", required=True); p.add_argument("--ds_stride", type=int, default=4)
args = p.parse_args()
sys.path.insert(0, args.code)
from resifuse.preprocessing import mabe as RM                                                   # noqa: E402

cfg = RM.ProcessingConfig(); cfg.window_size, cfg.stride = RM.STAGE_WINDOWS["finetune"]
label_data = RM.load_label_data(args.labels)
sp = json.load(open(args.split)); sp = sp.get("splits", sp)
h5_of = {re.match(r"([A-Z0-9]{20})DLC", os.path.basename(f)).group(1): f for f in glob.glob(os.path.join(args.h5_dir, "*.h5"))}
rows = {"clip": [], "split": [], "frame": [], "labels": []}; clip_ids = []; report = {}
wins = {"clip": [], "split": [], "start": [], "labels": [], "mask": []}                         # his fine-tuning samples (window, ds 4)
for s_i, split in enumerate(("train", "val", "test")):
    n_frames_split = 0
    for vid in sp[split]:
        df = pd.read_hdf(h5_of[vid]); mouse_ids = df.columns.get_level_values("individuals").unique().tolist()
        lm = RM.get_video_label_matrix(vid, label_data)
        pose = {m: RM.extract_mouse_data(df, m) for m in mouse_ids}
        num_frames = min(len(pose[mouse_ids[0]][0]), lm.shape[1]); lm = lm[:, :num_frames]
        windows, frame_labels = {}, {}
        for m in mouse_ids:
            kp, conf, conf_nonfinite = (x[:num_frames] for x in pose[m])
            bad = ~np.isfinite(kp).all(axis=(1, 2)) | conf_nonfinite
            if bad.any():
                conf = conf.copy(); conf[bad] = 0
            segments = RM.find_valid_segments(conf.mean(axis=1), cfg)
            valid_mask = RM._segments_to_mask(segments, num_frames) & ~bad
            frame_labels[m] = RM.build_frame_labels(lm, valid_mask)                              # (T, 4), -1 = not scored
            windows[m] = [w for w in RM.generate_labeled_windows(lm, segments, cfg) if w["start"] + cfg.window_size <= num_frames]
        common = sorted(set.intersection(*({(w["start"], w["end"]) for w in windows[m]} for m in mouse_ids)))
        seen = {}; ci = len(clip_ids)
        for start, end in common:
            L_w = np.full((cfg.window_size // args.ds_stride, 4), -1, np.int8); M_w = np.zeros(cfg.window_size // args.ds_stride, bool)
            for j, t in enumerate(range(0, cfg.window_size, args.ds_stride)):
                f = start + t
                if f >= num_frames:
                    continue
                labs = np.stack([frame_labels[m][f] for m in mouse_ids])                         # 3, 4
                valid = labs != -1
                if not valid.any():
                    continue
                pos = ((labs == 1) & valid).sum(0); cnt = valid.sum(0)
                lab_f = (pos > cnt / 2.0).astype(np.int8); L_w[j] = lab_f; M_w[j] = True
                if f not in seen:
                    seen[f] = lab_f
            wins["clip"].append(ci); wins["split"].append(s_i); wins["start"].append(start); wins["labels"].append(L_w); wins["mask"].append(M_w)
        clip_ids.append(vid)
        for f, lab in sorted(seen.items()):
            rows["clip"].append(ci); rows["split"].append(s_i); rows["frame"].append(f); rows["labels"].append(lab)
        n_frames_split += len(seen)
    report[split] = n_frames_split
lab = np.stack(rows["labels"]); spl = np.array(rows["split"])
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez(args.out, clip_ids=np.array(clip_ids), split=spl, clip=np.array(rows["clip"]), frame=np.array(rows["frame"]), labels=lab,
         label_names=np.array(label_data["label_names"]), win_clip=np.array(wins["clip"]), win_split=np.array(wins["split"]),
         win_start=np.array(wins["start"]), win_labels=np.stack(wins["labels"]), win_mask=np.stack(wins["mask"]), ds_stride=args.ds_stride)
print("windows per split:", {s: int((np.array(wins["split"]) == i).sum()) for i, s in enumerate(("train", "val", "test"))})
print("scored frames per split:", report)
for s_i, split in enumerate(("train", "val", "test")):
    L = lab[spl == s_i]; print(f"  {split}: " + ", ".join(f"{n} {L[:, i].mean():.3f}" for i, n in enumerate(label_data["label_names"])) + f"; all-positive macro F1 {np.mean(2 * L.mean(0) / (1 + L.mean(0))):.3f}")
print(f"wrote {args.out}")
