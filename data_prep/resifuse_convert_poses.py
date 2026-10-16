"""Convert the ResiFuse re-estimated DeepLabCut poses of the MABe22 'submission' clips into the MABe npy format read by
this pipeline, and write the OFFICIAL keypoints of the same clips in the same format for a paired comparison.

Input : Hugging Face WeihanLikk/ResiFuse-MABe22, deeplabcut_filtered/<clip_id>DLC_...h5 (one pandas HDF per clip, 3 individuals
        x 12 bodyparts x (x, y, likelihood) in 448x448-video pixels, undetected = NaN, 1735-1801 frames);
        mouse_submission_labels.npy + mouse_submission_keypoints.npy (CaltechDATA record rdsa8-rde65).
Output: data/resifuse/mouse_submission_<tag>.npy   DLC poses   {"vocabulary": ["chases", "lights"], "sequences": {clip: {"keypoints": (1800, 3, 12, 2), "annotations": (2, 1800)}}}
        data/resifuse/mouse_submission_sub.npy      official keypoints of the same clips, same format
        data/resifuse/submission_split.json         70/15/15 clip split (SubmissionTrain / publicTest / privateTest) for readouts
        data/resifuse/mouse_submission_labels_subset.npy   the 13-task labels in the same clip order

Conventions: MABe bodypart order (nose, left ear, right ear, neck, left forepaw, right forepaw, center_back, left hindpaw,
right hindpaw, tail base, tail mid, tail tip); with --resifuse_validity 0.2 the ResiFuse validity rule is applied (a mouse-frame
is dropped whole when any keypoint is non-finite or the mean likelihood is below 0.2); gaps of <= --max_gap frames are linearly
interpolated per keypoint; remaining untracked keypoints are (0, 0) (the MABe encoding); coordinates are rescaled so that the
median nose - tail-base length equals --body_px (70.4 px, the value of the 710-px user-train arena), so the event thresholds
and arena_px = 710 of mabe_event_windows.py apply unchanged to both sources.
Usage (from the repository root; the two protocol variants used in the paper):
    python data_prep/resifuse_convert_poses.py --max_gap 30                                      # tag dlc  (gap-filled)
    python data_prep/resifuse_convert_poses.py --max_gap 0 --resifuse_validity 0.2 --tag dlcrf   # ResiFuse validity rule
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--h5_dir", default=os.path.join(ROOT, "data", "resifuse", "deeplabcut_filtered"))
p.add_argument("--labels", default=os.path.join(ROOT, "data", "mabe22", "mouse_submission_labels.npy"))
p.add_argument("--official", default=os.path.join(ROOT, "data", "mabe22", "mouse_submission_keypoints.npy"))
p.add_argument("--out_dir", default=os.path.join(ROOT, "data", "resifuse"))
p.add_argument("--max_gap", type=int, default=30); p.add_argument("--body_px", type=float, default=70.4)
p.add_argument("--min_likelihood", type=float, default=0.0); p.add_argument("--max_clips", type=int, default=0); p.add_argument("--seed", type=int, default=42)
p.add_argument("--resifuse_validity", type=float, default=0.0, help="> 0: ResiFuse rule (preprocessing/mabe.py): a mouse-frame is invalid if any keypoint "
               "is non-finite or its mean likelihood over the 12 keypoints is below this value (ResiFuse: 0.2); invalid mouse-frames are dropped whole")
p.add_argument("--tag", default="dlc", help="output file mouse_submission_<tag>.npy")
args = p.parse_args()
MABE_ORDER = ["nose", "left ear", "right ear", "neck", "left forepaw", "right forepaw", "center back", "left hindpaw", "right hindpaw", "tail base", "tail middle", "tail tip"]
T = 1800; os.makedirs(args.out_dir, exist_ok=True)

lab = np.load(args.labels, allow_pickle=True).item()
fm, L, vocab = lab["frame_number_map"], lab["label_array"], list(lab["vocabulary"])
i_chase, i_light = vocab.index("chases"), vocab.index("lights")
official = np.load(args.official, allow_pickle=True).item()["sequences"]


def fill_gaps(xy, max_gap):
    """xy (T, 2) with NaN rows -> linear interpolation of interior gaps <= max_gap frames"""
    x = xy.copy(); ok = np.isfinite(x[:, 0]); n = len(x)
    if ok.sum() < 2 or max_gap <= 0:
        return x
    idx = np.arange(n); d = np.diff(np.concatenate([[0], (~ok).astype(int), [0]])); s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    for a, b in zip(s, e):
        if b - a <= max_gap and a > 0 and b < n:
            for c in range(2):
                x[a:b, c] = np.interp(idx[a:b], [a - 1, b], [x[a - 1, c], x[b, c]])
    return x


def body_scale(seqs):
    """px per 70.4-px body length: median nose - tail-base distance over tracked frames (keypoints as NaN where untracked)"""
    bl = []
    for kp in seqs.values():
        d = np.linalg.norm(kp[:, :, 0] - kp[:, :, 9], axis=-1); bl.append(d[np.isfinite(d)])
    bl = np.concatenate(bl); return args.body_px / float(np.median(bl)), float(np.median(bl))


files = sorted(glob.glob(os.path.join(args.h5_dir, "*.h5")))
if args.max_clips:
    files = files[:args.max_clips]
dlc, stats, t0 = {}, [], time.time()
for i, f in enumerate(files):
    cid = re.match(r"([A-Z0-9]{20})DLC", os.path.basename(f)).group(1)
    if cid not in fm or cid not in official:
        continue
    df = pd.read_hdf(f); sc = df.columns.get_level_values(0)[0]; inds = df.columns.get_level_values(1).unique().tolist()
    if len(inds) != 3:
        continue
    kp = np.full((T, 3, 12, 2), np.nan, np.float32); n = min(len(df), T); lik = np.zeros((T, 3, 12), np.float32)
    for m, ind in enumerate(inds):
        for k, bp in enumerate(MABE_ORDER):
            x = df[(sc, ind, bp, "x")].to_numpy()[:n]; y = df[(sc, ind, bp, "y")].to_numpy()[:n]; lk = np.nan_to_num(df[(sc, ind, bp, "likelihood")].to_numpy()[:n], nan=0.0)
            ok = np.isfinite(x) & np.isfinite(y) & (lk >= args.min_likelihood)
            kp[:n, m, k, 0] = np.where(ok, x, np.nan); kp[:n, m, k, 1] = np.where(ok, y, np.nan); lik[:n, m, k] = lk
    if args.resifuse_validity > 0:                                                         # ResiFuse rule, whole mouse-frame
        bad = ~np.isfinite(kp).all((2, 3)) | (lik.mean(2) < args.resifuse_validity)       # T,3
        kp[bad] = np.nan
    for m in range(3):
        for k in range(12):
            kp[:, m, k] = fill_gaps(kp[:, m, k], args.max_gap)
    dlc[cid] = kp
    stats.append(np.isfinite(kp[:, :, [0, 3, 6, 9], 0]).all((1, 2)).mean())
    if i % 300 == 0:
        print(f"  {i}/{len(files)} {time.time() - t0:.0f}s", flush=True)
ids = sorted(dlc)
off = {}
for cid in ids:
    o = np.asarray(official[cid]["keypoints"], np.float32)[:T].copy(); o[(np.abs(o).sum(-1) == 0)] = np.nan; off[cid] = o
s_dlc, bl_dlc = body_scale(dlc); s_off, bl_off = body_scale(off)
print(f"{len(ids)} clips | DLC complete-frame coverage after gap fill: mean {np.mean(stats):.3f} | body length: DLC {bl_dlc:.1f} px (scale {s_dlc:.4f}), official {bl_off:.1f} px (scale {s_off:.4f})")


def pack(kp, s):
    out = kp * s; out[np.isnan(kp)] = 0.0; return out.astype(np.float32)


def ann(cid):
    a, b = fm[cid]; return np.stack([L[i_chase, a:b], L[i_light, a:b]]).astype(np.float32)


for name, seqs, s, src in ((args.tag, dlc, s_dlc, "ResiFuse DeepLabCut re-estimation (gap fill <= %d frames, resifuse_validity %.2f)" % (args.max_gap, args.resifuse_validity)),
                           ("sub", off, s_off, "official MABe22 submission keypoints, same clips")):
    out = os.path.join(args.out_dir, f"mouse_submission_{name}.npy")
    np.save(out, {"vocabulary": ["chases", "lights"], "sequences": {cid: {"keypoints": pack(seqs[cid], s), "annotations": ann(cid)} for cid in ids},
                  "meta": {"source": src, "scale_to_710px_space": s, "clips": len(ids), "arena_px": 710.0}}, allow_pickle=True)
    print(f"wrote {out}")
rng = np.random.default_rng(args.seed); perm = [ids[i] for i in rng.permutation(len(ids))]; n_tr = int(0.7 * len(perm)); n_pu = int(0.15 * len(perm))
split = {"SubmissionTrain": perm[:n_tr], "publicTest": perm[n_tr:n_tr + n_pu], "privateTest": perm[n_tr + n_pu:]}
json.dump(split, open(os.path.join(args.out_dir, "submission_split.json"), "w")); print("split:", {k: len(v) for k, v in split.items()})
lab_sub = {"frame_number_map": {cid: (i * T, (i + 1) * T) for i, cid in enumerate(ids)}, "label_array": np.concatenate([L[:, fm[cid][0]:fm[cid][1]] for cid in ids], 1), "vocabulary": vocab, "task_type": lab["task_type"]}
np.save(os.path.join(args.out_dir, "mouse_submission_labels_subset.npy"), lab_sub, allow_pickle=True); print("labels subset:", lab_sub["label_array"].shape)
