"""MABe22 mouse triplets -> event-centred, sequence-split training windows (the ERR training set).

Input : the MABe22 trajectory-track training file  mouse_triplet_train.npy (called user_train_r1.npy in the research code)
        {"sequences": {id: {"keypoints": (1800, 3, 12, 2) px, "annotations": (2, 1800)}}}, 1600 sequences of 60 s at 30 Hz,
        710-px arena (CaltechDATA record 8kdn3-95j37).  Other sources (ResiFuse DLC poses, official 'submission' keypoints)
        are registered in DATASET_META after data_prep/resifuse_convert_poses.py.
Output: data/windows/dataset_social_<dataset>_w20_fs4_seqsplit.npz          4-kp windows (nose, neck, center_back, tail_base)
        data/windows/dataset_social_<dataset>_w20_fs4_seqsplit_12kpfull.npz  the same windows with all 12 keypoints

    {split}_data (N, 20, 3, 4 | 12, 2)   coordinates / arena_px, 7.5 Hz (fs = 4), 10 observed + 10 predicted steps

Two design decisions:
  1. SPLIT BY SEQUENCE.  Every 60-s sequence goes to exactly one split (80/10/10, seed 42), so no frame is shared
     between train and test (sliding windows shuffled window-wise would share 89 % of the test frames with train).
  2. EVENT-CENTRED WINDOWS.  Each social event between an ordered pair (focal = target mouse a, partner = acting mouse b)
     gives one window in which the event frame falls at a random FUTURE step 1..6 (0.13-0.8 s after the last observed
     frame), i.e. where the partner's trajectory depends on the focal mouse.  Background windows without any event are
     sampled 1:1 per sequence (activity filter) so training is mixed and evaluation can be stratified.  Two events of
     different pairs can draw the same start frame; they then give two windows with the same frames and different event
     metadata (as in the data of the released checkpoints; --dedup_starts 1 shifts the second one instead).
With the defaults this script rebuilds, byte for byte, the window files the released checkpoints were trained and tested on
(train 31,976 / val 3,761 / test 4,203; the files in data/windows).

Per-window metadata (used by the trainer's event-stratified metrics):
    {split}_event_type   int32 bitmask (0 = background)
                         1 approach   partner's nose enters 80 px of focal's center_back while closing > 5 px/step
                         2 contact    partner's nose enters 40 px of any focal body kp
                         4 anogenital partner's nose enters 50 px of focal's tail_base
                         8 nose_nose  noses enter 50 px (focal = lower index)
                        16 following  partner starts moving (> 4 px/step) toward where focal was 0.5 s ago, both moving, < 200 px
                        32 chase      annotated chase bout starts (pair unknown: -1)
    {split}_focal / _partner   int32 mouse indices (-1 for background / chase)
    {split}_event_step   int32 index of the event frame inside the window (10..19), -1 if none
    {split}_min_nb_dist  float32 min partner-nose -> any-body distance over the window (px)
    {split}_seq_id / _t_start   sequence index (into meta_seq_keys) and window start (7.5-Hz frame)
    {split}_lights / _chase / _activity   the two public labels and the mean centre step

Usage (from the repository root):
    python data_prep/mabe_event_windows.py                          # r1: 4-kp + 12-kp twins (~10 min)
    python data_prep/mabe_event_windows.py --no_12kp --max_seqs 200  # quick check
    python data_prep/mabe_event_windows.py --dataset dlcrf --split_json data/resifuse/code/splits/mabe_pretrain_split.json --tag rfpre
"""

import argparse
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME_RATE = 30
N_MICE = 3
KP4 = [0, 3, 6, 9]                      # nose, neck, center_back (tail_1), tail_base (tail_4) in the MABe 12-kp skeleton
KP4_NAMES = ["nose", "neck", "center_back", "tail_base"]
NOSE, NECK, CB, TAIL = 0, 3, 6, 9
DATASET_META = {"r1": {"file": os.path.join(ROOT, "data", "mabe22", "mouse_triplet_train.npy"), "arena_px": 710},   # = user_train_r1.npy
                # MABe22 'submission' clips (1810 with video): ResiFuse DeepLabCut poses and the official keypoints of the same
                # clips, both rescaled to the 710-px body-length space by data_prep/resifuse_convert_poses.py
                "dlc": {"file": os.path.join(ROOT, "data", "resifuse", "mouse_submission_dlc.npy"), "arena_px": 710},
                "dlcrf": {"file": os.path.join(ROOT, "data", "resifuse", "mouse_submission_dlcrf.npy"), "arena_px": 710},
                "sub": {"file": os.path.join(ROOT, "data", "resifuse", "mouse_submission_sub.npy"), "arena_px": 710}}

EV_APPROACH, EV_CONTACT, EV_ANOGEN, EV_NOSENOSE, EV_FOLLOW, EV_CHASE = 1, 2, 4, 8, 16, 32
EV_NAMES = {1: "approach", 2: "contact", 4: "anogenital", 8: "nose_nose", 16: "following", 32: "chase"}


# ───────────────────────────── event detection ─────────────────────────────

def onsets(cond):
    """indices where a boolean series switches False -> True"""
    return np.where(cond[1:] & ~cond[:-1])[0] + 1


def detect_events(kp, ann, p):
    """kp (T,3,12,2) px @7.5Hz; ann (2,T).  Returns list of (t, a, b, type_bit)."""
    T = kp.shape[0]
    nose, cb, tail = kp[:, :, NOSE], kp[:, :, CB], kp[:, :, TAIL]
    body = kp[:, :, KP4]                                                 # T,3,4,2
    ev = []
    for a in range(N_MICE):
        for b in range(N_MICE):
            if a == b:
                continue
            d_ncb = np.linalg.norm(nose[:, b] - cb[:, a], axis=-1)
            d_nb = np.linalg.norm(nose[:, b, None] - body[:, a], axis=-1).min(-1)
            d_nt = np.linalg.norm(nose[:, b] - tail[:, a], axis=-1)
            d_cc = np.linalg.norm(cb[:, b] - cb[:, a], axis=-1)
            closing = np.concatenate([[0.0], -np.diff(d_ncb)])
            for t in onsets(d_ncb < p.approach_r):
                if closing[t] > p.approach_v:
                    ev.append((t, a, b, EV_APPROACH))
            ev += [(t, a, b, EV_CONTACT) for t in onsets(d_nb < p.contact_r)]
            ev += [(t, a, b, EV_ANOGEN) for t in onsets(d_nt < p.anogen_r)]
            if a < b:
                d_nn = np.linalg.norm(nose[:, b] - nose[:, a], axis=-1)
                ev += [(t, a, b, EV_NOSENOSE) for t in onsets(d_nn < p.nosenose_r)]
            vb = np.concatenate([np.zeros((1, 2)), np.diff(cb[:, b], axis=0)])
            va = np.concatenate([np.zeros((1, 2)), np.diff(cb[:, a], axis=0)])
            to_a_past = np.roll(cb[:, a], p.follow_lag, axis=0) - cb[:, b]
            cosang = (vb * to_a_past).sum(-1) / (np.linalg.norm(vb, axis=-1) * np.linalg.norm(to_a_past, axis=-1) + 1e-6)
            follow = ((cosang > 0.9) & (np.linalg.norm(vb, axis=-1) > 4) & (np.linalg.norm(va, axis=-1) > 4)
                      & (d_cc < 200))
            follow[:p.follow_lag] = False
            ev += [(t, a, b, EV_FOLLOW) for t in onsets(follow)]
    ev += [(t, -1, -1, EV_CHASE) for t in onsets(ann[0] > 0.5)]
    return ev


def merge_events(ev, tol=2):
    """Events of the same ordered pair within +-tol steps are merged into one anchor (earliest t) carrying a bitmask."""
    ev.sort(key=lambda e: (e[1], e[2], e[0]))
    merged = []
    for t, a, b, bit in ev:
        if merged and merged[-1][1] == a and merged[-1][2] == b and t - merged[-1][0] <= tol:
            merged[-1][3] |= bit
        else:
            merged.append([t, a, b, bit])
    return merged


# ───────────────────────────── windows ─────────────────────────────

def window_activity(win4, arena_px):
    """mean center_back step of the 3 mice / arena"""
    return np.linalg.norm(np.diff(win4[:, :, 2, :], axis=0), axis=-1).mean() / arena_px


def min_nose_body(kp_win):
    nose = kp_win[:, :, NOSE]; body = kp_win[:, :, KP4]
    best = np.inf
    for a in range(N_MICE):
        for b in range(N_MICE):
            if a != b:
                best = min(best, np.linalg.norm(nose[:, b, None] - body[:, a], axis=-1).min())
    return float(best)


def build_sequence(kp, ann, valid, arena_px, p, rng, seq_id):
    """returns the list of window dicts of one sequence"""
    T = kp.shape[0]; W, OBS = p.window_size, p.obs_len
    events = merge_events(detect_events(kp, ann, p))
    out = []
    used_starts = set()
    for t, a, b, bits in events:
        k = int(rng.integers(p.min_future, p.max_future + 1))       # event at future step k
        start = t - (OBS - 1) - k
        # --dedup_starts 1 (NOT used for the released data): when two events (other pair / type) land on the same start frame,
        # shift by a step inside the allowed future range instead of writing a second window with the same frames
        if p.dedup_starts and start in used_starts:
            for alt in range(p.min_future, p.max_future + 1):
                if t - (OBS - 1) - alt not in used_starts:
                    start = t - (OBS - 1) - alt
                    break
            else:
                continue
        if start < 0 or start + W > T or not valid[start:start + W].all():
            continue
        used_starts.add(start)
        out.append(dict(start=start, type=bits, focal=a, partner=b, step=t - start))
    n_ev = len(out)
    # background: sliding windows with no event anchor inside, activity filter
    anchors = np.zeros(T, bool)
    for t, *_ in events:
        anchors[t] = True
    bg = []
    for start in range(0, T - W, p.bg_stride):
        if not valid[start:start + W].all() or anchors[start:start + W].any():
            continue
        if window_activity(kp[start:start + W][:, :, KP4], arena_px) < p.static_thresh:
            continue
        bg.append(start)
    rng.shuffle(bg)
    for start in bg[:int(round(n_ev * p.bg_ratio))]:
        out.append(dict(start=start, type=0, focal=-1, partner=-1, step=-1))
    for w in out:
        s = w["start"]; win = kp[s:s + W]
        w.update(seq_id=seq_id, lights=int(ann[1][s:s + W].mean() > 0.5), chase=int(ann[0][s:s + W].sum() > 0),
                 activity=float(window_activity(win[:, :, KP4], arena_px)), min_nb=min_nose_body(win),
                 data4=(win[:, :, KP4] / arena_px).astype(np.float32),
                 data12=(win / arena_px).astype(np.float32) if p.save_12kp else None)
    return out


def to_arrays(ws, key12):
    arr = {"data": np.stack([w["data4"] for w in ws]) if ws else np.empty((0, 20, 3, 4, 2), np.float32)}
    if key12:
        arr["data"] = np.stack([w["data12"] for w in ws]) if ws else np.empty((0, 20, 3, 12, 2), np.float32)
    for k, src, dt in [("lights", "lights", np.int32), ("chase", "chase", np.int32), ("activity", "activity", np.float32),
                       ("event_type", "type", np.int32), ("focal", "focal", np.int32), ("partner", "partner", np.int32),
                       ("event_step", "step", np.int32), ("min_nb_dist", "min_nb", np.float32),
                       ("seq_id", "seq_id", np.int32), ("t_start", "start", np.int32)]:
        arr[k] = np.array([w[src] for w in ws], dtype=dt)
    return arr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="r1", choices=list(DATASET_META))
    ap.add_argument("--raw", default=None, help="sequences npy to read instead of DATASET_META[dataset]['file'] (same arena_px)")
    ap.add_argument("--fs", type=int, default=4)
    ap.add_argument("--window_size", type=int, default=20)
    ap.add_argument("--obs_len", type=int, default=10)
    ap.add_argument("--min_future", type=int, default=1, help="event lands at future step >= this")
    ap.add_argument("--max_future", type=int, default=6)
    ap.add_argument("--approach_r", type=float, default=80.0)
    ap.add_argument("--approach_v", type=float, default=5.0)
    ap.add_argument("--contact_r", type=float, default=40.0)
    ap.add_argument("--anogen_r", type=float, default=50.0)
    ap.add_argument("--nosenose_r", type=float, default=50.0)
    ap.add_argument("--follow_lag", type=int, default=4)
    ap.add_argument("--bg_ratio", type=float, default=1.0, help="background windows per event window (per sequence)")
    ap.add_argument("--bg_stride", type=int, default=10)
    ap.add_argument("--static_thresh", type=float, default=2e-3)
    ap.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    ap.add_argument("--split_json", default=None, help="fixed clip split json ({train,val,test} lists or {splits: {...}}); overrides --split")
    ap.add_argument("--dedup_starts", type=int, default=0, help="1 = shift event windows that share a start frame (default 0 = the data of the "
                                                                "released checkpoints, where such events give two windows with the same frames)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--no_12kp", action="store_true")
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "data", "windows"))
    ap.add_argument("--tag", default="social")
    p = ap.parse_args(); p.save_12kp = not p.no_12kp
    meta = DATASET_META[p.dataset]
    os.makedirs(p.out_dir, exist_ok=True)
    t0 = time.time()
    raw = np.load(p.raw or meta["file"], allow_pickle=True).item()["sequences"]
    keys = sorted(raw.keys())
    if p.max_seqs:
        keys = keys[:p.max_seqs]
    print(f"loaded {len(keys)} sequences in {time.time() - t0:.0f}s")

    # sequence-level split
    rng = np.random.default_rng(p.seed)
    split_of = {}
    if p.split_json:                                  # externally fixed clip split, e.g. ResiFuse splits/mabe_pretrain_split.json
        sj = json.load(open(p.split_json)); sj = sj.get("splits", sj)
        where = {cid: s for s in ("train", "val", "test") for cid in sj.get(s, [])}
        for si, k in enumerate(keys):
            if k in where:
                split_of[si] = where[k]
        print(f"split from {p.split_json}: " + ", ".join(f"{s} {sum(v == s for v in split_of.values())}" for s in ("train", "val", "test")) +
              f"; {len(keys) - len(split_of)} clips not in the split file are excluded")
    else:
        perm = rng.permutation(len(keys))
        n_tr, n_va = int(len(keys) * p.split[0]), int(len(keys) * p.split[1])
        for i, si in enumerate(perm):
            split_of[si] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")

    windows = {"train": [], "val": [], "test": []}
    for si, k in enumerate(keys):
        if si not in split_of:
            continue
        kp = np.asarray(raw[k]["keypoints"], dtype=np.float32)[::p.fs]
        ann = np.asarray(raw[k]["annotations"])[:, ::p.fs]
        valid = (np.abs(kp[:, :, KP4, :]).sum(-1) >= 1).all((1, 2))          # untracked keypoints are stored as (0, 0)
        windows[split_of[si]] += build_sequence(kp, ann, valid, meta["arena_px"], p, rng, si)
        if si % 200 == 0:
            print(f"  {si}/{len(keys)} sequences, {sum(len(v) for v in windows.values())} windows", flush=True)

    base = f"{p.tag}_{p.dataset}_w{p.window_size}_fs{p.fs}_seqsplit"
    for key12, suffix in [(False, ""), (True, "_12kpfull")]:
        if key12 and not p.save_12kp:
            continue
        arrays = {}
        for split, ws in windows.items():
            for kk, v in to_arrays(ws, key12).items():
                arrays[f"{split}_{kk}"] = v
            arrays[f"meta_split_seq_ids_{split}"] = np.array(sorted(si for si, s in split_of.items() if s == split), np.int32)
        arrays.update(meta_window_size=np.array(p.window_size), meta_stride=np.array(-1), meta_fs=np.array(p.fs),
                      meta_hz=np.array(FRAME_RATE / p.fs), meta_n_nodes=np.array(N_MICE * (12 if key12 else 4)),
                      meta_kp_indices=np.array(list(range(12)) if key12 else KP4),
                      meta_kp_names=np.array([f"kp{i}" for i in range(12)] if key12 else KP4_NAMES),
                      meta_seq_keys=np.array(keys), meta_event_bits=np.array([f"{b}:{n}" for b, n in EV_NAMES.items()]),
                      meta_args=np.array(str(vars(p))))
        out = os.path.join(p.out_dir, f"dataset_{base}{suffix}.npz")
        np.savez_compressed(out, **arrays)
        print(f"saved {out} ({os.path.getsize(out) / 2**20:.0f} MB)")

    print("\nsummary")
    for split, ws in windows.items():
        types = np.array([w["type"] for w in ws])
        line = f"  {split:5s} windows={len(ws):6d} background={int((types == 0).sum()):6d} "
        line += " ".join(f"{n}={int(((types & b) > 0).sum())}" for b, n in EV_NAMES.items())
        print(line)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
