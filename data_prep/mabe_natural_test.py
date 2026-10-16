"""Natural-distribution test windows from the TEST sequences of the event-centred split (no event centring, no filtering).

Sliding windows (stride --stride steps at 7.5 Hz) over the held-out test sequences of
data/windows/dataset_social_r1_w20_fs4_seqsplit.npz, same frame rate / window / normalisation, 4-kp and 12-kp twins.
Use them with err/train.py --test --npz <this file> to score a trained model on all windows instead of the
event-centred ones.  Every window is annotated with the event detector so it can be stratified:
    test_event_type     bitmask of event onsets ANYWHERE in the window (0 = none)
    test_event_step     step (0..19) of the first onset, -1 if none;  < 10 = inside the observed segment
    test_focal / _partner  pair of the first onset (-1 if none)
    test_n_events       number of onsets in the window
    test_in_contact_obs 1 if some nose is within 40 px of another mouse's body during the observed segment
Usage (from the repository root):  python data_prep/mabe_natural_test.py --stride 5
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from mabe_event_windows import DATASET_META, KP4, KP4_NAMES, EV_NAMES, N_MICE, NOSE, detect_events, merge_events, min_nose_body, window_activity  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--social_npz", default=os.path.join(ROOT, "data", "windows", "dataset_social_r1_w20_fs4_seqsplit.npz"))
ap.add_argument("--dataset", default="r1", choices=list(DATASET_META))
ap.add_argument("--raw", default=None, help="sequences npy to read instead of DATASET_META[dataset]['file']")
ap.add_argument("--stride", type=int, default=5)
ap.add_argument("--out_tag", default="natural")
ap.add_argument("--out_dir", default=os.path.join(ROOT, "data", "windows"))
p = ap.parse_args()
# detector thresholds = mabe_event_windows defaults
p.fs, p.window_size, p.obs_len = 4, 20, 10
p.approach_r, p.approach_v, p.contact_r, p.anogen_r, p.nosenose_r, p.follow_lag = 80.0, 5.0, 40.0, 50.0, 50.0, 4

t0 = time.time()
z = np.load(p.social_npz)
keys = [str(k) for k in z["meta_seq_keys"]]; test_ids = z["meta_split_seq_ids_test"]
meta = DATASET_META[p.dataset]; arena = meta["arena_px"]
raw = np.load(p.raw or meta["file"], allow_pickle=True).item()["sequences"]
print(f"loaded raw in {time.time() - t0:.0f}s; test sequences {len(test_ids)}")
W = p.window_size
rows = []
for si in test_ids:
    k = keys[si]; kp = np.asarray(raw[k]["keypoints"], dtype=np.float32)[::p.fs]; ann = np.asarray(raw[k]["annotations"])[:, ::p.fs]
    valid = (np.abs(kp[:, :, KP4, :]).sum(-1) >= 1).all((1, 2))
    events = merge_events(detect_events(kp, ann, p))
    T = kp.shape[0]
    for start in range(0, T - W + 1, p.stride):
        if not valid[start:start + W].all():
            continue
        win = kp[start:start + W]
        ev_in = [(t - start, a, b, bits) for t, a, b, bits in events if start <= t < start + W and a >= 0]
        ev_in.sort()
        bits_all = 0
        for _, _, _, bits in ev_in:
            bits_all |= bits
        first = ev_in[0] if ev_in else (-1, -1, -1, 0)
        nose = win[:p.obs_len, :, NOSE]; body = win[:p.obs_len, :, KP4]
        inc = any(np.linalg.norm(nose[:, b, None] - body[:, a], axis=-1).min() < p.contact_r for a in range(N_MICE) for b in range(N_MICE) if a != b)
        rows.append(dict(data4=(win[:, :, KP4] / arena).astype(np.float32), data12=(win / arena).astype(np.float32),
                         event_type=bits_all, event_step=first[0], focal=first[1], partner=first[2], n_events=len(ev_in),
                         in_contact_obs=int(inc), activity=float(window_activity(win[:, :, KP4], arena)), min_nb=min_nose_body(win), seq_id=int(si), t_start=start,
                         lights=int(ann[1][start:start + W].mean() > 0.5), chase=int(ann[0][start:start + W].sum() > 0)))
print(f"{len(rows)} natural windows")
for key12, suffix in ((False, ""), (True, "_12kpfull")):
    arr = {"test_data": np.stack([r["data12" if key12 else "data4"] for r in rows])}
    for kk, dt in (("event_type", np.int32), ("event_step", np.int32), ("focal", np.int32), ("partner", np.int32), ("n_events", np.int32), ("in_contact_obs", np.int32),
                   ("activity", np.float32), ("min_nb", np.float32), ("seq_id", np.int32), ("t_start", np.int32), ("lights", np.int32), ("chase", np.int32)):
        arr["test_" + ("min_nb_dist" if kk == "min_nb" else kk)] = np.array([r[kk] for r in rows], dtype=dt)
    arr.update(meta_window_size=np.array(W), meta_stride=np.array(p.stride), meta_fs=np.array(p.fs), meta_hz=np.array(30 / p.fs),
               meta_n_nodes=np.array(N_MICE * (12 if key12 else 4)), meta_kp_indices=np.array(list(range(12)) if key12 else KP4),
               meta_kp_names=np.array([f"kp{i}" for i in range(12)] if key12 else KP4_NAMES), meta_seq_keys=z["meta_seq_keys"],
               meta_split_seq_ids_test=test_ids, meta_event_bits=np.array([f"{b}:{n}" for b, n in EV_NAMES.items()]), meta_args=np.array(str(vars(p))))
    out = os.path.join(p.out_dir, f"dataset_{p.out_tag}_{p.dataset}_w20_fs4_test{suffix}.npz")
    np.savez_compressed(out, **arr); print(f"saved {out} ({os.path.getsize(out) / 2**20:.0f} MB)")
et, es, pa = np.array([r["event_type"] for r in rows]), np.array([r["event_step"] for r in rows]), np.array([r["partner"] for r in rows])
print(f"none {int((et == 0).sum())}  event in observed segment (step<10) {int(((es >= 0) & (es < 10)).sum())}  in future (step>=10) {int((es >= 10).sum())}  "
      f"in contact during obs {int(sum(r['in_contact_obs'] for r in rows))}")
print("by type:", {n: int(((et & b) > 0).sum()) for b, n in EV_NAMES.items()})
