"""Frame embeddings of a trained ERR for the MABe22 mouse-triplet representation benchmark.

MABe22 (Sun et al., ICML 2023) scores a representation by training a linear readout (ridge) on frozen per-frame
embeddings for 13 hidden tasks.  This script turns a trained ERR into such embeddings: every 30-Hz frame of every
sequence gets one vector read from the model's internal state.

Input  : a MABe sequences file  {"sequences": {id: {"keypoints": (1800, 3, 12, 2) px}}}  (e.g. mouse_triplet_test.npy)
Output : <out>.npy  float16 memmap (sum of clip lengths, F)  +  <out>.json  {ids, offsets, feature names}
         rows are ordered by the label file's frame_number_map (if --labels is given) else by sequence order.

Modes (--mode):
  window  (default = the training regime)  for frame f the model observes the 10 frames f-36, f-32, ..., f
          (1.33 s at 7.5 Hz, left-padded by repeating the first frame) from a zero state; the embedding is the state
          after the last observed frame.
  long    the model is run teacher-forced over the whole sequence at 7.5 Hz (4 phase offsets), never reset.

Features per frame (F = 216 for both 36- and 12-node models):
  h_mean[m]      3 x D   node memory h_i averaged over mouse m's keypoints          (order: mouse 0, 1, 2)
  share[m->m']   6       big-graph attention share of m on m' (rows sum to 1)        'who'
  p_contact[m]   3       1 - P(none) of mouse m's nose (intention head)              'when'
  intent[m->m']  6       intention mass of m's nose on the nodes of m'               'who / where'
  surprise[m]    3       |x_{t+1} - (x_t + dx_mean)| averaged over m's keypoints (one-step error, arena units)
  speed_hat[m]   3       |dx_mean| averaged over m's keypoints
  tracked[m]     3       1 if mouse m is tracked in the frame
Untracked keypoints (stored as (0,0)) are linearly interpolated in time inside the sequence; a mouse that is never
tracked sits at the arena centre.
Usage (from the repository root):
  python eval/mabe22/embed.py --ck checkpoints/err_36kp.pth.tar --npy data/mabe22/mouse_triplet_test.npy \
      --labels data/mabe22/mouse_triplets_test_labels.npy --out embeddings/mabe22/err_36kp
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from err.checkpoint import load_checkpoint                                                 # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--ck", required=True); p.add_argument("--npy", required=True); p.add_argument("--out", required=True)
p.add_argument("--labels", default=None, help="labels npy with frame_number_map: rows follow this map")
p.add_argument("--mode", default="window", choices=["window", "long"])
p.add_argument("--fs", type=int, default=4, help="subsampling (30 Hz / fs)"); p.add_argument("--obs", type=int, default=10)
p.add_argument("--arena_px", type=float, default=710.0)
p.add_argument("--batch", type=int, default=0, help="sequences per batch (0 = auto)")
p.add_argument("--max_seqs", type=int, default=0)
args = p.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KP4 = [0, 3, 6, 9]


def impute(kp):
    """kp (T, 3, 12, 2) int px -> float normalised, untracked (0,0) entries interpolated in time; returns (x, tracked (T,3))."""
    x = kp.astype(np.float32) / args.arena_px
    miss = np.abs(kp).sum(-1) == 0                                                         # T,3,12
    tracked = ~miss.all(2)                                                                 # T,3
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
            else:                                                                          # keypoint never tracked
                src = x[:, m, 6] if not miss[:, m, 6].all() else None
                x[:, m, k] = src if src is not None else 0.5
    return x, tracked.astype(np.float32)


def share_of(w, kp):
    """w (B, N, N) inter attention -> (B, 3, 3) share of each mouse's inter-mouse attention on each other mouse."""
    B = w.shape[0]
    v = w.view(B, 3, kp, 3, kp).sum((2, 4))
    return v / v.sum(2, keepdim=True).clamp(min=1e-9)


@torch.no_grad()
def features(m, kp, buf_steps, x_next):
    """buf_steps (B, N, L, 2) teacher-forced input; x_next (B, N, 2) truth after the last step (for surprise).
    Returns (B, F) features read after the last step."""
    B, N, L, _ = buf_steps.shape
    types = m.types(); mask = torch.ones(B, N, device=dev)
    h = torch.zeros(B, N, m.cell.D, device=dev); s = torch.zeros(B, N, N, m.cell.D, device=dev)
    buf = buf_steps[:, :, :1].expand(-1, -1, m.cell.k, -1).contiguous()
    for t in range(L):
        buf = torch.cat([buf[:, :, 1:], buf_steps[:, :, t:t + 1]], 2)
        dx, dx_mean, h, s, aux = m.cell(buf, h, s, types, mask, noise_on=False)
    return _collect(m, kp, h, aux, dx_mean, buf_steps[:, :, -1], x_next), h, s


def _collect(m, kp, h, aux, dx_mean, x_last, x_next):
    B, N, D = h.shape
    nose = torch.arange(3, device=dev) * kp
    h_mean = h.view(B, 3, kp, D).mean(2).reshape(B, 3 * D)
    sh = share_of(aux["w_inter"], kp); off = ~torch.eye(3, dtype=torch.bool, device=dev)
    share = sh[:, off]                                                                      # B,6
    P = aux["goal"]                                                                         # B,N,N+1
    Pn = P[:, nose]                                                                         # B,3,N+1
    p_contact = 1 - Pn[..., N]
    intent = Pn[..., :N].view(B, 3, 3, kp).sum(-1)[:, off]                                  # B,6
    err = (x_next - (x_last + dx_mean)).norm(dim=-1).view(B, 3, kp).mean(2)                 # B,3
    spd = dx_mean.norm(dim=-1).view(B, 3, kp).mean(2)
    return torch.cat([h_mean, share, p_contact, intent, err, spd], -1)


@torch.no_grad()
def embed_long(m, kp, X):
    """X (B, T30, N, 2) normalised, full sequences -> (B, T30, F)"""
    B, T30, N, _ = X.shape; fs = args.fs
    out = torch.zeros(B, T30, FEAT, device=dev)
    types = m.types(); mask = torch.ones(B, N, device=dev)
    for phase in range(fs):
        xs = X[:, phase::fs].permute(0, 2, 1, 3).contiguous()                               # B,N,L,2
        L = xs.shape[2]
        h = torch.zeros(B, N, m.cell.D, device=dev); s = torch.zeros(B, N, N, m.cell.D, device=dev)
        buf = xs[:, :, :1].expand(-1, -1, m.cell.k, -1).contiguous()
        for t in range(L):
            buf = torch.cat([buf[:, :, 1:], xs[:, :, t:t + 1]], 2)
            dx, dx_mean, h, s, aux = m.cell(buf, h, s, types, mask, noise_on=False)
            x_next = xs[:, :, min(t + 1, L - 1)]
            out[:, phase + fs * t] = _collect(m, kp, h, aux, dx_mean, xs[:, :, t], x_next)
    return out


@torch.no_grad()
def embed_window(m, kp, X):
    """X (B, T30, N, 2) -> (B, T30, F): for every frame, the state after observing the obs-step window ending there."""
    B, T30, N, _ = X.shape; fs, L = args.fs, args.obs
    out = torch.zeros(B, T30, FEAT, device=dev)
    for phase in range(fs):
        xs = X[:, phase::fs]                                                                # B,Tp,N,2
        Tp = xs.shape[1]
        pad = torch.cat([xs[:, :1].expand(-1, L - 1, -1, -1), xs], 1)                        # left pad by repeating frame 0
        win = pad.unfold(1, L, 1)                                                            # B,Tp,N,2,L
        win = win.permute(0, 1, 2, 4, 3).reshape(B * Tp, N, L, 2)                            # windows ending at every step
        nxt = torch.cat([xs[:, 1:], xs[:, -1:]], 1).reshape(B * Tp, N, 2)
        f, _, _ = features(m, kp, win, nxt)
        out[:, phase + fs * torch.arange(Tp, device=dev)] = f.view(B, Tp, FEAT)
    return out


def main():
    global FEAT
    m, ck = load_checkpoint(args.ck, device=dev); kp = m.kp
    assert m.M == 3, "MABe22 embeddings need a 3-mouse checkpoint (all12 / all36)"
    D = m.cell.D; FEAT = 3 * D + 6 + 3 + 6 + 3 + 3
    names = [f"h{m_}_{d}" for m_ in range(3) for d in range(D)] + [f"share_{a}{b}" for a in range(3) for b in range(3) if a != b] + \
            [f"pcontact_{m_}" for m_ in range(3)] + [f"intent_{a}{b}" for a in range(3) for b in range(3) if a != b] + \
            [f"surprise_{m_}" for m_ in range(3)] + [f"speed_hat_{m_}" for m_ in range(3)] + [f"tracked_{m_}" for m_ in range(3)]
    raw = np.load(args.npy, allow_pickle=True).item()["sequences"]
    if args.labels:
        fm = np.load(args.labels, allow_pickle=True).item()["frame_number_map"]
        ids = sorted(fm.keys(), key=lambda k: fm[k][0]); offsets = [fm[k] for k in ids]
    else:
        ids = list(raw.keys()); offsets = []; o = 0
        for k in ids:
            n = len(raw[k]["keypoints"]); offsets.append((o, o + n)); o += n
    if args.max_seqs:
        ids, offsets = ids[:args.max_seqs], offsets[:args.max_seqs]
    total = offsets[-1][1] if not args.max_seqs else sum(e - s for s, e in offsets)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    F_out = FEAT + 3
    emb = np.lib.format.open_memmap(args.out + ".npy", mode="w+", dtype=np.float16, shape=(total, F_out))
    B = args.batch or ((4 if kp == 12 else 12) if args.mode == "window" else (64 if kp == 12 else 192))
    t0 = time.time(); done = 0
    for i0 in range(0, len(ids), B):
        chunk = ids[i0:i0 + B]
        xs, trs = zip(*(impute(np.asarray(raw[k]["keypoints"])) for k in chunk))
        X = np.stack(xs)                                                                    # B,T,3,12,2
        if kp == 4:
            X = X[:, :, :, KP4]
        X = torch.tensor(X.reshape(X.shape[0], X.shape[1], 3 * kp, 2), device=dev)
        out = (embed_window if args.mode == "window" else embed_long)(m, kp, X).cpu().numpy()
        for j, k in enumerate(chunk):
            s, e = offsets[i0 + j] if not args.max_seqs else (done, done + out.shape[1])
            emb[s:e] = np.concatenate([out[j], trs[j]], -1).astype(np.float16); done += out.shape[1]
        if (i0 // B) % 20 == 0:
            el = time.time() - t0; print(f"  {i0 + len(chunk)}/{len(ids)} sequences, {el:.0f}s, eta {el / (i0 + len(chunk)) * (len(ids) - i0 - len(chunk)):.0f}s", flush=True)
    emb.flush()
    json.dump({"ids": ids, "offsets": [list(map(int, o)) for o in offsets], "features": names, "ck": args.ck, "mode": args.mode, "kp": kp},
              open(args.out + ".json", "w"))
    print(f"wrote {args.out}.npy {emb.shape} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
