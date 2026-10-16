"""Frame embeddings of an ERR checkpoint on CalMS21 task 1 (2 mice x 7 keypoints, resident-intruder assay).

Two uses:
  --nodes 4  ZERO-SHOT TRANSFER of the MABe22-trained 4-keypoint model (checkpoints/err_12kp.pth.tar): the CalMS21 keypoints
             (nose, left ear, right ear, neck, left hip, right hip, tail base) are mapped onto the MABe legacy skeleton
             nose, neck, centre_back = midpoint of the hips, tail_base.  The model is built for 2 entities (8 nodes) and
             loaded from the 3-mouse checkpoint: every parameter is per node / per edge, only the entity buffers differ.
             Coordinates are scaled so that the median body length (nose - tail base) of the training videos equals the MABe
             training value (70.4 px / 710 = 0.0992 units), which puts speeds and distances on the scale the model was trained on.
  --nodes 7  the CalMS21-trained 14-node model (checkpoints/err_calms14.pth.tar) on all 7 keypoints, at its own training scale.
Window mode as in mabe22/embed.py: for every 30-Hz frame, the state after observing the 10 frames f-36 ... f (7.5 Hz).
Features: h_mean[m] (2 x D), p_contact[m] (2), intent of m's nose on the other mouse's parts (2 x KP), surprise[m] (2),
          speed_hat[m] (2); plus a hand-crafted geometry block (30 dims, always on the 4-kp skeleton) for the geometry baseline.
Output: <out>_<split>.npy float16 (frames, F), <out>_<split>_labels.npy int8 (attack 0, investigation 1, mount 2, other 3),
        <out>_<split>.json (ids, offsets, names, scale).
Usage (from the repository root):
  python eval/calms21/embed.py --ck checkpoints/err_12kp.pth.tar --out embeddings/calms21/err_zeroshot              # zero-shot
  python eval/calms21/embed.py --ck checkpoints/err_calms14.pth.tar --nodes 7 --out embeddings/calms21/err_calms14    # CalMS21-trained
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
p.add_argument("--ck", required=True); p.add_argument("--out", required=True)
p.add_argument("--json_dir", default=os.path.join(ROOT, "data", "calms21", "task1_classic_classification"))
p.add_argument("--fs", type=int, default=4); p.add_argument("--obs", type=int, default=10)
p.add_argument("--body_len", type=float, default=70.4 / 710.0, help="target median body length in model units (zero-shot scaling)")
p.add_argument("--max_seqs", type=int, default=0)
p.add_argument("--nodes", type=int, default=4, choices=[4, 7], help="4 = MABe-trained 4-kp model on the mapped skeleton; 7 = CalMS21-trained 14-node model on all 7 keypoints")
p.add_argument("--scale_px", type=float, default=0.0, help="px per model unit (0 = from --body_len on the training videos; a CalMS21-trained checkpoint uses its arena_px)")
args = p.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M, KP = 2, args.nodes
KP7 = ["nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip", "tail_base"]


def to_nodes(kp):
    """kp (T, 2 mice, 2 xy, 7 parts) px -> (T, 2, 4, 2): nose, neck, hip midpoint, tail base (the geometry block always uses this)"""
    x = np.transpose(kp, (0, 1, 3, 2)).astype(np.float32)                                  # T,2,7,2
    cb = 0.5 * (x[:, :, 4] + x[:, :, 5])
    return np.stack([x[:, :, 0], x[:, :, 3], cb, x[:, :, 6]], 2)


def to_nodes7(kp):
    """kp (T, 2, 2 xy, 7) px -> (T, 2, 7, 2): all CalMS21 keypoints in their native order (the 14-node model's input)"""
    return np.transpose(kp, (0, 1, 3, 2)).astype(np.float32)


def geometry(x):
    """hand-crafted block on the 4-kp nodes (T, 2, 4, 2) in model units: per mouse speed cb / nose, body length, turning;
    pair: nose-nose, nose0-tail1, nose1-tail0, cb-cb, closing speed, facing cosines (2); 1-s smoothed copies."""
    d = lambda a, b: np.linalg.norm(a - b, axis=-1)
    vel = np.gradient(x, axis=0)
    head = x[:, :, 0] - x[:, :, 3]; hn = head / (np.linalg.norm(head, axis=-1, keepdims=True) + 1e-6)
    ang = np.arctan2(hn[..., 1], hn[..., 0]); turn = np.abs(np.gradient(np.unwrap(ang, axis=0), axis=0))
    f = []
    for m in range(2):
        f += [np.linalg.norm(vel[:, m, 2], axis=-1), np.linalg.norm(vel[:, m, 0], axis=-1), d(x[:, m, 0], x[:, m, 3]), turn[:, m]]
    dc = d(x[:, 0, 2], x[:, 1, 2]); to1 = x[:, 1, 2] - x[:, 0, 2]; to1 = to1 / (np.linalg.norm(to1, axis=-1, keepdims=True) + 1e-6)
    f += [d(x[:, 0, 0], x[:, 1, 0]), d(x[:, 0, 0], x[:, 1, 3]), d(x[:, 1, 0], x[:, 0, 3]), dc, np.gradient(dc), (hn[:, 0] * to1).sum(-1), -(hn[:, 1] * to1).sum(-1)]
    F = np.stack(f, 1).astype(np.float32)
    h = 15; P = np.concatenate([np.repeat(F[:1], h, 0), F, np.repeat(F[-1:], h, 0)], 0).astype(np.float64)
    c = np.cumsum(np.concatenate([np.zeros((1, F.shape[1])), P], 0), 0); Fs = ((c[31:] - c[:-31]) / 31).astype(np.float32)
    return np.concatenate([F, Fs], 1)


GEO_NAMES = [f"{g}_{m}" for m in range(2) for g in ("spd_cb", "spd_nose", "body_len", "turn")] + ["nn", "n0_t1", "n1_t0", "cc", "closing", "facing0", "facing1"]
GEO_NAMES = GEO_NAMES + [n + "_s" for n in GEO_NAMES]


@torch.no_grad()
def collect(m, h, aux, dx_mean, x_last, x_next):
    B, N, D = h.shape; nose = torch.arange(M, device=dev) * KP
    h_mean = h.view(B, M, KP, D).mean(2).reshape(B, M * D)
    if "goal" in aux:
        P = aux["goal"][:, nose]                                                           # B,M,N+1
        p_contact = 1 - P[..., N]
        intent = P[..., :N].view(B, M, M, KP)                                              # mass of m's nose on (m', part)
        other = torch.stack([intent[:, 0, 1], intent[:, 1, 0]], 1).reshape(B, M * KP)
    else:                                                                                  # no intention head: zero columns
        p_contact = torch.zeros(B, M, device=dev); other = torch.zeros(B, M * KP, device=dev)
    err = (x_next - (x_last + dx_mean)).norm(dim=-1).view(B, M, KP).mean(2)
    spd = dx_mean.norm(dim=-1).view(B, M, KP).mean(2)
    return torch.cat([h_mean, p_contact, other, err, spd], -1)


@torch.no_grad()
def embed_window(m, X):
    """X (B, T30, N, 2) -> (B, T30, F)"""
    B, T30, N, _ = X.shape; fs, L = args.fs, args.obs; D = m.cell.D; F = M * D + 2 + M * KP + 2 + 2
    out = torch.zeros(B, T30, F, device=dev); types = m.types()
    for phase in range(fs):
        xs = X[:, phase::fs]; Tp = xs.shape[1]
        pad = torch.cat([xs[:, :1].expand(-1, L - 1, -1, -1), xs], 1)
        win = pad.unfold(1, L, 1).permute(0, 1, 2, 4, 3).reshape(B * Tp, N, L, 2)
        nxt = torch.cat([xs[:, 1:], xs[:, -1:]], 1).reshape(B * Tp, N, 2)
        Bw = win.shape[0]; mask = torch.ones(Bw, N, device=dev)
        h = torch.zeros(Bw, N, D, device=dev); s = torch.zeros(Bw, N, N, D, device=dev)
        buf = win[:, :, :1].expand(-1, -1, m.cell.k, -1).contiguous()
        for t in range(L):
            buf = torch.cat([buf[:, :, 1:], win[:, :, t:t + 1]], 2)
            dx, dx_mean, h, s, aux = m.cell(buf, h, s, types, mask, noise_on=False)
        out[:, phase + fs * torch.arange(Tp, device=dev)] = collect(m, h, aux, dx_mean, win[:, :, -1], nxt).view(B, Tp, F)
    return out


def main():
    m, ck = load_checkpoint(args.ck, n_ent=M, device=dev); a = ck["args"]; D = m.cell.D
    assert m.kp == KP, f"checkpoint has {m.kp} keypoints per mouse, --nodes {KP}"
    if args.scale_px <= 0 and a["node_mode"] == "calms14":
        args.scale_px = float(a["arena_px"])                                               # the scale the model was trained with
    part_names = ("nose", "neck", "cb", "tail") if KP == 4 else tuple(KP7)
    names = [f"h{mm}_{d}" for mm in range(M) for d in range(D)] + [f"pcontact_{mm}" for mm in range(M)] + \
            [f"intent_{mm}_{k}" for mm in range(M) for k in part_names] + [f"surprise_{mm}" for mm in range(M)] + [f"speed_hat_{mm}" for mm in range(M)]
    scale = args.scale_px if args.scale_px > 0 else None
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for split in ("train", "test"):
        d = json.load(open(os.path.join(args.json_dir, f"calms21_task1_{split}.json")))["annotator-id_0"]
        ids = list(d.keys())[:args.max_seqs or None]
        seqs = [(k, to_nodes(np.asarray(d[k]["keypoints"])), to_nodes7(np.asarray(d[k]["keypoints"])), np.asarray(d[k]["annotations"], np.int8)) for k in ids]
        if scale is None:                                                                  # from the training split
            bl = np.concatenate([np.linalg.norm(x[:, :, 0] - x[:, :, 3], axis=-1).ravel() for _, x, _, _ in seqs])
            scale = float(np.median(bl)) / args.body_len
        print(f"{split}: scale {scale:.1f} px per model unit; model input = {KP} keypoints per mouse")
        total = sum(len(y) for _, _, _, y in seqs); F = len(names) + len(GEO_NAMES)
        emb = np.lib.format.open_memmap(f"{args.out}_{split}.npy", mode="w+", dtype=np.float16, shape=(total, F))
        labels = np.zeros(total, np.int8); offsets = []; o = 0; t0 = time.time()
        for k, x4, x7, y in seqs:
            xs = x4 / scale; xin = (x7 if KP == 7 else x4) / scale; T = len(y)
            X = torch.tensor(xin.reshape(1, T, M * KP, 2), device=dev)
            feats = []
            for c0 in range(0, T, 4000):                                                   # chunk long videos (windows x nodes x D)
                c1 = min(T, c0 + 4000); lo = max(0, c0 - 40)
                f = embed_window(m, X[:, lo:c1])[0, c0 - lo:].cpu().numpy(); feats.append(f)
            f = np.concatenate([np.concatenate(feats, 0), geometry(xs)], 1)
            emb[o:o + T] = f.astype(np.float16); labels[o:o + T] = y; offsets.append((o, o + T)); o += T
            print(f"  {split} {k} T={T} {time.time() - t0:.0f}s", flush=True)
        emb.flush(); np.save(f"{args.out}_{split}_labels.npy", labels)
        json.dump({"ids": ids, "offsets": offsets, "features": names + GEO_NAMES, "scale_px_per_unit": scale, "ck": args.ck}, open(f"{args.out}_{split}.json", "w"))
        print(f"wrote {args.out}_{split}.npy {emb.shape}")


if __name__ == "__main__":
    main()
