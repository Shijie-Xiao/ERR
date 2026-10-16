"""ResiFuse's own BiGRU classification head on frozen per-animal features, trained and scored exactly as its Mouse Triplet
fine-tuning (resifuse/training/mabe.py, configs/mabe_finetune.yaml):

  samples   his fine-tuning windows (200 frames, ds_stride 4 -> 50 steps, 3 mice), from protocol_frames.py
  head      per animal: Linear(D_in -> 128) feature adapter (his motion features are 128-d) -> his BiGRUEncoder (residual
            bidirectional GRU, hidden 256, 1 layer, dropout 0, LayerNorm) -> concat over the 3 animals -> his MLPHead
            (hidden 256, dropout 0) -> 4 logits per step.  BiGRUEncoder / MLPHead source is loaded verbatim from his
            resifuse/models/model.py (the module itself needs transformers + flash-attn, so the two classes are exec'd alone)
  loss      BCE with logits, pos_weight = neg / pos over the valid training frames, mean over valid frames
  optimiser AdamW lr 1e-4, weight decay 1e-2, batch 32, 50 epochs (his cosine schedule has eta_min = lr -> constant)
  metric    per-class F1 at sigmoid >= 0.5 on the valid test frames, each frame counted once per clip (first window in order),
            macro over the 4 classes; reported at the epoch with the best validation macro F1; several seeds.
Per-animal features: ERR embedding (mabe22/embed.py): h_m (D), shares / intention of m on the two others, P(contact)_m,
surprise_m, predicted speed_m, tracked_m; hand-crafted (mabe22/handcrafted.py): m's own 5, its 2 outgoing pairs x 5, the 4 scene
features, raw + 1-s smoothed, tracked_m.  Inputs standardised on the training windows.
Usage: python eval/resifuse/bigru_readout.py --code data/resifuse/code --frames data/resifuse/finetune_scored_frames.npz \
           --emb embeddings/resifuse/err_36kp --kind err [--seeds 0 1 2 3 4] --out results/resifuse/err_36kp_bigru.json
"""
import argparse
import ast
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--code", required=True); p.add_argument("--frames", required=True); p.add_argument("--emb", required=True)
p.add_argument("--kind", required=True, choices=["err", "hand"]); p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
p.add_argument("--epochs", type=int, default=50); p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--wd", type=float, default=1e-2)
p.add_argument("--batch", type=int, default=32); p.add_argument("--d_z", type=int, default=128)
p.add_argument("--out", default=None); p.add_argument("--tag", default="")
args = p.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- his BiGRUEncoder and MLPHead, verbatim from resifuse/models/model.py ---
src = open(os.path.join(args.code, "resifuse", "models", "model.py"), encoding="utf-8").read()
tree = ast.parse(src); ns = {"torch": torch, "nn": nn, "F": F, "pack_padded_sequence": pack_padded_sequence, "pad_packed_sequence": pad_packed_sequence}
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name in ("BiGRUEncoder", "MLPHead"):
        exec(compile(ast.Module(body=[node], type_ignores=[]), "resifuse/models/model.py", "exec"), ns)
BiGRUEncoder, MLPHead = ns["BiGRUEncoder"], ns["MLPHead"]

Z = np.load(args.frames, allow_pickle=True)
clip_ids = [str(x) for x in Z["clip_ids"]]; names = list(Z["label_names"]); ds = int(Z["ds_stride"])
wc, wsplit, wstart, WL, WM = Z["win_clip"], Z["win_split"], Z["win_start"], Z["win_labels"].astype(np.int64), Z["win_mask"]
E = np.load(args.emb + ".npy", mmap_mode="r"); J = json.load(open(args.emb + ".json")); off = {k: o for k, o in zip(J["ids"], J["offsets"])}; fn = J["features"]
idx = {n: i for i, n in enumerate(fn)}


def animal_cols(m):
    o = [x for x in range(3) if x != m]
    if args.kind == "err":
        D = sum(1 for n in fn if n.startswith("h0_"))
        c = [idx[f"h{m}_{d}"] for d in range(D)] + [idx[f"share_{m}{x}"] for x in o] + [idx[f"pcontact_{m}"]] + [idx[f"intent_{m}{x}"] for x in o]
        c += [idx[f"surprise_{m}"], idx[f"speed_hat_{m}"], idx[f"tracked_{m}"]]
    else:
        base = [f"{g}_{m}" for g in ("spd_cb", "spd_nose", "body_len", "head_cos", "turn")] + [f"{g}_{m}{x}" for x in o for g in ("nn", "ntail", "cc", "closing", "facing")] + ["min_nn", "min_ntail", "min_cc", "n_tracked"]
        c = [idx[n] for n in base] + [idx[n + "_s"] for n in base] + [idx[f"tracked_{m}"]]
    return c


cols = [animal_cols(m) for m in range(3)]; D_in = len(cols[0]); T = WL.shape[1]
X = np.zeros((len(wc), 3, T, D_in), np.float32)
for i, (c, s) in enumerate(zip(wc, wstart)):
    a, b = off[clip_ids[c]]; fr = np.minimum(s + ds * np.arange(T), b - a - 1)
    rows = np.asarray(E[a:b], dtype=np.float32)[fr]
    for m in range(3):
        X[i, m] = rows[:, cols[m]]
X[~np.isfinite(X)] = 0
tr, va, te = wsplit == 0, wsplit == 1, wsplit == 2
mu = X[tr].reshape(-1, D_in).mean(0); sd = X[tr].reshape(-1, D_in).std(0) + 1e-6; X = (X - mu) / sd
Xt, Lt, Mt = torch.tensor(X), torch.tensor(WL), torch.tensor(WM)
lab_tr, m_tr = WL[tr], WM[tr]
pos = ((lab_tr == 1) & m_tr[..., None]).sum((0, 1)).astype(np.float64); neg = ((lab_tr == 0) & m_tr[..., None]).sum((0, 1)).astype(np.float64)
pos_weight = torch.tensor(np.where(pos > 0, neg / (pos + 1e-6), 0.0), dtype=torch.float32, device=dev)
print(f"[{args.kind}] per-animal features {D_in}; windows train {tr.sum()} val {va.sum()} test {te.sum()}; pos_weight {np.round(pos_weight.cpu().numpy(), 2).tolist()}")


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapt = nn.Linear(D_in, args.d_z)
        self.gru = BiGRUEncoder(input_dim=args.d_z, hidden_dim=256, num_layers=1, dropout=0.0)
        self.head = MLPHead(3 * args.d_z, 4, 256, 0.0)

    def forward(self, x):                                                                     # x: B,3,T,D
        B = x.shape[0]
        z = self.gru(self.adapt(x).reshape(B * 3, T, -1)).view(B, 3, T, -1)
        return self.head(z.permute(0, 2, 1, 3).reshape(B, T, -1))                             # B,T,4


@torch.no_grad()
def frame_f1(model, sel):
    model.eval(); ii = np.nonzero(sel)[0]; logits = []
    for k in range(0, len(ii), 64):
        logits.append(model(Xt[ii[k:k + 64]].to(dev)).cpu())
    P = (torch.sigmoid(torch.cat(logits)) >= 0.5).numpy(); counts = np.zeros((4, 3)); seen = {}
    for j, w in enumerate(ii):                                                                # windows already in (clip, start) order
        for t in range(T):
            if not WM[w, t]:
                continue
            key = (int(wc[w]), int(wstart[w]) + ds * t)
            if key in seen:
                continue
            seen[key] = 1; g = WL[w, t] == 1; pr = P[j, t]
            counts[:, 0] += pr & g; counts[:, 1] += pr & ~g; counts[:, 2] += ~pr & g
    f1 = 2 * counts[:, 0] / np.maximum(2 * counts[:, 0] + counts[:, 1] + counts[:, 2], 1e-8)
    return float(f1.mean()), f1.tolist()


results = []
for seed in args.seeds:
    torch.manual_seed(seed); np.random.seed(seed); t0 = time.time()
    model = Head().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    tri = np.nonzero(tr)[0]; best = (-1, None, None, -1)
    for ep in range(args.epochs):
        model.train(); perm = np.random.permutation(tri)
        for k in range(0, len(perm), args.batch):
            b = perm[k:k + args.batch]; x = Xt[b].to(dev); y = Lt[b].to(dev).float().clamp(min=0); m = Mt[b].to(dev).float()
            loss_raw = F.binary_cross_entropy_with_logits(model(x), y, reduction="none", pos_weight=pos_weight)
            loss = (loss_raw * m.unsqueeze(-1)).sum() / m.sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        vm, _ = frame_f1(model, va); tm, tf = frame_f1(model, te)
        if vm > best[0]:
            best = (vm, tm, tf, ep)
    results.append({"seed": seed, "val_macro": best[0], "test_macro": best[1], "test_per_class": dict(zip(names, best[2])), "best_epoch": best[3]})
    print(f"  seed {seed}: best val {best[0]:.3f} @ep{best[3]} -> test macro {best[1]:.3f} | " + " ".join(f"{n} {v:.3f}" for n, v in zip(names, best[2])) + f" ({time.time() - t0:.0f}s)", flush=True)
tm = np.array([r["test_macro"] for r in results]); pc = np.array([[r["test_per_class"][n] for n in names] for r in results])
print(f"BIGRU(ResiFuse head) test macro F1 {tm.mean():.3f} +- {tm.std():.3f} | " + "  ".join(f"{n} {v:.3f}" for n, v in zip(names, pc.mean(0))) + f"   [{args.tag or os.path.basename(args.emb)}]")
if args.out:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"args": vars(args), "results": results, "macro_mean": float(tm.mean()), "macro_std": float(tm.std()), "per_class_mean": dict(zip(names, pc.mean(0).tolist()))}, open(args.out, "w"), indent=1)
