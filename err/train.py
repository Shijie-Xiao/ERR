"""Train / evaluate ERR on the window datasets built by data_prep/.

Loss = joint variety (K_train samples, best sample per scene)  +  lambda_obs x observed one-step error
       + lambda_body x body-band hinge                           +  lambda_goal x intention cross-entropy (observed steps)

Every epoch the val split is scored under the standard protocol (learned attention everywhere, K_eval samples) and the
checkpoint is selected by val minADE20 on center_back.  The final test run adds the take-over analysis sources
('sample' = hard decisions drawn from the head, 'oracle' = the true contact target given).

Trajectory metrics are reported on center_back (cb), on the 4 legacy keypoints (nose, neck, center_back, tail_base = nodes
m*KP + [0,3,6,9] in the 12-kp layout) so 4-kp and 12-kp models are comparable, and on all nodes.  Recognition metrics of the
intention head: contact AUC, contact recall, part / group / entity top-1, plus the NEW-contact split (contacts that have not
started yet) against the nearest-part-now geometry baseline.  Part-level attention metrics (actor nose's big-graph row over
the contacted mouse's parts) are reported for the 12-kp layout.

Usage (from the repository root):
  python err/train.py --exp_name err_12kp --node_mode all12                         # 4 kp / mouse, 12 nodes
  python err/train.py --exp_name err_36kp --node_mode all36 --batch_size 32 --accum 2 \
         --npz data/windows/dataset_social_r1_w20_fs4_seqsplit_12kpfull.npz        # 12 kp / mouse, 36 nodes
  python err/train.py --exp_name err_calms14 --node_mode calms14 --arena_px 1827.7 \
         --npz data/windows/dataset_calms21_w20_fs4.npz                              # CalMS21, 2 mice x 7 kp
  python err/train.py --exp_name err_36kp --node_mode all36 --npz ... --test          # evaluate the best checkpoint
A '<exp>_last.pth.tar' is written every epoch; --resume continues from it (use it for jobs that may be interrupted).
Training guards (needed for the 36-node runs): gradient clipping, skipping singular updates, a batch-level loss-spike
rollback to the last snapshot, and an epoch-level divergence rollback with lr halving.
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from err.model import (ERR, variety_loss, energy_loss, body_pairs, body_band, body_hinge,                 # noqa: E402
                        goal_labels, goal_targets, goal_loss, goal_rho, goal_stats, pool_goal_stats)
from err.data import MiceWindowDataset, NODE_MODES                                                       # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--exp_name", default="err_12kp")
p.add_argument("--npz", default=os.path.join(ROOT, "data", "windows", "dataset_social_r1_w20_fs4_seqsplit.npz"))
p.add_argument("--stats_npz", default=None, help="npz whose train split provides the contact radius, body band and steering unit when --npz has no "
                                                "train split (e.g. --test on a natural-window test file); default: the training file of the same layout")
p.add_argument("--node_mode", default="all12", choices=list(NODE_MODES), help="all12 / all36 = MABe22 3 mice x 4 / 12 kp; calms14 = CalMS21 2 mice x 7 kp")
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (effective batch = batch_size x accum)")
p.add_argument("--eval_batch_size", type=int, default=64)
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--past_length", type=int, default=10)
p.add_argument("--future_length", type=int, default=10)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--epoch_decay", type=int, default=2)
p.add_argument("--lr_gamma", type=float, default=0.9)
p.add_argument("--max_epochs_no_improve", type=int, default=15)
p.add_argument("--clip", type=float, default=5.0, help="gradient-norm clip")
p.add_argument("--skip_gnorm", type=float, default=1000.0, help="skip an update whose raw gradient norm exceeds this (singular step; typical norms are 2-80)")
p.add_argument("--diverge_factor", type=float, default=1.5, help="val cb > factor x best -> roll back to the last healthy epoch, halve lr")
p.add_argument("--max_rollbacks", type=int, default=5)
# batch-level guard: a loss spike (the onset of the step-saturation collapse) restores the last snapshot (<= snap_every batches old)
p.add_argument("--spike_var", type=float, default=4.0, help="batch variety loss > factor x EMA -> spike (0 = off)")
p.add_argument("--spike_body", type=float, default=50.0, help="batch body hinge > factor x EMA + 1e-3 -> spike")
p.add_argument("--snap_every", type=int, default=50, help="batches between weight/optimizer snapshots")
p.add_argument("--max_spikes", type=int, default=3, help="spikes per epoch after which the lr is halved")
p.add_argument("--spike_warmup", type=int, default=100, help="batches (from the start of the process) before the batch spike guard is armed")
# model (defaults = ERR; each switch is an ablation)
p.add_argument("--D", type=int, default=64)
p.add_argument("--C", type=int, default=8)
p.add_argument("--k", type=int, default=3)
p.add_argument("--attn", default="both", choices=["both", "intra", "inter", "none"])
p.add_argument("--edge_state", type=int, default=1)
p.add_argument("--inter_msg", type=int, default=1)
p.add_argument("--noise_share", default="entity", choices=["entity", "node"])
p.add_argument("--vec_nl", type=int, default=1)
p.add_argument("--attn_clamp", type=float, default=5.0)
p.add_argument("--attn_temp", type=float, default=1.0)
p.add_argument("--goal", type=int, default=1, help="intention head (0 = trunk only)")
p.add_argument("--goal_detach", type=int, default=0, help="1 = intention loss does not reach the trunk (ablation)")
p.add_argument("--lambda_goal", type=float, default=0.01)
p.add_argument("--goal_rho_frac", type=float, default=0.5, help="contact radius = frac x median body length")
p.add_argument("--rho_kp", default="legacy", choices=["legacy", "all"], help="body length from the legacy 4 kp (same radius for 4-kp and 12-kp runs) or from all kp")
p.add_argument("--goal_H", type=int, default=0, help="label horizon in steps (0 = future_length)")
p.add_argument("--steer", type=int, default=1, help="1 = ERR: steering term + intention head after the state update; 0 = no steering term, head read before the update (ablation)")
p.add_argument("--mode_sample", type=int, default=1, help="1 = ERR mode sampling: one Gumbel perturbation of the intention logits per (sample, entity) when predicting")
p.add_argument("--goal_scale", default="auto", help="unit of the steering step in arena units; 'auto' = mean center_back step of the train split")
p.add_argument("--soft_tau", type=float, default=0.0, help="px; 0 = hard intention targets (ERR); > 0 = parts within tau of the closest share the target mass")
p.add_argument("--equiv", type=int, default=1, help="0 = remove the symmetry (absolute coordinates into the MLPs, free 2-D readout): design-axis ablation")
p.add_argument("--oneshot", type=int, default=0, help="1 = one-shot decoding of all future steps at the last observed frame (no closed loop): design-axis ablation")
p.add_argument("--augment_rot", type=int, default=0, help="1 = random rotation + reflection of every training window about the arena centre (augmentation control)")
p.add_argument("--train_frac", type=float, default=1.0, help="use only the first fraction of the training windows (data-efficiency curve)")
# loss / selection
p.add_argument("--K_train", type=int, default=5)
p.add_argument("--K_eval", type=int, default=20)
p.add_argument("--traj_loss", default="variety", choices=["variety", "energy"], help="energy = proper scoring rule over the K_train samples instead of min-of-K (ablation)")
p.add_argument("--select", default="cb", choices=["cb", "es", "part"], help="model selection on val minADE20(cb), the val energy score (cb) or the part-level attention metric")
p.add_argument("--joint_min", type=int, default=1)
p.add_argument("--lambda_obs", type=float, default=1.0)
p.add_argument("--lambda_body", type=float, default=1.0)
p.add_argument("--body_p", type=float, nargs=2, default=[1.0, 99.0])
# io
p.add_argument("--model_save_dir", default=os.path.join(ROOT, "runs"))
p.add_argument("--arena_px", type=float, default=710.0, help="px per arena unit of the npz (MABe22 user_train_r1: 710; CalMS21 file: printed by data_prep/calms21_windows.py)")
p.add_argument("--eval_split", default="test", choices=["val", "test"])
p.add_argument("--test", action="store_true", help="evaluate the best checkpoint on --eval_split (all decision sources)")
p.add_argument("--resume", action="store_true", help="continue from <exp>_last.pth.tar if it exists")
p.add_argument("--num_workers", type=int, default=0)
p.add_argument("--max_batches", type=int, default=0, help="debug: limit the batches per epoch / per evaluation")
p.add_argument("--fill_missing", type=int, default=1)
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.model_save_dir, exist_ok=True)
KP = {"all12": 4, "all36": 12, "cb": 1, "calms14": 7}[args.node_mode]
N_ENT = {"calms14": 2}.get(args.node_mode, 3)
N_NODES = N_ENT * KP
LEGACY_KP = {4: [0, 1, 2, 3], 12: [0, 3, 6, 9], 1: [0], 7: [0, 3, 4, 6]}[KP]    # nose, neck, center_back, tail_base (CalMS21: nose, neck, left hip, tail base)
CB_KP = {4: 2, 12: 6, 1: 0, 7: 3}[KP]                                           # centre node for the cb metrics (CalMS21: neck)
GROUP_OF = torch.tensor({4: [0, 1, 2, 3], 12: [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], 1: [0], 7: [0, 0, 0, 1, 2, 2, 3]}[KP], device=device)
GROUP_NAMES = {4: ["nose", "neck", "center_back", "tail_base"], 12: ["head", "shoulders", "back", "tail"], 1: ["cb"], 7: ["head", "neck", "hips", "tail"]}[KP]
EV_BITS = [(1, "approach"), (2, "contact"), (4, "anogenital"), (8, "nose_nose"), (16, "following")]
if args.goal_H == 0:
    args.goal_H = args.future_length
_z = np.load(args.npz)
if "train_data" in _z.files:
    _train_raw = _z["train_data"]
else:                                                                           # test-only file: statistics from the training file
    args.stats_npz = args.stats_npz or os.path.join(ROOT, "data", "windows", "dataset_social_r1_w20_fs4_seqsplit" + ("_12kpfull" if KP == 12 else "") + ".npz")
    _train_raw = np.load(args.stats_npz)["train_data"]
    print(f"--npz has no train split: rho / body band / steering unit taken from {args.stats_npz}", flush=True)
args.goal_rho = goal_rho(_train_raw, args.goal_rho_frac, LEGACY_KP if args.rho_kp == "legacy" else None)
if args.goal_scale == "auto":                                                   # steering unit = mean centre step of the train split
    args.goal_scale = float(np.linalg.norm(np.diff(_train_raw[:, :, :, CB_KP], axis=1), axis=-1).mean())
else:
    args.goal_scale = float(args.goal_scale)
print(f"nodes={N_NODES} (KP={KP}, entities={N_ENT})  goal: rho={args.goal_rho:.4f} ({args.goal_rho * args.arena_px:.1f} px, from {args.rho_kp} kp)  "
      f"H={args.goal_H}  soft_tau={args.soft_tau}px  lambda_goal={args.lambda_goal}  detach={args.goal_detach}  "
      f"steer={args.steer} (unit {args.goal_scale:.4f} = {args.goal_scale * args.arena_px:.1f} px)  mode_sample={args.mode_sample}", flush=True)


def setup_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)


def build_model():
    return ERR(N_NODES, KP, D=args.D, C=args.C, k=args.k, device=device, goal=bool(args.goal), goal_detach=bool(args.goal_detach),
               attn=args.attn, edge_state=bool(args.edge_state), inter_msg=bool(args.inter_msg), noise_share=args.noise_share,
               vec_nl=bool(args.vec_nl), attn_clamp=args.attn_clamp, attn_temp=args.attn_temp,
               equiv=bool(args.equiv), oneshot_T=args.future_length if args.oneshot else 0,
               steer=bool(args.steer), goal_scale=args.goal_scale, mode_sample=bool(args.mode_sample))


def rand_rigid(loc, loc_end, centre=0.5):
    """random rotation + reflection of each window (observed and future together) about the arena centre."""
    B = loc.size(0); dev = loc.device
    th = torch.rand(B, device=dev) * 2 * math.pi; refl = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    c, s_ = torch.cos(th), torch.sin(th)
    R = torch.stack([torch.stack([c, -s_ * refl], -1), torch.stack([s_, c * refl], -1)], -2)                       # B,2,2 (rotation @ diag(1, refl))
    f = lambda z: torch.einsum('bntd,bed->bnte', z - centre, R) + centre
    return f(loc), f(loc_end)


def build_loaders():
    z = np.load(args.npz); loaders, meta = {}, {}
    for split in ("train", "val", "test"):
        if f"{split}_data" not in z.files:                                                         # test-only files (natural windows)
            continue
        ds = MiceWindowDataset(args.npz, split=split, obs_len=args.past_length, pred_len=args.future_length,
                               node_mode=args.node_mode, fill_missing=bool(args.fill_missing))
        if ds.n_filled:
            print(f"  [{split}] imputed {ds.n_filled} untracked keypoint entries", flush=True)
        train = split == "train"
        if train and args.train_frac < 1.0:
            n_sub = int(len(ds) * args.train_frac)
            ds = torch.utils.data.Subset(ds, list(range(n_sub)))                                   # first windows = first sequences
            print(f"  [train] using the first {n_sub} windows ({args.train_frac:.0%})", flush=True)
        loaders[split] = DataLoader(ds, batch_size=args.batch_size if train else args.eval_batch_size, shuffle=train,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=train)
        if f"{split}_event_type" in z.files:
            meta[split] = {k: z[f"{split}_{k}"] for k in ("event_type", "focal", "partner")}
    return loaders, meta


class Body:
    """body-band hinge: every intra-entity keypoint distance is kept inside its [p1, p99] training band."""

    def __init__(self):
        self.on = args.lambda_body > 0 and KP > 1
        if self.on:
            self.ii, self.jj = [t.to(device) for t in body_pairs(N_NODES, KP)]
            self.lo, self.hi = [t.to(device) for t in body_band(_train_raw, *args.body_p)]

    def loss(self, pred, obs_pred):
        if not self.on:
            return torch.zeros((), device=device)
        return body_hinge(pred, self.ii, self.jj, self.lo, self.hi) + body_hinge(obs_pred, self.ii, self.jj, self.lo, self.hi)

    @torch.no_grad()
    def outside_frac(self, pred):
        """fraction of (window, sample, step, entity) with any intra pair outside the band."""
        if not self.on:
            return float("nan")
        d = (pred[:, self.ii] - pred[:, self.jj]).norm(dim=-1)                                    # B,P,K,T
        out = (d < self.lo.view(1, -1, 1, 1)) | (d > self.hi.view(1, -1, 1, 1))
        return float(out.view(out.shape[0], N_ENT, -1, *out.shape[2:]).any(2).float().mean())


@torch.no_grad()
def speed_ratio(pred, loc, loc_end):
    """mean predicted step / mean true step (all nodes), overall and at the last step."""
    x0 = loc[:, :, -1]
    sp = torch.cat([pred[:, :, :, :1] - x0[:, :, None, None], pred[:, :, :, 1:] - pred[:, :, :, :-1]], 3).norm(dim=-1)
    st = torch.cat([loc_end[:, :, :1] - x0[:, :, None], loc_end[:, :, 1:] - loc_end[:, :, :-1]], 2).norm(dim=-1)
    return float(sp.mean() / st.mean().clamp(min=1e-9)), float(sp[..., -1].mean() / st[..., -1].mean().clamp(min=1e-9))


def snapshot(model, opt):
    return ({k: v.detach().clone() for k, v in model.state_dict().items()}, copy.deepcopy(opt.state_dict()))


def train_epoch(model, opt, epoch, loader, body):
    model.train()
    tot = {"var": 0.0, "obs": 0.0, "body": 0.0, "goal": 0.0}; cnt = 0; t0 = time.time()
    gn_max, n_skip, n_acc, n_spike, n_halve = 0.0, 0, 0, 0, 0
    ema = train_epoch.__dict__.setdefault("ema", {"var": None, "body": None, "n": 0})                # persists across epochs
    snap = snapshot(model, opt); cur_lr = opt.param_groups[0]["lr"]

    def step():
        nonlocal gn_max, n_skip
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip))
        if not np.isfinite(gn) or gn > args.skip_gnorm:                                             # never apply a non-finite / singular update
            n_skip += 1
            print(f"    skipped update: raw grad norm {gn:.3g}", flush=True)
        else:
            gn_max = max(gn_max, gn); opt.step()
        opt.zero_grad()

    opt.zero_grad()
    for bi, (loc, loc_end, nv) in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        loc, loc_end, nv = loc.to(device), loc_end.to(device), nv.to(device)
        if args.augment_rot:
            loc, loc_end = rand_rigid(loc, loc_end)
        pred, obs_pred, aux = model(loc, nv, args.future_length, K=args.K_train, noise=True)
        l_var = variety_loss(pred, loc_end, joint=bool(args.joint_min)) if args.traj_loss == "variety" else energy_loss(pred, loc_end)
        l_obs = (obs_pred - loc[:, :, 1:]).norm(dim=-1).mean()
        l_body = body.loss(pred, obs_pred)
        if args.goal:
            full = torch.cat([loc, loc_end], 2)
            tg = goal_targets(full, model.inter, args.goal_rho, args.goal_H, args.past_length, args.soft_tau / args.arena_px)
            l_goal = goal_loss(aux["goal_obs"], tg)
        else:
            l_goal = torch.zeros((), device=device)
        lv, lb = l_var.item(), l_body.item()
        spike = args.spike_var > 0 and ema["var"] is not None and ema["n"] >= args.spike_warmup and (lv > args.spike_var * ema["var"] or lb > args.spike_body * ema["body"] + 1e-3)
        if spike:                                                                                    # restore the last healthy snapshot, drop this batch
            n_spike += 1
            print(f"    !! loss spike at batch {bi}: variety={lv * args.arena_px:.1f}px (ema {ema['var'] * args.arena_px:.1f}) body={lb:.2e} (ema {ema['body']:.1e}) "
                  f"-> restored snapshot #{n_spike}", flush=True)
            model.load_state_dict(snap[0]); opt.load_state_dict(snap[1]); opt.zero_grad(); n_acc = 0
            if n_spike % args.max_spikes == 0:
                cur_lr *= 0.5; n_halve += 1
                print(f"    !! {n_spike} spikes this epoch -> lr halved to {cur_lr:.2e}", flush=True)
            for g_ in opt.param_groups:                                                              # the optimizer snapshot carries the old lr
                g_["lr"] = cur_lr
            continue
        loss = l_var + args.lambda_obs * l_obs + args.lambda_body * l_body + args.lambda_goal * l_goal
        (loss / args.accum).backward(); n_acc += 1
        if n_acc == args.accum:
            step(); n_acc = 0
        for k_, v_ in (("var", lv), ("body", lb)):
            ema[k_] = v_ if ema[k_] is None else 0.98 * ema[k_] + 0.02 * v_
        ema["n"] += 1
        if (bi + 1) % args.snap_every == 0:
            snap = snapshot(model, opt)
        B = loc.size(0); cnt += B
        for k, v in (("var", l_var), ("obs", l_obs), ("body", l_body), ("goal", l_goal)):
            tot[k] += v.item() * B
        if bi % 100 == 0:
            print(f"  epoch {epoch} batch {bi}/{len(loader)} variety={l_var.item() * args.arena_px:.2f}px obs1step={l_obs.item() * args.arena_px:.2f}px "
                  f"body={l_body.item():.2e} goalCE={l_goal.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
    if n_acc > 0:                                                                                    # flush a partial accumulation (mean over the micro-batches seen)
        for p_ in model.parameters():
            if p_.grad is not None:
                p_.grad.mul_(args.accum / n_acc)
        step()
    out = {k: v / max(cnt, 1) for k, v in tot.items()}
    out.update(grad_norm_max=gn_max, skipped=n_skip, spikes=n_spike, lr_halvings=n_halve)
    return out


@torch.no_grad()
def evaluate(model, loader, body, meta, split, sources=("none",)):
    """sources: 'none' (protocol), 'sample' / 'argmax' (decisions from the head), 'oracle' (true decision)."""
    model.eval()
    cb = torch.tensor([m * KP + CB_KP for m in range(N_ENT)], device=device)
    leg = torch.tensor([m * KP + k for m in range(N_ENT) for k in LEGACY_KP], device=device)
    R = {s: {} for s in sources}; mean_cb, gst, ev_all, pa_all = [], [], [], []; off = 0
    ap_top, ap_geo, ap_tv, ap_has = [], [], [], []                                                  # part-level attention metrics
    tail_kp = LEGACY_KP[3] if KP > 1 else 0

    def add(src, k, v):
        R[src].setdefault(k, []).append(v)

    for bi, (loc, loc_end, nv) in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        loc, loc_end, nv = loc.to(device), loc_end.to(device), nv.to(device); B = loc.size(0); ar = torch.arange(B, device=device)
        full = torch.cat([loc, loc_end], 2)
        labels = goal_labels(full, model.inter, args.goal_rho, args.goal_H, args.past_length)
        pm, _, aux_m = model(loc, nv, args.future_length, K=1, noise=False)
        mean_cb.append((pm[:, cb, 0] - loc_end[:, cb]).norm(dim=-1).mean((1, 2)).cpu())
        dnow = (loc[:, :, None, -1] - loc[:, None, :, -1]).norm(dim=-1).masked_fill(~model.inter[None], float("inf"))
        if args.goal:
            gst.append(goal_stats(aux_m["goal_obs"], labels, N_NODES, KP, GROUP_OF, near_node=dnow.argmin(-1), in_contact=dnow.amin(-1) < args.goal_rho))
        pa = meta["partner"][off:off + B] if meta is not None else np.full(B, -1); pa_t = torch.tensor(np.where(pa < 0, 0, pa), device=device)
        nose = pa_t * KP; body_nodes = pa_t[:, None] * KP + torch.tensor(LEGACY_KP, device=device)[None]
        true_part = labels[ar, nose, -1]                                                            # actor nose's true contact part (or N)
        # part-level attention metrics: the actor nose's big-graph row over the contacted mouse's parts at the last observed step
        w_last = aux_m["w_inter"][..., args.past_length - 1]                                       # B,N,N
        has_c = true_part < N_NODES; tpc = true_part.clamp(max=N_NODES - 1); tgt_e = tpc // KP
        tcols = tgt_e[:, None] * KP + torch.arange(KP, device=device)[None]
        rn = w_last[ar, nose].gather(1, tcols); rt = w_last[ar, pa_t * KP + tail_kp].gather(1, tcols)
        gm = torch.zeros(B, int(GROUP_OF.max()) + 1, device=device).index_add_(1, GROUP_OF, rn)
        geo_g = GROUP_OF[dnow[ar, nose].gather(1, tcols).argmin(1)]; true_g = GROUP_OF[tpc % KP]
        ap_top.append((gm.argmax(1) == true_g).float().cpu()); ap_geo.append((geo_g == true_g).float().cpu()); ap_has.append(has_c.cpu())
        ap_tv.append((0.5 * (rn / rn.sum(1, keepdim=True).clamp(min=1e-9) - rt / rt.sum(1, keepdim=True).clamp(min=1e-9)).abs().sum(1)).cpu())
        for src in sources:
            torch.manual_seed(1000 + bi)
            if src == "oracle":
                pred, _, _ = model(loc, nv, args.future_length, K=args.K_eval, noise=True, goal_source="label", pred_labels=labels[..., -1])
            else:
                pred, _, _ = model(loc, nv, args.future_length, K=args.K_eval, noise=True, goal_source=src)
            e_all = (pred - loc_end[:, :, None]).norm(dim=-1)                                       # B,N,K,T
            e_cb = e_all[:, cb]; j = e_cb.mean(3).mean(1); best = j.argmin(1)
            pc = pred[:, cb]; Kp = pc.shape[2]                                                      # energy score per (node, step) on center_back
            d_xx = (pc[:, :, :, None] - pc[:, :, None]).norm(dim=-1).sum((2, 3)) / (Kp * (Kp - 1))
            add(src, "es", (e_cb.mean(2) - 0.5 * d_xx).mean((1, 2)).cpu())
            add(src, "cb", j.min(1).values.cpu()); add(src, "fde", e_cb[..., -1].mean(1).min(1).values.cpu())
            add(src, "ade4", e_all[:, leg].mean(3).mean(1).min(1).values.cpu()); add(src, "ade_all", e_all.mean(3).mean(1).min(1).values.cpu())
            eb = e_all[ar, :, best]                                                                 # B,N,T  (the cb-best sample)
            add(src, "nose", eb[ar, nose].mean(1).cpu()); add(src, "body", eb[ar[:, None], body_nodes].mean((1, 2)).cpu())
            add(src, "spd", torch.tensor([speed_ratio(pred, loc, loc_end)])); add(src, "out", torch.tensor([body.outside_frac(pred)]))
            tp = true_part.clamp(max=N_NODES - 1)
            dmin = (pred[ar, nose] - loc_end[ar, tp][:, None]).norm(dim=-1).min(-1).values            # B,K
            add(src, "hit", torch.where(true_part < N_NODES, (dmin < args.goal_rho).any(1).float(), torch.full_like(dmin[:, 0], float("nan"))).cpu())
        ev_all.append(meta["event_type"][off:off + B] if meta is not None else np.zeros(B, int)); pa_all.append(pa); off += B
    ev_type = np.concatenate(ev_all); pa_all = np.concatenate(pa_all); ev = (ev_type > 0) & (pa_all >= 0)
    out = {"meanADE_cb": float(torch.cat(mean_cb).mean() * args.arena_px), "n_windows": int(len(ev_type)), "n_event_windows": int(ev.sum())}
    m_ap = ev & torch.cat(ap_has).numpy().astype(bool)
    f_ap = lambda L: float(torch.cat(L).numpy()[m_ap].mean()) if m_ap.any() else float("nan")
    out["attn_part"] = {"group_top1": f_ap(ap_top), "geo_group_top1": f_ap(ap_geo), "sender_tv": f_ap(ap_tv), "n": int(m_ap.sum())}
    if gst:
        out["goal"] = pool_goal_stats(gst)
    for src in sources:
        c = lambda k: torch.cat(R[src][k]).numpy()
        o = {"minADE20_cb": float(c("cb").mean() * args.arena_px), "minFDE20_cb": float(c("fde").mean() * args.arena_px), "ES_cb": float(c("es").mean() * args.arena_px),
             "minADE20_legacy4": float(c("ade4").mean() * args.arena_px), "minADE20_all": float(c("ade_all").mean() * args.arena_px),
             "actor_nose_min20": float(c("nose")[ev].mean() * args.arena_px), "actor_body_min20": float(c("body")[ev].mean() * args.arena_px),
             "speed_ratio": float(c("spd")[:, 0].mean()), "speed_ratio_step10": float(c("spd")[:, 1].mean()),
             "contact_realised_any20": float(np.nanmean(c("hit")[ev])) if ev.any() else float("nan"), "body_outside": float(c("out").mean())}
        for bit, name in EV_BITS:
            m = ev & ((ev_type & bit) > 0)
            if m.any():
                o[f"actor_nose_min20_{name}"] = float(c("nose")[m].mean() * args.arena_px)
        out[src] = o
    g = out.get("goal", {})
    ap = out["attn_part"]
    print(f"  [{split}] mean={out['meanADE_cb']:.2f} | attn-part: top1={ap['group_top1']:.3f} geo={ap['geo_group_top1']:.3f} tv={ap['sender_tv']:.3f} n={ap['n']} | goal: auc={g.get('goal_contact_auc', float('nan')):.3f} recall={g.get('goal_contact_recall', float('nan')):.3f} "
          f"part@1={g.get('goal_part_top1', float('nan')):.3f} group@1={g.get('goal_group_top1', float('nan')):.3f} "
          f"NEW group@1={g.get('goal_new_group_top1', float('nan')):.3f} (geo {g.get('goal_new_geo_top1', float('nan')):.3f}, recall {g.get('goal_new_recall', float('nan')):.3f}) "
          f"ongoing {g.get('goal_ong_group_top1', float('nan')):.3f} (geo {g.get('goal_ong_geo_top1', float('nan')):.3f}) false-alarm {g.get('goal_false_alarm', float('nan')):.3f} | " +
          " | ".join(f"{s}: cb={out[s]['minADE20_cb']:.2f} fde={out[s]['minFDE20_cb']:.2f} es={out[s]['ES_cb']:.2f} leg4={out[s]['minADE20_legacy4']:.2f} all={out[s]['minADE20_all']:.2f} "
                     f"nose={out[s]['actor_nose_min20']:.2f} body={out[s]['actor_body_min20']:.2f} spd={out[s]['speed_ratio']:.2f} hit={out[s]['contact_realised_any20']:.2f} "
                     f"out={out[s]['body_outside']:.3f}" for s in sources), flush=True)
    return out


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp); os.replace(tmp, path)


def rng_state():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(), "python": random.getstate()}


def set_rng_state(st):
    torch.set_rng_state(st["torch"]); np.random.set_state(st["numpy"]); random.setstate(st["python"])
    if st["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def main():
    setup_seed(args.seed)
    print(args, flush=True)
    loaders, meta = build_loaders()
    model = build_model(); body = Body()
    print(f"ERR parameters: {sum(p.numel() for p in model.parameters()):,}  device={device}", flush=True)
    best_path = os.path.join(args.model_save_dir, args.exp_name + ".pth.tar")
    last_path = os.path.join(args.model_save_dir, args.exp_name + "_last.pth.tar")
    if args.test:
        ck = torch.load(best_path, map_location="cpu", weights_only=False); model.load_state_dict(ck["state_dict"])
        print(f"best checkpoint @ epoch {ck['epoch']} (val cb {ck['best_ade']:.2f})", flush=True)
        r = evaluate(model, loaders[args.eval_split], body, meta.get(args.eval_split), args.eval_split, sources=("none", "sample", "oracle"))
        r["best_epoch"] = ck["epoch"]; r["args"] = vars(args)
        with open(os.path.join(args.model_save_dir, f"{args.exp_name}_{args.eval_split}.json"), "w") as f:
            json.dump(r, f, indent=2)
        print(json.dumps(r, indent=1)); return
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-12)
    best, best_ep, no_imp, start, lr_now, best_cb = 1e9, -1, 0, 0, args.lr, 1e9
    log = ["epoch,lr,sec,train_var_px,train_obs_px,train_body,train_goalCE,val_cb,val_fde,val_leg4,val_all,val_mean,val_nose,val_body,val_spd,val_hit,val_out,"
           "goal_auc,goal_recall,goal_part1,goal_group1,goal_none_pred,goal_none_true,new_group1,new_geo1,new_recall,ong_group1,ong_geo1,false_alarm,val_es,"
           "attn_group1,attn_geo1,attn_tv"]
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["optimizer"])
        start, best, best_ep, no_imp, lr_now, log = ck["epoch"] + 1, ck["best"], ck["best_ep"], ck["no_imp"], ck["lr_now"], ck["log"]
        best_cb = ck.get("best_cb", 1e9)
        if ck["args"].get("select", "cb") != args.select:                                         # selection metric changed on resume: restart the best/patience bookkeeping
            print(f"  selection metric changed ({ck['args'].get('select', 'cb')} -> {args.select}): best checkpoint and patience reset", flush=True)
            best, best_ep, no_imp = 1e9, -1, 0
        set_rng_state(ck["rng"]); n_rb = ck.get("n_rollback", 0)
        print(f"resumed from {last_path}: next epoch {start}, best {best:.2f} @ {best_ep}", flush=True)
    else:
        n_rb = 0
    good = (copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()))                 # last healthy state (rollback target)
    for ep in range(start, args.epochs):
        if no_imp >= args.max_epochs_no_improve:
            break
        if ep % args.epoch_decay == 0 and ep > 0:
            lr_now *= args.lr_gamma
        for g in opt.param_groups:
            g["lr"] = lr_now
        t0 = time.time(); tr = train_epoch(model, opt, ep, loaders["train"], body); t_train = time.time() - t0
        lr_now *= 0.5 ** tr["lr_halvings"]
        r = evaluate(model, loaders["val"], body, meta.get("val"), "val"); v = r["none"]; g = r.get("goal", {})
        sv = {"cb": v["minADE20_cb"], "es": v["ES_cb"], "part": 1.0 - r["attn_part"]["group_top1"]}[args.select]   # selection metric (lower = better)
        mem = f" peak_mem={torch.cuda.max_memory_allocated() / 2**30:.1f}GB" if torch.cuda.is_available() else ""
        print(f"== epoch {ep} (train {t_train:.0f}s, eval {time.time() - t0 - t_train:.0f}s{mem}) lr={lr_now:.2e} "
              f"train={tr['var'] * args.arena_px:.2f}px goalCE={tr['goal']:.3f} gnorm_max={tr['grad_norm_max']:.2f} skipped={tr['skipped']} spikes={tr['spikes']} "
              f"| val cb={v['minADE20_cb']:.2f} es={v['ES_cb']:.2f} part={r['attn_part']['group_top1']:.3f}/geo {r['attn_part']['geo_group_top1']:.3f} tv={r['attn_part']['sender_tv']:.3f}", flush=True)
        if not np.isfinite(v["minADE20_cb"]) or not np.isfinite(sv) or (best_cb < 1e9 and v["minADE20_cb"] > args.diverge_factor * best_cb):   # divergence is judged on cb
            n_rb += 1
            model.load_state_dict(good[0]); opt.load_state_dict(good[1]); lr_now *= 0.5
            print(f"  !! DIVERGED (val {args.select} {sv:.2f} > {args.diverge_factor} x best {best:.2f}) -> ROLLBACK #{n_rb} to the last healthy "
                  f"state, lr -> {lr_now:.2e}", flush=True)
            if n_rb > args.max_rollbacks:
                print("  too many rollbacks -> stop training", flush=True); break
            atomic_save({"epoch": ep, "state_dict": model.state_dict(), "optimizer": opt.state_dict(), "args": vars(args), "best": best,
                         "best_ep": best_ep, "no_imp": no_imp, "lr_now": lr_now, "log": log, "rng": rng_state(), "n_rollback": n_rb, "best_cb": best_cb}, last_path)
            continue
        good = (copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()))
        log.append(",".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in
                            [ep, lr_now, time.time() - t0, tr["var"] * args.arena_px, tr["obs"] * args.arena_px, tr["body"], tr["goal"],
                             v["minADE20_cb"], v["minFDE20_cb"], v["minADE20_legacy4"], v["minADE20_all"], r["meanADE_cb"], v["actor_nose_min20"],
                             v["actor_body_min20"], v["speed_ratio"], v["contact_realised_any20"], v["body_outside"],
                             g.get("goal_contact_auc", float("nan")), g.get("goal_contact_recall", float("nan")), g.get("goal_part_top1", float("nan")),
                             g.get("goal_group_top1", float("nan")), g.get("goal_none_pred", float("nan")), g.get("goal_none_true", float("nan")),
                             g.get("goal_new_group_top1", float("nan")), g.get("goal_new_geo_top1", float("nan")), g.get("goal_new_recall", float("nan")),
                             g.get("goal_ong_group_top1", float("nan")), g.get("goal_ong_geo_top1", float("nan")), g.get("goal_false_alarm", float("nan")),
                             v["ES_cb"], r["attn_part"]["group_top1"], r["attn_part"]["geo_group_top1"], r["attn_part"]["sender_tv"]]))
        best_cb = min(best_cb, v["minADE20_cb"])
        if sv < best:
            best, best_ep, no_imp = sv, ep, 0
            atomic_save({"epoch": ep, "state_dict": model.state_dict(), "args": vars(args), "best_ade": best, "val": r}, best_path)
            print(f"  ** new best val {args.select}={best:.2f}px @ {ep}", flush=True)
        else:
            no_imp += 1
        with open(os.path.join(args.model_save_dir, f"{args.exp_name}_log.csv"), "w") as f:
            f.write("\n".join(log))
        atomic_save({"epoch": ep, "state_dict": model.state_dict(), "optimizer": opt.state_dict(), "args": vars(args), "best": best,
                     "best_ep": best_ep, "no_imp": no_imp, "lr_now": lr_now, "log": log, "rng": rng_state(), "n_rollback": n_rb}, last_path)
        if no_imp >= args.max_epochs_no_improve:
            print(f"  early stop at {ep} (best {best_ep})", flush=True); break
    ck = torch.load(best_path, map_location="cpu", weights_only=False); model.load_state_dict(ck["state_dict"])
    print(f"\nBest epoch {ck['epoch']} (val cb {ck['best_ade']:.2f}) -> final test:", flush=True)
    r = evaluate(model, loaders["test"], body, meta.get("test"), "test", sources=("none", "sample", "oracle"))
    r["best_epoch"] = ck["epoch"]; r["args"] = vars(args)
    with open(os.path.join(args.model_save_dir, f"{args.exp_name}_test.json"), "w") as f:
        json.dump(r, f, indent=2)
    print("TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
