"""Write an UNTRAINED ERR checkpoint (random initialisation, same architecture and args as the trained ones) for the
'architecture prior' control: the readouts then show how much of a benchmark score comes from training.
Usage: python eval/random_checkpoint.py --node_mode all12 --out checkpoints/random_12kp.pth.tar [--seed 0]
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from err.checkpoint import build_from_args                                                 # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--node_mode", default="all12", choices=["all12", "all36", "calms14"]); p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=0); p.add_argument("--goal_scale", type=float, default=0.008, help="steering unit (the trained 4-kp value)")
args = p.parse_args()
torch.manual_seed(args.seed)
a = {"node_mode": args.node_mode, "D": 64, "C": 8, "k": 3, "attn": "both", "edge_state": 1, "inter_msg": 1, "noise_share": "entity", "vec_nl": 1,
     "attn_clamp": 5.0, "attn_temp": 1.0, "goal": 1, "goal_detach": 0, "equiv": 1, "oneshot": 0, "future_length": 10, "past_length": 10,
     "steer": 1, "mode_sample": 1, "goal_scale": args.goal_scale, "arena_px": 710.0, "exp_name": f"random_{args.node_mode}", "seed": args.seed}
m, kp, n_ent = build_from_args(a)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
torch.save({"epoch": -1, "state_dict": m.state_dict(), "args": a, "best_ade": None, "source": "random initialisation (untrained control)"}, args.out)
print(f"wrote {args.out}: {sum(p.numel() for p in m.parameters()):,} parameters, {n_ent} x {kp} nodes, untrained")
