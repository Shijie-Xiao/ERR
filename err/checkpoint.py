"""Build an ERR model from the arguments stored in a checkpoint and load its weights.

Checkpoints written by err/train.py are dicts {epoch, state_dict, args, best_ade, ...}; `args` holds the full
command line, so the model can always be rebuilt without remembering flags.  The released checkpoints in
checkpoints/ follow the same format.

n_ent overrides the number of entities (zero-shot transfer of a 3-mouse model to a 2-mouse assay): every parameter
of ERR is per node / per edge, only the entity-index buffers (kp_id, ent, intra, inter) depend on N, and those are
rebuilt instead of loaded.
"""
import torch

from err.model import ERR

KP_OF_MODE = {"all12": 4, "all36": 12, "cb": 1, "calms14": 7}
ENT_OF_MODE = {"calms14": 2}
ENTITY_BUFFERS = ("kp_id", "ent", "intra", "inter")


def build_from_args(a, n_ent=None, device="cpu"):
    """ERR model from a trainer args dict (keys as in err/train.py) -> (model, kp_per_entity, n_ent)"""
    kp = KP_OF_MODE[a["node_mode"]]
    n_ent = n_ent or ENT_OF_MODE.get(a["node_mode"], 3)
    for key in ("body_frame", "part_query", "goal_attn", "lambda_attn_sup"):
        assert not a.get(key, 0), f"checkpoint uses the dropped variant switch {key}; load it with the research code"
    m = ERR(n_ent * kp, kp, D=a["D"], C=a["C"], k=a["k"], device=device, goal=bool(a["goal"]), goal_detach=bool(a.get("goal_detach", 0)),
            attn=a["attn"], edge_state=bool(a["edge_state"]), inter_msg=bool(a["inter_msg"]), noise_share=a["noise_share"],
            vec_nl=bool(a["vec_nl"]), attn_clamp=a["attn_clamp"], attn_temp=a["attn_temp"], equiv=bool(a.get("equiv", 1)),
            oneshot_T=(a["future_length"] if a.get("oneshot", 0) else 0), steer=bool(a.get("steer", 0)),
            goal_scale=float(a.get("goal_scale", 0.01) or 0.01), mode_sample=bool(a.get("mode_sample", 0)))
    return m, kp, n_ent


def load_checkpoint(path, n_ent=None, device="cpu", verbose=True):
    """-> (model in eval mode, checkpoint dict).  Entity buffers are rebuilt when n_ent differs from the training value;
    'offdiag' (an unused buffer of the original ERR research class) is ignored."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m, kp, n_ent = build_from_args(ck["args"], n_ent, device)
    own = m.state_dict()
    sd = {k: v for k, v in ck["state_dict"].items() if k in own and own[k].shape == v.shape}
    skipped = [k for k in ck["state_dict"] if k not in sd]
    res = m.load_state_dict(sd, strict=False)
    bad = [k for k in list(res.missing_keys) + skipped if k not in ENTITY_BUFFERS + ("offdiag",)]
    assert not bad, f"{path}: tensors not loaded {bad}"
    if verbose:
        print(f"loaded {path}: epoch {ck.get('epoch')}, node_mode {ck['args']['node_mode']} ({kp} kp x {n_ent} entities)"
              + (f", entity buffers rebuilt: {skipped}" if skipped else ""), flush=True)
    m.eval()
    return m, ck
