"""ERR — Equivariant Relational Recurrent model with a self-supervised intention head.

One recurrent cell is shared by all nodes of a scene (node = entity x keypoint).  Per step the cell

  1. lifts the last k positions of every node to C E(2)-equivariant vector channels V_i (relative vectors only),
  2. computes invariant node features (speeds, turning cosine, |V|) and invariant edge features
     (distance, projections of V on the edge direction, <V_i, V_j>, keypoint types, same/other entity),
  3. updates a per-edge GRU memory s_ij and a per-node GRU memory h_i,
  4. attends in two streams: the small graph (own keypoints) and the big graph (other entities' keypoints),
     weights are softmax scores rescaled to the neighbour count, messages are scalars + weights on r_ij and V_j,
  5. reads the intention P_i over {other-entity node j, none} from (s_ij, h_i) AFTER the state update,
  6. emits the next displacement dx_i = bound( readout(h_i) . V_i + gain(h_i) * sum_j P_ij r_hat_ij * goal_scale )
     plus an entity-shared noise term for sampling.

Everything that multiplies a vector is a scalar function of invariants, so the model is exactly E(2)-equivariant
(tests/test_model.py).  The intention head is trained with a self-supervised contact label (the closest other-entity
node within rho over the next H steps, else 'none'); its loss flows into the trunk.  During sampling, one Gumbel
perturbation of the intention logits per (sample, entity) selects a 'mode' of interacting (mode sampling).
The take-over option (goal_source != 'none') replaces a node's big-graph attention row by a one-hot decision; it is an
analysis / intervention tool used only at evaluation time.

Ablation switches (ERR = all defaults): steer=False (head before the update, no steering term),
equiv=False (absolute coordinates, free 2-D readout), oneshot_T>0 (one-shot decoding instead of closed loop),
attn / edge_state / inter_msg / noise_share / vec_nl / goal / goal_detach.

Shapes: x (B, N, T, 2) with node = entity * kp_per_entity + kp, coordinates in arena units (px / arena_px).
Python >= 3.8, torch >= 1.13.
"""

import itertools

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

EPS = 1e-6
KP_EMB = 8


def snorm(v, keepdim=False):
    return torch.sqrt((v * v).sum(-1, keepdim=keepdim) + 1e-12)


class ERRCell(nn.Module):
    """One recurrent step for all nodes of a scene.

    buf (B,N,k,2) last k positions, h (B,N,D) node state, s (B,N,N,D) edge state,
    types {kp, ent, intra, inter}, mask (B,N) valid nodes.
    """

    def __init__(self, D=64, C=8, k=3, n_kp_types=4, attn="both", edge_state=True, inter_msg=True,
                 noise_share="entity", vec_nl=True, attn_clamp=5.0, attn_temp=1.0, goal=True,
                 act=nn.SiLU(), equiv=True, oneshot_T=0, steer=True, goal_scale=0.01):
        """D node / edge state size, C vector channels, k position-buffer length, n_kp_types keypoints per entity.
        steer / goal_scale: the goal-directed steering term dx_i += m_i * sum_j P_ij r_hat_ij * goal_scale with
            m_i = Linear(h_i) zero-initialised (== the plain model at init); the intention head is then read after
            the state update.  steer=False (ablation) reads the head before the update and has no steering term.
        Design-axis switches (everything else identical):
          equiv=False  the symmetry is removed: absolute positions and raw relative vectors enter the node / edge MLPs
                       and the displacement (and its noise) is a free 2-D readout of h
          oneshot_T>0  no closed-loop prediction: at the last observed step the cell emits all oneshot_T future
                       displacements at once from the same encoder state"""
        super().__init__()
        self.D, self.C, self.k = D, C, k
        self.dx_max = 0.2                                                   # radial step bound (arena units)
        self.attn_intra = attn in ("both", "intra")
        self.attn_inter = attn in ("both", "inter")
        self.edge_state, self.inter_msg, self.noise_share, self.vec_nl = edge_state, inter_msg, noise_share, vec_nl
        self.attn_clamp, self.attn_temp, self.goal = attn_clamp, attn_temp, goal
        self.equiv, self.oneshot_T = equiv, oneshot_T
        self.xy_scale = 0.03                                                 # output scale of the free 2-D readout / noise (arena units,
                                                                             # ~ a typical step) so the non-equivariant variant starts at the
                                                                             # same step / noise magnitude as the vector-channel readout

        # lift: k relative vectors (k-1 past-relative frames + centroid offset) -> C vector channels
        w = torch.randn(C, k) * 0.1
        w[0].zero_(); w[0, k - 2] = -1.0                                    # V^0 = x_t - x_{t-1}
        if C > 1:
            w[1].zero_(); w[1, k - 1] = 1.0                                 # V^1 = x_t - centroid
        self.lift = nn.Parameter(w)
        self.node_emb = nn.Sequential(nn.Linear(k + 1 + C + (0 if equiv else 2 + 2 * k), D), act)

        # invariant edge features -> edge state
        self.kp_emb = nn.Embedding(n_kp_types, KP_EMB)
        self.rel_emb = nn.Embedding(2, 4)
        self.edge_mlp = nn.Sequential(nn.Linear(1 + 3 * C + 2 * KP_EMB + 4 + (0 if equiv else 6), D), act, nn.Linear(D, D), act)
        if edge_state:
            self.edge_gru = nn.GRUCell(D, D)
        if not equiv:                                                        # free 2-D readout / noise (residual on the last velocity)
            self.readout_xy = nn.Linear(D, 2)
            self.noise_xy = nn.Linear(D, 2)
        if oneshot_T:                                                        # one-shot multi-step readout / noise
            if equiv:
                self.readout_multi = nn.Linear(D, oneshot_T * C)
                self.noise_multi = nn.Linear(D, oneshot_T * C)
            else:
                self.readout_multi = nn.Linear(D, oneshot_T * 2)
                self.noise_multi = nn.Linear(D, oneshot_T * 2)

        # two-stream additive attention: score_ij = v . tanh(Q h_i + K s_ij), softmax within the stream
        self.attn_q = nn.Linear(D, D)
        self.attn_k_intra, self.attn_v_intra = nn.Linear(D, D), nn.Linear(D, 1)
        self.attn_k_inter, self.attn_v_inter = nn.Linear(D, D), nn.Linear(D, 1)

        # messages: scalar part (D) + weights on r_ij (C) and on V_j (C); one head per stream
        self.msg_dim = D + 2 * C
        self.msg_intra = nn.Sequential(nn.Linear(D, D), act, nn.Linear(D, self.msg_dim))
        self.msg_inter = nn.Sequential(nn.Linear(D, D), act, nn.Linear(D, self.msg_dim))

        # node state, vector non-linearity, readout, noise
        self.gru = nn.GRUCell(2 * D, D)
        if vec_nl:
            self.Wq = nn.Linear(C, C, bias=False); self.Wk = nn.Linear(C, C, bias=False)
            self.vec_gate = nn.Linear(D, C)
        self.readout = nn.Linear(D, C)
        self.noise = nn.Linear(D, C)

        # intention head: logit_ij = Linear(s_ij) for other-entity j, logit_none = Linear(h_i)
        if goal:
            self.goal_lin = nn.Linear(D, 1)
            self.goal_none = nn.Linear(D, 1)
        self.steer, self.goal_scale = bool(goal and steer), goal_scale
        if self.steer:
            self.steer_lin = nn.Linear(D, 1)                                 # steering gain m_i = Linear(h_i)

        with torch.no_grad():
            if self.steer:
                self.steer_lin.weight.zero_(); self.steer_lin.bias.zero_()                       # == the plain model at init
            if vec_nl:
                self.Wq.weight.copy_(torch.eye(C)); self.Wk.weight.copy_(torch.eye(C))
                self.vec_gate.weight.zero_(); self.vec_gate.bias.zero_()
            self.readout.weight.zero_(); self.readout.bias.zero_(); self.readout.bias[0] = 1.0   # constant velocity at init
            self.noise.weight.zero_(); self.noise.bias.fill_(-3.0)
            if not equiv:
                self.readout_xy.weight.zero_(); self.readout_xy.bias.zero_()                     # dx = last velocity at init
                self.noise_xy.weight.zero_(); self.noise_xy.bias.fill_(-3.0)
            if oneshot_T:
                self.readout_multi.weight.zero_(); self.readout_multi.bias.zero_()
                if equiv:
                    self.readout_multi.bias.view(oneshot_T, C)[:, 0] = 1.0                       # constant-velocity extrapolation at init
                self.noise_multi.weight.zero_(); self.noise_multi.bias.fill_(-3.0)
            for m in (self.msg_intra[-1], self.msg_inter[-1]):
                m.weight.zero_(); m.bias.zero_()                                                 # messages start at 0
            for m in (self.attn_v_intra, self.attn_v_inter):
                m.weight.zero_(); m.bias.zero_()                                                 # uniform attention at init

    # ------------------------------------------------------------------
    def vec_nonlinear(self, V):
        """gated vector non-linearity (equivariant): q where <q,k> >= 0, else q minus its projection on k."""
        q = self.Wq(V.transpose(-1, -2)).transpose(-1, -2)
        kk = self.Wk(V.transpose(-1, -2)).transpose(-1, -2)
        prod = (q * kk).sum(-1, keepdim=True)
        pos = (prod >= 0).to(V.dtype)
        return pos * q + (1 - pos) * (q - prod / ((kk * kk).sum(-1, keepdim=True) + 1e-4) * kk)

    def _goal(self, s, h, m_inter, goal_detach):
        """intention logits: logit_ij = Linear(s_ij) for other-entity j, logit_none = Linear(h_i) -> (softmax P, logits)"""
        s_g, h_g = (s.detach(), h.detach()) if goal_detach else (s, h)
        lg = torch.cat([self.goal_lin(s_g).squeeze(-1).masked_fill(m_inter == 0, -1e9), self.goal_none(h_g)], -1)
        return F.softmax(lg, -1), lg

    def _attend(self, h, s, stream_mask, use_attn, Kv, Kk):
        """weights (B,N,N) that sum to the neighbour count of the row within the stream."""
        n_nb = stream_mask.sum(-1, keepdim=True)
        if not use_attn:
            return stream_mask
        q = self.attn_q(h)[:, :, None]
        sc = Kv(torch.tanh(q + Kk(s))).squeeze(-1)
        sc = sc.clamp(-self.attn_clamp, self.attn_clamp) / self.attn_temp
        sc = sc.masked_fill(stream_mask == 0, -1e9)
        return F.softmax(sc, -1) * stream_mask * n_nb

    def sample_eps(self, B, N, ent, dim=None):
        dim = dim or self.C
        if self.noise_share == "entity":
            e = torch.randn(B, int(ent.max()) + 1, dim, device=ent.device)
            return e[:, ent]                                                 # shared by the entity's nodes
        return torch.randn(B, N, dim, device=ent.device)

    def _bound(self, dx):
        n = snorm(dx, keepdim=True)
        return dx * (self.dx_max * torch.tanh(n / self.dx_max) / n)

    def forward(self, buf, h, s, types, mask, noise_on, decision=None, goal_detach=False, multi=False, goal_noise=None):
        """decision (B,N) long or None: target node index for a taken-over row, N for 'keep the learned attention'.
        goal_noise (B,N,N+1) or None: Gumbel perturbation of the intention logits used by the steering term (mode sampling).
        Returns dx (sampled), dx_mean, h, s, aux{w_intra, w_inter, pull[, goal, steer_gain]};
        multi=True (one-shot variant): dx and dx_mean are (B,N,oneshot_T,2), all future steps at once."""
        B, N, k, _ = buf.shape
        C, D = self.C, self.D
        xt = buf[:, :, -1]
        m2 = mask[:, :, None]
        centroid = (xt * m2).sum(1, keepdim=True) / mask.sum(1).clamp(min=1)[:, None, None]
        u = torch.cat([buf[:, :, :-1] - xt[:, :, None], (xt - centroid)[:, :, None]], 2)      # B,N,k,2 relative vectors
        V = torch.einsum('bnmd,cm->bncd', u, self.lift)                      # B,N,C,2

        d_last = buf[:, :, -1] - buf[:, :, -2]
        d_prev = buf[:, :, -2] - buf[:, :, -3] if k >= 3 else d_last
        cos_turn = (d_last * d_prev).sum(-1) / ((snorm(d_last) + EPS) * (snorm(d_prev) + EPS))
        n_feat = [snorm(u), cos_turn[..., None], snorm(V)]
        if not self.equiv:
            n_feat += [xt, u.reshape(B, N, -1)]                              # absolute position + raw relative vectors
        n_in = self.node_emb(torch.cat(n_feat, -1))

        # invariant edge features
        r = xt[:, None, :, :] - xt[:, :, None, :]                            # x_j - x_i
        dist = snorm(r, keepdim=True)
        uij = r / (dist + EPS)
        proj_i = torch.einsum('bicd,bijd->bijc', V, uij)
        proj_j = torch.einsum('bjcd,bijd->bijc', V, uij)
        vv = torch.einsum('bicd,bjcd->bijc', V, V)
        kp_i = self.kp_emb(types['kp'])[None, :, None].expand(B, N, N, -1)
        kp_j = self.kp_emb(types['kp'])[None, None, :].expand(B, N, N, -1)
        rel = self.rel_emb((~types['intra']).long())[None].expand(B, N, N, -1)
        e_feat = [dist, proj_i, proj_j, vv, kp_i, kp_j, rel]
        if not self.equiv:
            e_feat += [r, xt[:, :, None].expand(B, N, N, 2), xt[:, None].expand(B, N, N, 2)]    # raw r_ij, x_i, x_j
        e = self.edge_mlp(torch.cat(e_feat, -1))
        s = self.edge_gru(e.reshape(B * N * N, D), s.reshape(B * N * N, D)).reshape(B, N, N, D) if self.edge_state else e

        # graph weights (small graph / big graph)
        valid = mask[:, :, None] * mask[:, None, :]
        m_intra = types['intra'][None].to(buf.dtype) * valid
        m_inter = types['inter'][None].to(buf.dtype) * valid
        w_intra = self._attend(h, s, m_intra, self.attn_intra, self.attn_v_intra, self.attn_k_intra)
        w_inter = self._attend(h, s, m_inter, self.attn_inter, self.attn_v_inter, self.attn_k_inter)
        aux = {}; steer_vec = None
        if self.goal and not self.steer:                                     # steer=False: intention head read before the state update
            aux['goal'] = self._goal(s, h, m_inter, goal_detach)[0]         # B,N,N+1  (index N = none)
        if decision is not None:                                             # take-over (analysis only)
            took = decision < N
            onehot = F.one_hot(decision.clamp(max=N - 1), N).to(buf.dtype) * m_inter.sum(-1, keepdim=True)
            w_inter = torch.where(took[..., None], onehot, w_inter)

        # messages and state updates
        msg = self.msg_intra(s) * w_intra[..., None]
        if self.inter_msg:
            msg = msg + self.msg_inter(s) * w_inter[..., None]
        m_s, a, b = msg[..., :D], msg[..., D:D + C], msg[..., D + C:]
        M = m_s.sum(2)
        dV = torch.einsum('bijc,bijd->bicd', a, r) + torch.einsum('bijc,bjcd->bicd', b, V)
        h = self.gru(torch.cat([n_in, M], -1).reshape(B * N, -1), h.reshape(B * N, -1)).reshape(B, N, D)
        V = V + dV
        if self.vec_nl:
            V = self.vec_nonlinear(V) * torch.sigmoid(self.vec_gate(h))[..., None] * 2
        if self.goal and self.steer:                                         # ERR: head and steering read the UPDATED state h_t
            P, lg = self._goal(s, h, m_inter, goal_detach)
            aux['goal'] = P
            P_use = F.softmax(lg + goal_noise, -1) if goal_noise is not None else P
            gain = self.steer_lin(h).squeeze(-1)
            aux['steer_gain'] = gain
            steer_vec = gain[..., None] * torch.einsum('bij,bijd->bid', P_use[..., :N] * m_inter, uij) * self.goal_scale

        coef = self.readout(h)
        aux.update(w_intra=w_intra, w_inter=w_inter, pull=torch.einsum('bic,bijc->bij', coef, a))
        if multi:                                                            # one-shot: all oneshot_T displacements from this state
            Tm = self.oneshot_T
            if self.equiv:
                cm = self.readout_multi(h).view(B, N, Tm, C)
                dx_mean = self._bound(torch.einsum('bntc,bncd->bntd', cm, V))
                if noise_on:
                    sig = torch.sigmoid(self.noise_multi(h)).view(B, N, Tm, C)
                    eps = self.sample_eps(B, N, types['ent'])[:, :, None]
                    dx = self._bound(dx_mean + torch.einsum('bntc,bncd->bntd', sig * eps, V))
                else:
                    dx = dx_mean
            else:
                dx_mean = self._bound(d_last[:, :, None] + self.xy_scale * self.readout_multi(h).view(B, N, Tm, 2))
                if noise_on:
                    sig = torch.sigmoid(self.noise_multi(h)).view(B, N, Tm, 2)
                    dx = self._bound(dx_mean + self.xy_scale * sig * self.sample_eps(B, N, types['ent'], 2)[:, :, None])
                else:
                    dx = dx_mean
            return dx, dx_mean, h, s, aux
        if self.equiv:
            step = torch.einsum('bnc,bncd->bnd', coef, V)
            if steer_vec is not None:
                step = step + steer_vec
            dx_mean = self._bound(step)
        else:
            dx_mean = self._bound(d_last + self.xy_scale * self.readout_xy(h))
        if noise_on:
            if self.equiv:
                sig = torch.sigmoid(self.noise(h))
                eps = self.sample_eps(B, N, types['ent'])
                dx = self._bound(dx_mean + torch.einsum('bnc,bncd->bnd', sig * eps, V))
            else:
                dx = self._bound(dx_mean + self.xy_scale * torch.sigmoid(self.noise_xy(h)) * self.sample_eps(B, N, types['ent'], 2))
        else:
            dx = dx_mean
        return dx, dx_mean, h, s, aux


class ERR(nn.Module):
    """The full model: node-type buffers + the shared cell, rolled out over observed and predicted steps."""

    def __init__(self, n_nodes, kp_per_entity, D=64, C=8, k=3, device='cpu', goal=True, goal_detach=False, equiv=True, oneshot_T=0,
                 mode_sample=True, **cell_kw):
        super().__init__()
        self.N, self.kp, self.k = n_nodes, kp_per_entity, k
        self.M = n_nodes // kp_per_entity
        self.goal_detach, self.mode_sample = goal_detach, mode_sample
        self.cell = ERRCell(D, C, k, n_kp_types=kp_per_entity, goal=goal, equiv=equiv, oneshot_T=oneshot_T, **cell_kw)
        idx = torch.arange(n_nodes)
        ent = idx // kp_per_entity
        eye = torch.eye(n_nodes, dtype=torch.bool)
        self.register_buffer('kp_id', idx % kp_per_entity)
        self.register_buffer('ent', ent)
        self.register_buffer('intra', (ent[:, None] == ent[None, :]) & ~eye)
        self.register_buffer('inter', ent[:, None] != ent[None, :])
        self.to(device)

    def types(self):
        return {'kp': self.kp_id, 'ent': self.ent, 'intra': self.intra, 'inter': self.inter}

    def sample_goal_noise(self, BK, dtype, device):
        """mode sampling: one Gumbel vector per (sample, entity), shared by the entity's keypoints, kept for the rollout."""
        g = -torch.log(-torch.log(torch.rand(BK, self.M, self.N + 1, device=device, dtype=dtype).clamp(1e-10, 1 - 1e-10)))
        return g[:, self.ent]                                                                 # BK,N,N+1

    def forward(self, x_obs, num_valid, T_fut, K=1, noise=True, goal_source='none', pred_labels=None):
        """x_obs (B,N,T,2) -> pred (B,N,K,T_fut,2), obs_pred (B,N,T-1,2), aux (sample 0):
             w_intra / w_inter / pull (B,N,N,T+T_fut-1), goal (B,N,N+1,T+T_fut-1),
             goal_obs (B,N,N+1,T) intention at the observed steps (for the loss / recognition metrics)
        goal_source (prediction phase only, observed steps are never taken over; the decision is taken once, from the
        latest intention available before the first prediction, i.e. after the cell has ingested frame T-2):
             'none'    learned attention everywhere (training and the standard protocol)
             'label'   rows of pred_labels (B,N) < N are taken over by that target (intervention / oracle)
             'sample'  one hard draw per (sample, node) from that intention
             'argmax'  the most likely intention                    -> aux['decision'] (B,K,N)"""
        B, N, T, _ = x_obs.shape
        dev, dt = x_obs.device, x_obs.dtype
        types = self.types()
        x = x_obs.repeat_interleave(K, 0)
        nv = num_valid.to(dt).repeat_interleave(K, 0)
        mask = (torch.arange(N, device=dev)[None] < nv[:, None]).to(dt)
        BK = B * K
        buf = x[:, :, :1].expand(-1, -1, self.k, -1).contiguous()
        h = torch.zeros(BK, N, self.cell.D, device=dev, dtype=dt)
        s = torch.zeros(BK, N, N, self.cell.D, device=dev, dtype=dt)
        use = self.cell.goal and goal_source != 'none'
        oneshot = self.cell.oneshot_T > 0
        gnoise = self.sample_goal_noise(BK, dt, dev) if (self.mode_sample and self.cell.goal and self.cell.steer and noise) else None
        if oneshot:
            assert self.cell.oneshot_T == T_fut, "one-shot variant: T_fut must equal the model's oneshot_T"
        preds, obs_pred, aux_t, goal_obs = [], [], {}, []
        x_next, dec, last_goal = None, None, None
        for t in range(T if oneshot else T + T_fut - 1):
            xt = x[:, :, t] if t < T else x_next
            buf = torch.cat([buf[:, :, 1:], xt[:, :, None]], 2)
            predicting = t >= T - 1
            if use and predicting and dec is None:                           # decide once, at the last observed frame
                if goal_source == 'label':
                    dec = pred_labels.repeat_interleave(K, 0)
                elif goal_source == 'argmax':
                    dec = last_goal.argmax(-1)
                else:
                    g = -torch.log(-torch.log(torch.rand_like(last_goal).clamp(1e-10, 1 - 1e-10)))
                    dec = (last_goal.clamp(min=1e-12).log() + g).argmax(-1)
            dx, dx_mean, h, s, aux = self.cell(buf, h, s, types, mask, noise_on=(noise and predicting),
                                               decision=dec if predicting else None, goal_detach=self.goal_detach,
                                               multi=oneshot and predicting, goal_noise=gnoise if predicting else None)
            if 'goal' in aux:
                last_goal = aux['goal']
                if t < T:
                    goal_obs.append(aux['goal'])
            for kk, v in aux.items():
                aux_t.setdefault(kk, []).append(v)
            if predicting and oneshot:
                preds = xt[:, :, None] + dx.cumsum(2)                        # BK,N,T_fut,2
            elif predicting:
                x_next = xt + dx
                preds.append(x_next)
            else:
                obs_pred.append(xt + dx_mean)
        pred = (preds if oneshot else torch.stack(preds, 2)).view(B, K, N, T_fut, 2).permute(0, 2, 1, 3, 4)
        obs_pred = torch.stack(obs_pred, 2).view(B, K, N, T - 1, 2)[:, 0]
        aux_out = {}
        for kk, v in aux_t.items():
            st = torch.stack(v, -1)
            aux_out[kk] = st.view(B, K, *st.shape[1:])[:, 0]
        if goal_obs:
            st = torch.stack(goal_obs, -1)
            aux_out['goal_obs'] = st.view(B, K, *st.shape[1:])[:, 0]
        if dec is not None:
            aux_out['decision'] = dec.view(B, K, N)
        return pred, obs_pred, aux_out


# ───────────────────────────── losses ─────────────────────────────

def variety_loss(pred, gt, joint=True):
    """pred (B,N,K,T,2), gt (B,N,T,2).  joint: one best sample per scene (= the evaluation protocol)."""
    err = (pred - gt[:, :, None]).norm(dim=-1).mean(3)                      # B,N,K
    if joint:
        return err.mean(1).min(1)[0].mean()
    return err.min(2)[0].mean()


def energy_loss(pred, gt):
    """energy score per (node, step): E||X-y|| - 1/2 E||X-X'|| over the K samples (unbiased pairwise mean), averaged.
    A strictly proper scoring rule: the ablation objective that rewards a calibrated spread instead of the best sample."""
    K = pred.shape[2]
    d_xy = (pred - gt[:, :, None]).norm(dim=-1).mean(2)                                         # B,N,T
    d_xx = (pred[:, :, :, None] - pred[:, :, None, :]).norm(dim=-1).sum((2, 3)) / (K * (K - 1))   # B,N,T
    return (d_xy - 0.5 * d_xx).mean()


def body_pairs(n_nodes, kp_per_entity):
    ii, jj = [], []
    for e in range(n_nodes // kp_per_entity):
        for a, b in itertools.combinations(range(kp_per_entity), 2):
            ii.append(e * kp_per_entity + a); jj.append(e * kp_per_entity + b)
    return torch.tensor(ii), torch.tensor(jj)


def body_band(data, p_lo=1.0, p_hi=99.0):
    """data (W,T,M,KP,2): [lo, hi] percentiles per intra pair type (pooled over entities), in body_pairs order."""
    KP, n_ent = data.shape[3], data.shape[2]
    if KP < 2:
        return None
    pairs = list(itertools.combinations(range(KP), 2))
    d = np.stack([np.linalg.norm(data[..., a, :] - data[..., b, :], axis=-1) for a, b in pairs], -1).reshape(-1, len(pairs))
    lo, hi = np.percentile(d, p_lo, axis=0), np.percentile(d, p_hi, axis=0)
    return torch.tensor(np.tile(lo, n_ent), dtype=torch.float32), torch.tensor(np.tile(hi, n_ent), dtype=torch.float32)


def body_hinge(pos, idx_i, idx_j, lo, hi):
    """pos (B,N,...,2): zero inside [lo,hi], relative quadratic outside, mean over everything."""
    d = (pos[:, idx_i] - pos[:, idx_j]).norm(dim=-1)
    shp = (1, -1) + (1,) * (d.dim() - 2)
    lo, hi = lo.view(shp), hi.view(shp)
    return ((F.relu(lo - d) / lo) ** 2 + (F.relu(d - hi) / hi) ** 2).mean()


# ─────────────────────── self-supervised intention targets ───────────────────────

def goal_rho(data, frac=0.5, kp_subset=None):
    """contact radius = frac x median body length (largest intra-entity pairwise distance), computed on
    kp_subset (default: all keypoints) so that 4-kp and 12-kp runs can share the same radius."""
    if data.shape[3] == 1:
        return 100.0 / 710.0
    d = data if kp_subset is None else data[:, :, :, kp_subset]
    bl = np.linalg.norm(d[:, :, :, :, None] - d[:, :, :, None, :], axis=-1).max((-1, -2))
    return float(frac * np.median(bl))


@torch.no_grad()
def future_min_dist(full, inter, H, T_obs):
    """full (B,N,T,2) true positions -> (B,N,N,T_obs): for each observed step t the smallest distance
    between i and other-entity node j over frames t+1..t+H (inf for same-entity pairs)."""
    B, N, T, _ = full.shape
    d = (full[:, :, None] - full[:, None]).norm(dim=-1).masked_fill(~inter[None, :, :, None], float('inf'))
    return torch.stack([d[..., t + 1:min(t + H, T - 1) + 1].amin(-1) for t in range(T_obs)], -1)


@torch.no_grad()
def goal_labels(full, inter, rho, H, T_obs):
    """hard labels (B,N,T_obs) long: closest other-entity node within rho over the next H frames, else N."""
    dm = future_min_dist(full, inter, H, T_obs)
    dmin, j = dm.min(2)
    return torch.where(dmin < rho, j, torch.full_like(j, full.shape[1]))


@torch.no_grad()
def goal_targets(full, inter, rho, H, T_obs, tau=0.0):
    """targets (B,N,T_obs,N+1): tau = 0 (ERR) -> one-hot of goal_labels; tau > 0 -> the mass is shared among
    other-entity nodes within rho as exp(-(d - d_min) / tau) (tau in arena units); all on 'none' if nothing is within rho."""
    dm = future_min_dist(full, inter, H, T_obs).permute(0, 1, 3, 2)          # B,N,T_obs,N
    dmin = dm.min(-1, keepdim=True).values
    w = torch.exp(-(dm - dmin) / tau) if tau > 0 else (dm == dmin).to(dm.dtype)
    w = w.masked_fill(dm >= rho, 0.0)
    none = (dmin >= rho).to(dm.dtype)
    w = torch.cat([w, none], -1)
    return w / w.sum(-1, keepdim=True).clamp(min=1e-9)


def goal_loss(goal_obs, targets):
    """goal_obs (B,N,N+1,T_obs) probabilities, targets (B,N,T_obs,N+1) -> cross-entropy, mean over all rows."""
    lp = torch.log(goal_obs.clamp(min=1e-9)).permute(0, 1, 3, 2)
    return -(targets * lp).sum(-1).mean()


# ─────────────────────── recognition metrics of the intention head ───────────────────────

def rank_auc(score, pos):
    """Mann-Whitney AUC of score (1-D) for the binary mask pos; nan if one class is empty."""
    score, pos = np.asarray(score, float), np.asarray(pos, bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    order = score.argsort(kind='mergesort'); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s_sorted = score[order]; i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


@torch.no_grad()
def goal_stats(goal_obs, labels, N, kp_per_entity, group_of=None, near_node=None, in_contact=None):
    """recognition metrics at the last observed step.
    goal_obs (B,N,N+1,T_obs), labels (B,N,T_obs) hard labels, group_of (KP,) long: coarse part group
    of each keypoint (None -> every keypoint is its own group).  near_node (B,N) long: currently nearest
    other-entity node, in_contact (B,N) bool: an ongoing contact now -> NEW-contact metrics (the predictive
    part: contacts that have not started yet) with the nearest-node-now geometry baseline.
    Returns per-batch sums + counts so the caller can pool over batches."""
    P = goal_obs[..., -1]; y = labels[..., -1]
    pred = P.argmax(-1)
    contact = y < N
    out = {'n': y.numel(), 'acc': float((pred == y).float().sum()), 'none_pred': float((pred == N).float().sum()),
           'none_true': float((~contact).float().sum()), 'n_contact': int(contact.sum())}
    out['p_contact'] = (1.0 - P[..., N]).flatten().cpu().numpy(); out['y_contact'] = contact.flatten().cpu().numpy()
    if contact.any():
        part = P[..., :N].argmax(-1)
        out['part_top1'] = float((part == y)[contact].float().sum())
        out['contact_recall'] = float((pred != N)[contact].float().sum())
        g = (torch.arange(N, device=y.device) % kp_per_entity) if group_of is None else group_of[torch.arange(N, device=y.device) % kp_per_entity]
        ent = torch.arange(N, device=y.device) // kp_per_entity
        yt = y.clamp(max=N - 1)
        same = (ent[part] == ent[yt]) & (g[part] == g[yt])
        out['group_top1'] = float(same[contact].float().sum())
        out['entity_top1'] = float((ent[part] == ent[yt])[contact].float().sum())
        if near_node is not None and in_contact is not None:
            new = contact & ~in_contact; ong = contact & in_contact
            geo = (ent[near_node] == ent[yt]) & (g[near_node] == g[yt])
            out['new_n'] = int(new.sum()); out['new_top1'] = float(same[new].float().sum()); out['new_geo'] = float(geo[new].float().sum())
            out['new_recall'] = float((pred != N)[new].float().sum())
            out['ong_n'] = int(ong.sum()); out['ong_top1'] = float(same[ong].float().sum()); out['ong_geo'] = float(geo[ong].float().sum())
            notc = ~contact & ~in_contact                                      # no contact now and none coming: false alarms
            out['nc_n'] = int(notc.sum()); out['nc_pred_contact'] = float((pred != N)[notc].float().sum())
    return out


def pool_goal_stats(stats):
    """pool per-batch goal_stats into rates + contact AUC."""
    n = sum(s['n'] for s in stats); nc = sum(s['n_contact'] for s in stats)
    out = {'goal_acc': sum(s['acc'] for s in stats) / n, 'goal_none_pred': sum(s['none_pred'] for s in stats) / n,
           'goal_none_true': sum(s['none_true'] for s in stats) / n}
    for k in ('part_top1', 'contact_recall', 'group_top1', 'entity_top1'):
        out['goal_' + k] = sum(s.get(k, 0.0) for s in stats) / max(nc, 1)
    out['goal_contact_auc'] = rank_auc(np.concatenate([s['p_contact'] for s in stats]), np.concatenate([s['y_contact'] for s in stats]))
    for pre in ('new', 'ong'):
        nn_ = sum(s.get(pre + '_n', 0) for s in stats)
        if nn_:
            out[f'goal_{pre}_n'] = nn_
            out[f'goal_{pre}_group_top1'] = sum(s.get(pre + '_top1', 0.0) for s in stats) / nn_
            out[f'goal_{pre}_geo_top1'] = sum(s.get(pre + '_geo', 0.0) for s in stats) / nn_
    nn_ = sum(s.get('new_n', 0) for s in stats)
    if nn_:
        out['goal_new_recall'] = sum(s.get('new_recall', 0.0) for s in stats) / nn_
    nn_ = sum(s.get('nc_n', 0) for s in stats)
    if nn_:
        out['goal_false_alarm'] = sum(s.get('nc_pred_contact', 0.0) for s in stats) / nn_
    return out
