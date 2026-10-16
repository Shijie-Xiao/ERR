"""Checks for the ERR model (err/model.py).  CPU, double precision:  python tests/test_model.py

  1. init = constant-velocity extrapolation (steering gain and messages start at zero)
  2. exact E(2) equivariance of the mean rollout (rotation / reflection + translation), also with the steering term active
     and with a take-over; intention and attention weights invariant; sampled rollout equivariant for a matched noise draw
  3. the intention head reads the UPDATED state (steer=1) and the steering term moves the trajectory once the gain is non-zero
  4. mode sampling: samples differ, the Gumbel draw is shared by an entity's keypoints and fixed over the rollout
  5. take-over mechanics: row = K_inter * onehot(target) in the prediction phase only, decisions persist
  6. self-supervised labels / targets on a synthetic scene; hard == tau 0; contact radius from the legacy keypoints
  7. the intention loss reaches the trunk (and not with goal_detach)
  8. recognition metrics on synthetic inputs
  9. 36-node path on the real 12-kp windows (if data/windows is present): legacy nodes == the 4-kp file, imputation
 10. ablation switches run; overfit on 32 real windows (variety + goal CE decrease)
"""
import math
import os
import sys
import types as _t

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from err.model import (ERR, variety_loss, goal_labels, goal_targets, goal_loss, goal_rho, goal_stats, pool_goal_stats,    # noqa: E402
                        body_pairs, body_band, body_hinge, rank_auc)
from err.data import MiceWindowDataset, fill_missing_keypoints                                                         # noqa: E402

torch.manual_seed(0)
dev = torch.device('cpu')
T_OBS, T_FUT, N, KP, M = 10, 10, 12, 4, 3
DATA = os.path.join(ROOT, 'data', 'windows')
NPZ4 = os.path.join(DATA, 'dataset_social_r1_w20_fs4_seqsplit.npz')
NPZ12 = os.path.join(DATA, 'dataset_social_r1_w20_fs4_seqsplit_12kpfull.npz')
HAVE_DATA = os.path.exists(NPZ4) and os.path.exists(NPZ12)


def ok(name, cond, detail=''):
    print(f'  [{"PASS" if cond else "FAIL"}] {name} {detail}', flush=True)
    return bool(cond)


def scene(B, n=N, kp=KP, m=M):
    x = ((torch.rand(B, n, T_OBS, 2, dtype=torch.double) - 0.5).cumsum(2) * 0.02 + 0.3)
    x = x + (torch.arange(m, dtype=torch.double).repeat_interleave(kp) * 0.25)[None, :, None, None] * torch.tensor([1.0, 0.3], dtype=torch.double)
    return x + (torch.rand(n, 2, dtype=torch.double) * 0.04)[None, :, None]


def perturb(model, scale=0.05):
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * scale)


def fixed_bank(models, B, K, seed=3):
    """monkeypatch sample_eps on each model to draw from the same pre-drawn bank (restore with del)."""
    torch.manual_seed(seed); bank = [torch.randn(B * K, M, 8, dtype=torch.double) for _ in range(T_FUT)]
    for mm in models:
        it = iter(bank); mm.cell.sample_eps = _t.MethodType(lambda self, B_, N_, ent, _it=it: next(_it)[:, ent], mm.cell)


def main():
    all_ok = True
    B = 4; x = scene(B); nv = torch.full((B,), N, dtype=torch.double)
    dec = torch.full((B, N), N, dtype=torch.long); dec[:, 0] = 7; dec[:, 5] = 10

    # ── 1. constant-velocity init ──
    mi = ERR(N, KP).double().eval()
    with torch.no_grad():
        pi, oi, ai = mi(x, nv, T_FUT, K=1, noise=False)
    v = x[:, :, -1] - x[:, :, -2]; cv = x[:, :, -1, None] + v[:, :, None] * torch.arange(1, T_FUT + 1, dtype=torch.double)[None, None, :, None]
    d = (pi[:, :, 0] - cv).abs().max().item()
    all_ok &= ok('init == constant-velocity extrapolation', d < 1e-3, f'max|d|={d:.1e} (bounded step)')
    all_ok &= ok('init: steering gain is zero, intention head present', ai['steer_gain'].abs().max().item() == 0.0 and ai['goal'].shape == (B, N, N + 1, T_OBS + T_FUT - 1))

    # ── 2. equivariance (random weights, steering gain non-zero) ──
    m = ERR(N, KP).double(); perturb(m); m.eval()
    with torch.no_grad():
        m.cell.steer_lin.weight.normal_(0, 0.5); m.cell.steer_lin.bias.fill_(1.0)
    th = 0.9; R = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=torch.double)
    t = torch.tensor([0.4, -0.7], dtype=torch.double)
    for src, lab in (('none', None), ('label', dec)):
        with torch.no_grad():
            pred, _, aux = m(x, nv, T_FUT, K=1, noise=False, goal_source=src, pred_labels=lab)
        for name, Mt in [('rotation', R), ('reflection', torch.tensor([[-1., 0.], [0., 1.]], dtype=torch.double))]:
            with torch.no_grad():
                pred2, _, aux2 = m(x @ Mt.T + t, nv, T_FUT, K=1, noise=False, goal_source=src, pred_labels=lab)
            d = (pred2 - (pred @ Mt.T + t)).abs().max().item(); dw = max((aux2[k] - aux[k]).abs().max().item() for k in ('goal', 'w_inter', 'w_intra', 'steer_gain'))
            all_ok &= ok(f'E(2) equivariance [{src}] ({name}+translation); intention / attention / gain invariant', d < 1e-8 and dw < 1e-8, f'max|d|={d:.1e} / {dw:.1e}')
    # sampled rollout: equivariant in distribution -> exactly equivariant for a matched noise bank (and the same Gumbel draw)
    fixed_bank([m], B, 2); torch.manual_seed(5)
    with torch.no_grad():
        ps, _, _ = m(x, nv, T_FUT, K=2, noise=True)
    del m.cell.sample_eps; fixed_bank([m], B, 2); torch.manual_seed(5)
    with torch.no_grad():
        ps2, _, _ = m(x @ R.T + t, nv, T_FUT, K=2, noise=True)
    del m.cell.sample_eps
    d = (ps2 - (ps @ R.T + t)).abs().max().item(); all_ok &= ok('sampled rollout (noise + mode sampling) equivariant for a matched draw', d < 1e-8, f'max|d|={d:.1e}')

    # ── 3. head placement and steering ──
    m0 = ERR(N, KP).double(); m0.load_state_dict(m.state_dict()); m0.eval()
    with torch.no_grad():
        m0.cell.steer_lin.weight.zero_(); m0.cell.steer_lin.bias.zero_()
        p_on, _, a_on = m(x, nv, T_FUT, K=1, noise=False); p_off, _, a_off = m0(x, nv, T_FUT, K=1, noise=False)
    all_ok &= ok('steering term moves the prediction; intention at the observed steps unchanged by the gain (same h, s)',
                 (p_on - p_off).abs().max().item() > 1e-6 and (a_on['goal_obs'] - a_off['goal_obs']).abs().max().item() < 1e-12)
    mf = ERR(N, KP, steer=False).double(); res = mf.load_state_dict(m.state_dict(), strict=False)
    all_ok &= ok("steer=False differs only by the steering gain parameters", set(res.unexpected_keys) == {'cell.steer_lin.weight', 'cell.steer_lin.bias'} and not res.missing_keys)
    mf.eval()
    with torch.no_grad():
        _, _, a_f = mf(x, nv, T_FUT, K=1, noise=False)
    all_ok &= ok('steer=False reads the head before the state update -> different intention than ERR', (a_f['goal_obs'] - a_on['goal_obs']).abs().max().item() > 1e-6)

    # ── 4. mode sampling ──
    g = m.sample_goal_noise(B, torch.double, dev)
    same_ent = all(torch.equal(g[:, e * KP], g[:, e * KP + k]) for e in range(M) for k in range(KP))
    all_ok &= ok('mode sampling: one Gumbel vector per entity, shared by its keypoints', same_ent and g.shape == (B, N, N + 1))
    torch.manual_seed(2)
    with torch.no_grad():
        pk, _, _ = m(x, nv, T_FUT, K=3, noise=True)
    all_ok &= ok('samples differ (noise + mode)', (pk[:, :, 0] - pk[:, :, 1]).abs().max().item() > 1e-6)
    mns = ERR(N, KP, mode_sample=False).double(); mns.load_state_dict(m.state_dict()); mns.eval()
    fixed_bank([m, mns], B, 1)
    with torch.no_grad():
        a1 = m(x, nv, T_FUT, K=1, noise=True)[0]; a2 = mns(x, nv, T_FUT, K=1, noise=True)[0]
    del m.cell.sample_eps; del mns.cell.sample_eps
    all_ok &= ok('mode_sample=False: same noise, different trajectory (the Gumbel term is the only difference)', (a1 - a2).abs().max().item() > 1e-8)

    # ── 5. take-over mechanics ──
    with torch.no_grad():
        pn, _, an = m(x, nv, T_FUT, K=1, noise=False); pl, _, al = m(x, nv, T_FUT, K=1, noise=False, goal_source='label', pred_labels=dec)
    w = al['w_inter']; row0 = w[:, 0, :, T_OBS - 1:]
    all_ok &= ok('take-over: row 0 == K_inter * onehot(7) in the prediction phase', torch.allclose(row0[:, 7], torch.full_like(row0[:, 7], 8.0)) and row0[:, [j for j in range(N) if j != 7]].abs().max() < 1e-12)
    all_ok &= ok('take-over: observed steps untouched', torch.allclose(w[:, 0, :, :T_OBS - 1], an['w_inter'][:, 0, :, :T_OBS - 1]))
    all_ok &= ok("take-over: 'none' rows keep the learned attention (first predicted step)", torch.allclose(w[:, 1, :, T_OBS - 1], an['w_inter'][:, 1, :, T_OBS - 1]))
    with torch.no_grad():
        torch.manual_seed(1); ps, _, as_ = m(x, nv, T_FUT, K=6, noise=True, goal_source='sample'); pa_, _, aa = m(x, nv, T_FUT, K=1, noise=False, goal_source='argmax')
    decs = as_['decision']; took = decs[:, 0] < N; row = as_['w_inter'][:, :, :, T_OBS - 1:]
    # the decision is taken before the cell ingests the last observed frame, i.e. from the intention after frame T-2 (goal_obs[..., -2])
    all_ok &= ok("'argmax' decision == argmax of the latest intention available at decision time (after frame T-2)", torch.equal(aa['decision'][:, 0], aa['goal_obs'][..., -2].argmax(-1)))
    all_ok &= ok("'sample': decisions differ across samples and persist over the rollout", (decs[:, 0] != decs[:, 1]).any().item() and
                 all(torch.allclose(row[b, i], row[b, i, :, :1].expand_as(row[b, i])) for b in range(B) for i in range(N) if took[b, i]))

    # ── 6. labels / targets ──
    inter = ERR(N, KP).inter
    full = torch.zeros(1, N, T_OBS + T_FUT, 2, dtype=torch.double)
    for e in range(M):                                                                         # three entities far apart
        full[0, e * KP:(e + 1) * KP, :, 0] = e * 1.0 + torch.arange(KP, dtype=torch.double)[:, None] * 0.02
    full[0, 0, :, 0] = torch.cat([torch.zeros(T_OBS, dtype=torch.double), torch.linspace(0.0, 1.0, T_FUT, dtype=torch.double)])   # node 0 runs to entity 1
    rho = 0.03
    lab = goal_labels(full, inter, rho, T_FUT, T_OBS)
    still = [1, 2, 3, 6, 7, 8, 9, 10, 11]                                                        # never within rho of another entity
    all_ok &= ok('label: approaching node -> closest other-entity node (4 = entity 1 nose) once the contact is within the horizon; '
                 'approached nodes 4/5 -> node 0; others -> none',
                 lab[0, 0, -1].item() == 4 and lab[0, 0, 0].item() == N and (lab[0, still, :] == N).all().item() and
                 lab[0, 4, -1].item() == 0 and lab[0, 5, -1].item() == 0 and (lab[0, 4:6, 0] == N).all().item())
    tg0 = goal_targets(full, inter, rho, T_FUT, T_OBS, 0.0); tg1 = goal_targets(full, inter, rho, T_FUT, T_OBS, 0.03)
    all_ok &= ok('targets: sum to 1, hard (tau=0) == onehot(label), soft mass only within rho with argmax = label',
                 torch.allclose(tg0.sum(-1), torch.ones_like(tg0.sum(-1))) and torch.allclose(tg1.sum(-1), torch.ones_like(tg1.sum(-1))) and
                 torch.equal(tg0.argmax(-1), lab) and torch.equal(tg1.argmax(-1), lab) and (tg1[0, 0, -1, 4] < 1.0).item() and
                 (tg1[0, 0, -1, [j for j in range(N + 1) if j not in (4, 5, 6, 7)]].sum() < 1e-12).item())
    if HAVE_DATA:
        r4 = goal_rho(np.load(NPZ4)['train_data'][:2000]); r12 = goal_rho(np.load(NPZ12)['train_data'][:2000], kp_subset=[0, 3, 6, 9]); r12all = goal_rho(np.load(NPZ12)['train_data'][:2000])
        all_ok &= ok('rho: legacy-kp body length on the 12-kp file == 4-kp file; all-kp would be larger', abs(r4 - r12) < 1e-7 and r12all > 1.3 * r4, f'{r4 * 710:.1f} / {r12 * 710:.1f} / {r12all * 710:.1f} px')

    # ── 7. loss into the trunk ──
    xf = x.float(); nvf = torch.full((B,), float(N)); fullf = torch.cat([xf, xf[:, :, -1:].expand(-1, -1, T_FUT, -1)], 2)
    for detach in (False, True):
        mt = ERR(N, KP, goal_detach=detach).train(); perturb(mt, 0.02)
        _, _, aux = mt(xf, nvf, T_FUT, K=1, noise=False)
        goal_loss(aux['goal_obs'], goal_targets(fullf, mt.inter, 0.3, T_FUT, T_OBS, 0.0)).backward()
        trunk = mt.cell.edge_mlp[0].weight.grad; head = mt.cell.goal_lin.weight.grad
        all_ok &= ok(f'goal loss -> trunk gradient {"absent" if detach else "present"} (goal_detach={detach}), head gradient present',
                     (head is not None and head.abs().sum() > 0) and ((trunk is None or trunk.abs().sum() == 0) if detach else (trunk is not None and trunk.abs().sum() > 0)))

    # ── 8. recognition metrics ──
    all_ok &= ok('rank AUC: perfect 1.0, inverted 0.0, ties 0.5', rank_auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0 and rank_auc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0 and rank_auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5)
    P = torch.zeros(1, N, N + 1, 1); y = torch.full((1, N, 1), N, dtype=torch.long)
    P[0, 0, 5, 0] = 1.0; y[0, 0, 0] = 4                                                         # predicted entity-1 neck, true entity-1 nose
    P[0, 1, 9, 0] = 1.0; y[0, 1, 0] = 4                                                         # predicted entity-2 neck, true entity-1 nose
    P[0, 2:, N, 0] = 1.0
    g12 = torch.tensor([0, 0, 1, 1])                                                             # groups: (nose, neck) / (cb, tail)
    st = pool_goal_stats([goal_stats(P, y, N, KP, g12)])
    all_ok &= ok('group / entity top-1 use the keypoint groups and the entity', abs(st['goal_group_top1'] - 0.5) < 1e-9 and abs(st['goal_entity_top1'] - 0.5) < 1e-9 and
                 abs(st['goal_part_top1'] - 0.0) < 1e-9 and abs(st['goal_contact_recall'] - 1.0) < 1e-9 and st['goal_contact_auc'] == 1.0)

    # ── 9. 36-node path on real windows ──
    if HAVE_DATA:
        ds = MiceWindowDataset(NPZ12, 'test', node_mode='all36')
        z12, z4 = np.load(NPZ12)['test_data'][:8], np.load(NPZ4)['test_data'][:8]
        leg = [mm * 12 + k for mm in range(3) for k in (0, 3, 6, 9)]
        seq = torch.stack([ds[i][0] for i in range(8)]); fut = torch.stack([ds[i][1] for i in range(8)])
        all_ok &= ok(f'all36: legacy nodes {leg[:4]}.. == the 4-kp file (same windows); {ds.n_filled} entries imputed',
                     np.allclose(seq.numpy()[:, leg], z4[:, :T_OBS].reshape(8, T_OBS, 12, 2).transpose(0, 2, 1, 3)) and ds.n_filled > 0)
        raw = z12.copy(); raw[2, 5:9, 1, 11] = 0.0; raw[3, :, 0, 2] = 0.0
        filled, n = fill_missing_keypoints(raw)
        lin = raw[2, 4, 1, 11] + (raw[2, 9, 1, 11] - raw[2, 4, 1, 11]) * np.arange(1, 5)[:, None] / 5.0
        all_ok &= ok('imputation: linear in time inside the window; whole-window gap -> center_back', n >= 24 and np.allclose(filled[2, 5:9, 1, 11], lin, atol=1e-6) and np.allclose(filled[3, :, 0, 2], raw[3, :, 0, 6]))
        m36 = ERR(36, 12).train(); nv36 = torch.full((8,), 36.0)
        pred, obs_pred, aux = m36(seq, nv36, T_FUT, K=2, noise=True)
        ii, jj = body_pairs(36, 12); lo, hi = body_band(np.load(NPZ12)['train_data'][:3000])
        rho12 = goal_rho(np.load(NPZ12)['train_data'][:3000], kp_subset=[0, 3, 6, 9])
        tg = goal_targets(torch.cat([seq, fut], 2), m36.inter, rho12, T_FUT, T_OBS, 0.0)
        loss = variety_loss(pred, fut) + body_hinge(pred, ii, jj, lo, hi) + 0.01 * goal_loss(aux['goal_obs'], tg); loss.backward()
        all_ok &= ok('all36 forward/backward: shapes and finite gradients', pred.shape == (8, 36, 2, T_FUT, 2) and aux['goal_obs'].shape == (8, 36, 37, T_OBS) and
                     all(p.grad is None or torch.isfinite(p.grad).all() for p in m36.parameters()) and torch.isfinite(loss).item())
    else:
        print('  [SKIP] real-data checks (data/windows not found)')

    # ── 10. ablations + overfit ──
    for kw in (dict(attn='none'), dict(edge_state=False), dict(inter_msg=False), dict(goal=False), dict(vec_nl=False), dict(noise_share='node'), dict(attn='intra'),
               dict(steer=False), dict(mode_sample=False), dict(equiv=False), dict(oneshot_T=T_FUT)):
        ma = ERR(N, KP, **kw).train(); pr, ob, au = ma(xf, nvf, T_FUT, K=2, noise=True)
        (variety_loss(pr, xf) + ob.abs().mean()).backward()
        all_ok &= ok(f'ablation {kw} runs', torch.isfinite(pr).all().item() and pr.shape == (B, N, 2, T_FUT, 2))
    if HAVE_DATA:
        rho = goal_rho(np.load(NPZ4)['train_data'][:3000])
        dd = np.load(NPZ4)['train_data'][:32].astype(np.float32)
        seq = torch.from_numpy(dd.reshape(32, 20, N, 2)).permute(0, 2, 1, 3); xo, xf_ = seq[:, :, :T_OBS], seq[:, :, T_OBS:]; nv32 = torch.full((32,), float(N))
        ii, jj = body_pairs(N, KP); lo, hi = body_band(np.load(NPZ4)['train_data'][:3000])
        tg = goal_targets(seq, ERR(N, KP).inter, rho, T_FUT, T_OBS, 0.0)
        model = ERR(N, KP).train(); opt = torch.optim.Adam(model.parameters(), lr=1e-3); lv, lg = [], []
        for it in range(120):
            pred, obs_pred, aux = model(xo, nv32, T_FUT, K=3, noise=True)
            l_var = variety_loss(pred, xf_); l_goal = goal_loss(aux['goal_obs'], tg)
            loss = l_var + (obs_pred - xo[:, :, 1:]).norm(dim=-1).mean() + body_hinge(pred, ii, jj, lo, hi) + 0.01 * l_goal
            opt.zero_grad(); loss.backward(); opt.step(); lv.append(l_var.item()); lg.append(l_goal.item())
        all_ok &= ok(f'overfit 32 windows: variety {lv[0] * 710:.1f} -> {lv[-1] * 710:.1f} px, goal CE {lg[0]:.2f} -> {lg[-1]:.2f}', lv[-1] < 0.85 * lv[0] and lg[-1] < 0.8 * lg[0])
    print('\nALL OK' if all_ok else '\nSOME CHECKS FAILED')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
