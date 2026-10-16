"""Window dataset for ERR.

Reads the npz window files written by data_prep/ (all coordinates already divided by the arena size):

    {split}_data  (N, 20, 3, 4, 2)   MABe22, 4 keypoints / mouse   (nose, neck, center_back, tail_base)
    {split}_data  (N, 20, 3, 12, 2)  MABe22, all 12 keypoints / mouse (same windows, '_12kpfull' twin file)
    {split}_data  (N, 20, 2, 7, 2)   CalMS21, 7 keypoints / mouse

Node layouts (node = entity * KP + kp):
    all12    3 mice x 4 kp  -> 12 nodes, center_back nodes [2, 6, 10]
    all36    3 mice x 12 kp -> 36 nodes, center_back nodes [6, 18, 30] (kp 6 = 'tail_1' of the MABe skeleton)
    cb       center_back only -> 3 nodes (single-point control)
    calms14  2 mice x 7 kp  -> 14 nodes

A sample is (past (A, T_obs, 2), future (A, T_fut, 2), num_valid) with A nodes; all nodes are valid.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

N_MICE = 3
N_KPS = 4
N_NODES = N_MICE * N_KPS                                   # 12
N_KPS_FULL = 12
N_NODES_FULL = N_MICE * N_KPS_FULL                         # 36
CB_NODES = [m * N_KPS + 2 for m in range(N_MICE)]          # [2, 6, 10]
CB_NODES_36 = [m * N_KPS_FULL + 6 for m in range(N_MICE)]  # [6, 18, 30]
NODE_MODES = ("all12", "all36", "cb", "calms14")


def fill_missing_keypoints(raw, thresh=1.5e-3, cb_kp=6):
    """raw (N, T, M, K, 2) normalised coordinates.  Window validity was checked on the 4 legacy keypoints only, so a
    12-kp window can still contain a few untracked keypoints stored as (0, 0) (0.06 % of the entries, tail tip / ears).
    Those are replaced by the linear interpolation in time of the same keypoint inside the window (nearest valid frame
    at the ends); if the keypoint is missing for the whole window it takes the mouse's center_back.
    Returns a copy and the number of filled entries."""
    raw = raw.copy()
    missing = np.abs(raw).sum(-1) < thresh                          # N,T,M,K
    n_fix = int(missing.sum())
    if n_fix == 0:
        return raw, 0
    T = raw.shape[1]
    for n, m, k in {(n, m, k) for n, _, m, k in zip(*np.nonzero(missing))}:
        bad = missing[n, :, m, k]
        good = ~bad
        if good.any():
            t_all = np.arange(T)
            for c in range(2):
                raw[n, bad, m, k, c] = np.interp(t_all[bad], t_all[good], raw[n, good, m, k, c])
        else:
            raw[n, bad, m, k] = raw[n, bad, m, cb_kp]
    return raw, n_fix


class MiceWindowDataset(Dataset):
    def __init__(self, npz_path, split="train", obs_len=10, pred_len=10, node_mode="all12", fill_missing=True):
        assert split in ("train", "val", "test")
        assert node_mode in NODE_MODES, f"node_mode must be one of {NODE_MODES}"
        d = np.load(npz_path)
        raw = d[f"{split}_data"].astype(np.float32)             # (N, T, M, K, 2)
        N, T = raw.shape[0], raw.shape[1]
        assert T == obs_len + pred_len, f"window {T} != obs+pred {obs_len + pred_len}"
        self.n_filled = 0

        if node_mode == "all36":
            assert raw.shape[2] == 3 and raw.shape[3] == 12, f"all36 needs a 12-kp npz, got {raw.shape[1:4]}"
            if fill_missing:
                raw, self.n_filled = fill_missing_keypoints(raw)
            nodes = raw.reshape(N, T, N_NODES_FULL, 2)
            self.n_agents = N_NODES_FULL
        elif node_mode == "calms14":
            assert raw.shape[2] == 2 and raw.shape[3] == 7, f"calms14 needs a (N,T,2,7,2) npz, got {raw.shape[1:4]}"
            nodes = raw.reshape(N, T, 14, 2)
            self.n_agents = 14
        else:
            assert raw.shape[3] == 4, f"all12/cb need the 4-kp npz, got kp={raw.shape[3]}"
            nodes = raw.reshape(N, T, N_NODES, 2)
            if node_mode == "cb":
                nodes = nodes[:, :, CB_NODES, :]
                self.n_agents = len(CB_NODES)
            else:
                self.n_agents = N_NODES

        self.all_data = torch.from_numpy(nodes.transpose(0, 2, 1, 3).copy())     # (N, A, T, 2)
        self.obs_len = obs_len
        self.pred_len = pred_len

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, idx):
        seq = self.all_data[idx]                    # (A, T, 2)
        past = seq[:, : self.obs_len]
        fut = seq[:, self.obs_len:]
        return past, fut, torch.tensor(float(self.n_agents))
