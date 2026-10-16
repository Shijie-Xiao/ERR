# Evaluation outputs of the released checkpoints

* `trajectory/` — `err/train.py --test` outputs produced with this package on `checkpoints/*.pth.tar`:
  event-centred MABe22 test split (`*_event_test.json`), natural sliding windows of the 160 held-out test sequences
  (`*_natural_test.json`) and the CalMS21 test videos (`err_calms14_test.json`).  Sources `none` (learned attention),
  `sample` (decisions drawn from the intention head) and `oracle` (true contact target), 20 samples, per-event-type metrics.
* `mabe22/` — `official_evaluator_err_36kp_pca128.csv`: the organisers' round-1 evaluator (github.com/damaggu/MABe2022) on the
  PCA-128 submission of `err_36kp` (`eval/mabe22/embed.py` + `make_submission.py`; 'Private Score' = the leaderboard
  column); `probe_*.json`: our re-implementation of the protocol (`eval/mabe22/probe.py`) on the full 216-d embedding, PCA-128,
  node memories only (h), graph features only, + hand-crafted geometry, and the hand-crafted baseline alone.
* `calms21/` — `eval/calms21/probe.py` outputs (linear + GBM, +-15/+-100-frame context, readout seeds s0-s2): zero-shot
  `err_12kp` (`err_zeroshot_err` = model features, `_all` = + 30-d geometry), CalMS21-trained `err_calms14`, geometry only,
  official MARS features, MARS + ERR, and the untrained control (`random_init_err_*`: `eval/random_checkpoint.py
  --node_mode all12 --seed 0`, model features only).
* `resifuse/` — ResiFuse Mouse Triplet protocol (`eval/resifuse/*`): linear / GBM readouts and ResiFuse's BiGRU head (5 seeds)
  for the released 36-node model (zero-shot on the corrected DeepLabCut poses) and for the hand-crafted features.

Files produced before the code was packaged keep research-repository paths in their `args`; the scripts and inputs are the
same as in this repository.
