# ERR — Equivariant Relational Recurrent model for multi-animal interaction

ERR is a small (103 k parameters) **E(2)-equivariant relational recurrent network** that predicts the future poses of
several interacting animals from their recent trajectories, **without any behaviour labels**.  While learning to predict, it
learns a readable interaction structure — *who* attends to whom (two-stream graph attention), *when* a contact is coming
(a self-supervised intention head) and *where* on the other body it will land.  Its frozen internal state, read out with
simple classifiers, transfers to labelled behaviour benchmarks: on MABe22 it scores at the level of the round-1 challenge
winner, on CalMS21 it exceeds the published supervised keypoint baselines even zero-shot (trained on a different assay), and
it is complementary to the hand-designed MARS features.

This repository is the self-contained, final version of the model used in the paper: the model, the preprocessed data,
the data preparation, the training recipe, the three released checkpoints and every evaluation.  The included data and
checkpoints reproduce every number reported below (`paper_results/`); training and trajectory evaluation run directly after
cloning.

## The model in one paragraph

A scene is a graph whose nodes are keypoints (`node = animal x KP + keypoint`).  One recurrent cell is shared by all nodes;
per time step it (1) lifts the last three positions of each node into `C = 8` vector channels built only from relative
vectors, (2) computes invariant node and edge features (speeds, turning, distances, projections, keypoint types), (3) updates
a per-edge GRU memory `s_ij` and a per-node GRU memory `h_i` (`D = 64`), (4) attends in two streams — the *small graph*
(a body's own keypoints) and the *big graph* (the other animals' keypoints) — with messages that are scalars plus weights on
the relative vectors, (5) reads the **intention** `P_i` over {other-animal keypoint `j`, none} from `(s_ij, h_i)`, and
(6) emits the next displacement `dx_i = bound( readout(h_i) · V_i + gain(h_i) · Σ_j P_ij r̂_ij · goal_scale )`, plus an
animal-shared noise term for sampling.  Every vector is a scalar-weighted combination of relative vectors, so the model is
exactly equivariant to rotations, reflections and translations.  Training is self-supervised: best-of-K trajectory loss,
an observed one-step loss, a body-shape hinge, and a cross-entropy on the intention head whose target is the closest
other-animal keypoint that comes within half a body length in the next 10 steps (else *none*).  During sampling, one Gumbel
perturbation of the intention logits per (sample, animal) selects a *mode* of interacting.

## Repository layout

```
err/                    the package
  model.py              ERRCell / ERR, losses, self-supervised intention targets, recognition metrics
  data.py               window dataset (MABe22 4-kp / 12-kp, CalMS21 7-kp layouts)
  train.py              training + trajectory evaluation (resumable, with the guards used for the 36-node runs)
  checkpoint.py         rebuild a model from a checkpoint's stored args (also for zero-shot transfer to 2 animals)
data/                   preprocessed datasets used by the released checkpoints (see data/README.md)
data_prep/              raw files -> the datasets in data/
  mabe_event_windows.py MABe22 -> event-centred, sequence-split windows (the training set)
  mabe_natural_test.py  natural sliding-window test set from the held-out sequences
  calms21_windows.py    CalMS21 task 1 -> 14-node windows
  resifuse_convert_poses.py / resifuse_subset_clips.py   ResiFuse DeepLabCut poses -> MABe format
eval/
  mabe22/               frame embeddings, hand-crafted baseline, 13-task linear readout, official-evaluator submission
  calms21/              zero-shot / trained embeddings, linear + GBM readouts, MARS feature import
  resifuse/             ResiFuse Mouse Triplet protocol: scored frames, linear/GBM readouts, ResiFuse's own BiGRU head
  random_checkpoint.py  untrained-weights control
tests/                  model checks (equivariance, steering, labels, ...)
checkpoints/            the three released ERR checkpoints (see checkpoints/README.md)
paper_results/          evaluation outputs of the released checkpoints produced with this code
```

## Installation

```bash
conda create -n err python=3.10 && conda activate err
pip install torch numpy scikit-learn            # core + readouts
pip install pandas tables huggingface_hub       # only to rebuild the ResiFuse files
```
Tested with Python 3.10, PyTorch 2.7 (CUDA 12.8) and scikit-learn 1.7; the model itself needs only `torch >= 1.13` and numpy.
All commands below are run from the repository root; a GPU with 8 GB is enough for the 12-node model, 16 GB for 36 nodes
(batch 32, `--accum 2`).

## Data

**Included.**  `data/` holds the preprocessed files the released checkpoints were trained and evaluated on, so training,
trajectory evaluation and the ResiFuse readouts run directly (details in `data/README.md`):

| file | content |
|---|---|
| `data/windows/dataset_social_r1_w20_fs4_seqsplit.npz` | MABe22 event-centred windows, 3 mice x 4 keypoints; train 31,976 / val 3,761 / test 4,203 |
| `data/windows/dataset_social_r1_w20_fs4_seqsplit_12kpfull.npz` | the same windows with all 12 keypoints (36 nodes) |
| `data/windows/dataset_natural_r1_w20_fs4_test{,_12kpfull}.npz` | natural sliding windows of the 160 held-out test sequences (8,137) |
| `data/windows/dataset_calms21_w20_fs4.npz` | CalMS21 task 1, 2 mice x 7 keypoints; train 22,514 / val 2,639 / test 13,041 |
| `data/resifuse/finetune108_dlcrf.npy`, `finetune_scored_frames.npz` | the 108 ResiFuse fine-tuning clips (DeepLabCut poses) and the frames / labels its protocol scores |

**Raw data** (only needed to rebuild these files and for the MABe22 / CalMS21 representation benchmarks, whose inputs are
too large to include):

| dataset | what to download | where to put it |
|---|---|---|
| MABe22 mouse triplets, trajectory track | `mouse_triplet_train.npy` (1,600 sequences, used for training; called `user_train_r1.npy` in the research code), `mouse_triplet_test.npy` + `mouse_triplets_test_labels.npy` (3736 sequences with the 13 hidden-task labels), evaluator metadata `jax_split.json`, `jax_tasks.json` — CaltechDATA record **8kdn3-95j37** and github.com/damaggu/MABe2022 (`round1/metadata/`) | `data/mabe22/` |
| MABe22 'submission' clips (video track, used only for the ResiFuse protocol) | `mouse_submission_keypoints.npy`, `mouse_submission_labels.npy` — CaltechDATA record **rdsa8-rde65** | `data/mabe22/` |
| CalMS21 task 1 | `task1_classic_classification.zip` (and `task1_MARS_features.zip` for the MARS comparison) — CaltechDATA record **s0vdx-0k302** | `data/calms21/` |
| ResiFuse re-estimated poses | Hugging Face `WeihanLikk/ResiFuse-MABe22` (`deeplabcut_filtered/*.h5`) and the code/splits of github.com/BRAINML-GT/ResiFuse | `data/resifuse/deeplabcut_filtered/`, `data/resifuse/code/` |

Rebuild the window datasets (coordinates are divided by the arena size, 7.5 Hz, 20-step windows = 10 observed + 10
predicted; the scripts reproduce the included files byte for byte):
```bash
python data_prep/mabe_event_windows.py          # reads data/mabe22/mouse_triplet_train.npy (or --raw <file>)
                                                # -> data/windows/dataset_social_r1_w20_fs4_seqsplit{,_12kpfull}.npz
                                                #    train 31,976 / val 3,761 / test 4,203 windows, split by sequence
python data_prep/mabe_natural_test.py           # -> data/windows/dataset_natural_r1_w20_fs4_test{,_12kpfull}.npz (8,137 windows)
python data_prep/calms21_windows.py             # -> data/windows/dataset_calms21_w20_fs4.npz (prints the px/unit scale: 1827.7)
```
The MABe22 training windows are **event-centred**: for every detected social event (approach, contact, anogenital, nose-nose,
following; thresholds in `data_prep/mabe_event_windows.py`) one window is cut so that the event falls 1–6 steps after the
last observed frame, plus one background window per event.  The split is by 60-s sequence (80/10/10, seed 42), so no frame
is shared between train and test.  The 12-kp file contains the same windows with all 12 keypoints.

## Training

```bash
# 3 mice x 4 keypoints (12 nodes)                                        ~2-4 min / epoch
python err/train.py --exp_name err_12kp --node_mode all12
# 3 mice x 12 keypoints (36 nodes)                                       ~9-14 min / epoch, fits in 16 GB
python err/train.py --exp_name err_36kp --node_mode all36 --batch_size 32 --accum 2 \
    --npz data/windows/dataset_social_r1_w20_fs4_seqsplit_12kpfull.npz
# CalMS21, 2 mice x 7 keypoints (14 nodes)
python err/train.py --exp_name err_calms14 --node_mode calms14 --arena_px 1827.7 \
    --npz data/windows/dataset_calms21_w20_fs4.npz
```
Defaults are the ERR recipe: Adam 1e-3 decayed x0.9 every 2 epochs, 60 epochs with early stopping (15), `K_train = 5`
samples, `lambda_goal = 0.01`, hard intention targets (`--soft_tau 0`), contact radius = half the median body length of the
4 legacy keypoints (35 px), steering unit = the mean centre step of the training set (`--goal_scale auto`), model
selection on the validation minADE20 of center_back, seed 42.  Outputs in `runs/`: `<exp>.pth.tar` (best epoch),
`<exp>_last.pth.tar` (resume point, `--resume`), `<exp>_log.csv` (per-epoch metrics) and `<exp>_test.json` (final test).
Training is seeded, but GPU kernels are not deterministic: a retrained model differs slightly from the released
checkpoints (two trainings of the same configuration differed by about 2 % in test minADE20 in our runs).

Evaluate a checkpoint (all decision sources: learned attention / sampled intentions / oracle intention):
```bash
python err/train.py --exp_name err_36kp --node_mode all36 --model_save_dir checkpoints --test \
    --npz data/windows/dataset_social_r1_w20_fs4_seqsplit_12kpfull.npz
python err/train.py --exp_name err_12kp --node_mode all12 --model_save_dir checkpoints --test \
    --npz data/windows/dataset_natural_r1_w20_fs4_test.npz              # natural (non-event-centred) windows
```

## Representation evaluation

**MABe22 (13 hidden tasks, official protocol).**  Frozen per-frame embeddings (216 dims: per-mouse mean node memory 3x64,
attention shares, P(contact), intention masses, one-step surprise, predicted speed, tracked flags) are read out with the
organisers' ridge protocol: trained on the 2668 `SubmissionTrain` sequences, scored on the 1068 test sequences.
```bash
python eval/mabe22/embed.py --ck checkpoints/err_36kp.pth.tar --npy data/mabe22/mouse_triplet_test.npy \
    --labels data/mabe22/mouse_triplets_test_labels.npy --out embeddings/mabe22/err_36kp       # ~1 h on a GPU
python eval/mabe22/handcrafted.py --npy data/mabe22/mouse_triplet_test.npy \
    --labels data/mabe22/mouse_triplets_test_labels.npy --out embeddings/mabe22/handcrafted     # geometry baseline
bash eval/mabe22/run_probes.sh err_36kp                                                          # our re-implementation
python eval/mabe22/make_submission.py --emb embeddings/mabe22/err_36kp --labels data/mabe22/mouse_triplets_test_labels.npy \
    --split data/mabe22/jax_split.json --out results/mabe22/submission_err_36kp.npy             # PCA-128, official format
# official evaluator (github.com/damaggu/MABe2022; with scikit-learn >= 1.6 patch round1/training/train_and_eval.py:
# alpha_scores.append(float(res[0])) ):
cd <MABe2022 clone> && python round1_evaluator.py --task mouse --submission <...>/submission_err_36kp.npy \
    --labels <...>/mouse_triplets_test_labels.npy --output-dir results_err
```

**CalMS21 task 1 (attack / investigation / mount, 70 train / 19 test videos).**  Two settings: zero-shot transfer of the
MABe22-trained 12-node model (CalMS21 keypoints mapped to nose / neck / hip midpoint / tail base, coordinates scaled to the
MABe body length, 3-mouse weights loaded into a 2-mouse graph) and the model trained on CalMS21 itself.  Readouts: logistic
regression and histogram gradient boosting on the frozen features with ±15 / ±100-frame moving-average context.
```bash
python eval/calms21/embed.py --ck checkpoints/err_12kp.pth.tar --out embeddings/calms21/err_zeroshot                # zero-shot
python eval/calms21/embed.py --ck checkpoints/err_calms14.pth.tar --nodes 7 --out embeddings/calms21/err_calms14     # trained
for s in 0 1 2; do python eval/calms21/probe.py --emb embeddings/calms21/err_zeroshot --feats all --seed $s \
    --out results/calms21/err_zeroshot_all_s$s.json; done             # --feats err (model only) | geo (geometry only)
python eval/calms21/mars_feats.py --mars_dir data/calms21/calms21_task1_features --ref embeddings/calms21/err_zeroshot \
    --out embeddings/calms21/mars                                      # official MARS features and MARS + ERR
python eval/random_checkpoint.py --node_mode all12 --out checkpoints/random_12kp.pth.tar   # untrained control -> embed + probe
```

**ResiFuse Mouse Triplet protocol** (4 classes, frame macro-F1 on the ResiFuse re-estimated DeepLabCut poses; the frames,
windows and splits are reproduced with the ResiFuse code so numbers are comparable within this pipeline).  The 108 clips
and the scored frames are included in `data/resifuse/`; the BiGRU head also needs a clone of github.com/BRAINML-GT/ResiFuse:
```bash
python eval/mabe22/embed.py --ck checkpoints/err_36kp.pth.tar --npy data/resifuse/finetune108_dlcrf.npy --out embeddings/resifuse/err_36kp
python eval/resifuse/readout.py --frames data/resifuse/finetune_scored_frames.npz --emb embeddings/resifuse/err_36kp --out results/resifuse/err_36kp.json
git clone https://github.com/BRAINML-GT/ResiFuse data/resifuse/code
python eval/resifuse/bigru_readout.py --code data/resifuse/code --frames data/resifuse/finetune_scored_frames.npz \
    --emb embeddings/resifuse/err_36kp --kind err --out results/resifuse/err_36kp_bigru.json
# rebuilding the two included files from the ResiFuse poses (Hugging Face) and the MABe22 'submission' labels:
python data_prep/resifuse_convert_poses.py --max_gap 0 --resifuse_validity 0.2 --tag dlcrf
python data_prep/resifuse_subset_clips.py --npy data/resifuse/mouse_submission_dlcrf.npy \
    --split data/resifuse/code/splits/mabe_finetune_split.json --out data/resifuse/finetune108_dlcrf.npy
python eval/resifuse/protocol_frames.py --code data/resifuse/code --h5_dir data/resifuse/deeplabcut_filtered \
    --labels data/mabe22/mouse_submission_labels.npy --split data/resifuse/code/splits/mabe_finetune_split.json \
    --out data/resifuse/finetune_scored_frames.npz
```

## Results of the released checkpoints

**Trajectory prediction** — 10 observed → 10 predicted steps at 7.5 Hz; minADE20 / minFDE20 = best of 20 joint samples on
the centre node (MABe22 center_back, CalMS21 neck); contact AUC = the intention head's contact-vs-none AUC at the last
observed step; part top-1 = the contacted keypoint (group = head / shoulders / back / tail); MABe22 in pixels of the
710-px arena, CalMS21 in video pixels (`paper_results/trajectory/`, produced with `err/train.py --test`):

| model | test set | minADE20 cb | minFDE20 cb | minADE20 all nodes | contact AUC | part top-1 |
|---|---|---|---|---|---|---|
| `err_12kp` (12 nodes) | MABe22 event-centred test split (4,203 windows) | 14.51 | 25.16 | 17.88 | 0.937 | 0.683 |
| `err_36kp` (36 nodes) | MABe22 event-centred test split | 14.70 | 25.54 | 18.33 | 0.956 | 0.512 (group 0.671) |
| `err_12kp` | MABe22 natural windows of the 160 test sequences (8,137) | 11.78 | 20.56 | 14.68 | 0.964 | 0.786 |
| `err_36kp` | MABe22 natural windows | 11.91 | 20.76 | 15.03 | 0.972 | 0.638 (group 0.765) |
| `err_calms14` (14 nodes) | CalMS21 test videos (13,041 windows) | 45.60 | 71.32 | 48.32 | 0.948 | 0.623 (group 0.751) |

The json files also contain the single-sample mean error, the energy score, the event-type breakdown and the 'sample' /
'oracle' take-over sources.

**MABe22 representation benchmark** — official evaluator, PCA-128 embedding of `err_36kp`, private test
(`paper_results/mabe22/`): mean F1 over the 11 binary tasks **0.307** (public 0.308).  Per task F1: strain .708, lights .680,
close .746, contact .597, huddles .318, watching .178, approaches .044, chases .032, oral-ear .012, oral-genital .025,
oral-oral .031; MSE experiment-day .093, time-of-day .094.  The round-1 leaderboard winner scored 0.303, the MABe22 paper's
T-BERT 0.297, PCA 0.196 (BAMS, NeurIPS 2023, 0.308 with transductive training on the test sequences).  ERR only sees
the MABe22 training file (1,280 of its 1,600 sequences are used for training), never a label or the test keypoints.  With our re-implementation of
the protocol the full 216-d embedding gives 0.314, node memories alone 0.314, graph features alone 0.248, ERR + hand-crafted
geometry 0.325, hand-crafted geometry alone 0.273.

**CalMS21 task 1** — class-averaged F1 / mean AP over attack, investigation, mount on the 19 test videos, frozen features,
GBM readout, mean of 3 readout seeds (`paper_results/calms21/`):

| features | training data of the features | avg F1 | MAP |
|---|---|---|---|
| ERR 12-node, zero-shot | MABe22 (different assay: 3 mice in an open arena; no CalMS21 frame seen) | 0.869 | 0.931 |
| ERR 12-node zero-shot + geometry (30-d) | — | 0.871 | 0.933 |
| ERR 14-node, trained on CalMS21 videos | CalMS21 train (no labels) | 0.871 | 0.933 |
| ERR 14-node + geometry | — | 0.874 | 0.936 |
| geometry only | — | 0.860 | 0.919 |
| ERR architecture, random weights (control) | none (`eval/random_checkpoint.py`, seed 0) | 0.800 | 0.865 |
| MARS official features (316-d) | hand-designed | 0.879 | 0.939 |
| MARS + ERR zero-shot | — | 0.885 | 0.941 |
| MARS + ERR CalMS21-trained | — | 0.885 | 0.944 |
| published: 1D-conv on keypoints / + task programming / MABe21 winner | supervised | 0.793 / 0.829 / 0.864 | 0.856 / 0.889 / 0.914 |

**ResiFuse Mouse Triplet protocol** — 4-class macro F1 (chases, huddles, oral contact, oral-genital contact) on the 1,328
scored frames of the 43 test clips, corrected DeepLabCut poses (`paper_results/resifuse/`).  The released `err_36kp` is
applied zero-shot: it was trained on the MABe22 training file, i.e. on other clips and on the official keypoints.

| features | ResiFuse's BiGRU head (5 seeds) | linear readout |
|---|---|---|
| ERR 36-node, zero-shot | **0.758 ± 0.004** (chases .849, huddles .919, oral .608, oral-genital .656) | 0.706 |
| hand-crafted geometry (39-d per animal) | 0.720 ± 0.009 | 0.671 |

The ResiFuse paper reports, with its own pipeline, hand-crafted 0.684, VideoPrism 0.722 and ResiFuse 0.784; these are
not calibrated against this pipeline (the same hand-crafted features score 0.720 here), so compare within the table.

## Ablations and variants (all switches of `err/train.py`)

| flag | default | meaning |
|---|---|---|
| `--steer 0` | 1 | no steering term; intention head read *before* the state update |
| `--mode_sample 0` | 1 | no Gumbel mode perturbation when sampling |
| `--soft_tau 10` | 0 | soft intention targets (parts within 10 px of the closest share the mass) |
| `--goal 0` | 1 | no intention head (trunk only) |
| `--goal_detach 1` | 0 | the intention loss does not reach the trunk |
| `--equiv 0` | 1 | symmetry removed: absolute coordinates into the MLPs, free 2-D readout |
| `--oneshot 1` | 0 | all future steps decoded at once from the last observed state (no closed loop) |
| `--attn intra/inter/none`, `--edge_state 0`, `--inter_msg 0`, `--noise_share node`, `--vec_nl 0` | | component ablations |
| `--traj_loss energy`, `--select es` | variety, cb | proper-scoring-rule objective / selection |
| `--augment_rot 1`, `--train_frac f` | 0, 1 | rotation-augmentation control, data-efficiency curve |

The research repository additionally explored part-level relation switches (body-frame invariants, part-typed queries,
attention supervision, onset-only intention loss); none improved the 36-node model and they are not part of this release.

## Tests

```bash
python tests/test_model.py                   # constant-velocity init, exact E(2) equivariance (also with steering, take-over,
                                             # sampling), head placement, mode sampling, labels / targets, loss into the trunk,
                                             # metrics, 36-node path on real windows, ablation switches, overfit on 32 windows
```

## License

The code and the released checkpoints are under the MIT License (`LICENSE`).  The files in `data/` are derived
from MABe22, CalMS21 and the ResiFuse poses and remain subject to the terms of those datasets (see `data/README.md`).

## Citation

Paper and citation entry to be added.
